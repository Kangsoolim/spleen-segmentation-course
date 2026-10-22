"""Shared I/O and spatial transforms for MSD Task09 Spleen.

Conventions
-----------
* NIfTI arrays are kept in their native voxel-axis order when loaded.
* CT slices are reoriented to closest RAS axis order, then sliced along axis 2.
* CT input is clipped to a configurable HU window and normalized to [0, 1].
* Slice resizing uses scipy.ndimage.zoom (order=1 for CT, order=0 for masks).
* Restored masks are returned to the original voxel grid and can be saved with
  the original CT affine. Resizing is lossy; restoring the grid does not recover
  details removed by downsampling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from nibabel.orientations import (
    apply_orientation,
    axcodes2ornt,
    inv_ornt_aff,
    io_orientation,
    ornt_transform,
)
from scipy.ndimage import zoom

SCHEMA_VERSION = "1.0"
AFFINE_RTOL = 1e-5
AFFINE_ATOL = 1e-4


def _case_id(path: str | Path | None) -> str | None:
    if path is None:
        return None
    name = Path(path).name
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    if name.lower().endswith(".nii"):
        return name[:-4]
    return Path(name).stem


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _resize_2d(array: np.ndarray, output_shape: tuple[int, int], order: int) -> np.ndarray:
    """Resize one 2D array to an exact shape using SciPy zoom interpolation."""
    arr = np.asarray(array)
    if arr.ndim != 2:
        raise ValueError(f"_resize_2d expects a 2D array, got shape={arr.shape}")
    out_h, out_w = (int(output_shape[0]), int(output_shape[1]))
    if out_h < 1 or out_w < 1:
        raise ValueError(f"output_shape must be positive, got {output_shape}")
    if arr.shape == (out_h, out_w):
        return arr.copy()

    result = zoom(
        arr,
        zoom=(out_h / arr.shape[0], out_w / arr.shape[1]),
        order=order,
        mode="nearest",
        prefilter=False,
        grid_mode=False,
    )
    # scipy zoom normally returns the requested dimensions. This guard makes
    # the API robust to rounding differences across SciPy versions.
    if result.shape != (out_h, out_w):
        fixed = np.zeros((out_h, out_w), dtype=result.dtype)
        copy_h = min(out_h, result.shape[0])
        copy_w = min(out_w, result.shape[1])
        fixed[:copy_h, :copy_w] = result[:copy_h, :copy_w]
        if copy_h < out_h and copy_h > 0:
            fixed[copy_h:, :copy_w] = result[copy_h - 1:copy_h, :copy_w]
        if copy_w < out_w and copy_w > 0:
            fixed[:, copy_w:] = fixed[:, copy_w - 1:copy_w]
        result = fixed
    return result


def _orientation_info(affine: np.ndarray, shape: tuple[int, int, int]) -> dict[str, Any]:
    original_ornt = io_orientation(affine)
    ras_ornt = axcodes2ornt(("R", "A", "S"))
    to_ras_ornt = ornt_transform(original_ornt, ras_ornt)
    canonical_shape = tuple(int(x) for x in apply_orientation(np.zeros(shape, dtype=np.uint8), to_ras_ornt).shape)
    canonical_affine = np.asarray(affine, dtype=np.float64) @ inv_ornt_aff(to_ras_ornt, shape)
    canonical_ornt = io_orientation(canonical_affine)
    to_original_ornt = ornt_transform(canonical_ornt, original_ornt)
    return {
        "original_ornt": original_ornt,
        "to_ras_ornt": to_ras_ornt,
        "to_original_ornt": to_original_ornt,
        "canonical_shape": canonical_shape,
        "canonical_affine": canonical_affine,
        "original_axcodes": tuple(str(x) for x in nib.aff2axcodes(affine)),
        "canonical_axcodes": tuple(str(x) for x in nib.aff2axcodes(canonical_affine)),
    }


def load_case(
    image_path: str | Path,
    label_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load a CT case and optional binary spleen label, retaining native geometry.

    Parameters
    ----------
    image_path:
        Path to a 3D CT NIfTI (.nii or .nii.gz).
    label_path:
        Optional path to the corresponding 3D label NIfTI.

    Returns
    -------
    dict
        Includes ``ct_original`` (float32), optional ``gt_original`` (uint8),
        original shape/affine/header/forms/zooms/orientation, and source paths.
    """
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"CT file not found: {image_path}")

    image_nii = nib.load(str(image_path))
    if len(image_nii.shape) != 3:
        raise ValueError(f"CT must be 3D, got shape={image_nii.shape}: {image_path}")

    ct = np.asarray(image_nii.get_fdata(dtype=np.float32), dtype=np.float32)
    if not np.isfinite(ct).all():
        raise ValueError(f"CT contains NaN or Inf values: {image_path}")

    orig_shape = tuple(int(x) for x in image_nii.shape)
    orig_affine = np.asarray(image_nii.affine, dtype=np.float64).copy()
    if orig_affine.shape != (4, 4) or not np.isfinite(orig_affine).all():
        raise ValueError(f"Invalid CT affine: {image_path}")

    qform, qform_code = image_nii.get_qform(coded=True)
    sform, sform_code = image_nii.get_sform(coded=True)
    orient = _orientation_info(orig_affine, orig_shape)

    gt = None
    label_nii = None
    if label_path is not None:
        label_path = Path(label_path)
        if not label_path.is_file():
            raise FileNotFoundError(f"GT file not found: {label_path}")
        label_nii = nib.load(str(label_path))
        if len(label_nii.shape) != 3:
            raise ValueError(f"GT must be 3D, got shape={label_nii.shape}: {label_path}")
        if tuple(label_nii.shape) != orig_shape:
            raise ValueError(
                f"CT/GT shape mismatch for {_case_id(image_path)}: "
                f"CT={orig_shape}, GT={label_nii.shape}"
            )
        if not np.allclose(label_nii.affine, orig_affine, rtol=AFFINE_RTOL, atol=AFFINE_ATOL):
            raise ValueError(
                f"CT/GT affine mismatch for {_case_id(image_path)}; "
                f"rtol={AFFINE_RTOL}, atol={AFFINE_ATOL}"
            )
        label_data = np.asarray(label_nii.get_fdata(dtype=np.float32), dtype=np.float32)
        if not np.isfinite(label_data).all():
            raise ValueError(f"GT contains NaN or Inf values: {label_path}")
        rounded = np.rint(label_data)
        if not np.allclose(label_data, rounded, atol=1e-5):
            raise ValueError(f"GT labels are not discrete integers: {label_path}")
        unique = set(int(x) for x in np.unique(rounded))
        if not unique.issubset({0, 1}):
            raise ValueError(f"Expected binary GT values 0/1, found {sorted(unique)}: {label_path}")
        if not np.any(rounded == 1):
            raise ValueError(f"GT contains no foreground spleen voxels: {label_path}")
        gt = rounded.astype(np.uint8)

    return {
        "case_id": _case_id(image_path),
        "image_path": image_path,
        "label_path": Path(label_path) if label_path is not None else None,
        "ct_original": ct,
        "gt_original": gt,
        "orig_shape": orig_shape,
        "orig_affine": orig_affine,
        "orig_header": image_nii.header.copy(),
        "orig_axcodes": orient["original_axcodes"],
        "orig_zooms": tuple(float(x) for x in image_nii.header.get_zooms()[:3]),
        "spatial_unit": image_nii.header.get_xyzt_units()[0] or "unknown",
        "qform": None if qform is None else np.asarray(qform, dtype=np.float64).copy(),
        "qform_code": int(qform_code),
        "sform": None if sform is None else np.asarray(sform, dtype=np.float64).copy(),
        "sform_code": int(sform_code),
        "to_ras_ornt": np.asarray(orient["to_ras_ornt"], dtype=np.float64),
        "to_original_ornt": np.asarray(orient["to_original_ornt"], dtype=np.float64),
        "canonical_shape": orient["canonical_shape"],
        "canonical_affine": orient["canonical_affine"],
        "canonical_axcodes": orient["canonical_axcodes"],
    }


def prepare_slices(
    case: dict[str, Any],
    size: tuple[int, int] = (256, 256),
    hu_window: tuple[float, float] = (-57.0, 164.0),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Prepare CT as float32 ``[N, 1, H, W]`` slices in [0, 1].

    The native volume is reoriented to closest RAS axis order without spacing
    resampling, then sliced along canonical axis 2. Each slice is resized to
    ``size`` with linear interpolation after HU clipping/normalization.
    """
    if "ct_original" not in case or "orig_affine" not in case:
        raise KeyError("case must come from load_case()")
    out_h, out_w = int(size[0]), int(size[1])
    if out_h < 1 or out_w < 1:
        raise ValueError(f"size must be positive, got {size}")
    low, high = float(hu_window[0]), float(hu_window[1])
    if not low < high:
        raise ValueError(f"hu_window must satisfy low < high, got {hu_window}")

    ct_native = np.asarray(case["ct_original"], dtype=np.float32)
    to_ras = np.asarray(case["to_ras_ornt"], dtype=np.float64)
    ct_ras = apply_orientation(ct_native, to_ras)
    clipped = np.clip(ct_ras, low, high)
    normalized = ((clipped - low) / (high - low)).astype(np.float32, copy=False)

    n_slices = int(normalized.shape[2])
    prepared = np.empty((n_slices, 1, out_h, out_w), dtype=np.float32)
    for z in range(n_slices):
        resized = _resize_2d(normalized[:, :, z], (out_h, out_w), order=1)
        prepared[z, 0] = resized.astype(np.float32, copy=False)

    meta = {
        "schema_version": SCHEMA_VERSION,
        "case_id": case.get("case_id"),
        "image_filename": Path(case["image_path"]).name if case.get("image_path") else None,
        "label_filename": Path(case["label_path"]).name if case.get("label_path") else None,
        "original_shape": [int(x) for x in case["orig_shape"]],
        "original_affine": np.asarray(case["orig_affine"], dtype=float).tolist(),
        "original_axcodes": list(case["orig_axcodes"]),
        "original_voxel_spacing_mm_or_header_units": [float(x) for x in case["orig_zooms"]],
        "spatial_unit": case.get("spatial_unit", "unknown"),
        "qform_code": int(case["qform_code"]),
        "sform_code": int(case["sform_code"]),
        "original_qform": None if case.get("qform") is None else np.asarray(case["qform"], dtype=float).tolist(),
        "original_sform": None if case.get("sform") is None else np.asarray(case["sform"], dtype=float).tolist(),
        "canonical_shape": [int(x) for x in ct_ras.shape],
        "canonical_affine": np.asarray(case["canonical_affine"], dtype=float).tolist(),
        "canonical_axcodes": list(case["canonical_axcodes"]),
        "to_ras_ornt": to_ras.tolist(),
        "to_original_ornt": np.asarray(case["to_original_ornt"], dtype=float).tolist(),
        "slice_axis": 2,
        "slice_count": n_slices,
        "prepared_shape": [int(x) for x in prepared.shape],
        "target_slice_size_hw": [out_h, out_w],
        "hu_window": [low, high],
        "normalization": "clip to [low, high], then (x-low)/(high-low) to [0,1]",
        "ct_interpolation": "scipy.ndimage.zoom, order=1 (linear), mode=nearest, grid_mode=False",
        "mask_interpolation": "scipy.ndimage.zoom, order=0 (nearest), mode=nearest, grid_mode=False",
        "orientation_transform": "axis permutation/flip to closest RAS; no voxel-spacing resampling",
        "affine_rtol": AFFINE_RTOL,
        "affine_atol": AFFINE_ATOL,
        "warning": (
            "Resize from native in-plane dimensions to target_slice_size_hw is lossy. "
            "restore_mask restores the voxel grid and geometry, not discarded image detail."
        ),
    }
    return prepared, meta


def restore_mask(
    pred_slices: np.ndarray,
    spatial_meta: dict[str, Any],
    threshold: float = 0.5,
) -> np.ndarray:
    """Restore slice predictions to the original CT voxel grid as uint8 mask.

    Accepted shapes: ``[N,1,H,W]`` or ``[N,H,W]``. Input can be probabilities
    or binary values. Thresholding occurs before nearest-neighbor resizing.
    The number of slices must match the canonical volume's axis-2 length.
    """
    pred = np.asarray(pred_slices)
    if pred.ndim == 4:
        if pred.shape[1] != 1:
            raise ValueError(f"Expected one prediction channel, got shape={pred.shape}")
        pred = pred[:, 0]
    elif pred.ndim != 3:
        raise ValueError(f"Expected [N,H,W] or [N,1,H,W], got shape={pred.shape}")

    original_shape = tuple(int(x) for x in spatial_meta["original_shape"])
    canonical_shape = tuple(int(x) for x in spatial_meta["canonical_shape"])
    expected_n = canonical_shape[2]
    if pred.shape[0] != expected_n:
        raise ValueError(
            f"Prediction has {pred.shape[0]} slices, expected {expected_n} "
            f"from canonical_shape={canonical_shape}"
        )
    if not np.isfinite(pred).all():
        raise ValueError("Prediction contains NaN or Inf values")

    target_hw = canonical_shape[:2]
    canonical_mask = np.zeros(canonical_shape, dtype=np.uint8)
    for z in range(expected_n):
        binary = (pred[z] >= float(threshold)).astype(np.uint8)
        restored_slice = _resize_2d(binary, target_hw, order=0)
        canonical_mask[:, :, z] = (restored_slice > 0).astype(np.uint8)

    to_original = np.asarray(spatial_meta["to_original_ornt"], dtype=np.float64)
    mask_original = apply_orientation(canonical_mask, to_original).astype(np.uint8, copy=False)
    if tuple(mask_original.shape) != original_shape:
        raise RuntimeError(
            f"Restored mask shape mismatch: {mask_original.shape} != {original_shape}"
        )
    values = set(int(x) for x in np.unique(mask_original))
    if not values.issubset({0, 1}):
        raise RuntimeError(f"Restored mask is not binary: {sorted(values)}")
    return mask_original


def save_original_mask(
    mask_original: np.ndarray,
    case: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Save a binary mask with the original CT shape and effective affine.

    The saved file is reopened and validated before this function returns.
    If the source header has no coded spatial form, an sform is written using
    the original effective affine so the saved mask retains the same geometry.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.asarray(mask_original)
    expected_shape = tuple(int(x) for x in case["orig_shape"])
    if tuple(mask.shape) != expected_shape:
        raise ValueError(f"Mask shape {mask.shape} does not match CT {expected_shape}")
    if not np.isfinite(mask).all():
        raise ValueError("Mask contains NaN or Inf values")
    mask = (mask > 0).astype(np.uint8)

    affine = np.asarray(case["orig_affine"], dtype=np.float64)
    header = case["orig_header"].copy()
    header.set_data_dtype(np.uint8)
    out_nii = nib.Nifti1Image(mask, affine, header=header)

    qform = case.get("qform")
    qcode = int(case.get("qform_code", 0))
    sform = case.get("sform")
    scode = int(case.get("sform_code", 0))
    if qform is not None and qcode > 0:
        out_nii.set_qform(np.asarray(qform), code=qcode)
    else:
        out_nii.set_qform(None, code=0)
    if sform is not None and scode > 0:
        out_nii.set_sform(np.asarray(sform), code=scode)
    else:
        # Ensure the effective affine is explicitly encoded in the output.
        fallback_code = qcode if qcode > 0 else 1
        out_nii.set_sform(affine, code=fallback_code)

    nib.save(out_nii, str(output_path))
    check = nib.load(str(output_path))
    if tuple(check.shape) != expected_shape:
        raise RuntimeError(f"Saved mask shape check failed: {check.shape} != {expected_shape}")
    if not np.allclose(check.affine, affine, rtol=0.0, atol=1e-5):
        raise RuntimeError(
            "Saved mask affine does not match source CT affine. "
            f"max_abs_diff={float(np.max(np.abs(check.affine - affine)))}"
        )
    check_values = set(int(x) for x in np.unique(check.get_fdata(dtype=np.float32)))
    if not check_values.issubset({0, 1}):
        raise RuntimeError(f"Saved mask has unexpected values: {sorted(check_values)}")
    return output_path


def save_spatial_json(spatial_meta: dict[str, Any], output_path: str | Path) -> Path:
    """Write the spatial metadata to a UTF-8 JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _to_jsonable(spatial_meta)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(output_path)
    return output_path
