"""Patient-volume-level segmentation metrics and three-plane overlay utilities."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from nibabel.orientations import apply_orientation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _unit_to_mm_scale(unit: str | None) -> float | None:
    normalized = (unit or "unknown").lower()
    mapping = {"mm": 1.0, "meter": 1000.0, "micron": 0.001}
    return mapping.get(normalized)


def voxel_volume_mm3(affine: np.ndarray, spatial_unit: str | None) -> float | None:
    """Compute physical voxel volume from the affine determinant, converted to mm^3."""
    scale = _unit_to_mm_scale(spatial_unit)
    if scale is None:
        return None
    voxel_volume_native = abs(float(np.linalg.det(np.asarray(affine, dtype=np.float64)[:3, :3])))
    return voxel_volume_native * (scale ** 3)


def evaluate_arrays(
    gt: np.ndarray,
    pred: np.ndarray,
    affine: np.ndarray,
    spatial_unit: str | None,
    case_id: str = "",
    method: str = "",
) -> dict[str, Any]:
    """Evaluate binary masks on the same original voxel grid; no slice averaging."""
    gt_arr = np.asarray(gt)
    pred_arr = np.asarray(pred)
    if gt_arr.shape != pred_arr.shape:
        raise ValueError(f"GT/pred shape mismatch: {gt_arr.shape} != {pred_arr.shape}")
    if gt_arr.ndim != 3:
        raise ValueError(f"Expected 3D masks, got {gt_arr.ndim}D")
    if not np.isfinite(gt_arr).all() or not np.isfinite(pred_arr).all():
        raise ValueError("Masks contain non-finite values")
    gt_arr = gt_arr > 0
    pred_arr = pred_arr > 0

    tp = int(np.logical_and(gt_arr, pred_arr).sum())
    fp = int(np.logical_and(~gt_arr, pred_arr).sum())
    fn = int(np.logical_and(gt_arr, ~pred_arr).sum())
    tn = int(np.logical_and(~gt_arr, ~pred_arr).sum())

    dice_denom = 2 * tp + fp + fn
    union = tp + fp + fn
    dice = 1.0 if dice_denom == 0 else (2.0 * tp / dice_denom)
    iou = 1.0 if union == 0 else (tp / union)
    precision = 1.0 if (tp + fp) == 0 else (tp / (tp + fp))
    recall = 1.0 if (tp + fn) == 0 else (tp / (tp + fn))

    gt_voxels = int(gt_arr.sum())
    pred_voxels = int(pred_arr.sum())
    voxel_mm3 = voxel_volume_mm3(affine, spatial_unit)
    if voxel_mm3 is not None:
        gt_volume_ml = gt_voxels * voxel_mm3 / 1000.0
        pred_volume_ml = pred_voxels * voxel_mm3 / 1000.0
        volume_error_ml = pred_volume_ml - gt_volume_ml
        abs_volume_error_ml = abs(volume_error_ml)
        rave_percent = (
            None if gt_volume_ml == 0
            else 100.0 * abs_volume_error_ml / gt_volume_ml
        )
    else:
        gt_volume_ml = pred_volume_ml = volume_error_ml = abs_volume_error_ml = rave_percent = None

    return {
        "case_id": case_id,
        "method": method,
        "spatial_unit": spatial_unit or "unknown",
        "shape": "x".join(str(x) for x in gt_arr.shape),
        "tp_voxels": tp,
        "fp_voxels": fp,
        "fn_voxels": fn,
        "tn_voxels": tn,
        "gt_voxels": gt_voxels,
        "pred_voxels": pred_voxels,
        "dice_3d": float(dice),
        "iou_3d": float(iou),
        "precision_3d": float(precision),
        "recall_3d": float(recall),
        "voxel_volume_mm3": voxel_mm3,
        "gt_volume_ml": gt_volume_ml,
        "pred_volume_ml": pred_volume_ml,
        "volume_error_ml_pred_minus_gt": volume_error_ml,
        "abs_volume_error_ml": abs_volume_error_ml,
        "RAVE_percent": rave_percent,
    }


def save_metrics_csv(rows: list[dict[str, Any]], output_csv: str | Path) -> Path:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No metric rows to save")
    fieldnames = list(rows[0].keys())
    with output_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_csv


def evaluate_nifti_files(
    gt_path: str | Path,
    pred_path: str | Path,
    case_id: str = "",
    method: str = "",
) -> dict[str, Any]:
    gt_nii = nib.load(str(gt_path))
    pred_nii = nib.load(str(pred_path))
    if tuple(gt_nii.shape) != tuple(pred_nii.shape):
        raise ValueError(f"GT/pred shape mismatch: {gt_nii.shape} != {pred_nii.shape}")
    if not np.allclose(gt_nii.affine, pred_nii.affine, rtol=1e-5, atol=1e-4):
        raise ValueError("GT/pred affine mismatch; refusing to compare different grids")
    unit = gt_nii.header.get_xyzt_units()[0] or "unknown"
    gt = np.asarray(gt_nii.get_fdata(dtype=np.float32))
    pred = np.asarray(pred_nii.get_fdata(dtype=np.float32))
    if not np.isfinite(gt).all() or not np.isfinite(pred).all():
        raise ValueError("GT/pred contains NaN or Inf")
    return evaluate_arrays(
        gt, pred, gt_nii.affine, unit,
        case_id=case_id or Path(gt_path).name,
        method=method,
    )


def save_three_plane_overlay(
    ct_native: np.ndarray,
    gt_native: np.ndarray,
    pred_native: np.ndarray,
    to_ras_ornt: np.ndarray,
    output_png: str | Path,
    hu_window: tuple[float, float] = (-57.0, 164.0),
    title: str = "",
) -> Path:
    """Save axial/coronal/sagittal overlay with GT green and prediction red."""
    ct = apply_orientation(np.asarray(ct_native), np.asarray(to_ras_ornt))
    gt = apply_orientation((np.asarray(gt_native) > 0).astype(np.uint8), np.asarray(to_ras_ornt))
    pred = apply_orientation((np.asarray(pred_native) > 0).astype(np.uint8), np.asarray(to_ras_ornt))
    if ct.shape != gt.shape or gt.shape != pred.shape:
        raise ValueError(f"Overlay shapes differ: CT={ct.shape}, GT={gt.shape}, pred={pred.shape}")

    coords = np.argwhere(gt > 0)
    if coords.size:
        i, j, k = [int(round(float(v))) for v in np.median(coords, axis=0)]
    else:
        i, j, k = [int(s // 2) for s in gt.shape]

    low, high = hu_window
    planes = [
        ("Axial", ct[:, :, k].T, gt[:, :, k].T, pred[:, :, k].T,
         f"k={k}"),
        ("Coronal", ct[:, j, :].T, gt[:, j, :].T, pred[:, j, :].T,
         f"j={j}"),
        ("Sagittal", ct[i, :, :].T, gt[i, :, :].T, pred[i, :, :].T,
         f"i={i}"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, (name, image, gt_plane, pred_plane, index_label) in zip(axes, planes):
        ax.imshow(image, cmap="gray", vmin=low, vmax=high, origin="lower")
        if np.any(gt_plane):
            ax.contour(gt_plane, levels=[0.5], colors=["lime"], linewidths=1.0)
        if np.any(pred_plane):
            ax.contour(pred_plane, levels=[0.5], colors=["red"], linewidths=1.0)
        ax.set_title(f"{name} ({index_label})")
        ax.set_axis_off()
    fig.suptitle(title or "GT: green | Prediction: red")
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_png
