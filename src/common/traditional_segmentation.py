"""Semi-automatic 3D ConfidenceConnected baseline constrained by a manual ROI.

Coordinate contract:
- CT arrays are native NiBabel voxel order [i, j, k].
- The algorithm operates on the closest-RAS canonical voxel array [i, j, k].
- Manual seed is (i, j, k); ROI bounds are half-open:
  i in [i0, i1), j in [j0, j1), k in [k0, k1).
- SimpleITK image arrays are [z, y, x], so conversion uses transpose (2, 1, 0).
- The final mask is inverse-oriented to the original CT grid before saving.
"""
from __future__ import annotations

from typing import Any
import time

import numpy as np
import SimpleITK as sitk
from nibabel.orientations import apply_orientation


def confidence_connected_roi(
    case: dict[str, Any],
    seed_ijk: tuple[int, int, int],
    roi_ijk: tuple[int, int, int, int, int, int],
    iterations: int = 2,
    multiplier: float = 2.5,
    initial_neighborhood_radius: int = 2,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a binary mask on the original CT grid plus a run record.

    Parameters are deliberately explicit because this is a semi-automatic
    baseline. The function never reads or uses ``case['gt_original']``.
    """
    if "ct_original" not in case or "to_ras_ornt" not in case:
        raise KeyError("case must be produced by load_case()")
    if iterations < 0:
        raise ValueError("iterations must be >= 0")
    if multiplier <= 0:
        raise ValueError("multiplier must be > 0")
    if initial_neighborhood_radius < 1:
        raise ValueError("initial_neighborhood_radius must be >= 1")

    ct_native = np.asarray(case["ct_original"], dtype=np.float32)
    ct_ras = apply_orientation(
        ct_native, np.asarray(case["to_ras_ornt"], dtype=np.float64)
    )
    shape = tuple(int(v) for v in ct_ras.shape)

    if len(seed_ijk) != 3 or len(roi_ijk) != 6:
        raise ValueError("seed_ijk must have 3 values; roi_ijk must have 6 values")
    seed = tuple(int(v) for v in seed_ijk)
    i0, i1, j0, j1, k0, k1 = (int(v) for v in roi_ijk)

    if not (0 <= i0 < i1 <= shape[0]
            and 0 <= j0 < j1 <= shape[1]
            and 0 <= k0 < k1 <= shape[2]):
        raise ValueError(
            f"ROI {roi_ijk} must be valid half-open bounds within canonical shape {shape}"
        )
    if not (i0 <= seed[0] < i1
            and j0 <= seed[1] < j1
            and k0 <= seed[2] < k1):
        raise ValueError(f"Seed {seed} must lie inside ROI {roi_ijk}")
    if not all(0 <= seed[d] < shape[d] for d in range(3)):
        raise ValueError(f"Seed {seed} is outside canonical shape {shape}")

    crop = np.ascontiguousarray(ct_ras[i0:i1, j0:j1, k0:k1], dtype=np.float32)
    if not np.isfinite(crop).all():
        raise ValueError("ROI CT contains NaN or Inf")

    # SimpleITK expects NumPy [z, y, x], while our canonical contract is [i,j,k].
    sitk_array_zyx = np.transpose(crop, (2, 1, 0))
    sitk_image = sitk.GetImageFromArray(sitk_array_zyx, isVector=False)
    seed_sitk_xyz = (seed[0] - i0, seed[1] - j0, seed[2] - k0)

    start = time.perf_counter()
    filt = sitk.ConfidenceConnectedImageFilter()
    filt.SetSeedList([tuple(int(v) for v in seed_sitk_xyz)])
    filt.SetNumberOfIterations(int(iterations))
    filt.SetMultiplier(float(multiplier))
    filt.SetInitialNeighborhoodRadius(int(initial_neighborhood_radius))
    filt.SetReplaceValue(1)
    # Keep the default face-connected neighborhood; ROI constrains leakage.
    result_sitk = filt.Execute(sitk_image)
    elapsed = time.perf_counter() - start

    result_zyx = sitk.GetArrayFromImage(result_sitk).astype(np.uint8, copy=False)
    result_ijk = np.transpose(result_zyx, (2, 1, 0))
    if tuple(result_ijk.shape) != tuple(crop.shape):
        raise RuntimeError(
            f"SimpleITK crop output shape mismatch: {result_ijk.shape} != {crop.shape}"
        )

    canonical_mask = np.zeros(shape, dtype=np.uint8)
    canonical_mask[i0:i1, j0:j1, k0:k1] = (result_ijk > 0).astype(np.uint8)
    to_original = np.asarray(case["to_original_ornt"], dtype=np.float64)
    original_mask = apply_orientation(canonical_mask, to_original).astype(np.uint8, copy=False)
    if tuple(original_mask.shape) != tuple(case["orig_shape"]):
        raise RuntimeError(
            f"Original mask shape mismatch: {original_mask.shape} != {case['orig_shape']}"
        )

    record = {
        "method": "SimpleITK ConfidenceConnected within manually defined 3D ROI",
        "case_id": case.get("case_id"),
        "coordinate_convention": (
            "seed=(i,j,k) in closest-RAS canonical voxel indices; ROI bounds are "
            "half-open [i0,i1), [j0,j1), [k0,k1); not world-coordinate millimeters"
        ),
        "canonical_shape": list(shape),
        "seed_ijk": list(seed),
        "roi_ijk_half_open": [i0, i1, j0, j1, k0, k1],
        "sitk_seed_xyz_within_roi": list(seed_sitk_xyz),
        "iterations": int(iterations),
        "multiplier": float(multiplier),
        "initial_neighborhood_radius": int(initial_neighborhood_radius),
        "roi_shape_ijk": list(crop.shape),
        "roi_voxels": int(crop.size),
        "foreground_voxels_canonical": int(np.count_nonzero(canonical_mask)),
        "elapsed_seconds": round(float(elapsed), 6),
        "uses_ground_truth": False,
        "note": (
            "Semi-automatic baseline: seed and ROI are manually selected from CT. "
            "The ROI limits leakage but does not guarantee anatomical separation."
        ),
    }
    return original_mask, record
