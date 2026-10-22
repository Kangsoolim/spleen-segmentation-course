"""Conservative, optional CT-only region-grow mask postprocessing.

No GT, metric or trained model is used. Operates on the closest-RAS canonical
[i,j,k] grid *within* a manually selected half-open 3D ROI. A physically sized
in-plane opening may cut narrow bridges, then only the 3D face-connected region
containing the original seed survives. Morphological closing is optional.

This is exploratory: no method guarantees that all erroneous tissue is removed.
"""
from __future__ import annotations

from typing import Any
import numpy as np
from scipy import ndimage as ndi
from nibabel.orientations import apply_orientation


def physical_disk(radius_mm: float, spacing_ij_mm: tuple[float, float]) -> np.ndarray:
    """Return a centered 2D elliptical footprint with physical radius radius_mm."""
    if radius_mm <= 0:
        raise ValueError("radius_mm must be > 0")
    si, sj = map(float, spacing_ij_mm)
    if si <= 0 or sj <= 0 or not np.isfinite([si, sj]).all():
        raise ValueError("Invalid in-plane voxel spacing")
    ni, nj = int(np.ceil(radius_mm / si)), int(np.ceil(radius_mm / sj))
    x, y = np.ogrid[-ni:ni + 1, -nj:nj + 1]
    footprint = ((x * si) ** 2 + (y * sj) ** 2) <= radius_mm ** 2 + 1e-10
    return footprint.astype(bool)


def postprocess_region_native(
    pred_native: np.ndarray,
    case: dict[str, Any],
    seed_ijk: tuple[int, int, int],
    roi_ijk_half_open: tuple[int, int, int, int, int, int],
    opening_radius_mm: float = 0.0,
    closing_radius_mm: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return native-geometry uint8 mask and logging record.

    The original seed and ROI are closest-RAS canonical voxel indices, not
    native voxel indices and not world coordinates. Opening and closing are
    performed independently on in-plane slices (not across thick-slice z).

    If an opening destroys the seed's connected component, the operation
    returns the *unchanged* source mask with an explicit skipped warning.
    """
    if opening_radius_mm < 0 or closing_radius_mm < 0:
        raise ValueError("Morphological radii cannot be negative")
    pred_native = np.asarray(pred_native)
    shape = tuple(case["orig_shape"])
    if tuple(pred_native.shape) != shape:
        raise ValueError(f"Prediction shape {pred_native.shape} != CT shape {shape}")
    if not np.isfinite(pred_native).all():
        raise ValueError("Prediction contains NaN or Inf")
    if not np.all(np.isin(np.unique(pred_native), [0, 1])):
        raise ValueError("Prediction must be a binary 0/1 mask")

    original = pred_native.astype(bool, copy=False)
    to_ras = np.asarray(case["to_ras_ornt"], dtype=float)
    canonical = apply_orientation(original, to_ras)
    canonical_shape = tuple(canonical.shape)
    i0, i1, j0, j1, k0, k1 = map(int, roi_ijk_half_open)
    si, sj, sk = map(int, seed_ijk)
    if not (0 <= i0 < i1 <= canonical_shape[0] and
            0 <= j0 < j1 <= canonical_shape[1] and
            0 <= k0 < k1 <= canonical_shape[2]):
        raise ValueError(f"ROI invalid for canonical shape {canonical_shape}")
    if not (i0 <= si < i1 and j0 <= sj < j1 and k0 <= sk < k1):
        raise ValueError("Seed is not inside ROI")
    crop = canonical[i0:i1, j0:j1, k0:k1].copy()
    seed_local = si - i0, sj - j0, sk - k0
    if not crop[seed_local]:
        raise ValueError("Original seed is not in original predicted mask")

    affine = np.asarray(case["canonical_affine"], dtype=float)
    space_unit = case.get("spatial_unit")
    factor = {"mm": 1.0, "meter": 1000.0, "micron": 0.001}.get(space_unit)
    if factor is None:
        raise ValueError("Unknown NIfTI spatial units; cannot apply mm morphology")
    ij_spacing = (
        float(np.linalg.norm(affine[:3, 0]) * factor),
        float(np.linalg.norm(affine[:3, 1]) * factor),
    )
    result = crop.copy()
    for name, radius, operator in (
        ("opening", opening_radius_mm, ndi.binary_opening),
        ("closing", closing_radius_mm, ndi.binary_closing),
    ):
        if radius > 0:
            disk = physical_disk(radius, ij_spacing)
            # If footprint degenerates to a single pixel, no visible effect.
            for k in range(result.shape[2]):
                result[:, :, k] = operator(
                    result[:, :, k], structure=disk, border_value=0
                )

    # 6-connected in [i,j,k]. No floating GT or anatomical priors used.
    structure = ndi.generate_binary_structure(3, 1)
    labels, component_count = ndi.label(result, structure=structure)
    seeded_label = int(labels[seed_local])
    record: dict[str, Any] = {
        "uses_ground_truth": False,
        "coordinate_system": "closest-RAS canonical voxel indices",
        "seed_ijk": list(map(int, seed_ijk)),
        "roi_ijk_half_open": list(map(int, roi_ijk_half_open)),
        "opening_radius_mm": float(opening_radius_mm),
        "closing_radius_mm": float(closing_radius_mm),
        "inplane_spacing_mm": list(ij_spacing),
        "morphology_applied_slice_wise": True,
        "seeded_components_before_selection": int(component_count),
        "raw_mask_voxels": int(original.sum()),
    }
    if seeded_label == 0:
        record["status"] = "SKIPPED_SEED_REMOVED"
        record["warning"] = "Morphology removed seed region; original mask retained, NOT a successful edit"
        record["output_mask_voxels"] = int(original.sum())
        return original.astype(np.uint8), record

    selected_crop = labels == seeded_label
    final_canonical = np.zeros(canonical_shape, dtype=np.uint8)
    final_canonical[i0:i1, j0:j1, k0:k1] = selected_crop.astype(np.uint8)
    final = apply_orientation(
        final_canonical, np.asarray(case["to_original_ornt"], dtype=float)
    ).astype(np.uint8, copy=False)
    if tuple(final.shape) != shape:
        raise RuntimeError("Unexpected native shape after inverse orientation")
    record["status"] = "APPLIED"
    record["output_mask_voxels"] = int(final.sum())
    record["change_voxels"] = int(np.count_nonzero(original ^ final.astype(bool)))
    record["note"] = ("No GT used. Opening may cut narrow connections, but also "
                      "remove true spleen; test only on train/val first.")
    return final, record
