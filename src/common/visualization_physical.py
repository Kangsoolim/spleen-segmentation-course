"""CT + prediction (+ optional GT) axial/coronal/sagittal physical-aspect overlays.

Displays only: no resampling, changes to source CT, GT, predictions, or NIfTI affine.
All indexing (seed, ROI) is on the closest-RAS canonical [i,j,k] grid.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from nibabel.orientations import apply_orientation


def _spacing_mm(case: dict) -> tuple[float, float, float]:
    unit = (case.get('spatial_unit') or 'unknown').lower()
    scale = {'mm': 1.0, 'meter': 1000.0, 'micron': 0.001}.get(unit)
    if scale is None:
        raise ValueError('Unknown NIfTI spatial unit: cannot render physical spacing')
    affine = np.asarray(case['canonical_affine'], dtype=np.float64)
    space = tuple(float(np.linalg.norm(affine[:3, c]) * scale) for c in range(3))
    if not all(np.isfinite(v) and v > 0 for v in space):
        raise ValueError(f'Invalid voxel spacing: {space}')
    return space


def _plane(ax, image: np.ndarray, pred: np.ndarray, gt: np.ndarray | None,
           spacing_y: float, spacing_x: float, title: str,
           hu_window: tuple[float, float]) -> None:
    rows, cols = image.shape
    extent = (0.0, cols * spacing_x, 0.0, rows * spacing_y)
    ax.imshow(image, origin='lower', cmap='gray', vmin=hu_window[0],
              vmax=hu_window[1], extent=extent, interpolation='nearest',
              aspect='equal')
    ys = (np.arange(rows, dtype=float) + 0.5) * spacing_y
    xs = (np.arange(cols, dtype=float) + 0.5) * spacing_x
    for mask, color in ((gt, 'lime'), (pred, 'red')):
        if mask is not None and np.any(mask) and not np.all(mask):
            ax.contour(xs, ys, mask, levels=(0.5,), colors=(color,), linewidths=1.0)
    ax.set_xlim(0, cols * spacing_x)
    ax.set_ylim(0, rows * spacing_y)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title, fontsize=11)
    ax.axis('off')


def _extended(start: int, stop: int, pad: int, max_length: int) -> slice:
    return slice(max(0, start - pad), min(max_length, stop + pad))


def save_physical_overlay(
    case: dict,
    pred_native: np.ndarray,
    output_path: str | Path,
    seed_ijk: tuple[int, int, int],
    roi_ijk_half_open: tuple[int, int, int, int, int, int],
    gt_native: np.ndarray | None = None,
    hu_window: tuple[float, float] = (-57.0, 164.0),
    title: str = '',
) -> Path:
    """Show full axial view plus ROI-context coronal/sagittal close-ups.

    Slice positions and crop windows depend ONLY on manual CT seed/ROI, not GT.
    GT is used for the green contour only after predictions exist. Spatial
    extents use the canonical affine's mm/voxel for both image and contours.
    """
    ornt = np.asarray(case['to_ras_ornt'], dtype=float)
    ct = apply_orientation(np.asarray(case['ct_original']), ornt)
    pred = apply_orientation(np.asarray(pred_native) > 0, ornt)
    if gt_native is None:
        gt = None
    else:
        gt = apply_orientation(np.asarray(gt_native) > 0, ornt)
    if ct.shape != pred.shape or (gt is not None and gt.shape != ct.shape):
        raise ValueError('CT/GT/prediction shapes do not match')
    si, sj, sk = _spacing_mm(case)
    i, j, k = (int(v) for v in seed_ijk)
    i0, i1, j0, j1, k0, k1 = map(int, roi_ijk_half_open)
    if not (i0 <= i < i1 <= ct.shape[0] and
            j0 <= j < j1 <= ct.shape[1] and
            k0 <= k < k1 <= ct.shape[2]):
        raise ValueError('Invalid seed/ROI for canonical CT')
    ic = _extended(i0, i1, 20, ct.shape[0])
    jc = _extended(j0, j1, 20, ct.shape[1])
    kc = _extended(k0, k1, 4, ct.shape[2])
    planes = [
        ('Axial (full CT, k=%d)' % k,
         ct[:, :, k].T, pred[:, :, k].T,
         None if gt is None else gt[:, :, k].T, sj, si),
        ('Coronal (ROI context, j=%d)' % j,
         ct[ic, j, kc].T, pred[ic, j, kc].T,
         None if gt is None else gt[ic, j, kc].T, sk, si),
        ('Sagittal (ROI context, i=%d)' % i,
         ct[i, jc, kc].T, pred[i, jc, kc].T,
         None if gt is None else gt[i, jc, kc].T, sk, sj),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.7),
                             gridspec_kw={'width_ratios': [1.5, 1, 1]},
                             constrained_layout=True)
    for ax, (name, image, mask, label, spacing_y, spacing_x) in zip(axes, planes):
        _plane(ax, image, mask, label, spacing_y, spacing_x, name, hu_window)
    fig.suptitle(title or ('Prediction red; GT green' if gt is not None else 'Prediction red; CT only'), fontsize=13)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return output
