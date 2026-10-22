#!/usr/bin/env python3
"""Semi-automatic spleen baseline V2: ConfidenceConnected + optional 1mm opening.

Segmentation loads ONLY CT, seed and ROI. GT is not loaded until after both
predictions have been saved, and only when --evaluate is explicitly requested
for a training or validation case. Never optimize from test labels.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from nibabel.orientations import apply_orientation

from src.common.io_spatial import load_case, prepare_slices, save_original_mask, save_spatial_json
from src.common.traditional_segmentation import confidence_connected_roi
from src.common.region_postprocess import postprocess_region_native
from src.common.evaluation import evaluate_arrays, save_metrics_csv
from src.common.visualization_physical import save_physical_overlay


def find_nii(folder: Path, case_id: str) -> Path:
    for suffix in ('.nii.gz', '.nii'):
        candidate = folder / f'{case_id}{suffix}'
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f'No {case_id}.nii.gz or .nii in {folder}')


def split_status(path: Path, case_id: str) -> tuple[str, bool, str]:
    if not path.is_file():
        return 'unknown', False, 'missing'
    split = json.loads(path.read_text(encoding='utf-8-sig'))
    matches = [s for s in ('train', 'val', 'test') if case_id in split.get(s, [])]
    if len(matches) > 1:
        raise ValueError(f'Case {case_id} is in multiple splits: {matches}')
    return (matches[0] if matches else 'unknown',
            split.get('is_frozen') is True,
            str(split.get('status', 'unknown')))


def touching_roi_faces(mask_native: np.ndarray, case: dict, roi: tuple[int, ...]) -> dict[str, bool]:
    # CT-only quality-control flag; touching ROI can indicate truncation.
    m = apply_orientation(np.asarray(mask_native) > 0, np.asarray(case['to_ras_ornt']))
    i0, i1, j0, j1, k0, k1 = roi
    return {
        'i_low': bool(m[i0, j0:j1, k0:k1].any()),
        'i_high': bool(m[i1 - 1, j0:j1, k0:k1].any()),
        'j_low': bool(m[i0:i1, j0, k0:k1].any()),
        'j_high': bool(m[i0:i1, j1 - 1, k0:k1].any()),
        'k_low': bool(m[i0:i1, j0:j1, k0].any()),
        'k_high': bool(m[i0:i1, j0:j1, k1 - 1].any()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--case-id', required=True)
    ap.add_argument('--seed-roi-json', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, default=Path('outputs/traditional_v2_open1mm'))
    ap.add_argument('--split-json', type=Path, default=Path('data/split.json'))
    ap.add_argument('--radius', type=int, default=1)
    ap.add_argument('--iterations', type=int, default=2)
    ap.add_argument('--multiplier', type=float, default=2.0)
    ap.add_argument('--opening-mm', type=float, default=1.0)
    ap.add_argument('--closing-mm', type=float, default=0.0)
    ap.add_argument('--evaluate', action='store_true',
                    help='Only train/val: load GT AFTER segmentation to generate comparison metrics/green contour')
    ap.add_argument('--overwrite', action='store_true',
                    help='Overwrite outputs of this V2 case only, never legacy outputs')
    args = ap.parse_args()
    if args.opening_mm < 0 or args.closing_mm < 0:
        raise ValueError('Morphological radii must be nonnegative')
    role, frozen, split_state = split_status(args.split_json, args.case_id)
    if args.evaluate and role not in ('train', 'val'):
        raise ValueError(f'--evaluate is restricted to train/val, not role={role}. '
                         'Freeze the evaluation protocol before final test assessment.')
    out_dir = args.output_dir / args.case_id
    if out_dir.exists() and not args.overwrite:
        raise FileExistsError(f'{out_dir} already exists. Use a fresh --output-dir, '
                              'or --overwrite if you intend to replace this V2 run.')
    if args.overwrite and args.output_dir.resolve() in {
            (ROOT / 'outputs' / 'traditional').resolve(),
            (ROOT / 'outputs' / 'traditional_c20').resolve(),
            (ROOT / 'outputs' / 'traditional_c225').resolve()}:
        raise ValueError('Refusing to overwrite previously generated baseline directories')

    sel = json.loads(args.seed_roi_json.read_text(encoding='utf-8-sig'))
    if sel.get('case_id') != args.case_id:
        raise ValueError('Seed/ROI case ID differs from requested case')
    seed = tuple(int(x) for x in sel['seed_ijk'])
    roi = tuple(int(x) for x in sel['roi_ijk_half_open'])
    ct_path = find_nii(args.data_root / 'imagesTr', args.case_id)

    # IMPORTANT: segmentation sees CT only, never GT.
    case = load_case(ct_path, label_path=None)
    raw_mask, raw_record = confidence_connected_roi(
        case, seed, roi,
        iterations=args.iterations,
        multiplier=args.multiplier,
        initial_neighborhood_radius=args.radius,
    )
    if args.opening_mm == 0.0 and args.closing_mm == 0.0:
        improved_mask = raw_mask.copy()
        post_record = {'status': 'DISABLED_RAW', 'uses_ground_truth': False,
                       'opening_radius_mm': 0.0, 'closing_radius_mm': 0.0}
    else:
        improved_mask, post_record = postprocess_region_native(
            raw_mask, case, seed, roi, args.opening_mm, args.closing_mm
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = save_original_mask(raw_mask, case, out_dir / 'raw_mask.nii.gz')

    if post_record['status'] == 'SKIPPED_SEED_REMOVED':
        (out_dir / 'failure_postprocess.json').write_text(
            json.dumps({'case_id': args.case_id, 'postprocess': post_record,
                        'raw_path': str(raw_path)}, indent=2, ensure_ascii=False),
            encoding='utf-8')
        raise RuntimeError('Postprocessing removed the seed-containing component. '
                           'No improved mask/metrics were produced. Raw mask saved; '
                           'inspect failure_postprocess.json.')
    improved_path = save_original_mask(improved_mask, case, out_dir / 'traditional_mask.nii.gz')
    # Preserve the shared U-Net I/O geometry contract and metadata without using GT.
    _, spatial_meta = prepare_slices(case)
    save_spatial_json(spatial_meta, out_dir / 'spatial.json')

    raw_flags = touching_roi_faces(raw_mask, case, roi)
    processed_flags = touching_roi_faces(improved_mask, case, roi)
    record = {
        'schema_version': '2.0',
        'method': 'SimpleITK ConfidenceConnected ROI + optional seed-connected physical in-plane morphology',
        'case_id': args.case_id,
        'split_role': role,
        'split_is_frozen': frozen,
        'split_status': split_state,
        'selection_source': sel.get('selection_source', 'unspecified'),
        'seed_coordinate_convention': 'closest-RAS canonical [i,j,k]',
        'seed_ijk': list(seed),
        'roi_ijk_half_open': list(roi),
        'base_segmentation': raw_record,
        'postprocess': post_record,
        'raw_mask_path': str(raw_path),
        'improved_mask_path': str(improved_path),
        'raw_roi_boundary_contact': raw_flags,
        'improved_roi_boundary_contact': processed_flags,
        'boundary_warning': ('At least one prediction touches the ROI boundary: '
                             'consider whether ROI truncation is possible. CT-only QC; '
                             'do not adjust ROI based on test ground truth.'),
        'segmentation_uses_gt': False,
        'evaluation_requested': bool(args.evaluate),
        'not_final_benchmark': not (role == 'test' and frozen),
    }

    gt = None
    if args.evaluate:
        # GT is loaded ONLY now, after both predictions are finalized/saved.
        gt_path = find_nii(args.data_root / 'labelsTr', args.case_id)
        labelled_case = load_case(ct_path, gt_path)  # verifies shape/affine/labels
        gt = labelled_case['gt_original']
        rows = []
        for name, mask in (('raw', raw_mask), ('postprocessed', improved_mask)):
            metric = evaluate_arrays(gt, mask, case['orig_affine'],
                                     case['spatial_unit'], args.case_id,
                                     f'ConfidenceConnected_ROI_{name}')
            metric.update({'split_role': role, 'is_final_test_benchmark': False,
                           'opening_mm': args.opening_mm if name == 'postprocessed' else 0.0,
                           'closing_mm': args.closing_mm if name == 'postprocessed' else 0.0,
                           'postprocess_status': post_record['status'] if name == 'postprocessed' else 'ORIGINAL_UNMODIFIED'})
            rows.append(metric)
        save_metrics_csv(rows, out_dir / 'metrics.csv')
        for row in rows:
            print(f"{row['method']}: Dice={row['dice_3d']:.4f} "
                  f"IoU={row['iou_3d']:.4f} FP={row['fp_voxels']} FN={row['fn_voxels']} "
                  f"RAVE={row['RAVE_percent']:.2f}%")
    save_physical_overlay(
        case, improved_mask, out_dir / 'overlay_physical.png', seed, roi,
        gt_native=gt,
        title=f'{args.case_id} | V2 opening {args.opening_mm:g} mm | '
              + ('GT green, prediction red' if gt is not None else 'prediction red (no GT)'),
    )
    (out_dir / 'traditional_run_record.json').write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    print('V2 PIPELINE COMPLETED (CT-only segmentation).')
    print(f'Postprocessing status: {post_record["status"]}')
    print(f'Raw mask voxels: {np.count_nonzero(raw_mask)}; V2 mask voxels: {np.count_nonzero(improved_mask)}')
    print('ROI boundary contact (processed): ' + json.dumps(processed_flags))
    if any(processed_flags.values()):
        print('WARNING: Mask touches ROI boundary; CT-only ROI review recommended before fixing protocol.')
    print(f'Outputs: {out_dir}')
    print('Files: raw_mask.nii.gz, traditional_mask.nii.gz, spatial.json, '
          'traditional_run_record.json, overlay_physical.png'
          + (', metrics.csv' if args.evaluate else ''))
    print('NOTE: This run is development/protocol validation, not final held-out testing.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(1)
