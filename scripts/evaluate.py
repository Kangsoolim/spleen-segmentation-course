#!/usr/bin/env python3
"""Evaluate two binary NIfTI masks on the same native voxel grid."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.evaluation import evaluate_nifti_files, save_metrics_csv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True, help="Ground-truth NIfTI mask")
    parser.add_argument("--pred", type=Path, required=True, help="Predicted NIfTI mask")
    parser.add_argument("--case-id", default="")
    parser.add_argument("--method", default="unspecified")
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    metrics = evaluate_nifti_files(args.gt, args.pred, args.case_id, args.method)
    save_metrics_csv([metrics], args.output_csv)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nCreated: {args.output_csv}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
