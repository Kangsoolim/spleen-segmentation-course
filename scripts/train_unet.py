"""Config-driven U-Net training or checkpoint verification; no test evaluation."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.unet_pipeline import run_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/unet_runs") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--allow-legacy-amp-fix", action="store_true",
                        help="Explicitly migrate only the approved pre-AMP-fix source hashes; never arbitrary code")
    args = parser.parse_args()
    result = run_training(args.config, args.output_dir, args.resume, args.verify_only, args.allow_legacy_amp_fix)
    print(json.dumps({k: v for k, v in result.items() if k not in ("state", "validation")}, indent=2))


if __name__ == "__main__":
    main()
