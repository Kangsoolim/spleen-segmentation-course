"""Automatic full-volume CT-only prediction using a full-pipeline checkpoint."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.models.unet import build_unet
from src.training import set_seed
from src.unet_pipeline import predict_native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, choices=(1, 2, 4), default=4)
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if saved.get("schema_version") not in (1, 2):
        raise ValueError("Expected full-pipeline checkpoint")
    if saved.get("checkpoint_boundary", "completed_epoch") != "completed_epoch":
        raise ValueError("Partial diagnostic checkpoint is not a completed training model")
    set_seed(saved["config"]["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = build_unet().to(args.device)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    path, check = predict_native(model, args.image, args.output_dir, args.device,
                                 args.batch_size, saved["config"]["threshold"])
    print(json.dumps({"prediction": str(path), "checks": check, "GT_used": False}, indent=2))


if __name__ == "__main__":
    main()
