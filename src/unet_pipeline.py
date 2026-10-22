"""Patient-volume validation and reproducible training orchestration.

Spatial processing belongs exclusively to common.io_spatial and unet_data.
Checkpoints represent completed epochs; an interrupted partial epoch is replayed
from the preceding checkpoint. No held-out test images are opened here.
"""
import csv
import hashlib
import json
import logging
import math
import platform
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import monai
import nibabel as nib
import numpy as np
import scipy
import torch
from torch.utils.data import DataLoader

from src.common.evaluation import evaluate_nifti_files, save_metrics_csv
from src.common.io_spatial import load_case, prepare_slices, restore_mask, save_original_mask, save_spatial_json
from src.models.unet import build_unet
from src.training import (AMP_MAX_CONSECUTIVE_SKIPS, AMP_POLICY_VERSION, DiceBCELoss,
                          load_config, set_seed, train, validate_step_counters)
from src.unet_data import PatientSliceDataset, read_manifest


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def environment():
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "monai": monai.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
            "nibabel": nib.__version__, "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def capture_rng():
    np_state = np.random.get_state()
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy": {"name": np_state[0], "keys": torch.tensor(np_state[1].astype(np.int64)),
                      "pos": np_state[2], "gauss": np_state[3], "cached": np_state[4]}}


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    s = state["numpy"]
    np.random.set_state((s["name"], s["keys"].cpu().numpy().astype(np.uint32), s["pos"], s["gauss"], s["cached"]))


def training_signature(config, split):
    # Epoch budget, output paths, and audit timestamps may change on resume;
    # patient membership and every learning/data/preprocessing setting may not.
    keys = ("mode", "seed", "batch_size", "learning_rate", "dice_weight", "bce_weight",
            "amp", "train_case_ids", "val_case_ids", "max_train_batches", "threshold",
            "patience", "min_delta")
    settings = {k: config[k] for k in keys}
    # Preserve historical signatures for the default patient-grouped schedule.
    if config.get("batch_order", "patient_grouped") != "patient_grouped":
        settings.update(batch_order=config["batch_order"], schedule_seed=config["schedule_seed"])
    return json_hash({"settings": settings,
                      "split": {k: split[k] for k in ("train", "val", "test")},
                      "architecture": "MONAI_2D_UNet_16_32_64_128_256_res2",
                      "preprocessing": "closestRAS_HU-57_164_256_linear_CT_nearest_GT_v1"})


SOURCE_PATHS = ("src/unet_pipeline.py", "src/unet_data.py", "src/training.py",
                "src/models/unet.py", "src/common/io_spatial.py", "src/common/evaluation.py",
                "scripts/train_unet.py", "src/unet_schedule.py")
LEGACY_AMP_SOURCES = {
    "src/training.py": "38f0851e2d9e1f2f65a1aed2161975da7dd6093e1eba7f8062e9519aaf7a13c3",
    "src/unet_pipeline.py": "e38ed54b8996f612fd1b01fcebc2c6be550579f00ba2e6ea6e7859bd7f16ba23",
    "src/unet_data.py": "4690a6f37ddebb6d2c7b5f4d649c845bb59d4097abb5f321248d8f50e0a4b26e",
    "src/models/unet.py": "73ee43a87cd166766e0ff1f5edee190b76bb9f0ef1ddc952cbded1cfec688e16",
    "src/common/io_spatial.py": "dd83764778c7f73c01f4cfdd2546d4aa506545debc82a2666aad22b3a492a7bb",
    "src/common/evaluation.py": "289ac2f4e91c9c0028773609329f24b4f2b7967b4d5ccee3f66dd66aabce9132"}
COMPATIBILITY_MANIFEST = Path("configs/checkpoint_compat_amp_v2.json")
COUNTER_KEYS = ("global_step", "optimizer_updates", "batch_attempts", "amp_skipped_steps", "consecutive_amp_skips")


def source_fingerprints():
    return {p: file_sha256(p) for p in SOURCE_PATHS}


def precision_policy():
    return {"version": AMP_POLICY_VERSION, "max_consecutive_amp_skips": AMP_MAX_CONSECUTIVE_SKIPS,
            "global_step": "actual_optimizer_updates", "epoch_loss": "all_finite_forward_attempts_sample_weighted"}


def check_state_counters(state):
    if any(k not in state for k in COUNTER_KEYS) or state["optimizer_updates"] != state["global_step"]:
        raise ValueError("Missing/inconsistent checkpoint update counters")
    validate_step_counters(state["global_step"], state["batch_attempts"],
                           state["amp_skipped_steps"], state["consecutive_amp_skips"])


def save_run_checkpoint(path, model, optimizer, scaler, state, config, provenance,
                        boundary="completed_epoch"):
    check_state_counters(state)
    if provenance.get("source_sha256") != source_fingerprints():
        raise ValueError("Checkpoint provenance does not match executing source files")
    if boundary not in ("completed_epoch", "diagnostic_partial_epoch"):
        raise ValueError("Unknown checkpoint boundary")
    payload = {"schema_version": 2, "precision_policy": precision_policy(), "checkpoint_boundary": boundary,
               "model_state_dict": model.state_dict(),
               "optimizer_state_dict": optimizer.state_dict(), "scaler_state_dict": scaler.state_dict(),
               "state": state, "config": config, "provenance": provenance, "rng": capture_rng()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def read_run_checkpoint(path, expected_signature, expected_cases_sha, allow_legacy_amp_fix=False,
                        expected_boundary="completed_epoch"):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("schema_version") not in (1, 2):
        raise ValueError("Not a full-pipeline checkpoint (old smoke weights cannot resume this run)")
    if saved["provenance"]["training_signature"] != expected_signature:
        raise ValueError("Checkpoint training settings/patient split mismatch")
    if saved["provenance"]["cases_sha256"] != expected_cases_sha:
        raise ValueError("Checkpoint cases.csv mismatch")
    if saved["provenance"].get("environment") != environment():
        raise ValueError("Resume environment differs; exact continuation is not established")
    if saved.get("checkpoint_boundary", "completed_epoch") != expected_boundary:
        raise ValueError("Checkpoint boundary mismatch: partial diagnostics cannot resume a full epoch")
    current = source_fingerprints()
    if saved["schema_version"] == 1:
        if not allow_legacy_amp_fix or saved["provenance"].get("source_sha256") != LEGACY_AMP_SOURCES:
            raise ValueError("Legacy source mismatch or explicit allow-legacy-amp-fix required")
        manifest = json.loads(COMPATIBILITY_MANIFEST.read_text(encoding="utf-8"))
        if (manifest.get("migration_id") != "unet_amp_native_skip_v2_20261020"
                or manifest.get("from_source_sha256") != LEGACY_AMP_SOURCES
                or manifest.get("to_source_sha256") != current):
            raise ValueError("AMP migration target source hashes do not match approved manifest")
        for name in LEGACY_AMP_SOURCES.keys() - {"src/training.py", "src/unet_pipeline.py"}:
            if current[name] != LEGACY_AMP_SOURCES[name]:
                raise ValueError("AMP-only migration cannot change data/model/evaluation source")
        state = saved["state"]
        if any(k in state for k in COUNTER_KEYS[1:]):
            raise ValueError("Legacy checkpoint unexpectedly contains new counters")
        state.update(optimizer_updates=state["global_step"], batch_attempts=state["global_step"],
                     amp_skipped_steps=0, consecutive_amp_skips=0)
        saved["migration"] = {"id": manifest["migration_id"], "from_schema": 1, "to_schema": 2,
            "from_checkpoint_sha256": file_sha256(path), "from_source_sha256": LEGACY_AMP_SOURCES,
            "to_source_sha256": current, "manifest_sha256": file_sha256(COMPATIBILITY_MANIFEST),
            "defaults": {k: state[k] for k in COUNTER_KEYS},
            "reason": "Legacy code aborted on overflow; completed epochs had no skipped updates",
            "claim": "Modified-code continuation, not strict reproduction of the legacy execution"}
    else:
        if saved["provenance"].get("source_sha256") != current:
            raise ValueError("Checkpoint source hashes differ from executing code")
        if saved.get("precision_policy") != precision_policy():
            raise ValueError("Checkpoint AMP policy mismatch")
    check_state_counters(saved["state"])
    if not all(torch.isfinite(t).all() for t in saved["model_state_dict"].values()):
        raise FloatingPointError("Nonfinite checkpoint model tensors")
    for state in saved["optimizer_state_dict"]["state"].values():
        if any(isinstance(t, torch.Tensor) and not torch.isfinite(t).all() for t in state.values()):
            raise FloatingPointError("Nonfinite checkpoint optimizer tensors")
    scale_state = saved["scaler_state_dict"]
    if scale_state and (not math.isfinite(scale_state["scale"]) or scale_state["scale"] <= 0
                        or not 0 < scale_state["backoff_factor"] < 1
                        or scale_state["growth_factor"] <= 1
                        or scale_state["growth_interval"] < 1
                        or not 0 <= scale_state["_growth_tracker"] < scale_state["growth_interval"]):
        raise ValueError("Invalid checkpoint GradScaler state")
    return saved


def restore_run_checkpoint(path, model, optimizer, scaler, expected_signature, expected_cases_sha,
                           allow_legacy_amp_fix=False, expected_boundary="completed_epoch"):
    saved = read_run_checkpoint(path, expected_signature, expected_cases_sha,
                                allow_legacy_amp_fix, expected_boundary)
    if scaler.is_enabled() != bool(saved["scaler_state_dict"]):
        raise ValueError("Checkpoint scaler precision differs from requested execution")
    model.load_state_dict(saved["model_state_dict"], strict=True)
    optimizer.load_state_dict(saved["optimizer_state_dict"])
    scaler.load_state_dict(saved["scaler_state_dict"])
    restore_rng(saved["rng"])
    return saved


def patient_batches(dataset, batch_size, seed, epoch):
    """All positive AND negative slices, shuffled within shuffled patients.

    No GT-based slice selection, oversampling or augmentation is used. Only one
    patient's prepared arrays are resident, without persistent image caches.
    """
    rng = np.random.default_rng(seed + epoch)
    for patient in rng.permutation(len(dataset.case_ids)):
        case, pair, _ = dataset.load_patient(int(patient))
        del case
        positives = int((pair.labels.sum(dim=(1, 2, 3)) > 0).sum())
        logging.getLogger("unet_pipeline").info("TRAIN_PATIENT case=%s total_slices=%d positive=%d negative=%d",
                                                dataset.case_ids[int(patient)], len(pair), positives, len(pair) - positives)
        generator = torch.Generator().manual_seed(seed + epoch * 1000 + int(patient))
        yield from DataLoader(pair, batch_size=batch_size, shuffle=True,
                              generator=generator, num_workers=0)


@torch.inference_mode()
def predict_native(model, image_path, output_dir, device, batch_size=4, threshold=0.5):
    """CT-only automatic inference; GT cannot influence slices or predictions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    case = load_case(image_path)  # Deliberately no label path.
    images, meta = prepare_slices(case)
    probabilities = np.empty(images.shape, dtype=np.float32)
    model.eval()
    for start in range(0, len(images), batch_size):
        x = torch.from_numpy(images[start:start + batch_size]).to(device)
        logits = model(x)  # Validation/inference always FP32 for stable comparison.
        if tuple(logits.shape) != tuple(x.shape) or not torch.isfinite(logits).all():
            raise FloatingPointError("Invalid validation logits")
        probabilities[start:start + len(x)] = torch.sigmoid(logits).cpu().numpy()
    restored = restore_mask(probabilities, meta, threshold=threshold)
    path = save_original_mask(restored, case, output_dir / "prediction.nii.gz")
    saved = nib.load(str(path))
    if tuple(saved.shape) != tuple(case["orig_shape"]):
        raise ValueError("Export shape mismatch")
    np.testing.assert_allclose(saved.affine, case["orig_affine"], rtol=0, atol=1e-5)
    if tuple(nib.aff2axcodes(saved.affine)) != tuple(case["orig_axcodes"]):
        raise ValueError("Export orientation mismatch")
    if saved.header.get_xyzt_units()[0] != case["spatial_unit"]:
        raise ValueError("Export spatial units mismatch")
    np.testing.assert_allclose(saved.header.get_zooms()[:3], case["orig_zooms"])
    for form in ("qform", "sform"):
        matrix, code = getattr(saved, "get_" + form)(coded=True)
        if case[form + "_code"]:
            if int(code) != case[form + "_code"]:
                raise ValueError(f"Export {form} code mismatch")
            np.testing.assert_allclose(matrix, case[form], rtol=0, atol=1e-5)
    save_spatial_json(meta, output_dir / "spatial.json")
    return path, {"slice_count": len(images), "shape": list(saved.shape),
                  "affine_max_abs_diff": float(np.abs(saved.affine - case["orig_affine"]).max()),
                  "geometry_checks": "PASS"}


def mean_case_dice(rows, expected_ids):
    if len(rows) != len(expected_ids) or [r["case_id"] for r in rows] != list(expected_ids):
        raise ValueError("Validation must include every requested patient exactly once")
    scores = np.array([row["dice_3d"] for row in rows])
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Invalid patient Dice")
    return float(scores.mean())


def validate_cases(model, data_root, cases, split, case_ids, output_dir, device,
                   batch_size=4, threshold=0.5, logger=None):
    """Predict every slice first, then use GT only in B's NIfTI evaluator."""
    if not case_ids or len(set(case_ids)) != len(case_ids) or not set(case_ids).issubset(split["val"]):
        raise ValueError("Validation IDs must be unique members of val")
    output_dir = Path(output_dir)
    rows, checks = [], {}
    for case_id in case_ids:
        row = cases[case_id]
        prediction, check = predict_native(model, Path(data_root) / row["image_relpath"],
                                            output_dir / case_id, device, batch_size, threshold)
        metrics = evaluate_nifti_files(Path(data_root) / row["label_relpath"], prediction,
                                       case_id, "unet_validation")
        rows.append(metrics)
        checks[case_id] = check
        if logger:
            logger.info("VAL case=%s all_slices=%d dice_3d=%.8f pred_voxels=%d pred_volume_ml=%.4f gt_volume_ml=%.4f geometry=PASS",
                        case_id, check["slice_count"], metrics["dice_3d"], metrics["pred_voxels"],
                        metrics["pred_volume_ml"], metrics["gt_volume_ml"])
    mean = mean_case_dice(rows, case_ids)
    save_metrics_csv(rows, output_dir / "metrics.csv")
    result = {"case_ids": case_ids, "patient_count": len(rows), "mean_dice": mean,
              "aggregation": "arithmetic mean of patient 3D Dice, no slice averaging",
              "checks": checks, "rows": rows}
    (output_dir / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def update_selection(state, score, patience, min_delta):
    if not np.isfinite(score):
        raise ValueError("Nonfinite validation score")
    improved = score > state["best_mean_dice"]
    # best.pt ALWAYS follows the highest Dice, even when gain < min_delta;
    # min_delta only controls the patience counter.
    if score > state["patience_reference"] + min_delta:
        state["patience_reference"], state["bad_epochs"] = score, 0
    else:
        state["bad_epochs"] += 1
    if improved:
        state["best_mean_dice"] = score
        state["best_epoch"] = state["epoch"]
    return improved, state["bad_epochs"] >= patience


def check_config(config, split):
    if config.get("batch_order", "patient_grouped") not in ("patient_grouped", "global_batch_shuffle"):
        raise ValueError("Unknown batch_order")
    if config.get("batch_order") == "global_batch_shuffle" and type(config.get("schedule_seed")) is not int:
        raise ValueError("global_batch_shuffle requires an independent integer schedule_seed")
    if config["mode"] not in ("smoke", "full"):
        raise ValueError("mode must be smoke/full")
    if not 1 <= config["batch_size"] <= 4 or config["num_workers"] != 0:
        raise ValueError("Use batch size 1..4 and num_workers=0")
    if not 1 <= config["epochs"] <= 60 or config["patience"] < 1 or config["min_delta"] < 0:
        raise ValueError("Invalid epoch/early stopping settings")
    if (config["learning_rate"], config["dice_weight"], config["bce_weight"], config["threshold"]) != (1e-3, .5, .5, .5):
        raise ValueError("Preserve lr=1e-3, equal Dice/BCE, threshold=0.5")
    if config["mode"] == "smoke":
        if config["epochs"] != 1 or not 1 <= config["max_train_batches"] <= 8:
            raise ValueError("Real smoke is limited to one partial epoch and 1..8 batches")
        if not config["train_case_ids"] or not config["val_case_ids"]:
            raise ValueError("Smoke must explicitly declare train/val patient subsets")
    else:
        if not split.get("is_frozen"):
            raise ValueError("Full training awaits A/B split freeze; no split file was changed")
        if any(config[k] is not None for k in ("train_case_ids", "val_case_ids", "max_train_batches")):
            raise ValueError("Full training must use all train/all val and no batch cap")


def run_training(config_path, output_dir, resume=None, verify_only=False, allow_legacy_amp_fix=False):
    run_started = time.perf_counter()
    started_at_utc = datetime.now(timezone.utc).isoformat()
    config = load_config(config_path)
    cases, split = read_manifest(config["cases_csv"], config["split_json"])
    check_config(config, split)
    train_ids = config["train_case_ids"] or split["train"]
    val_ids = config["val_case_ids"] or split["val"]
    if not set(train_ids).issubset(split["train"]) or not set(val_ids).issubset(split["val"]):
        raise ValueError("Configured patients cross train/val boundaries")
    if len(set(train_ids)) != len(train_ids) or len(set(val_ids)) != len(val_ids):
        raise ValueError("Duplicate configured patients")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)  # Preserve every previous run.
    logger = logging.getLogger("unet_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(output_dir / "train.log", encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    set_seed(config["seed"])
    selected = config["device"]
    if selected not in ("auto", "cpu", "cuda"):
        raise ValueError("Invalid device")
    device = torch.device("cuda" if selected == "auto" and torch.cuda.is_available() else "cpu" if selected == "auto" else selected)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    env = environment()
    signature = training_signature(config, split)
    provenance = {"training_signature": signature, "cases_sha256": file_sha256(config["cases_csv"]),
                  "split_file_sha256": file_sha256(config["split_json"]),
                  "split_membership_sha256": json_hash({k: split[k] for k in ("train", "val", "test")}),
                  "split_frozen": split["is_frozen"], "train_ids": train_ids, "val_ids": val_ids,
                  "environment": env,
                  "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
                  "source_sha256": source_fingerprints(), "precision_policy": precision_policy()}
    (output_dir / "run_metadata.json").write_text(json.dumps({"config": config, "provenance": provenance}, indent=2), encoding="utf-8")
    model = build_unet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scaler = torch.amp.GradScaler("cuda", enabled=config["amp"] and device.type == "cuda")
    state = {"epoch": 0, "global_step": 0, "best_mean_dice": -1.0, "best_epoch": 0,
             "patience_reference": -1.0, "bad_epochs": 0, "history": [],
             "optimizer_updates": 0, "batch_attempts": 0, "amp_skipped_steps": 0, "consecutive_amp_skips": 0}
    if resume:
        saved = restore_run_checkpoint(resume, model, optimizer, scaler, signature, provenance["cases_sha256"],
                                        allow_legacy_amp_fix)
        state = saved["state"]
        provenance["resume"] = {"checkpoint": str(resume), "checkpoint_sha256": file_sha256(resume),
            "previous_source_sha256": saved["provenance"]["source_sha256"],
            "migration": saved.get("migration"),
            "prior_resume": saved["provenance"].get("resume")}
        if saved.get("migration"):
            logger.warning("CHECKPOINT_MIGRATION %s", json.dumps(saved["migration"]))
        # A run resumed in a new output directory still carries its previous best.
        previous_best = Path(resume).parent / "best.pt"
        if not verify_only:
            inherited = saved["provenance"].get("inherited_best", {})
            inherited_trusted = inherited.get("sha256") == file_sha256(previous_best)
            best_saved = read_run_checkpoint(previous_best, signature, provenance["cases_sha256"],
                                             allow_legacy_amp_fix or inherited_trusted)
            if best_saved["state"]["best_mean_dice"] != state["best_mean_dice"] or best_saved["state"]["epoch"] != state["best_epoch"]:
                raise ValueError("Matching previous best.pt is required to continue training")
            (output_dir / "best.pt").write_bytes(previous_best.read_bytes())
            provenance["inherited_best"] = {"sha256": file_sha256(previous_best),
                "source_sha256": best_saved["provenance"]["source_sha256"], "epoch": best_saved["state"]["epoch"]}
        logger.info("RESTORED epoch=%d global_step=%d best=%.8f scaler=%s", state["epoch"], state["global_step"], state["best_mean_dice"], scaler.state_dict())
    elif allow_legacy_amp_fix:
        raise ValueError("allow-legacy-amp-fix requires resume")
    (output_dir / "run_metadata.json").write_text(json.dumps({"config": config, "provenance": provenance}, indent=2), encoding="utf-8")
    if verify_only:
        if not resume:
            raise ValueError("verify-only requires resume checkpoint")
        result = validate_cases(model, config["data_root"], cases, split, val_ids,
                                output_dir / "reloaded_validation", device, config["batch_size"], config["threshold"], logger)
        summary = {"status": "CHECKPOINT_RELOAD_PASS", "new_optimizer_steps": 0,
                   "checkpoint_epoch": state["epoch"], "checkpoint_global_step": state["global_step"],
                   "validation": result, "scaler": scaler.state_dict()}
    else:
        if state["bad_epochs"] >= config["patience"]:
            raise ValueError("Checkpoint already early-stopped")
        if state["epoch"] >= config["epochs"]:
            raise ValueError("No epochs remain; use verify-only or increase the total epoch budget")
        dataset = PatientSliceDataset(config["data_root"], config["cases_csv"], config["split_json"], "train", train_ids)
        stop_reason = "max_epochs"
        for epoch in range(state["epoch"] + 1, config["epochs"] + 1):
            epoch_started = time.perf_counter()
            first_parameter = next(model.parameters()).detach().clone()
            if config.get("batch_order", "patient_grouped") == "global_batch_shuffle":
                from src.unet_schedule import shuffled_patient_batches
                batches = shuffled_patient_batches(dataset, config["batch_size"], config["seed"], epoch,
                    config["schedule_seed"], output_dir / f"batch_order_epoch_{epoch:03d}.json")
            else:
                batches = patient_batches(dataset, config["batch_size"], config["seed"], epoch)
            history = train(model, batches,
                            optimizer, DiceBCELoss(config["dice_weight"], config["bce_weight"]), device,
                            epochs=1, logger=logger, start_epoch=epoch - 1, global_step=state["global_step"],
                            amp=config["amp"], scaler=scaler, max_batches=config["max_train_batches"],
                            batch_attempts=state["batch_attempts"], amp_skipped_steps=state["amp_skipped_steps"],
                            consecutive_amp_skips=state["consecutive_amp_skips"])
            if history["global_step"] == state["global_step"]:
                raise RuntimeError("No optimizer updates in epoch; refusing to mark a completed training epoch")
            if torch.equal(first_parameter, next(model.parameters()).detach()):
                raise RuntimeError("Parameters unchanged after training")
            validation = validate_cases(model, config["data_root"], cases, split, val_ids,
                                        output_dir / f"validation_epoch_{epoch:03d}", device,
                                        config["batch_size"], config["threshold"], logger)
            state.update(epoch=epoch, **{k: history[k] for k in COUNTER_KEYS})
            improved, stop = update_selection(state, validation["mean_dice"], config["patience"], config["min_delta"])
            record = {"epoch": epoch, "train_loss": history["epochs"][0]["loss"],
                      "val_mean_dice": validation["mean_dice"], "val_cases": len(val_ids),
                      "learning_rate": optimizer.param_groups[0]["lr"], "global_step": state["global_step"],
                      "optimizer_updates": state["optimizer_updates"], "batch_attempts": state["batch_attempts"],
                      "amp_skipped_steps": state["amp_skipped_steps"], "consecutive_amp_skips": state["consecutive_amp_skips"],
                      "epoch_batch_attempts": history["epochs"][0]["batch_attempts"],
                      "epoch_optimizer_updates": history["epochs"][0]["optimizer_updates"],
                      "epoch_amp_skipped_steps": history["epochs"][0]["amp_skipped_steps"],
                      "best_epoch": state["best_epoch"], "best_mean_dice": state["best_mean_dice"],
                      "bad_epochs": state["bad_epochs"], "parameters_updated": True,
                      "epoch_seconds": time.perf_counter() - epoch_started,
                      "val_pred_volume_ml": sum(r["pred_volume_ml"] for r in validation["rows"]),
                      "val_empty_cases": sum(r["pred_voxels"] == 0 for r in validation["rows"]),
                      "best_path": str(output_dir / "best.pt"), "last_path": str(output_dir / "last.pt")}
            state["history"].append(record)
            if improved:
                save_run_checkpoint(output_dir / "best.pt", model, optimizer, scaler, state, config, provenance)
            save_run_checkpoint(output_dir / "last.pt", model, optimizer, scaler, state, config, provenance)
            with (output_dir / "history.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(record))
                writer.writeheader()
                writer.writerows({key: row.get(key, "") for key in record} for row in state["history"])
            logger.info("EPOCH %s early_stop=%s", json.dumps(record), stop)
            if stop:
                stop_reason = "early_stopping"
                break
            if (output_dir / "STOP_AFTER_EPOCH").exists():
                stop_reason = "user_pause_after_completed_epoch"
                logger.info("PAUSED after completed epoch=%d; resume last.pt in a NEW output directory", epoch)
                break
        summary = {"status": "TRAINING_PAUSED" if stop_reason == "user_pause_after_completed_epoch" else
                   "BOUNDED_REAL_PIPELINE_PASS" if config["mode"] == "smoke" else "TRAINING_FINISHED",
                   "stop_reason": stop_reason,
                   "state": state, "amp_enabled": scaler.is_enabled(), "scaler": scaler.state_dict(),
                   "validation": validation, "best_path": str(output_dir / "best.pt"),
                   "last_path": str(output_dir / "last.pt")}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        summary["gpu_peak_allocated_MiB"] = torch.cuda.max_memory_allocated(device) / 1024**2
        summary["gpu_peak_reserved_MiB"] = torch.cuda.max_memory_reserved(device) / 1024**2
    summary.update(started_at_utc=started_at_utc, finished_at_utc=datetime.now(timezone.utc).isoformat(),
                   session_wall_seconds=time.perf_counter() - run_started)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    return summary
