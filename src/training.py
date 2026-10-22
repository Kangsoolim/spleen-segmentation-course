"""Small FP32 training building blocks, independent of synthetic/CT generation."""

import logging
import json
import math
import random
from itertools import islice
from pathlib import Path

import numpy as np
import torch
import yaml
from monai.losses import DiceLoss
from torch import nn
from torch.utils.data import Dataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Training configuration must be a YAML mapping")
    return config


def validate_pair(images, labels):
    if images.dtype != torch.float32 or labels.dtype != torch.float32:
        raise ValueError("Images and labels must be float32")
    if images.ndim != 4 or tuple(images.shape[1:]) != (1, 256, 256):
        raise ValueError("Images must have shape [N,1,256,256]")
    if images.shape[0] == 0 or images.shape != labels.shape:
        raise ValueError("Images and labels must have identical nonempty shapes")
    if not torch.isfinite(images).all() or not torch.isfinite(labels).all():
        raise ValueError("Images/labels contain NaN or Inf")
    if not ((labels == 0) | (labels == 1)).all():
        raise ValueError("Labels must be binary (0/1)")


class SliceArrayDataset(Dataset):
    """Adapter for prepare_slices() float32 arrays; no spatial processing here.

    Arrays must already be [N,1,256,256]. The caller owns patient-level splits.
    """

    def __init__(self, images, labels):
        self.images = torch.as_tensor(images)
        self.labels = torch.as_tensor(labels)
        validate_pair(self.images, self.labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice = DiceLoss(sigmoid=True)
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, logits, labels):
        # Both components receive raw logits. DiceLoss applies its own sigmoid.
        return self.dice_weight * self.dice(logits, labels) + self.bce_weight * self.bce(logits, labels)


AMP_MAX_CONSECUTIVE_SKIPS = 3
AMP_POLICY_VERSION = 2


class AmpOverflowError(FloatingPointError):
    """The last overflow was safely skipped before the consecutive limit fired."""
    def __init__(self, result):
        super().__init__("Persistent nonfinite AMP gradients: require FP32 diagnostic")
        self.step_result = result


def validate_step_counters(global_step, attempts, skipped, consecutive):
    values = (global_step, attempts, skipped, consecutive)
    if any(type(v) is not int or v < 0 for v in values):
        raise ValueError("Step counters must be nonnegative integers")
    if attempts != global_step + skipped or consecutive > skipped or consecutive > AMP_MAX_CONSECUTIVE_SKIPS:
        raise ValueError("Inconsistent update/attempt/AMP skip counters")


def guarded_step(model, images, labels, optimizer, loss_fn, scaler, amp,
                 consecutive_skips=0, max_consecutive_skips=AMP_MAX_CONSECUTIVE_SKIPS):
    device = next(model.parameters()).device
    amp_enabled = bool(amp and device.type == "cuda")
    if scaler.is_enabled() != amp_enabled:
        raise ValueError("AMP/scaler settings disagree")
    if not torch.isfinite(images).all() or not torch.isfinite(labels).all():
        raise FloatingPointError("Nonfinite inputs or labels")
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError("Nonfinite parameters before forward")
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device.type, dtype=torch.float16, enabled=amp_enabled):
        logits = model(images)
    if logits.shape != labels.shape or (not amp_enabled and logits.dtype != torch.float32):
        raise ValueError("Model output must have the label shape (FP32 unless AMP)")
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Nonfinite logits")
    loss = loss_fn(logits.float(), labels)
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise FloatingPointError("Loss must be a finite scalar")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradients = [(name, p.grad) for name, p in model.named_parameters() if p.requires_grad]
    if not gradients or any(g is None for _, g in gradients):
        raise RuntimeError("Missing trainable parameter gradients")
    bad_names = [name for name, g in gradients if not torch.isfinite(g).all()]
    if bad_names and not amp_enabled:
        raise FloatingPointError("Nonfinite FP32 gradients")
    if not bad_names and not any(torch.count_nonzero(g).item() for _, g in gradients):
        raise RuntimeError("All gradients are zero")
    before = scaler.get_scale()
    # Public PyTorch APIs: unscale_ has already recorded nonfinite gradients.
    # step skips optimizer.step if needed; update applies the native backoff.
    scaler.step(optimizer)
    scaler.update()
    after = scaler.get_scale()
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError("Nonfinite parameters after step")
    if not math.isfinite(after) or after <= 0:
        raise FloatingPointError("Invalid AMP scale")
    skips = consecutive_skips+1 if bad_names else 0
    result = {"loss": loss.item(), "optimizer_step_skipped": bool(bad_names),
            "nonfinite_gradient_names": bad_names, "scale_before": before, "scale_after": after,
            "consecutive_skips": skips, "output_shape": tuple(logits.shape)}
    if skips > max_consecutive_skips:
        raise AmpOverflowError(result)
    return result


def train(model, loader, optimizer, loss_fn, device, epochs, logger=None,
          start_epoch=0, global_step=0, amp=False, scaler=None, max_batches=None,
          batch_attempts=None, amp_skipped_steps=0, consecutive_amp_skips=0):
    """Train with native AMP skip/backoff and actual-update global_step.

    Loss is sample-weighted over every finite forward batch, including skipped
    AMP attempts. Counters are cumulative and must be carried across epochs.
    Completed epochs alone are checkpoint recovery boundaries.
    """
    if epochs < 1 or start_epoch < 0 or global_step < 0 or (max_batches is not None and max_batches < 1):
        raise ValueError("Invalid epoch/step counters")
    batch_attempts = global_step + amp_skipped_steps if batch_attempts is None else batch_attempts
    validate_step_counters(global_step, batch_attempts, amp_skipped_steps, consecutive_amp_skips)
    device = torch.device(device)
    amp_enabled = bool(amp and device.type == "cuda")
    scaler = scaler if scaler is not None else torch.amp.GradScaler("cuda", enabled=amp_enabled)
    if scaler.is_enabled() != amp_enabled:
        raise ValueError("AMP/scaler settings disagree")
    logger = logger or logging.getLogger(__name__)
    history = {"steps": [], "epochs": []}
    logger.info("training device=%s precision=%s global_step=optimizer_updates loss_scope=all_finite_forward_attempts",
                device, "AMP_FP16" if amp_enabled else "FP32")
    for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
        model.train()
        total_loss, samples = 0.0, 0
        epoch_attempt_start, epoch_update_start, epoch_skip_start = batch_attempts, global_step, amp_skipped_steps
        batches = loader if max_batches is None else islice(loader, max_batches)
        for images, labels in batches:
            validate_pair(images, labels)
            images, labels = images.to(device), labels.to(device)
            batch_attempts += 1
            fatal_overflow = None
            try:
                result = guarded_step(model, images, labels, optimizer, loss_fn, scaler, amp_enabled,
                                      consecutive_amp_skips)
            except AmpOverflowError as exc:
                result, fatal_overflow = exc.step_result, exc
            skipped = result["optimizer_step_skipped"]
            amp_skipped_steps += int(skipped)
            global_step += int(not skipped)
            consecutive_amp_skips = result["consecutive_skips"]
            record = {"epoch": epoch, "step": global_step, "loss": result["loss"],
                      "batch_attempt": batch_attempts, "optimizer_updates": global_step,
                      "amp_skipped_steps": amp_skipped_steps,
                      "consecutive_amp_skips": consecutive_amp_skips,
                      "optimizer_step_skipped": skipped, "scale_before": result["scale_before"],
                      "scale_after": result["scale_after"],
                      "nonfinite_gradient_names": result["nonfinite_gradient_names"]}
            if skipped:
                logger.warning("AMP_OVERFLOW %s", json.dumps(record))
            logger.info("epoch=%d step=%d attempt=%d loss=%.8f input=%s output=%s skipped=%s scale_before=%s scale_after=%s",
                        epoch, global_step, batch_attempts, result["loss"], tuple(images.shape),
                        result["output_shape"], skipped, result["scale_before"], result["scale_after"])
            if fatal_overflow is not None:
                logger.error("STOP persistent AMP overflow after %d consecutive skips", consecutive_amp_skips)
                raise fatal_overflow
            total_loss += result["loss"] * len(images)
            samples += len(images)
            history["steps"].append(record)
        if not samples:
            raise ValueError("DataLoader yielded no batches")
        mean_loss = total_loss / samples
        history["epochs"].append({"epoch": epoch, "loss": mean_loss,
                                  "batch_attempts": batch_attempts-epoch_attempt_start,
                                  "optimizer_updates": global_step-epoch_update_start,
                                  "amp_skipped_steps": amp_skipped_steps-epoch_skip_start})
        logger.info("epoch=%d mean_loss=%.8f samples=%d attempts=%d updates=%d skipped=%d", epoch, mean_loss,
                    samples, batch_attempts-epoch_attempt_start, global_step-epoch_update_start,
                    amp_skipped_steps-epoch_skip_start)
    history.update(epoch=epoch, global_step=global_step, optimizer_updates=global_step,
                   batch_attempts=batch_attempts, amp_skipped_steps=amp_skipped_steps,
                   consecutive_amp_skips=consecutive_amp_skips)
    return history


def save_checkpoint(path, model, optimizer, epoch, global_step, seed, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch, "global_step": global_step,
                "seed": seed, "config": dict(config)}, path)
    return path


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
