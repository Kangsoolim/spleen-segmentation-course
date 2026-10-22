"""Reorder intact patient batches using an isolated, epoch-addressable RNG.

Array preparation and DataLoader sampling match patient_batches exactly. The
global list contains actual collated tensors, never rebatched slices. No disk
image cache, global RNG call, augmentation, sampling or dropped tail is used.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def tensor_sha256(tensor):
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def collect_patient_batches(dataset, batch_size, seed, epoch):
    batches, records = [], []
    rng = np.random.default_rng(seed + epoch)
    for patient in rng.permutation(len(dataset.case_ids)):
        patient = int(patient)
        case, pair, _ = dataset.load_patient(patient)
        del case
        generator = torch.Generator().manual_seed(seed + epoch * 1000 + patient)
        indexed = TensorDataset(pair.images, pair.labels, torch.arange(len(pair)))
        for images, labels, zs in DataLoader(indexed, batch_size=batch_size, shuffle=True,
                                             generator=generator, num_workers=0):
            batches.append((images, labels))
            records.append({"original_batch_index": len(records), "case_id": dataset.case_ids[patient],
                "slice_ids_canonical_z": zs.tolist(), "size": len(zs),
                "positive_slices": int((labels.sum((1, 2, 3)) > 0).sum()),
                "foreground_pixels": int(labels.sum()),
                "images_sha256": tensor_sha256(images), "labels_sha256": tensor_sha256(labels)})
    return batches, records


def batch_permutation(count, schedule_seed, epoch):
    return np.random.default_rng(np.random.SeedSequence([schedule_seed, epoch])).permutation(count).tolist()


def shuffled_patient_batches(dataset, batch_size, seed, epoch, schedule_seed, audit_path):
    batches, records = collect_patient_batches(dataset, batch_size, seed, epoch)
    order = batch_permutation(len(batches), schedule_seed, epoch)
    audit = {"epoch": epoch, "mode": "global_batch_shuffle", "schedule_seed": schedule_seed,
        "rng": "numpy.default_rng(SeedSequence([schedule_seed, epoch])) / PCG64",
        "batch_count": len(batches), "slice_count": sum(r["size"] for r in records),
        "execution_order_original_batch_indices": order,
        "batches_in_execution_order": [records[i] for i in order]}
    Path(audit_path).write_text(json.dumps(audit, indent=2), encoding="utf-8")
    for index in order:
        yield batches[index]
