"""Patient-split 2D Dataset using B's shared spatial implementation.

Only train/val are exposed here. No new orientation or resize convention is
introduced: GT uses the CT metadata and the shared nearest-neighbor resizer.
"""
import csv
import json
from bisect import bisect_right
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.orientations import apply_orientation
from torch.utils.data import Dataset

from src.common.io_spatial import _resize_2d, load_case, prepare_slices
from src.training import SliceArrayDataset


def read_manifest(cases_csv, split_json):
    """Validate the entire ID partition without opening any held-out CT/GT."""
    with Path(cases_csv).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    cases = {}
    for row in rows:
        case_id = row["case_id"]
        if not case_id or case_id in cases:
            raise ValueError(f"Empty/duplicate case_id: {case_id}")
        if row["pair_valid"].lower() != "true":
            continue
        for key, folder in (("image_relpath", "imagesTr"), ("label_relpath", "labelsTr")):
            path = Path(row[key].replace("\\", "/"))
            if path.parts not in ((folder, f"{case_id}.nii.gz"), (folder, f"{case_id}.nii")):
                raise ValueError(f"Unpaired or unsafe {key} for {case_id}: {path}")
        cases[case_id] = row
    split = json.loads(Path(split_json).read_text(encoding="utf-8-sig"))
    if split.get("split_unit") != "case_id":
        raise ValueError("Expected case_id-level patient split")
    seen = set()
    for name in ("train", "val", "test"):
        ids = split[name]
        if not isinstance(ids, list) or not ids or not all(isinstance(x, str) for x in ids):
            raise ValueError(f"Invalid/empty {name} partition")
        if len(ids) != len(set(ids)) or seen.intersection(ids):
            raise ValueError(f"Duplicate/overlapping patient IDs in {name}")
        if not set(ids).issubset(cases):
            raise ValueError(f"Unknown/invalid case IDs in {name}")
        if split.get("counts", {}).get(name) != len(ids):
            raise ValueError(f"Split count mismatch for {name}")
        seen.update(ids)
    if seen != set(cases) or split.get("valid_case_count") != len(cases):
        raise ValueError("Split must cover every valid manifest case exactly once")
    return cases, split


def prepare_pair(case):
    """Return the existing array Dataset plus CT metadata for one labeled case."""
    if case.get("gt_original") is None:
        raise ValueError("Training/validation requires paired GT")
    images, meta = prepare_slices(case, size=(256, 256), hu_window=(-57.0, 164.0))
    gt_ras = apply_orientation(case["gt_original"], np.asarray(meta["to_ras_ornt"]))
    if tuple(gt_ras.shape) != tuple(meta["canonical_shape"]):
        raise ValueError("Canonical CT/GT shape mismatch")
    labels = np.empty_like(images)
    for z in range(len(images)):
        labels[z, 0] = _resize_2d(gt_ras[:, :, z], (256, 256), order=0)
    if not (0 <= images.min() <= images.max() <= 1):
        raise ValueError("Normalized CT must be in [0,1]")
    return SliceArrayDataset(images, labels), meta


class PatientSliceDataset(Dataset):
    """All slices from explicitly selected train OR val patients.

    Builds its slice index from CT headers; caches one prepared patient to bound
    RAM use. For the initial integration use num_workers=0. Random shuffling can
    cause cache churn; tune data-loading performance only after real-data QA.
    ``is_frozen`` is reported, not silently changed. Case IDs are the only
    available patient identifiers; source-level patient independence still
    requires confirmation from the data provider.
    """

    def __init__(self, data_root, cases_csv="data/cases.csv", split_json="data/split.json",
                 partition="train", case_ids=None):
        if partition not in ("train", "val"):
            raise ValueError("This training adapter only permits train/val")
        self.cases, self.split = read_manifest(cases_csv, split_json)
        self.partition = partition
        self.case_ids = list(self.split[partition] if case_ids is None else case_ids)
        if not self.case_ids or len(self.case_ids) != len(set(self.case_ids)):
            raise ValueError("Expected nonempty unique case_ids")
        if not set(self.case_ids).issubset(self.split[partition]):
            raise ValueError(f"Requested cases must belong to {partition}")
        self.data_root = Path(data_root).resolve()
        self.paths, self.ends = [], []
        total = 0
        for case_id in self.case_ids:
            row = self.cases[case_id]
            image, label = [self.data_root / row[key].replace("\\", "/")
                            for key in ("image_relpath", "label_relpath")]
            for path in (image, label):
                if not path.resolve().is_relative_to(self.data_root):
                    raise ValueError(f"Path escapes data root: {path}")
                if not path.is_file():
                    raise FileNotFoundError(f"Missing raw data: {path}")
            nii = nib.load(str(image))
            if len(nii.shape) != 3:
                raise ValueError(f"Expected 3D CT: {image}")
            # Canonical S axis comes from this native axis; sign does not affect length.
            axcodes = nib.aff2axcodes(nii.affine)
            native_slice_axis = next(i for i, code in enumerate(axcodes) if code in ("S", "I"))
            total += int(nii.shape[native_slice_axis])
            self.ends.append(total)
            self.paths.append((image, label))
        self._cached_index = None
        self._cached_pair = None

    def __len__(self):
        return self.ends[-1]

    def load_patient(self, patient_index):
        """Return (native case, SliceArrayDataset, spatial_meta) via shared I/O."""
        case = load_case(*self.paths[patient_index])
        pair, meta = prepare_pair(case)
        start = 0 if patient_index == 0 else self.ends[patient_index - 1]
        if len(pair) != self.ends[patient_index] - start:
            raise ValueError("Prepared slice count disagrees with CT header index")
        return case, pair, meta

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        patient = bisect_right(self.ends, index)
        if self._cached_index != patient:
            _, self._cached_pair, _ = self.load_patient(patient)
            self._cached_index = patient
        start = 0 if patient == 0 else self.ends[patient - 1]
        return self._cached_pair[index - start]
