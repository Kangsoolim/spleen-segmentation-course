"""Synthetic spatial regression tests; none of these are real CT evidence."""
import csv
import json
import shutil
import unittest
import uuid
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from torch.utils.data import DataLoader

from src.common.evaluation import evaluate_nifti_files
from src.common.io_spatial import load_case, restore_mask, save_original_mask
from src.unet_data import PatientSliceDataset, prepare_pair, read_manifest


class TestUnetData(unittest.TestCase):
    def setUp(self):
        # Keep fixtures inside the explicitly writable workspace on Windows.
        fixture_parent = Path(__file__).resolve().parents[1] / "outputs"
        fixture_parent.mkdir(exist_ok=True)
        # tempfile's mode=0o700 produces an inaccessible ACL under this Windows
        # restricted token. Normal workspace mkdir inherits the usable ACL.
        self.root = fixture_parent / ("unet_data_test_" + uuid.uuid4().hex)
        self.root.mkdir()
        self.addCleanup(self.cleanup_fixture, fixture_parent)
        for folder in ("imagesTr", "labelsTr"):
            (self.root / folder).mkdir()
        # Native AIL -> canonical RAS requires both permutation and flips;
        # native shape[2] is deliberately NOT the canonical slice count.
        self.affine = np.array([[0, 0, -0.8, 31], [0.9, 0, 0, -22],
                                [0, -2.5, 0, 17], [0, 0, 0, 1]], dtype=float)
        self.gt = np.zeros((11, 17, 5), dtype=np.uint8)
        self.gt[2:7, 4:9, 1:4] = 1
        self.ct = (self.gt.astype(np.float32) * 221 - 57)
        self.rows = []
        for case_id in ("spleen_train", "spleen_val", "spleen_test"):
            row = {"case_id": case_id, "pair_valid": "True",
                   "image_relpath": f"imagesTr/{case_id}.nii.gz",
                   "label_relpath": f"labelsTr/{case_id}.nii.gz"}
            self.rows.append(row)
            # The held-out test raw files deliberately do not exist.
            if case_id != "spleen_test":
                for key, data in (("image_relpath", self.ct), ("label_relpath", self.gt)):
                    nii = nib.Nifti1Image(data, self.affine)
                    nii.header.set_xyzt_units("mm")
                    nii.set_qform(self.affine, code=1)
                    nii.set_sform(self.affine, code=1)
                    nib.save(nii, self.root / row[key])
        self.split = {"split_unit": "case_id", "train": ["spleen_train"],
                      "val": ["spleen_val"], "test": ["spleen_test"],
                      "counts": {"train": 1, "val": 1, "test": 1},
                      "valid_case_count": 3, "is_frozen": False}
        self.csv = self.root / "cases.csv"
        self.json = self.root / "split.json"
        self.write_manifest()

    def cleanup_fixture(self, fixture_parent):
        if self.root.resolve().parent != fixture_parent.resolve():
            raise RuntimeError("Refusing fixture cleanup outside outputs")
        shutil.rmtree(self.root)

    def write_manifest(self):
        with self.csv.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        self.json.write_text(json.dumps(self.split), encoding="utf-8")

    def dataset(self, **kwargs):
        return PatientSliceDataset(self.root, self.csv, self.json, **kwargs)

    def test_non_ras_pair_batch_and_native_export(self):
        dataset = self.dataset()
        case, pair, meta = dataset.load_patient(0)
        self.assertEqual(tuple(case["orig_axcodes"]), ("A", "I", "L"))
        self.assertEqual(len(dataset), 17)
        self.assertEqual(tuple(pair.images.shape), (17, 1, 256, 256))
        self.assertEqual(meta["canonical_axcodes"], ["R", "A", "S"])
        # Independent canonicalization and nearest-neighbor reference.
        canonical = nib.as_closest_canonical(nib.load(str(dataset.paths[0][1]))).get_fdata()
        expected = np.stack([zoom(canonical[:, :, z], (256/5, 256/11), order=0,
                                  prefilter=False, mode="nearest", grid_mode=False)
                             for z in range(17)])
        np.testing.assert_array_equal(pair.labels[:, 0].numpy(), expected)
        x, y = next(iter(DataLoader(dataset, batch_size=2, num_workers=0)))
        self.assertEqual(tuple(x.shape), (2, 1, 256, 256))
        self.assertEqual(x.shape, y.shape)
        self.assertEqual(float(pair.images.min()), 0.0)
        self.assertEqual(float(pair.images.max()), 1.0)
        self.assertTrue(set(np.unique(pair.labels)).issubset({0, 1}))
        restored = restore_mask(pair.labels.numpy(), json.loads(json.dumps(meta)))
        np.testing.assert_array_equal(restored, self.gt)
        output = save_original_mask(restored, case, self.root / "prediction.nii.gz")
        saved = nib.load(str(output))
        np.testing.assert_allclose(saved.affine, nib.load(str(dataset.paths[0][0])).affine, rtol=0, atol=1e-5)
        self.assertEqual(saved.header.get_xyzt_units()[0], "mm")
        self.assertEqual(saved.get_qform(coded=True)[1], 1)
        self.assertEqual(saved.get_sform(coded=True)[1], 1)
        metrics = evaluate_nifti_files(dataset.paths[0][1], output)
        self.assertEqual(metrics["dice_3d"], 1.0)
        self.assertAlmostEqual(metrics["gt_volume_ml"], int(self.gt.sum()) * abs(np.linalg.det(self.affine[:3, :3]))/1000, places=6)

    def test_ras_pair_and_clipping(self):
        case = load_case(self.root / self.rows[0]["image_relpath"], self.root / self.rows[0]["label_relpath"])
        case["ct_original"][0, 0, 0] = -2000
        case["ct_original"][-1, -1, -1] = 3000
        pair, _ = prepare_pair(case)
        self.assertGreaterEqual(float(pair.images.min()), 0)
        self.assertLessEqual(float(pair.images.max()), 1)
        # Also test an actual canonical-RAS NIfTI pair through shared load_case.
        for key in ("image_relpath", "label_relpath"):
            path = self.root / self.rows[0][key]
            nib.save(nib.as_closest_canonical(nib.load(str(path))), str(path))
        dataset = self.dataset()
        case, pair, meta = dataset.load_patient(0)
        self.assertEqual(case["orig_axcodes"], ("R", "A", "S"))
        np.testing.assert_array_equal(restore_mask(pair.labels.numpy(), meta), case["gt_original"])

    def test_patient_split_overlap_and_duplicates_rejected(self):
        for name, ids in (("val", ["spleen_train"]), ("train", ["spleen_train"] * 2)):
            original = self.split[name]
            self.split[name] = ids
            self.write_manifest()
            with self.assertRaisesRegex(ValueError, "Duplicate/overlapping"):
                read_manifest(self.csv, self.json)
            self.split[name] = original

    def test_pairing_and_path_escape_rejected(self):
        for wrong in ("labelsTr/spleen_val.nii.gz", "../spleen_train.nii.gz", "imagesTs/spleen_train.nii.gz"):
            self.rows[0]["label_relpath"] = wrong
            self.write_manifest()
            with self.assertRaisesRegex(ValueError, "Unpaired or unsafe"):
                self.dataset()

    def test_held_out_and_unknown_cases_rejected(self):
        self.assertEqual(len(self.dataset()), 17)  # No held-out raw data opened.
        self.assertEqual(len(self.dataset(partition="val")), 17)
        for kwargs in ({"partition": "test"}, {"case_ids": ["spleen_test"]}, {"case_ids": ["unknown"]}):
            with self.assertRaises(ValueError):
                self.dataset(**kwargs)

    def test_incomplete_split_rejected(self):
        self.split["valid_case_count"] = 4
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "cover every"):
            self.dataset()

    def test_missing_data_not_synthesized(self):
        (self.root / self.rows[0]["image_relpath"]).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Missing raw data"):
            self.dataset()

    def test_misaligned_gt_and_soft_labels_rejected(self):
        label = self.root / self.rows[0]["label_relpath"]
        shifted = self.affine.copy()
        shifted[0, 3] += 5
        nib.save(nib.Nifti1Image(self.gt, shifted), label)
        with self.assertRaisesRegex(ValueError, "affine mismatch"):
            self.dataset().load_patient(0)
        soft = self.gt.astype(np.float32)
        soft[0, 0, 0] = 0.5
        nib.save(nib.Nifti1Image(soft, self.affine), label)
        with self.assertRaisesRegex(ValueError, "not discrete"):
            self.dataset().load_patient(0)

    def test_case_boundary_index_and_cache(self):
        self.split["train"].append(self.split["val"].pop())
        # Manifest policy requires nonempty val; rename the existing test ID to val
        # and introduce one more metadata-only test ID, without loading either.
        self.split["val"] = self.split["test"]
        self.split["test"] = ["spleen_holdout"]
        self.rows.append({**self.rows[-1], "case_id": "spleen_holdout",
                          "image_relpath": "imagesTr/spleen_holdout.nii.gz",
                          "label_relpath": "labelsTr/spleen_holdout.nii.gz"})
        self.split["counts"]["train"] = 2
        self.split["valid_case_count"] = 4
        self.write_manifest()
        dataset = self.dataset()
        self.assertEqual(len(dataset), 34)
        for index, cached in ((0, 0), (16, 0), (17, 1), (-1, 1), (0, 0)):
            self.assertEqual(tuple(dataset[index][0].shape), (1, 256, 256))
            self.assertEqual(dataset._cached_index, cached)
        for index in (-35, 34):
            with self.assertRaises(IndexError):
                dataset[index]


if __name__ == "__main__":
    unittest.main()
