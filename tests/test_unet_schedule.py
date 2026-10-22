"""Intact-batch parity, membership, reproducibility and RNG isolation."""
import uuid
import copy
import json
import unittest
from pathlib import Path

import torch

from src.training import SliceArrayDataset, load_config
from src.unet_pipeline import patient_batches, capture_rng, training_signature, check_config
from src.unet_schedule import collect_patient_batches, batch_permutation, shuffled_patient_batches, tensor_sha256
from tests.helpers import equal


class FakePatients:
    case_ids = ['train_a', 'train_b', 'train_c']
    sizes = [3, 8, 11]

    def load_patient(self, patient):
        values = torch.arange(self.sizes[patient]).float() + patient * 20
        images = values[:, None, None, None].expand(-1, 1, 256, 256).clone()
        labels = (images.remainder(3) == 0).float()
        return {}, SliceArrayDataset(images, labels), {}


class TestBatchSchedule(unittest.TestCase):

    def test_intact_batches_once_and_global_rng_unchanged(self):
        ds = FakePatients()
        for epoch in (10, 11, 14):
            before = capture_rng()
            original = [(tensor_sha256(x), tensor_sha256(y)) for x, y in patient_batches(ds, 4, 42, epoch)]
            batches, records = collect_patient_batches(ds, 4, 42, epoch)
            self.assertEqual(original, [(r['images_sha256'], r['labels_sha256']) for r in records])
            self.assertEqual(len(batches), 6)
            self.assertEqual(sum(r['size'] for r in records), 22)
            for c, n in zip(ds.case_ids, ds.sizes):
                zs = [z for r in records if r['case_id'] == c for z in r['slice_ids_canonical_z']]
                self.assertEqual(sorted(zs), list(range(n)))
            Path('outputs/unet_runs').mkdir(parents=True, exist_ok=True)
            audit_path = Path('outputs/unet_runs') / ('schedule_unit_' + uuid.uuid4().hex + '.json')
            try:
                actual = list(shuffled_patient_batches(ds, 4, 42, epoch, 600042, audit_path))
            finally:
                audit_path.unlink(missing_ok=True)
            expected = [original[i] for i in batch_permutation(6, 600042, epoch)]
            self.assertEqual(expected, [(tensor_sha256(x), tensor_sha256(y)) for x, y in actual])
            self.assertTrue(equal(before, capture_rng()))

    def test_permutation_reproducible_and_epoch_specific(self):
        a = batch_permutation(676, 600042, 10)
        self.assertEqual(a, batch_permutation(676, 600042, 10))
        self.assertEqual(sorted(a), list(range(676)))
        self.assertNotEqual(a, batch_permutation(676, 600042, 11))

    def test_default_signature_and_config_guards(self):
        c = load_config('configs/train_unet_ampfix_epoch14_review.yaml')
        split = {'train':['a'], 'val':['b'], 'test':['c'], 'is_frozen':True}
        old = training_signature(c, split)
        c['batch_order'] = 'patient_grouped'
        self.assertEqual(old, training_signature(c, split))
        c['batch_order'] = 'global_batch_shuffle'
        with self.assertRaisesRegex(ValueError, 'schedule_seed'):
            check_config(c, split)
        c['schedule_seed'] = 600042
        check_config(c, split)
        self.assertNotEqual(old, training_signature(c, split))
        c['batch_order'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'batch_order'):
            check_config(c, split)


if __name__ == '__main__':
    unittest.main()
