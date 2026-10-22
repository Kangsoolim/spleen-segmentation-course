"""Focused tests for evaluation, model selection, and epoch-boundary recovery."""
import copy
import random
import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.common.evaluation import evaluate_arrays
from src.training import DiceBCELoss, load_config, set_seed, train
from src.unet_pipeline import (capture_rng, check_config, mean_case_dice, restore_rng,
                               restore_run_checkpoint, save_run_checkpoint,
                               training_signature, update_selection, source_fingerprints, environment)


class TestUnetPipeline(unittest.TestCase):
    def test_nan_inf_masks_rejected_before_binarization(self):
        good = np.zeros((2, 3, 4))
        for bad_value in (np.nan, np.inf, -np.inf):
            bad = good.copy()
            bad[0, 0, 0] = bad_value
            for gt, pred in ((bad, good), (good, bad)):
                with self.assertRaisesRegex(ValueError, "non-finite"):
                    evaluate_arrays(gt, pred, np.eye(4), "mm")
        self.assertEqual(evaluate_arrays(good, good, np.eye(4), "mm")["dice_3d"], 1)

    def test_patient_mean_not_voxel_weighted_and_complete(self):
        rows = [{"case_id": f"val{i}", "dice_3d": i/5, "gt_voxels": 10**i} for i in range(6)]
        ids = [r["case_id"] for r in rows]
        self.assertAlmostEqual(mean_case_dice(rows, ids), .5)
        with self.assertRaises(ValueError):
            mean_case_dice(rows[:-1], ids)
        rows[0]["dice_3d"] = np.nan
        with self.assertRaises(ValueError):
            mean_case_dice(rows, ids)

    def test_best_tracks_maximum_while_patience_uses_min_delta(self):
        state = {"epoch": 1, "best_mean_dice": -1., "patience_reference": -1., "bad_epochs": 0}
        self.assertEqual(update_selection(state, .5, 2, .1), (True, False))
        state["epoch"] = 2
        self.assertEqual(update_selection(state, .51, 2, .1), (True, False))
        state["epoch"] = 3
        self.assertEqual(update_selection(state, .49, 2, .1), (False, True))
        self.assertEqual(state["best_epoch"], 2)
        self.assertEqual(state["best_mean_dice"], .51)

    def test_config_rejects_long_smoke_and_unfrozen_full(self):
        smoke = load_config("configs/train_unet_smoke.yaml")
        check_config(smoke, {"is_frozen": False})
        smoke["epochs"] = 50
        with self.assertRaises(ValueError):
            check_config(smoke, {"is_frozen": False})
        full = load_config("configs/train_unet.yaml")
        with self.assertRaisesRegex(ValueError, "freeze"):
            check_config(full, {"is_frozen": False})
        check_config(full, {"is_frozen": True})

    def test_identity_excludes_timestamp_but_detects_membership_and_amp(self):
        config = load_config("configs/train_unet_smoke.yaml")
        split = {"train": ["a"], "val": ["b"], "test": ["c"], "created_at_utc": "old"}
        signature = training_signature(config, split)
        split["created_at_utc"] = "new"
        self.assertEqual(signature, training_signature(config, split))
        split["val"] = ["c"]
        self.assertNotEqual(signature, training_signature(config, split))
        split["val"] = ["b"]
        config["amp"] = not config["amp"]
        self.assertNotEqual(signature, training_signature(config, split))

    def test_rng_roundtrip_python_numpy_torch(self):
        set_seed(42)
        state = capture_rng()
        expected = (random.random(), float(np.random.rand()), torch.rand(3))
        restore_rng(state)
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(float(np.random.rand()), expected[1])
        torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)



if __name__ == "__main__":
    unittest.main()
