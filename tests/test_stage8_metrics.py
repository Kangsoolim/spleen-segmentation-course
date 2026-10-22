"""Meaningful metric edge cases and finite-data parity with B's evaluator."""
import unittest

import numpy as np
from src.common.evaluation import evaluate_arrays,voxel_volume_mm3


class TestStage8Metrics(unittest.TestCase):
    def test_counts_volume_and_relative_error(self):
        gt=np.zeros((2,2,2),dtype=np.uint8);pred=gt.copy()
        gt.flat[[0,1]]=1;pred.flat[[0,2,3]]=1
        r=evaluate_arrays(gt,pred,np.diag([2.,3.,4.,1.]),'mm')
        self.assertEqual((r['tp_voxels'],r['fp_voxels'],r['fn_voxels'],r['tn_voxels']),(1,2,1,4))
        self.assertAlmostEqual(r['dice_3d'],.4);self.assertAlmostEqual(r['iou_3d'],.25)
        self.assertAlmostEqual(r['precision_3d'],1/3);self.assertAlmostEqual(r['recall_3d'],.5)
        self.assertAlmostEqual(r['voxel_volume_mm3'],24)
        self.assertAlmostEqual(r['gt_volume_ml'],.048);self.assertAlmostEqual(r['pred_volume_ml'],.072)
        self.assertAlmostEqual(r['volume_error_ml_pred_minus_gt'],.024)
        self.assertAlmostEqual(r['abs_volume_error_ml'],.024);self.assertAlmostEqual(r['RAVE_percent'],50)

    def test_empty_prediction_convention_is_explicit(self):
        zero=np.zeros((2,2,2),dtype=np.uint8);gt=zero.copy();gt.flat[0]=1
        r=evaluate_arrays(gt,zero,np.eye(4),'mm')
        self.assertEqual([r[k] for k in ('dice_3d','iou_3d','precision_3d','recall_3d')],[0,0,1,0])
        self.assertEqual(r['RAVE_percent'],100)
        both=evaluate_arrays(zero,zero,np.eye(4),'mm')
        self.assertEqual([both[k] for k in ('dice_3d','iou_3d','precision_3d','recall_3d')],[1,1,1,1])
        self.assertIsNone(both['RAVE_percent'])

    def test_affine_determinant_units_and_unknown_units(self):
        for unit,factor in [('mm',1),('meter',.001),('micron',1000)]:
            self.assertAlmostEqual(voxel_volume_mm3(np.diag([factor,2*factor,3*factor,1]),unit),6)
        self.assertIsNone(voxel_volume_mm3(np.eye(4),'unknown'))



if __name__=='__main__':unittest.main()
