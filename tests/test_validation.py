#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2020 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of PANDORA
#
#     https://github.com/CNES/Pandora_pandora
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
This module contains functions to test the disparity map validation step.
"""

import unittest
import logging
import logging.config
import os
import json
import numpy as np
import xarray as xr

import pandora.validation as validation

from pandora.constants import *


class TestValidation(unittest.TestCase):
    """
    TestValidation class allows to test all the methods in the module Validation
    """
    def setUp(self):
        """
        Method called to prepare the test fixture

        """
        # Create reference and secondary disparity map
        self.ref = xr.Dataset({'disparity_map': (['row', 'col'], np.array([[0, -1, 1, -2],
                                                                           [2, 2, -1, 0]], dtype=np.float32)),
                               'confidence_measure': (['row', 'col', 'indicator'], np.full((2, 4, 1), np.nan)),
                               'validity_mask': (['row', 'col'],
                                                 np.array([[0, 0, 0, PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING],
                                                                           [0, 0, 0, 0]], dtype=np.uint16))},
                              coords={'row': [0, 1], 'col': np.arange(4)})
        self.ref.attrs['disp_min'] = -2
        self.ref.attrs['disp_max'] = 2

        self.sec = xr.Dataset({'disparity_map': (['row', 'col'], np.array([[0, 2, -1, -1],
                                                                           [1, 1, -2, -1]], dtype=np.float32)),
                               'confidence_measure': (['row', 'col', 'indicator'], np.full((2, 4, 1), np.nan)),
                               'validity_mask': (['row', 'col'], np.array([[0, 0, 0, 0],
                                                                           [0, 0, 0, 0]], dtype=np.uint16))},
                              coords={'row': [0, 1], 'col': np.arange(4)})
        self.sec.attrs['disp_min'] = -2
        self.sec.attrs['disp_max'] = 2

    def test_cross_checking(self):
        """
        Test the confidence measure and the validity_mask for the cross checking method,
                - If out & MSK_PIXEL_OCCLUSION != 0 : Invalid pixel : occluded pixel
                - If out & MSK_PIXEL_MISMATCH  != 0  : Invalid pixel : mismatched pixel
        """
        # Compute the cross checking confidence measure and validity mask
        validation_matcher = validation.AbstractValidation(**{'validation_method': 'cross_checking',
                                                              'cross_checking_threshold': 0.})

        ref = validation_matcher.disparity_checking(self.ref, self.sec)

        # Confidence measure ground truth
        gt_dist = np.array([[[np.nan, 0.],
                             [np.nan, 1.],
                             [np.nan, 0.],
                             [np.nan, 0.]],
                            [[np.nan, 0.],
                             [np.nan, 1.],
                             [np.nan, 0.],
                             [np.nan, 1.]]], dtype=np.float32)

        # Check if the calculated confidence measure is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(ref['confidence_measure'].data, gt_dist)

        # validity mask ground truth
        gt_mask = np.array([[0, PANDORA_MSK_PIXEL_MISMATCH, 0, PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING],
                            [0, PANDORA_MSK_PIXEL_MISMATCH, 0, PANDORA_MSK_PIXEL_OCCLUSION]], dtype=np.uint16)

        # Check if the calculated mask is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(ref['validity_mask'].data, gt_mask)

    def test_interpolate_occlusion_mc_cnn(self):
        """
        Test the disparity interpolation of occlusion
        """
        disp_data = np.array([[0, -1, 1, -2],
                              [2, 2, -1, 0]], dtype=np.float32)

        msk_data = np.array([[PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING, PANDORA_MSK_PIXEL_OCCLUSION,
                              PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING, 0],
                             [PANDORA_MSK_PIXEL_OCCLUSION, PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_REF, 0, PANDORA_MSK_PIXEL_OCCLUSION]],
                            dtype=np.uint16)
        # Create reference and secondary disparity map
        ref = xr.Dataset({'disparity_map': (['row', 'col'], disp_data),
                               'validity_mask': (['row', 'col'], msk_data)},
                              coords={'row': [0, 1], 'col': np.arange(4)})
        ref.attrs['disp_min'] = -2
        ref.attrs['disp_max'] = 2

        # Interpolate occlusions
        interpolation_matcher = validation.AbstractInterpolation(**{'interpolated_disparity': 'mc-cnn'})
        ref = interpolation_matcher.interpolated_disparity(ref)

        # validity mask after interpolation
        gt_mask_after_int = np.array([[PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING, PANDORA_MSK_PIXEL_FILLED_OCCLUSION,
                                       PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING, 0],
                                      [PANDORA_MSK_PIXEL_FILLED_OCCLUSION, PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_REF, 0,
                                       PANDORA_MSK_PIXEL_FILLED_OCCLUSION]], dtype=np.uint16)

        # Check if the calculated mask is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(ref['validity_mask'].data, gt_mask_after_int)

        # reference disparity map after interpolation
        gt_disp_after_int = np.array([[0, -2, 1, -2],
                                      [-1, 2, -1, -1]], dtype=np.float32)

        # Check if the calculated disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(ref['disparity_map'].data, gt_disp_after_int)

    def test_interpolate_mismatch_mc_cnn(self):
        """
        Test the disparity interpolation of mismatch
        """
        disp_data = np.array([[0, 1, -2, -1, -2],
                              [1, 0, 1, 0, 0],
                              [2, 1, -1, -2, -1],
                              [1, -1, 1, -1, -1]], dtype=np.float32)

        msk_data = np.array([[PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER, PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE,
                              0, PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION, 0],
                             [0, 0, PANDORA_MSK_PIXEL_MISMATCH, 0, 0],
                             [0, PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION, PANDORA_MSK_PIXEL_MISMATCH,
                              PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_REF, PANDORA_MSK_PIXEL_MISMATCH],
                             [0, PANDORA_MSK_PIXEL_MISMATCH, 0, 0, 0]], dtype=np.uint16)
        # Create reference and secondary disparity map
        ref = xr.Dataset({'disparity_map': (['row', 'col'], disp_data),
                               'validity_mask': (['row', 'col'], msk_data)},
                              coords={'row': np.arange(4), 'col': np.arange(5)})
        ref.attrs['disp_min'] = -2
        ref.attrs['disp_max'] = 2

        # Interpolate mistmatch
        interpolation_matcher = validation.AbstractInterpolation(**{'interpolated_disparity': 'mc-cnn'})
        ref = interpolation_matcher.interpolated_disparity(ref)

        # validity mask after interpolation
        gt_mask_after_int = np.array([[PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER,
                                       PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE, 0,
                                       PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION, 0],
                                      [0, 0, PANDORA_MSK_PIXEL_FILLED_MISMATCH, 0, 0],
                                      [0, (1 << 3), PANDORA_MSK_PIXEL_FILLED_MISMATCH,
                                       PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_REF,
                                       PANDORA_MSK_PIXEL_FILLED_MISMATCH],
                                      [0, PANDORA_MSK_PIXEL_FILLED_MISMATCH, 0, 0, 0]], dtype=np.uint16)

        # Check if the calculated mask is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(ref['validity_mask'].data, gt_mask_after_int)

        # reference disparity map after interpolation
        gt_disp_after_int = np.array([[0, 1, -2, -1, -2],
                                      [1, 0, np.median([1, 1, 0, 0, 0, 1, -2, -2, -2, -1, 0, 0, 0, -1, -1]), 0, 0],
                                      [2, 1,  np.median([1, 1, 1, 1, 1, 0, 1, -2, -1, 0, 0, -1, -1, 1]), -2,
                                       np.median([-1, -1, -1, 1, 1, 0, 0, 0, 0, 0])],
                                      [1, np.median([1, 1, 1, 2, 1, 1, 1, 0, 1, 1, 1]), 1, -1, -1]], dtype=np.float32)

        # Check if the calculated disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(ref['disparity_map'].data, gt_disp_after_int)

    def test_interpolate_occlusion_sgm(self):
        """
        Test the disparity interpolation of occlusion
        """
        disp_data = np.array([[0, 1, -2, -1, -2],
                              [1, 0, 1, 0, 0],
                              [2, 1, -1, -2, -1],
                              [1, -1, 1, -1, -1]], dtype=np.float32)

        msk_data = np.array([[PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER, PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE, 0,
                              PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION, 0],
                             [0, 0, PANDORA_MSK_PIXEL_OCCLUSION, 0, 0],
                             [0, PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION, PANDORA_MSK_PIXEL_OCCLUSION,
                              PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_REF, PANDORA_MSK_PIXEL_OCCLUSION],
                             [0, PANDORA_MSK_PIXEL_OCCLUSION, 0, 0, 0]], dtype=np.uint16)
        # Create reference and secondary disparity map
        ref = xr.Dataset({'disparity_map': (['row', 'col'], disp_data), 'validity_mask': (['row', 'col'], msk_data)},
                         coords={'row': np.arange(4), 'col': np.arange(5)})
        ref.attrs['disp_min'] = -2
        ref.attrs['disp_max'] = 2

        # Interpolate occlusion
        interpolation_matcher = validation.AbstractInterpolation(**{'interpolated_disparity': 'sgm'})
        ref = interpolation_matcher.interpolated_disparity(ref)

        # validity mask after interpolation
        gt_mask_after_int = np.array([[PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER,
                                       PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE, 0,
                                       PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION, 0],
                                      [0, 0, PANDORA_MSK_PIXEL_FILLED_OCCLUSION, 0, 0],
                                      [0, PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION, PANDORA_MSK_PIXEL_FILLED_OCCLUSION,
                                       PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_REF,
                                       PANDORA_MSK_PIXEL_FILLED_OCCLUSION],
                                      [0, PANDORA_MSK_PIXEL_FILLED_OCCLUSION, 0, 0, 0]], dtype=np.uint16)

        # Check if the calculated mask is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(ref['validity_mask'].data, gt_mask_after_int)

        # reference disparity map after interpolation
        gt_disp_after_int = np.array([[0, 1, -2, -1, -2],
                                      [1, 0, 0, 0, 0],
                                      [2, 1, 0, -2, 0],
                                      [1, 1, 1, -1, -1]], dtype=np.float32)

        # Check if the calculated disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(ref['disparity_map'].data, gt_disp_after_int)

    def test_interpolate_mismatch_sgm(self):
        """
        Test the disparity interpolation of mismatch
        """
        disp_data = np.array([[0, 1, -2, -1, -2],
                              [1, 0, 1, 0, 0],
                              [2, 1, -1, -2, -1],
                              [1, -1, 1, -1, -1]], dtype=np.float32)

        msk_data = np.array([[PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER, PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE,
                              0, PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION, 0],
                             [0, 0, PANDORA_MSK_PIXEL_MISMATCH, 0, 0],
                             [0, PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION, PANDORA_MSK_PIXEL_MISMATCH,
                              PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_REF,
                              PANDORA_MSK_PIXEL_MISMATCH],
                             [0, PANDORA_MSK_PIXEL_MISMATCH, 0, 0, 0]], dtype=np.uint16)
        # Create reference and secondary disparity map
        ref = xr.Dataset({'disparity_map': (['row', 'col'], disp_data), 'validity_mask': (['row', 'col'], msk_data)},
                         coords={'row': np.arange(4), 'col': np.arange(5)})
        ref.attrs['disp_min'] = -2
        ref.attrs['disp_max'] = 2

        # Interpolate mismatch
        interpolation_matcher = validation.AbstractInterpolation(**{'interpolated_disparity': 'sgm'})
        ref = interpolation_matcher.interpolated_disparity(ref)

        # validity mask after interpolation
        gt_mask_after_int = np.array([[PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER,
                                       PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE, 0,
                                       PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION, 0],
                                      [0, 0, PANDORA_MSK_PIXEL_FILLED_MISMATCH, 0, 0],
                                      [0, PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION, PANDORA_MSK_PIXEL_FILLED_MISMATCH,
                                       PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_REF, PANDORA_MSK_PIXEL_FILLED_MISMATCH],
                                      [0, PANDORA_MSK_PIXEL_FILLED_MISMATCH, 0, 0, 0]], dtype=np.uint16)

        # Check if the calculated mask is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(ref['validity_mask'].data, gt_mask_after_int)

        # reference disparity map after interpolation
        gt_disp_after_int = np.array([[0, 1, -2, -1, -2],
                                      [1, 0, np.median([1, 1, 0, 1, -2, -1, 0, -1]), 0, 0],
                                      [2, 1, np.median([1, 1, 0, -2, 0, -1]), -2, np.median([-1, -1, 1, 0, 0])],
                                      [1, np.median([1, 2, 1, 0, 1]), 1, -1, -1]], dtype=np.float32)

        # Check if the calculated disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(ref['disparity_map'].data, gt_disp_after_int)

    def test_interpolate_mismatch_and_occlusion_sgm(self):
        """
        Test the disparity interpolation of mismatch and occlusion
        """
        # Test with mismatched pixel that are direct neighbors of occluded pixels
        disp_data = np.array([[0, 1, -2, -1, -2],
                              [1, 0, 1, 0, 0],
                              [2, 1, -1, -2, -1],
                              [1, -1, 1, -1, -1]], dtype=np.float32)

        msk_data = np.array([[PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER, PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE,
                              0,  PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION, PANDORA_MSK_PIXEL_OCCLUSION],
                             [0, 0, PANDORA_MSK_PIXEL_MISMATCH, 0, 0],
                             [0, PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION, PANDORA_MSK_PIXEL_MISMATCH,
                              PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_REF, PANDORA_MSK_PIXEL_MISMATCH],
                             [PANDORA_MSK_PIXEL_OCCLUSION, PANDORA_MSK_PIXEL_MISMATCH, 0, 0, 0]], dtype=np.uint16)
        # Create reference and secondary disparity map
        ref = xr.Dataset({'disparity_map': (['row', 'col'], disp_data), 'validity_mask': (['row', 'col'], msk_data)},
                         coords={'row': np.arange(4), 'col': np.arange(5)})
        ref.attrs['disp_min'] = -2
        ref.attrs['disp_max'] = 2

        # Interpolate mismatch
        interpolation_matcher = validation.AbstractInterpolation(**{'interpolated_disparity': 'sgm'})
        ref = interpolation_matcher.interpolated_disparity(ref)

        # validity mask after interpolation
        gt_mask_after_int = np.array([[PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER,
                                       PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE, 0,
                                       PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION, PANDORA_MSK_PIXEL_FILLED_OCCLUSION],
                                      [0, 0, PANDORA_MSK_PIXEL_FILLED_MISMATCH, 0, 0],
                                      [0, PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION, PANDORA_MSK_PIXEL_FILLED_MISMATCH,
                                       PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_REF, PANDORA_MSK_PIXEL_FILLED_MISMATCH],
                                      [PANDORA_MSK_PIXEL_FILLED_OCCLUSION, PANDORA_MSK_PIXEL_FILLED_OCCLUSION, 0, 0,
                                       0]],
                                     dtype=np.uint16)

        # Check if the calculated mask is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(ref['validity_mask'].data, gt_mask_after_int)

        # reference disparity map after interpolation
        gt_disp_after_int = np.array([[0, 1, -2, -1, 0],
                                      [1, 0, np.median([1, 1, 0, 1, -2, -1, 0, -1]), 0, 0],
                                      [2, 1, np.median([1, 1, 0, -2, 0, -1]), -2, np.median([-1, -1, 1, 0, 0])],
                                      [1, 1, 1, -1, -1]], dtype=np.float32)

        # Check if the calculated disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(ref['disparity_map'].data, gt_disp_after_int)


def setup_logging(path='logging.json', default_level=logging.WARNING,):
    """
    Setup the logging configuration

    :param path: path to the configuration file
    :type path: string
    :param default_level: default level
    :type default_level: logging level
    """
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


if __name__ == '__main__':
    setup_logging()
    unittest.main()
