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
This module contains functions to test the disparity module.
"""

import unittest
import logging
import logging.config
import os
import json
import numpy as np
import xarray as xr

import pandora.disparity as disparity
import pandora.stereo as stereo
from pandora.img_tools import read_img
import pandora

from pandora.constants import *


class TestDisparity(unittest.TestCase):
    """
    TestDisparity class allows to test the disparity module
    """
    def setUp(self):
        """
        Method called to prepare the test fixture

        """
        # Create stereo images
        data = np.array(([[1, 2, 4, 6],
                          [2, 4, 1, 6],
                          [6, 7, 8, 10]]), dtype=np.float64)
        self.ref = xr.Dataset({'im': (['row', 'col'], data)},
                              coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})

        data = np.array(([[6, 1, 2, 4],
                          [6, 2, 4, 1],
                          [10, 6, 7, 8]]), dtype=np.float64)
        self.sec = xr.Dataset({'im': (['row', 'col'], data)},
                              coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})

    def test_to_disp(self):
        """
        Test the to disp method

        """
        # Create the left cost volume, with SAD measure window size 1, subpixel 1, disp_min -3 disp_max 1
        stereo_plugin = stereo.AbstractStereo(**{'stereo_method': 'sad', 'window_size': 1, 'subpix': 1})
        cv = stereo_plugin.compute_cost_volume(self.ref, self.sec, -3, 1, **{'valid_pixels': 0, 'no_data': 1})

        # Disparity map ground truth, for the images described in the setUp method
        gt_disp = np.array([[1, 1, 1, -3],
                            [1, 1, 1, -3],
                            [1, 1, 1, -3]])

        # Compute the disparity
        disp = disparity.to_disp(cv)

        # Check if the calculated disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(disp['disparity_map'].data, gt_disp)

        #
        # Test the to_disp method with negative disparity range
        #
        cv = stereo_plugin.compute_cost_volume(self.ref, self.sec, -3, -1, **{'valid_pixels': 0, 'no_data': 1})

        # Disparity map ground truth
        gt_disp = np.array([[0, -1, -2, -3],
                            [0, -1, -1, -3],
                            [0, -1, -2, -3]])

        # Compute the disparity
        disp = disparity.to_disp(cv, invalid_value=0)

        # Check if the calculated disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(disp['disparity_map'].data, gt_disp)

        #
        # Test the to_disp method with positive disparity range
        #
        cv = stereo_plugin.compute_cost_volume(self.ref, self.sec, 1, 3, **{'valid_pixels': 0, 'no_data': 1})

        # Disparity map ground truth
        gt_disp = np.array([[1, 1, 1, 0],
                            [1, 1, 1, 0],
                            [1, 1, 1, 0]])

        # Compute the disparity
        disp = disparity.to_disp(cv, invalid_value=0)

        # Check if the calculated disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(disp['disparity_map'].data, gt_disp)

        # Test disp_indices copy
        # Modify the disparity map
        disp['disparity_map'].data[0, 0] = -95
        # Check if the xarray disp_indices is equal to the ground truth disparity map
        np.testing.assert_array_equal(cv['disp_indices'].data, gt_disp)

    def test_argmin_split(self):
        """
        Test the argmin_split method

        """
        # Create the left cost volume, with SAD measure, window size 1, subpixel 2, disp_min -3 disp_max 1
        stereo_plugin = stereo.AbstractStereo(**{'stereo_method': 'sad', 'window_size': 1, 'subpix': 2})
        cv = stereo_plugin.compute_cost_volume(self.ref, self.sec, -3, 1, **{'valid_pixels': 0, 'no_data': 1})
        indices_nan = np.isnan(cv['cost_volume'].data)
        cv['cost_volume'].data[indices_nan] = np.inf

        # ground truth
        gt_disp = np.array([[1., 1., 1., -3.],
                            [1., -0.5, 1., -3.],
                            [1., 1., -1.5, -3]], dtype=np.float32)

        # Compute the disparity
        disp = disparity.argmin_split(cv)

        # Check if the calculated coefficient map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(gt_disp, disp)

    def test_argmax_split(self):
        """
        Test the argmax_split method

        """
        # Create the left cost volume, with ZNCC measure, window size 1, subpixel 2, disp_min -3 disp_max 1
        stereo_plugin = stereo.AbstractStereo(**{'stereo_method': 'zncc', 'window_size': 1, 'subpix': 2})
        cv = stereo_plugin.compute_cost_volume(self.ref, self.sec, -3, 1, **{'valid_pixels': 0, 'no_data': 1})
        indices_nan = np.isnan(cv['cost_volume'].data)
        cv['cost_volume'].data[indices_nan] = -np.inf

        # ground truth
        gt_disp = np.array([[0., -1., -2., -3.],
                            [0., -1., -2., -3.],
                            [0., -1., -2., -3.]], dtype=np.float32)

        # Compute the disparity
        disp = disparity.argmax_split(cv)

        # Check if the calculated coefficient map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(gt_disp, disp)

    def test_coefficient_map(self):
        """
        Test the method coefficient map

        """
        # Create the left cost volume, with SAD measure window size 1, subpixel 1, disp_min -3 disp_max 1
        stereo_plugin = stereo.AbstractStereo(**{'stereo_method': 'sad', 'window_size': 1, 'subpix': 1})
        cv = stereo_plugin.compute_cost_volume(self.ref, self.sec, -3, 1, **{'valid_pixels': 0, 'no_data': 1})

        # Compute the disparity
        disp = disparity.to_disp(cv)

        # Coefficient map ground truth, for the images described in the setUp method
        gt_coeff = np.array([[0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]])
        # Compute the disparity, and the coefficient map
        coeff = disparity.coefficient_map(cv)

        # Check if the calculated coefficient map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(coeff.data, gt_coeff)

    def test_approximate_right_disparity(self):
        """
        Test the approximate_right_disparity method

        """
        # Create the left cost volume, with SAD measure window size 3 and subpixel 1
        stereo_plugin = stereo.AbstractStereo(**{'stereo_method': 'sad', 'window_size': 3, 'subpix': 1})
        cv = stereo_plugin.compute_cost_volume(self.ref, self.sec, -2, 1, **{'valid_pixels': 0, 'no_data': 1})

        # Right disparity map ground truth, for the images described in the setUp method
        gt_disp = np.array([[0, -1]])

        # Compute the right disparity map
        disp_r = disparity.approximate_right_disparity(cv, self.sec)

        # Check if the calculated right disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(disp_r['disparity_map'].data, gt_disp)

    def test_right_disparity_subpixel(self):
        """
        Test the right disparity method, with subpixel disparity

        """
        # Create the left cost volume, with SAD measure window size 3 and subpixel 4
        stereo_plugin = stereo.AbstractStereo(**{'stereo_method': 'sad', 'window_size': 3, 'subpix': 4})
        cv = stereo_plugin.compute_cost_volume(self.ref, self.sec, -2, 1, **{'valid_pixels': 0, 'no_data': 1})

        # Right disparity map ground truth
        gt_disp = np.array([[0, -1]])

        # Compute the right disparity map
        disp_r = disparity.approximate_right_disparity(cv, self.sec)

        # Check if the calculated right disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(disp_r['disparity_map'].data, gt_disp)

    def test_right_disparity_comparaison(self):
        """
        Test the right disparity method by comparing the right disparity map calculated from scratch with the one
        calculated with the fast method

        """
        # Build the default configuration
        default_cfg = pandora.JSON_checker.default_short_configuration

        pandora_ref = read_img('tests/pandora/ref.png', no_data=np.nan, cfg=default_cfg['image'], mask=None)
        pandora_sec = read_img('tests/pandora/sec.png', no_data=np.nan, cfg=default_cfg['image'], mask=None)

        fast_cfg = {
            "stereo": {
                "stereo_method": "census"
            },
            "refinement": {
                "refinement_method": "vfit"
            },
            "validation": {
                "validation_method": "cross_checking",
                "right_left_mode": "approximate",
                "interpolated_disparity": "none",
                "filter_interpolated_disparities": True
            }
        }
        cfg = pandora.JSON_checker.update_conf(default_cfg, fast_cfg)
        ref, sec_fast = pandora.run(pandora_ref, pandora_sec, -60, 0, cfg)

        acc_cfg = {
            "stereo": {
                "stereo_method": "census"
            },
            "refinement": {
                "refinement_method": "vfit"
            },
            "validation": {
                "validation_method": "cross_checking",
                "right_left_mode": "accurate",
                "interpolated_disparity": "none",
                "filter_interpolated_disparities": True
            }
        }
        cfg = pandora.JSON_checker.update_conf(default_cfg, acc_cfg)
        ref, sec_acc = pandora.run(pandora_ref, pandora_sec, -60, 0, cfg)

        # Check if the calculated disparity map in fast mode is equal to the disparity map in accurate mode
        np.testing.assert_array_equal(sec_fast['disparity_map'].data, sec_acc['disparity_map'].data)

        # Check if the calculated coefficient map in fast mode is equal to the coefficient map in accurate mode
        np.testing.assert_array_equal(sec_fast['interpolated_coeff'].data, sec_acc['interpolated_coeff'].data)

    def test_to_disp_validity_mask(self):
        """
        Test the generated validity mask in the to_disp method

        # If bit 1 == 1 : Invalid pixel : the disparity interval is missing in the secondary image
        # If bit 2 == 1 : Information: the disparity interval is incomplete (edge reached in the secondary image)
        """
        # ------ Negative disparities ------
        # Create the left cost volume, with SAD measure window size 1, subpixel 1, disp_min -3 disp_max -1
        stereo_plugin = stereo.AbstractStereo(**{'stereo_method': 'sad', 'window_size': 1, 'subpix': 1})
        cv = stereo_plugin.compute_cost_volume(self.ref, self.sec, -3, -1, **{'valid_pixels': 0, 'no_data': 1})

        # Compute the disparity map and validity mask
        dataset = disparity.to_disp(cv)
        dataset = disparity.validity_mask(dataset, self.ref, self.sec, **{'valid_pixels': 0, 'no_data': 1})

        # Validity mask ground truth
        gt_mask = np.array([[PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE, 0],
                            [PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE, 0],
                            [PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE, 0]], dtype=np.uint16)

        # Check if the calculated disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(dataset['validity_mask'].data, gt_mask)

        # ------ Positive disparities ------
        # Create the left cost volume, with SAD measure window size 1, subpixel 1, disp_min 1 disp_max 2
        cv = stereo_plugin.compute_cost_volume(self.ref, self.sec, 1, 2, **{'valid_pixels': 0, 'no_data': 1})

        # Compute the disparity map and validity mask
        dataset = disparity.to_disp(cv)
        dataset = disparity.validity_mask(dataset, self.ref, self.sec, **{'valid_pixels': 0, 'no_data': 1})

        # Validity mask ground truth
        gt_mask = np.array([[0, 0, 1 << 2, PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING],
                            [0, 0, 1 << 2, PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING],
                            [0, 0, 1 << 2, PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING]], dtype=np.uint16)

        # Check if the calculated disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(dataset['validity_mask'].data, gt_mask)

        # ------ Negative and positive disparities ------
        # Create the left cost volume, with SAD measure window size 1, subpixel 1, disp_min -1 disp_max 1
        cv = stereo_plugin.compute_cost_volume(self.ref, self.sec, -1, 1, **{'valid_pixels': 0, 'no_data': 1})

        # Compute the disparity map and validity mask
        dataset = disparity.to_disp(cv)
        dataset = disparity.validity_mask(dataset, self.ref, self.sec, **{'valid_pixels': 0, 'no_data': 1})

        # Validity mask ground truth
        gt_mask = np.array([[PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE, 0, 0,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE],
                            [PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE, 0, 0,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE],
                            [PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE, 0, 0,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE]],
                           dtype=np.uint16)

        # Check if the calculated disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(dataset['validity_mask'].data, gt_mask)

    def test_approximate_right_disparity_validity_mask(self):
        """
        Test the generated validity mask in the right_disparity method

        # If bit 1 == 1 : Invalid pixel : the disparity interval is missing in the secondary image
        # If bit 2 == 1 : Information: the disparity interval is incomplete (edge reached in the secondary image)
        """
        # Create the left cost volume, with SAD measure window size 1 and subpixel 1
        stereo_plugin = stereo.AbstractStereo(**{'stereo_method': 'sad', 'window_size': 1, 'subpix': 1})

        # ------ Negative and positive disparities ------
        cv = stereo_plugin.compute_cost_volume(self.ref, self.sec, -2, 1, **{'valid_pixels': 0, 'no_data': 1})

        # Validity mask ground truth ( for disparities -1 0 1 2 )
        gt_mask = np.array([[PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE, 0,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE],
                            [PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE, 0,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE],
                            [PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE, 0,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE]], dtype=np.uint16)

        # Compute the right disparity map and the validity mask
        dataset = disparity.approximate_right_disparity(cv, self.sec)

        # Check if the calculated right disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(dataset['validity_mask'].data, gt_mask)

        # ------ Negative disparities ------
        cv = stereo_plugin.compute_cost_volume(self.ref, self.sec, 1, 2, **{'valid_pixels': 0, 'no_data': 1})

        # Validity mask ground truth ( for disparities -2 -1 )
        gt_mask = np.array([[PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE,
                             0, 0],
                            [PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE,
                             0, 0],
                            [PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE,
                             0, 0]], dtype=np.uint16)

        # Compute the right disparity map and the validity mask
        dataset = disparity.approximate_right_disparity(cv, self.sec)

        # Check if the calculated right disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(dataset['validity_mask'].data, gt_mask)

        # ------ Positive disparities ------
        cv = stereo_plugin.compute_cost_volume(self.ref, self.sec, -2, -1, **{'valid_pixels': 0, 'no_data': 1})

        # Validity mask ground truth ( for disparities 1 2 )
        gt_mask = np.array([[0, 0, PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING],
                            [0, 0, PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING],
                            [0, 0, PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING]], dtype=np.uint16)

        # Compute the right disparity map and the validity mask
        dataset = disparity.approximate_right_disparity(cv, self.sec)

        # Check if the calculated right disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(dataset['validity_mask'].data, gt_mask)

    def test_recover_size(self):
        """
        Test the recover_size method

        """
        ref = xr.Dataset({'disparity_map': (['row', 'col'], np.array([[1, -1]], dtype=np.float32)),
                          'confidence_measure': (['row', 'col', 'indicator'],
                                                 np.array([[[1., 0., 0.], [1., 0., 0.]]],
                                                          dtype=np.float32)),
                          'validity_mask': (['row', 'col'],
                                            np.array([[PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE,
                                                       PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE]],
                                                                     dtype=np.uint16))},
                         coords={'row': [2], 'col': [2, 3]})
        ref.attrs['offset_row_col'] = 2

        # Disparity map ground truth with the size of the input images
        gt_disp = np.array([[0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 1, -1, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0]], dtype=np.float32)

        # Confidence measure ground truth with the size of the input images
        gt_confidence = np.array([[[np.nan, np.nan, np.nan],
                                   [np.nan, np.nan, np.nan],
                                   [np.nan, np.nan, np.nan],
                                   [np.nan, np.nan, np.nan],
                                   [np.nan, np.nan, np.nan],
                                   [np.nan, np.nan, np.nan]],
                                  [[np.nan, np.nan, np.nan],
                                   [np.nan, np.nan, np.nan],
                                   [np.nan, np.nan, np.nan],
                                   [np.nan, np.nan, np.nan],
                                   [np.nan, np.nan, np.nan],
                                   [np.nan, np.nan, np.nan]],
                                  [[np.nan, np.nan, np.nan],
                                   [np.nan, np.nan, np.nan],
                                   [1., 0., 0.],
                                   [1., 0., 0.],
                                   [np.nan, np.nan, np.nan],
                                   [np.nan, np.nan, np.nan]],
                                  [[np.nan, np.nan, np.nan],
                                   [np.nan, np.nan, np.nan],
                                   [np.nan, np.nan, np.nan],
                                   [np.nan, np.nan, np.nan],
                                   [np.nan, np.nan, np.nan],
                                   [np.nan, np.nan, np.nan]],
                                  [[np.nan, np.nan, np.nan],
                                   [np.nan, np.nan, np.nan],
                                   [np.nan, np.nan, np.nan],
                                   [np.nan, np.nan, np.nan],
                                   [np.nan, np.nan, np.nan],
                                   [np.nan, np.nan, np.nan]]])

        # Validity mask ground truth with the size of the input images
        gt_mask = np.array([[PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER, PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER,
                             PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER, PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER,
                             PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER, PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER],
                            [PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER, PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER,
                             PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER, PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER,
                             PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER, PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER],
                            [PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER, PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER, PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER],
                            [PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER, PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER,
                             PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER, PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER,

                             PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER, PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER],
                            [PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER, PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER,
                             PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER, PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER,
                             PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER, PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER]],
                           dtype=np.uint16)

        # Resize the products
        reference_disparity = disparity.resize(ref, invalid_value=0)

        # Check if the products is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(reference_disparity['disparity_map'].data, gt_disp)
        np.testing.assert_array_equal(reference_disparity['confidence_measure'].data, gt_confidence)
        np.testing.assert_array_equal(reference_disparity['validity_mask'].data, gt_mask)

        sec = xr.Dataset({'disparity_map': (['row', 'col'], np.array([[1, -1], [2, -10]], dtype=np.float32)),
                          'confidence_measure': (['row', 'col', 'indicator'],
                                                 np.array([[[1., 0, 0], [1., 0, 0]],
                                                           [[1., 0, 0], [1., 0, 0]]],
                                                          dtype=np.float32)),
                          'validity_mask': (['row', 'col'],
                                            np.array([[0, PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE],
                                                                      [1 << 5, 0]],  dtype=bool))},
                         coords={'row': [1, 2], 'col': [1, 2]})

        sec.attrs['offset_row_col'] = 1

        # Disparity map ground truth with the size of the input images
        gt_disp_sec = np.array([[0, 0, 0, 0],
                               [0, 1, -1, 0],
                               [0, 2, -10, 0],
                               [0, 0, 0, 0]], dtype=np.float32)

        # Confidence measure ground truth with the size of the input images
        gt_confidence_sec = np.array([[[np.nan, np.nan, np.nan],
                                   [np.nan, np.nan, np.nan],
                                   [np.nan, np.nan, np.nan],
                                   [np.nan, np.nan, np.nan]],
                                  [[np.nan, np.nan, np.nan],
                                   [1., 0., 0.],
                                   [1., 0., 0.],
                                   [np.nan, np.nan, np.nan]],
                                  [[np.nan, np.nan, np.nan],
                                   [1., 0., 0.],
                                   [1., 0., 0.],
                                   [np.nan, np.nan, np.nan]],
                                  [[np.nan, np.nan, np.nan],
                                   [np.nan, np.nan, np.nan],
                                   [np.nan, np.nan, np.nan],
                                   [np.nan, np.nan, np.nan]]])

        # Validity mask ground truth with the size of the input images
        gt_mask_sec = np.array([[PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER, PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER,
                                 PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER, PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER],
                                [PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER, 0,
                                 PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE,
                                 PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER],
                                [PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER, PANDORA_MSK_PIXEL_FILLED_MISMATCH, 0,
                                 PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER],
                                [PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER, PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER,
                                 PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER, PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER]],
                               dtype=bool)

        # Resize the products
        secondary_disparity = disparity.resize(sec, invalid_value=0)

        # Check if the products is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(secondary_disparity['disparity_map'].data, gt_disp_sec)
        np.testing.assert_array_equal(secondary_disparity['confidence_measure'].data, gt_confidence_sec)
        np.testing.assert_array_equal(secondary_disparity['validity_mask'].data, gt_mask_sec)

    def test_validity_mask(self):
        """
        # If bit 0 == 1 : Invalid pixel : the disparity interval is missing in the secondary image
        # If bit 1 == 1 : Invalid pixel : the disparity interval is missing in the secondary image
        # If bit 2 == 1 : Information: the disparity interval is incomplete (edge reached in the secondary image)
        # If bit 6 == 1 : Invalid pixel : invalidated by the validity mask of the reference image given as input
        # If bit 7 == 1 : Invalid pixel : secondary positions invalidated by the mask of the secondary image given as
        #    input

        """
        # Masks convention
        # 1 = valid
        # 2 = no_data
        # ---------------------- Test with positive and negative disparity range ----------------------
        data = np.array(([[1, 2, 4, 6],
                          [2, 4, 1, 6],
                          [6, 7, 8, 10]]), dtype=np.float64)
        ref_mask = np.array([[2, 1, 1, 1],
                             [1, 2, 4, 1],
                             [5, 1, 1, 2]], dtype=np.uint8)
        ref = xr.Dataset({'im': (['row', 'col'], data),
                          'msk': (['row', 'col'], ref_mask)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})

        data = np.array(([[6, 1, 2, 4],
                          [6, 2, 4, 1],
                          [10, 6, 7, 8]]), dtype=np.float64)
        sec_mask = np.array([[1, 1, 3, 5],
                             [4, 1, 1, 1],
                             [2, 2, 4, 6]], dtype=np.uint8)

        sec = xr.Dataset({'im': (['row', 'col'], data),
                          'msk': (['row', 'col'], sec_mask)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})

        stereo_plugin = stereo.AbstractStereo(**{'stereo_method': 'sad', 'window_size': 1, 'subpix': 1})
        cv = stereo_plugin.compute_cost_volume(ref, sec, -1, 1, **{'valid_pixels': 1, 'no_data': 2})

        # Compute the disparity map and validity mask
        dataset = disparity.to_disp(cv)
        dataset = disparity.validity_mask(dataset, ref, sec, **{'valid_pixels': 1, 'no_data': 2})

        # Validity mask ground truth
        gt_mask = np.array([[PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE + PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER,
                             0, 0,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE + PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_SEC],
                            [PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE, PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER,
                             PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_REF, PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE],
                            [PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE + PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_REF +
                             PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING, 0, 0,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE + PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER +
                             PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_SEC]], dtype=np.uint16)

        # Check if the calculated validity mask is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(dataset['validity_mask'].data, gt_mask)

        # ---------------------- Test with negative disparity range ----------------------
        cv = stereo_plugin.compute_cost_volume(ref, sec, -2, -1, **{'valid_pixels': 1, 'no_data': 2})

        # Compute the disparity map and validity mask
        dataset = disparity.to_disp(cv)
        dataset = disparity.validity_mask(dataset, ref, sec, **{'valid_pixels': 1, 'no_data': 2})

        # Validity mask ground truth
        gt_mask = np.array([[PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING +
                             PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE, 0, 0],
                            [PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER +
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE + PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_SEC,
                             PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_REF, 0],
                            [PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING +
                             PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_REF,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER]],
                           dtype=np.uint16)

        # Check if the calculated validity mask is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(dataset['validity_mask'].data, gt_mask)

        # ---------------------- Test with positive disparity range ----------------------
        cv = stereo_plugin.compute_cost_volume(ref, sec, 1, 2, **{'valid_pixels': 1, 'no_data': 2})

        # Compute the disparity map and validity mask
        dataset = disparity.to_disp(cv)
        dataset = disparity.validity_mask(dataset, ref, sec, **{'valid_pixels': 1, 'no_data': 2})

        # Validity mask ground truth
        gt_mask = np.array([[PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER, PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_SEC,
                             PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_SEC + PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING],
                            [0, PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_REF,
                             PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING],
                            [PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_REF, PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_SEC,
                             PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_SEC + PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER +
                             PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING]],
                           dtype=np.uint16)

        # Check if the calculated validity mask is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(dataset['validity_mask'].data, gt_mask)

        # ---------------------- Test with positive and negative disparity range and window size = 3----------------
        data = np.array(([[1, 2, 4, 6, 1],
                          [2, 4, 1, 6, 1],
                          [6, 7, 8, 10, 1],
                          [0, 5, 6, 7, 8]]), dtype=np.float64)
        ref_mask = np.array([[2, 1, 1, 1, 1],
                             [1, 2, 4, 1, 1],
                             [5, 2, 1, 1, 1],
                             [1, 1, 1, 1, 1]], dtype=np.uint8)
        ref = xr.Dataset({'im': (['row', 'col'], data),
                          'msk': (['row', 'col'], ref_mask)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})

        data = np.array(([[6, 1, 2, 4, 1],
                          [6, 2, 4, 1, 6],
                          [10, 6, 7, 8, 1],
                          [5, 6, 7, 8, 0]]), dtype=np.float64)
        sec_mask = np.array([[1, 1, 1, 2, 1],
                             [5, 1, 1, 1, 1],
                             [2, 1, 1, 6, 1],
                             [0, 1, 1, 1, 1]], dtype=np.uint8)

        sec = xr.Dataset({'im': (['row', 'col'], data),
                          'msk': (['row', 'col'], sec_mask)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})

        stereo_plugin = stereo.AbstractStereo(**{'stereo_method': 'sad', 'window_size': 3, 'subpix': 1})
        cv = stereo_plugin.compute_cost_volume(ref, sec, -1, 1, **{'valid_pixels': 1, 'no_data': 2})

        # Compute the disparity map and validity mask
        dataset = disparity.to_disp(cv)
        dataset = disparity.validity_mask(dataset, ref, sec, **{'valid_pixels': 1, 'no_data': 2})

        # Validity mask ground truth
        gt_mask = np.array([[PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE + PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER +
                             PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER +
                             PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING +
                             PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_REF,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING],
                            [PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE + PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER,
                             PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER, PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE]],
                           dtype=np.uint16)

        # Check if the calculated validity mask is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(dataset['validity_mask'].data, gt_mask)

        # ---------------------- Test with positive and negative disparity range on flag 1 ----------------------
        # Masks convention
        # 1 = valid
        # 0 = no_data

        data = np.ones((10, 10), dtype=np.float64)
        ref_mask = np.ones((10, 10), dtype=np.uint8)

        ref = xr.Dataset({'im': (['row', 'col'], data),
                          'msk': (['row', 'col'], ref_mask)},
                         coords={'row': np.arange(5, data.shape[0] + 5), 'col': np.arange(4, data.shape[1] + 4)})

        data = np.ones((10, 10), dtype=np.float64)
        sec_mask = np.ones((10, 10), dtype=np.uint8)
        sec_mask = np.tril(sec_mask, -1.5)

        sec = xr.Dataset({'im': (['row', 'col'], data),
                          'msk': (['row', 'col'], sec_mask)},
                         coords={'row': np.arange(5, data.shape[0] + 5), 'col': np.arange(4, data.shape[1] + 4)})

        stereo_plugin = stereo.AbstractStereo(**{'stereo_method': 'sad', 'window_size': 3, 'subpix': 1})
        cv = stereo_plugin.compute_cost_volume(ref, sec, -3, 2, **{'valid_pixels': 1, 'no_data': 0})

        # Compute the disparity map and validity mask
        dataset = disparity.to_disp(cv)
        dataset = disparity.validity_mask(dataset, ref, sec, **{'valid_pixels': 1, 'no_data': 0})

        # Validity mask ground truth
        gt_mask = np.array([[PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING],
                            [PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING],
                            [PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING],
                            [PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE, 0,
                             PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING],
                            [PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE, 0, 0,
                             PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING],
                            [PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE, 0, 0, 0,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING],
                            [PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE, 0, 0, 0,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING],
                            [PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE, 0, 0, 0,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE]],
                           dtype=np.uint8)

        # Check if the calculated validity mask is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(dataset['validity_mask'].data, gt_mask)


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
