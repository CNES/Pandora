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

import json
import logging
import logging.config
import os
import unittest

import numpy as np
import pandora
import pandora.disparity as disparity
import pandora.stereo as stereo
import xarray as xr
from pandora.common import resize
from pandora.constants import *
from pandora.img_tools import read_img
from pandora.state_machine import PandoraMachine


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
        self.left = xr.Dataset({'im': (['row', 'col'], data)},
                               coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})
        self.left.attrs = {'valid_pixels': 0, 'no_data_mask': 1}

        data = np.array(([[6, 1, 2, 4],
                          [6, 2, 4, 1],
                          [10, 6, 7, 8]]), dtype=np.float64)
        self.right = xr.Dataset({'im': (['row', 'col'], data)},
                                coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})
        self.right.attrs = {'valid_pixels': 0, 'no_data_mask': 1}

    def test_to_disp(self):
        """
        Test the to disp method

        """
        # Create the left cost volume, with SAD measure window size 1, subpixel 1, disp_min -3 disp_max 1
        stereo_plugin = stereo.AbstractStereo(**{'stereo_method': 'sad', 'window_size': 1, 'subpix': 1})
        cv = stereo_plugin.compute_cost_volume(self.left, self.right, -3, 1)

        # Disparity map ground truth, for the images described in the setUp method
        gt_disp = np.array([[1, 1, 1, -3],
                            [1, 1, 1, -3],
                            [1, 1, 1, -3]])

        # Compute the disparity
        disparity_ = disparity.AbstractDisparity(**{'disparity_method': 'wta', 'invalid_disparity': 0})
        disp = disparity_.to_disp(cv)

        # Check if the calculated disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(disp['disparity_map'].data, gt_disp)

        #
        # Test the to_disp method with negative disparity range
        #
        cv = stereo_plugin.compute_cost_volume(self.left, self.right, -3, -1)

        # Disparity map ground truth
        gt_disp = np.array([[0, -1, -2, -3],
                            [0, -1, -1, -3],
                            [0, -1, -2, -3]])

        # Compute the disparity
        disp = disparity_.to_disp(cv)

        # Check if the calculated disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(disp['disparity_map'].data, gt_disp)

        #
        # Test the to_disp method with positive disparity range
        #
        cv = stereo_plugin.compute_cost_volume(self.left, self.right, 1, 3)

        # Disparity map ground truth
        gt_disp = np.array([[1, 1, 1, 0],
                            [1, 1, 1, 0],
                            [1, 1, 1, 0]])

        # Compute the disparity
        disp = disparity_.to_disp(cv)

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
        cv = stereo_plugin.compute_cost_volume(self.left, self.right, -3, 1)
        indices_nan = np.isnan(cv['cost_volume'].data)
        cv['cost_volume'].data[indices_nan] = np.inf

        # ground truth
        gt_disp = np.array([[1., 1., 1., -3.],
                            [1., -0.5, 1., -3.],
                            [1., 1., -1.5, -3]], dtype=np.float32)

        # Compute the disparity
        disparity_ = disparity.AbstractDisparity(**{'disparity_method': 'wta', 'invalid_disparity': 0})
        disp = disparity_.argmin_split(cv)

        # Check if the calculated coefficient map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(gt_disp, disp)

    def test_argmax_split(self):
        """
        Test the argmax_split method

        """
        # Create the left cost volume, with ZNCC measure, window size 1, subpixel 2, disp_min -3 disp_max 1
        stereo_plugin = stereo.AbstractStereo(**{'stereo_method': 'zncc', 'window_size': 1, 'subpix': 2})
        cv = stereo_plugin.compute_cost_volume(self.left, self.right, -3, 1)
        indices_nan = np.isnan(cv['cost_volume'].data)
        cv['cost_volume'].data[indices_nan] = -np.inf

        # ground truth
        gt_disp = np.array([[0., -1., -2., -3.],
                            [0., -1., -2., -3.],
                            [0., -1., -2., -3.]], dtype=np.float32)

        # Compute the disparity
        disparity_ = disparity.AbstractDisparity(**{'disparity_method': 'wta', 'invalid_disparity': 0})
        disp = disparity_.argmax_split(cv)

        # Check if the calculated coefficient map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(gt_disp, disp)

    def test_coefficient_map(self):
        """
        Test the method coefficient map

        """
        # Create the left cost volume, with SAD measure window size 1, subpixel 1, disp_min -3 disp_max 1
        stereo_plugin = stereo.AbstractStereo(**{'stereo_method': 'sad', 'window_size': 1, 'subpix': 1})
        cv = stereo_plugin.compute_cost_volume(self.left, self.right, -3, 1)

        # Compute the disparity
        disparity_ = disparity.AbstractDisparity(**{'disparity_method': 'wta', 'invalid_disparity': 0})
        disp = disparity_.to_disp(cv)

        # Coefficient map ground truth, for the images described in the setUp method
        gt_coeff = np.array([[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]])
        # Compute the disparity, and the coefficient map
        coeff = disparity_.coefficient_map(cv)

        # Check if the calculated coefficient map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(coeff.data, gt_coeff)

    def test_approximate_right_disparity(self):
        """
        Test the approximate_right_disparity method

        """
        # Create the left cost volume, with SAD measure window size 3 and subpixel 1
        stereo_plugin = stereo.AbstractStereo(**{'stereo_method': 'sad', 'window_size': 3, 'subpix': 1})
        cv = stereo_plugin.compute_cost_volume(self.left, self.right, -2, 1)

        # Right disparity map ground truth, for the images described in the setUp method
        gt_disp = np.array([[0, -1]])

        # Compute the right disparity map
        disparity_ = disparity.AbstractDisparity(**{'disparity_method': 'wta', 'invalid_disparity': 0})
        disp_r = disparity_.approximate_right_disparity(cv, self.right)

        # Check if the calculated right disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(disp_r['disparity_map'].data, gt_disp)

    def test_right_disparity_subpixel(self):
        """
        Test the right disparity method, with subpixel disparity

        """
        # Create the left cost volume, with SAD measure window size 3 and subpixel 4
        stereo_plugin = stereo.AbstractStereo(**{'stereo_method': 'sad', 'window_size': 3, 'subpix': 4})
        cv = stereo_plugin.compute_cost_volume(self.left, self.right, -2, 1)

        # Right disparity map ground truth
        gt_disp = np.array([[0, -1]])

        # Compute the right disparity map
        disparity_ = disparity.AbstractDisparity(**{'disparity_method': 'wta', 'invalid_disparity': 0})
        disp_r = disparity_.approximate_right_disparity(cv, self.right)

        # Check if the calculated right disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(disp_r['disparity_map'].data, gt_disp)

    def test_right_disparity_comparaison(self):
        """
        Test the right disparity method by comparing the right disparity map calculated from scratch with the one
        calculated with the fast method

        """
        # Build the default configuration
        default_cfg = pandora.JSON_checker.default_short_configuration

        pandora_left = read_img('tests/pandora/left.png', no_data=np.nan, cfg=default_cfg['image'], mask=None)
        pandora_right = read_img('tests/pandora/right.png', no_data=np.nan, cfg=default_cfg['image'], mask=None)

        fast_cfg = {
            "pipeline": {
                "right_disp_map": {
                    "method": "accurate"
                },
                "stereo": {
                    "stereo_method": "census"
                },
                "disparity": {
                    "disparity_method": "wta"
                },
                "refinement": {
                    "refinement_method": "vfit"
                },
                "validation": {
                    "validation_method": "cross_checking",
                    "right_left_mode": "approximate"
                }
            }
        }

        pandora_machine_fast = PandoraMachine()
        cfg = pandora.JSON_checker.update_conf(default_cfg, fast_cfg)
        left, right_fast = pandora.run(pandora_machine_fast, pandora_left, pandora_right, -60, 0, cfg)

        acc_cfg = {
            "pipeline":
                {
                    "right_disp_map": {
                        "method": "accurate"
                    },
                    "stereo": {
                        "stereo_method": "census"
                    },
                    "disparity": {
                        "disparity_method": "wta"
                    },
                    "refinement": {
                        "refinement_method": "vfit"
                    },
                    "validation": {
                        "validation_method": "cross_checking",
                        "right_left_mode": "accurate",
                    }
                }
        }

        pandora_machine_acc = PandoraMachine()
        cfg = pandora.JSON_checker.update_conf(default_cfg, acc_cfg)
        left, right_acc = pandora.run(pandora_machine_acc, pandora_left, pandora_right, -60, 0, cfg)
        # Check if the calculated disparity map in fast mode is equal to the disparity map in accurate mode
        np.testing.assert_array_equal(right_fast['disparity_map'].data, right_acc['disparity_map'].data)

        # Check if the calculated coefficient map in fast mode is equal to the coefficient map in accurate mode
        np.testing.assert_array_equal(right_fast['interpolated_coeff'].data, right_acc['interpolated_coeff'].data)

    def test_to_disp_validity_mask(self):
        """
        Test the generated validity mask in the to_disp method

        # If bit 1 == 1 : Invalid pixel : the disparity interval is missing in the right image
        # If bit 2 == 1 : Information: the disparity interval is incomplete (edge reached in the right image)
        """
        # ------ Negative disparities ------
        # Create the left cost volume, with SAD measure window size 1, subpixel 1, disp_min -3 disp_max -1
        stereo_plugin = stereo.AbstractStereo(**{'stereo_method': 'sad', 'window_size': 1, 'subpix': 1})
        cv = stereo_plugin.compute_cost_volume(self.left, self.right, -3, -1)

        # Compute the disparity map and validity mask
        disparity_ = disparity.AbstractDisparity(**{'disparity_method': 'wta', 'invalid_disparity': 0})
        dataset = disparity_.to_disp(cv)
        disparity_.validity_mask(dataset, self.left, self.right, cv)

        # Validity mask ground truth
        gt_mask = np.array([[PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE, 0],
                            [PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE, 0],
                            [PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE, 0]], dtype=np.uint16)

        # Check if the calculated disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(dataset['validity_mask'].data, gt_mask)

        # ------ Positive disparities ------
        # Create the left cost volume, with SAD measure window size 1, subpixel 1, disp_min 1 disp_max 2
        cv = stereo_plugin.compute_cost_volume(self.left, self.right, 1, 2)

        # Compute the disparity map and validity mask
        dataset = disparity_.to_disp(cv)
        disparity_.validity_mask(dataset, self.left, self.right, cv)

        # Validity mask ground truth
        gt_mask = np.array([[0, 0, 1 << 2, PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING],
                            [0, 0, 1 << 2, PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING],
                            [0, 0, 1 << 2, PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING]], dtype=np.uint16)

        # Check if the calculated disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(dataset['validity_mask'].data, gt_mask)

        # ------ Negative and positive disparities ------
        # Create the left cost volume, with SAD measure window size 1, subpixel 1, disp_min -1 disp_max 1
        cv = stereo_plugin.compute_cost_volume(self.left, self.right, -1, 1)

        # Compute the disparity map and validity mask
        dataset = disparity_.to_disp(cv)
        disparity_.validity_mask(dataset, self.left, self.right, cv)

        # Validity mask ground truth
        gt_mask = np.array([[PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE, 0, 0,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE],
                            [PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE, 0, 0,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE],
                            [PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE, 0, 0,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE]],
                           dtype=np.uint16)

        # Check if the calculated disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(dataset['validity_mask'].data, gt_mask)

        # ------ Variable grids of disparities ------
        # Disp_min and disp_max
        disp_min_grid = np.array([[-3, -2, -3, -1],
                                  [-2, -2, -1, -3],
                                  [-1, -2, -2, -3]])

        disp_max_grid = np.array([[-1, -1, -2, 0],
                                  [0, -1, 0, 0],
                                  [0, 0, -1, -1]])

        # Create the left cost volume, with SAD measure window size 1, subpixel 1, disp_min -3 disp_max -1
        stereo_plugin = stereo.AbstractStereo(**{'stereo_method': 'sad', 'window_size': 1, 'subpix': 1})
        dmin, dmax = stereo_plugin.dmin_dmax(disp_min_grid, disp_max_grid)
        cv = stereo_plugin.compute_cost_volume(self.left, self.right, dmin, dmax)
        stereo_plugin.cv_masked(self.left, self.right, cv, disp_min_grid, disp_max_grid)

        # Compute the disparity map and validity mask
        dataset = disparity_.to_disp(cv)
        disparity_.validity_mask(dataset, self.left, self.right, cv)

        # Validity mask ground truth
        gt_mask = np.array([[PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE, 0],
                            [PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE, 0],
                            [PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE, 0]], dtype=np.uint16)

        # Check if the calculated disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(dataset['validity_mask'].data, gt_mask)

    def test_approximate_right_disparity_validity_mask(self):
        """
        Test the generated validity mask in the right_disparity method

        # If bit 1 == 1 : Invalid pixel : the disparity interval is missing in the right image
        # If bit 2 == 1 : Information: the disparity interval is incomplete (edge reached in the right image)
        """
        # Create the left cost volume, with SAD measure window size 1 and subpixel 1
        stereo_plugin = stereo.AbstractStereo(**{'stereo_method': 'sad', 'window_size': 1, 'subpix': 1})

        # ------ Negative and positive disparities ------
        cv = stereo_plugin.compute_cost_volume(self.left, self.right, -2, 1)

        # Validity mask ground truth ( for disparities -1 0 1 2 )
        gt_mask = np.array([[PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE, 0,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE],
                            [PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE, 0,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE],
                            [PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE, 0,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE]], dtype=np.uint16)

        # Compute the right disparity map and the validity mask
        disparity_ = disparity.AbstractDisparity(**{'disparity_method': 'wta', 'invalid_disparity': 0})
        dataset = disparity_.approximate_right_disparity(cv, self.right)

        # Check if the calculated right disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(dataset['validity_mask'].data, gt_mask)

        # ------ Negative disparities ------
        cv = stereo_plugin.compute_cost_volume(self.left, self.right, 1, 2)

        # Validity mask ground truth ( for disparities -2 -1 )
        gt_mask = np.array([[PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                             0, 0],
                            [PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                             0, 0],
                            [PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                             0, 0]], dtype=np.uint16)

        # Compute the right disparity map and the validity mask
        dataset = disparity_.approximate_right_disparity(cv, self.right)

        # Check if the calculated right disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(dataset['validity_mask'].data, gt_mask)

        # ------ Positive disparities ------
        cv = stereo_plugin.compute_cost_volume(self.left, self.right, -2, -1)

        # Validity mask ground truth ( for disparities 1 2 )
        gt_mask = np.array([[0, 0, PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING],
                            [0, 0, PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING],
                            [0, 0, PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING]], dtype=np.uint16)

        # Compute the right disparity map and the validity mask
        dataset = disparity_.approximate_right_disparity(cv, self.right)

        # Check if the calculated right disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(dataset['validity_mask'].data, gt_mask)

    def test_recover_size(self):
        """
        Test the recover_size method

        """
        left = xr.Dataset({'disparity_map': (['row', 'col'], np.array([[1, -1]], dtype=np.float32)),
                           'confidence_measure': (['row', 'col', 'indicator'],
                                                  np.array([[[1., 0., 0.], [1., 0., 0.]]],
                                                           dtype=np.float32)),
                           'validity_mask': (['row', 'col'],
                                             np.array([[PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                                                        PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE]],
                                                      dtype=np.uint16))},
                          coords={'row': [2], 'col': [2, 3]})
        left.attrs['offset_row_col'] = 2

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
        gt_mask = np.array([[PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER, PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                             PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER, PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                             PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER, PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER],
                            [PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER, PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                             PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER, PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                             PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER, PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER],
                            [PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER, PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER, PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER],
                            [PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER, PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                             PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER, PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,

                             PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER, PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER],
                            [PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER, PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                             PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER, PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                             PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER, PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER]],
                           dtype=np.uint16)

        # Resize the products
        left_disparity = resize(left, border_disparity=0)

        # Check if the products is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(left_disparity['disparity_map'].data, gt_disp)
        np.testing.assert_array_equal(left_disparity['confidence_measure'].data, gt_confidence)
        np.testing.assert_array_equal(left_disparity['validity_mask'].data, gt_mask)

        right = xr.Dataset({'disparity_map': (['row', 'col'], np.array([[1, -1], [2, -10]], dtype=np.float32)),
                            'confidence_measure': (['row', 'col', 'indicator'],
                                                   np.array([[[1., 0, 0], [1., 0, 0]],
                                                             [[1., 0, 0], [1., 0, 0]]],
                                                            dtype=np.float32)),
                            'validity_mask': (['row', 'col'],
                                              np.array([[0, PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE],
                                                        [1 << 5, 0]], dtype=bool))},
                           coords={'row': [1, 2], 'col': [1, 2]})

        right.attrs['offset_row_col'] = 1

        # Disparity map ground truth with the size of the input images
        gt_disp_right = np.array([[0, 0, 0, 0],
                                  [0, 1, -1, 0],
                                  [0, 2, -10, 0],
                                  [0, 0, 0, 0]], dtype=np.float32)

        # Confidence measure ground truth with the size of the input images
        gt_confidence_right = np.array([[[np.nan, np.nan, np.nan],
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
        gt_mask_right = np.array([[PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER, PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                                   PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER, PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER],
                                  [PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER, 0,
                                   PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                                   PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER],
                                  [PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER, PANDORA_MSK_PIXEL_FILLED_MISMATCH, 0,
                                   PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER],
                                  [PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER, PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                                   PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER, PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER]],
                                 dtype=bool)

        # Resize the products
        right_disparity = resize(right, border_disparity=0)

        # Check if the products is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(right_disparity['disparity_map'].data, gt_disp_right)
        np.testing.assert_array_equal(right_disparity['confidence_measure'].data, gt_confidence_right)
        np.testing.assert_array_equal(right_disparity['validity_mask'].data, gt_mask_right)

    def test_validity_mask(self):
        """
        # If bit 0 == 1 : Invalid pixel : the disparity interval is missing in the right image
        # If bit 1 == 1 : Invalid pixel : the disparity interval is missing in the right image
        # If bit 2 == 1 : Information: the disparity interval is incomplete (edge reached in the right image)
        # If bit 6 == 1 : Invalid pixel : invalidated by the validity mask of the left image given as input
        # If bit 7 == 1 : Invalid pixel : right positions invalidated by the mask of the right image given as
        #    input

        """
        # Masks convention
        # 1 = valid
        # 2 = no_data
        # ---------------------- Test with positive and negative disparity range ----------------------
        data = np.array(([[1, 2, 4, 6],
                          [2, 4, 1, 6],
                          [6, 7, 8, 10]]), dtype=np.float64)
        left_mask = np.array([[2, 1, 1, 1],
                              [1, 2, 4, 1],
                              [5, 1, 1, 2]], dtype=np.uint8)
        left = xr.Dataset({'im': (['row', 'col'], data),
                           'msk': (['row', 'col'], left_mask)},
                          coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})
        left.attrs = {'valid_pixels': 1, 'no_data_mask': 2}

        data = np.array(([[6, 1, 2, 4],
                          [6, 2, 4, 1],
                          [10, 6, 7, 8]]), dtype=np.float64)
        right_mask = np.array([[1, 1, 3, 5],
                               [4, 1, 1, 1],
                               [2, 2, 4, 6]], dtype=np.uint8)

        right = xr.Dataset({'im': (['row', 'col'], data),
                            'msk': (['row', 'col'], right_mask)},
                           coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})
        right.attrs = {'valid_pixels': 1, 'no_data_mask': 2}

        stereo_plugin = stereo.AbstractStereo(**{'stereo_method': 'sad', 'window_size': 1, 'subpix': 1})
        cv = stereo_plugin.compute_cost_volume(left, right, -1, 1)

        # Compute the disparity map and validity mask
        disparity_ = disparity.AbstractDisparity(**{'disparity_method': 'wta', 'invalid_disparity': 0})
        dataset = disparity_.to_disp(cv)
        disparity_.validity_mask(dataset, left, right, cv)

        # Validity mask ground truth
        gt_mask = np.array(
            [[PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE + PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
              0, 0,
              PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE + PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_RIGHT],
             [PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE, PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
              PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT, PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE],
             [PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE + PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT +
              PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING, 0, 0,
              PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE + PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER +
              PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_RIGHT]], dtype=np.uint16)

        # Check if the calculated validity mask is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(dataset['validity_mask'].data, gt_mask)

        # ---------------------- Test with negative disparity range ----------------------
        cv = stereo_plugin.compute_cost_volume(left, right, -2, -1)

        # Compute the disparity map and validity mask
        dataset = disparity_.to_disp(cv)
        disparity_.validity_mask(dataset, left, right, cv)

        # Validity mask ground truth
        gt_mask = np.array([[PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING +
                             PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE, 0, 0],
                            [PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER +
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE + PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_RIGHT,
                             PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT, 0],
                            [PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING +
                             PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER]],
                           dtype=np.uint16)

        # Check if the calculated validity mask is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(dataset['validity_mask'].data, gt_mask)

        # ---------------------- Test with positive disparity range ----------------------
        cv = stereo_plugin.compute_cost_volume(left, right, 1, 2)

        # Compute the disparity map and validity mask
        dataset = disparity_.to_disp(cv)
        disparity_.validity_mask(dataset, left, right, cv)

        # Validity mask ground truth
        gt_mask = np.array([[PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER, PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_RIGHT,
                             PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_RIGHT + PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING],
                            [0, PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT,
                             PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING],
                            [PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT, PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_RIGHT,
                             PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_RIGHT + PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER +
                             PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING]],
                           dtype=np.uint16)

        # Check if the calculated validity mask is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(dataset['validity_mask'].data, gt_mask)

        # ---------------------- Test with positive and negative disparity range and window size = 3----------------
        data = np.array(([[1, 2, 4, 6, 1],
                          [2, 4, 1, 6, 1],
                          [6, 7, 8, 10, 1],
                          [0, 5, 6, 7, 8]]), dtype=np.float64)
        left_mask = np.array([[2, 1, 1, 1, 1],
                              [1, 2, 4, 1, 1],
                              [5, 2, 1, 1, 1],
                              [1, 1, 1, 1, 1]], dtype=np.uint8)
        left = xr.Dataset({'im': (['row', 'col'], data),
                           'msk': (['row', 'col'], left_mask)},
                          coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})
        left.attrs = {'valid_pixels': 1, 'no_data_mask': 2}

        data = np.array(([[6, 1, 2, 4, 1],
                          [6, 2, 4, 1, 6],
                          [10, 6, 7, 8, 1],
                          [5, 6, 7, 8, 0]]), dtype=np.float64)
        right_mask = np.array([[1, 1, 1, 2, 1],
                               [5, 1, 1, 1, 1],
                               [2, 1, 1, 6, 1],
                               [0, 1, 1, 1, 1]], dtype=np.uint8)

        right = xr.Dataset({'im': (['row', 'col'], data),
                            'msk': (['row', 'col'], right_mask)},
                           coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})
        right.attrs = {'valid_pixels': 1, 'no_data_mask': 2}

        stereo_plugin = stereo.AbstractStereo(**{'stereo_method': 'sad', 'window_size': 3, 'subpix': 1})
        cv = stereo_plugin.compute_cost_volume(left, right, -1, 1)

        # Compute the disparity map and validity mask
        dataset = disparity_.to_disp(cv)
        disparity_.validity_mask(dataset, left, right, cv)

        # Validity mask ground truth
        gt_mask = np.array(
            [[PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE + PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER +
              PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
              PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER +
              PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING +
              PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT,
              PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE +
              PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING],
             [PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE + PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
              PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER, PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE]],
            dtype=np.uint16)

        # Check if the calculated validity mask is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(dataset['validity_mask'].data, gt_mask)

        # ---------------------- Test with positive and negative disparity range on flag 1 ----------------------
        # Masks convention
        # 1 = valid
        # 0 = no_data

        data = np.ones((10, 10), dtype=np.float64)
        left_mask = np.ones((10, 10), dtype=np.uint8)

        left = xr.Dataset({'im': (['row', 'col'], data),
                           'msk': (['row', 'col'], left_mask)},
                          coords={'row': np.arange(5, data.shape[0] + 5), 'col': np.arange(4, data.shape[1] + 4)})
        left.attrs = {'valid_pixels': 1, 'no_data_mask': 0}

        data = np.ones((10, 10), dtype=np.float64)
        right_mask = np.ones((10, 10), dtype=np.uint8)
        right_mask = np.tril(right_mask, -1.5)

        right = xr.Dataset({'im': (['row', 'col'], data),
                            'msk': (['row', 'col'], right_mask)},
                           coords={'row': np.arange(5, data.shape[0] + 5), 'col': np.arange(4, data.shape[1] + 4)})
        right.attrs = {'valid_pixels': 1, 'no_data_mask': 0}

        stereo_plugin = stereo.AbstractStereo(**{'stereo_method': 'sad', 'window_size': 3, 'subpix': 1})
        cv = stereo_plugin.compute_cost_volume(left, right, -3, 2)

        # Compute the disparity map and validity mask
        dataset = disparity_.to_disp(cv)
        disparity_.validity_mask(dataset, left, right, cv)

        # Validity mask ground truth
        gt_mask = np.array([[PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING],
                            [PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING],
                            [PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING],
                            [PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE, 0,
                             PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING],
                            [PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE, 0, 0,
                             PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING],
                            [PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE, 0, 0, 0,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING],
                            [PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE, 0, 0, 0,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE +
                             PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING],
                            [PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE, 0, 0, 0,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                             PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE]],
                           dtype=np.uint8)

        # Check if the calculated validity mask is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(dataset['validity_mask'].data, gt_mask)


def setup_logging(path='logging.json', default_level=logging.WARNING, ):
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
