#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2020 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of PANDORA
#
#     https://github.com/CNES/Pandora
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
This module contains functions to test the disparity map filtering.
"""

import unittest

import numpy as np
import xarray as xr

import common
import pandora
import pandora.constants as cst
import pandora.filter as flt


class TestFilter(unittest.TestCase):
    """
    TestFilter class allows to test the filter module
    """

    def setUp(self):
        """
        Method called to prepare the test fixture

        """

    @staticmethod
    def test_median_filter():
        """
        Test the median method

        """
        disp = np.array([[5, 6, 7, 8, 9],
                         [6, 85, 1, 36, 5],
                         [5, 9, 23, 12, 2],
                         [6, 1, 9, 2, 4]], dtype=np.float32)

        valid = np.array([[0, 0, 0, 0, 0],
                          [0, cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE, 0, 0, 0],
                          [0, cst.PANDORA_MSK_PIXEL_FILLED_OCCLUSION, 0, 0, 0],
                          [0, 0, 0, 0, cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION]], dtype=np.uint16)

        disp_dataset = xr.Dataset({'disparity_map': (['row', 'col'], disp),
                                   'validity_mask': (['row', 'col'], valid)},
                                  coords={'row': np.arange(4), 'col': np.arange(5)})

        filter_median = flt.AbstractFilter(**{'filter_method': 'median', 'filter_size': 3})

        # Apply median filter to the disparity map. Median filter is only applied on valid pixels.
        filter_median.filter_disparity(disp_dataset)

        # Filtered disparity map ground truth
        gt_disp = np.array([[5, 6, 7, 8, 9],
                            [6, 6, 9, 8, 5],
                            [5, 6, 9, 5, 2],
                            [6, 1, 9, 2, 4]], dtype=np.float32)

        # Check if the calculated disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(disp_dataset['disparity_map'].data, gt_disp)

        disp = np.array([[7, 8, 4, 5, 5],
                         [5, 9, 4, 3, 8],
                         [5, 2, 7, 2, 2],
                         [6, 1, 9, 2, 4]], dtype=np.float32)

        valid = np.array([[cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE, 0,
                           cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                           cst.PANDORA_MSK_PIXEL_FILLED_OCCLUSION + cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER, 0],
                          [cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_RIGHT, cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                           cst.PANDORA_MSK_PIXEL_OCCLUSION, 0, 0],
                          [cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT, cst.PANDORA_MSK_PIXEL_MISMATCH,
                           cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                           cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                           + cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION,
                           0],
                          [cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                           cst.PANDORA_MSK_PIXEL_OCCLUSION,
                           cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT, 0,
                           cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING]],
                         dtype=np.uint16)

        disp_dataset = xr.Dataset({'disparity_map': (['row', 'col'], disp),
                                   'validity_mask': (['row', 'col'], valid)},
                                  coords={'row': np.arange(4), 'col': np.arange(5)})

        filter_median = flt.AbstractFilter(**{'filter_method': 'median', 'filter_size': 3})

        # Apply median filter to the disparity map. Median filter is only applied on valid pixels.
        filter_median.filter_disparity(disp_dataset)

        # Filtered disparity map ground truth
        gt_disp = np.array([[7, 8, 4, 5, 5],
                            [5, 9, 4, 3.5, 8],
                            [5, 2, 7, 2, 2],
                            [6, 1, 9, 2, 4]], dtype=np.float32)

        # Check if the calculated disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(disp_dataset['disparity_map'].data, gt_disp)

        disp = np.array([[7, 8, 4, 5, 5],
                         [5, 9, 4, 3, 8],
                         [5, 2, 7, 2, 2],
                         [6, 1, 9, 2, 4]], dtype=np.float32)

        valid = np.array([[cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE, 0,
                           cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                           cst.PANDORA_MSK_PIXEL_FILLED_OCCLUSION + cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER, 0],
                          [0, 0, cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION, 0, 0],
                          [0, 0, 0, cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE +
                           cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION, 0],
                          [cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_RIGHT, 0, 0, 0, 0]], dtype=np.uint16)

        disp_dataset = xr.Dataset({'disparity_map': (['row', 'col'], disp),
                                   'validity_mask': (['row', 'col'], valid)},
                                  coords={'row': np.arange(4), 'col': np.arange(5)})

        filter_median = flt.AbstractFilter(**{'filter_method': 'median', 'filter_size': 3})

        # Apply median filter to the disparity map. Median filter is only applied on valid pixels.
        filter_median.filter_disparity(disp_dataset)

        # Filtered disparity map ground truth
        gt_disp = np.array([[7, 8, 4, 5, 5],
                            [5, 5, 4, 4, 8],
                            [5, 5, 3, 4, 2],
                            [6, 1, 9, 2, 4]], dtype=np.float32)

        # Check if the calculated disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(disp_dataset['disparity_map'].data, gt_disp)

        # Test with window size 5
        disp = np.array([[7, 8, 4, 5, 5],
                         [5, 9, 4, 3, 8],
                         [5, 2, 7, 2, 2],
                         [6, 1, 9, 2, 4],
                         [1, 6, 2, 7, 8]], dtype=np.float32)

        valid = np.array([[cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE, 0,
                           cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                           cst.PANDORA_MSK_PIXEL_FILLED_OCCLUSION + cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER, 0],
                          [0, 0, cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION, 0, 0],
                          [0, 0, 0, cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE +
                           cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION, 0],
                          [cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_RIGHT, 0, 0, 0, 0],
                          [cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT, 0,
                           cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                           cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING +
                           cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION, 0]],
                         dtype=np.uint16)

        disp_dataset = xr.Dataset({'disparity_map': (['row', 'col'], disp),
                                   'validity_mask': (['row', 'col'], valid)},
                                  coords={'row': np.arange(5), 'col': np.arange(5)})

        filter_median = flt.AbstractFilter(**{'filter_method': 'median', 'filter_size': 5})

        # Apply median filter to the disparity map. Median filter is only applied on valid pixels.
        filter_median.filter_disparity(disp_dataset)

        # Filtered disparity map ground truth
        gt_disp = np.array([[7, 8, 4, 5, 5],
                            [5, 9, 4, 3, 8],
                            [5, 2, 5, 2, 2],
                            [6, 1, 9, 2, 4],
                            [1, 6, 2, 7, 8]], dtype=np.float32)

        # Check if the calculated disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(disp_dataset['disparity_map'].data, gt_disp)

    @staticmethod
    def test_bilateral_filter():
        """
        Test the bilateral method

        """
        disp = np.array([[5, 6, 7, 8, 9],
                         [6, 85, 1, 36, 5],
                         [5, 9, 23, 12, 2],
                         [6, 1, 9, 2, 4]], dtype=np.float32)

        valid = np.array([[0, 0, 0, 0, 0],
                          [0, cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE, 0, 0, 0],
                          [0, cst.PANDORA_MSK_PIXEL_FILLED_OCCLUSION, 0, 0, 0],
                          [0, 0, 0, 0, cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION]], dtype=np.uint16)

        disp_dataset = xr.Dataset({'disparity_map': (['row', 'col'], disp),
                                   'validity_mask': (['row', 'col'], valid)},
                                  coords={'row': np.arange(4), 'col': np.arange(5)})

        user_cfg = {
            'filter': {
                'filter_method': 'bilateral',
                'sigma_color': 4.,
                'sigma_space': 6.
            }
        }

        # Build the default configuration
        cfg = pandora.json_checker.default_short_configuration

        # Update the configuration with default values
        cfg = pandora.json_checker.update_conf(cfg, user_cfg)

        filter_bilateral = flt.AbstractFilter(**cfg['filter'])

        # Apply bilateral filter to the disparity map.
        filter_bilateral.filter_disparity(disp_dataset)


if __name__ == '__main__':
    common.setup_logging()
    unittest.main()
