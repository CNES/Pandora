# type:ignore
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
This module contains functions to test the pyramid class of the multiscale module.
"""

import unittest

import numpy as np
import xarray as xr

import tests.common as common
from pandora import multiscale
import pandora.constants as cst


class TestMultiScale(unittest.TestCase):
    """
    Test multiscale class
    """

    @staticmethod
    def test_disparity_range():
        """
        Test the disparity range method

        """

        multiscale_ = multiscale.AbstractMultiscale(
            **{"multiscale_method": "fixed_zoom_pyramid", "num_scales": 2, "scale_factor": 2, "marge": 0}
        )

        # Modify num_scales and scale_factor to properly test the function without zooming
        multiscale_._scale_factor = 1  # pylint:disable=protected-access
        multiscale_._num_scales = 1  # pylint:disable=protected-access

        disp_min = -30
        disp_max = 0

        # Disparity map ground truth with the size of the input images
        gt_disp = np.array(
            [
                [-1, -2, -3, -4, -5, -6],
                [-7, -8, -9, np.nan, -11, -12],
                [-13, -14, -15, -16, -17, -18],
                [-19, -20, -21, -22, -23, -24],
                [np.nan, -26, -27, -28, -29, -30],
            ],
            dtype=np.float32,
        )

        # Validity mask ground truth with the size of the input images
        gt_mask = np.array(
            [
                [
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,  # info
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                ],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    # invalid
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                ],
                [
                    cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION,
                    cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION,
                    # info
                    cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION,
                    cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION,
                    cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION,
                    cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION,
                ],
            ],
            dtype=np.uint16,
        )

        disp = xr.Dataset({"disparity_map": (["row", "col"], gt_disp), "validity_mask": (["row", "col"], gt_mask)})
        disp.attrs["window_size"] = 3

        # Ground truth output max disparity range
        # Margins can not be processed by disparity range
        gt_range_max = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, -1, -2, 0, -4, 0],
                [0, -7, -8, -9, -11, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )

        # Ground truth output max disparity range
        gt_range_min = np.array(
            [
                [-30, -30, -30, -30, -30, -30],
                [-30, -15, -16, -30, -18, -30],
                [-30, -15, -16, -17, -18, -30],
                [-30, -30, -30, -30, -30, -30],
                [-30, -30, -30, -30, -30, -30],
            ],
            dtype=np.float32,
        )

        # Compute disparity ranges
        disp_range_min, disp_range_max = multiscale_.disparity_range(disp, disp_min, disp_max)

        np.testing.assert_array_equal(disp_range_min, gt_range_min)
        np.testing.assert_array_equal(disp_range_max, gt_range_max)

    @staticmethod
    def test_mask_invalid_disparities():
        """
        Test the mask invalid disparities method

        """
        multiscale_ = multiscale.AbstractMultiscale(
            **{"multiscale_method": "fixed_zoom_pyramid", "num_scales": 2, "scale_factor": 2, "marge": 0}
        )

        # Modify num_scales and scale_factor to properly test the function without zooming
        multiscale_._scale_factor = 1  # pylint:disable=protected-access
        multiscale_._num_scales = 1  # pylint:disable=protected-access

        # Disparity map ground truth with the size of the input images
        gt_disp = np.array(
            [
                [-1, -2, -3, -4, -5, -6],
                [-7, -8, -9, -10, -11, -12],
                [-13, -14, -15, -16, np.nan, -18],
                [-19, -20, -21, -22, -23, -24],
                [-25, -26, -27, -28, -29, -30],
            ],
            dtype=np.float32,
        )

        # Validity mask ground truth with the size of the input images
        gt_mask = np.array(
            [
                [
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,  # invalid
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,  # info
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,  # info
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                ],  # info
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    # invalid
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                ],
                [
                    cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION,
                    cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION,
                    # info
                    cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION,
                    cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION,
                    cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION,
                    cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION,
                ],
            ],
            dtype=np.uint16,
        )

        disp = xr.Dataset({"disparity_map": (["row", "col"], gt_disp), "validity_mask": (["row", "col"], gt_mask)})
        disp.attrs["window_size"] = 3

        # Mask invalid disparities
        filtered_disp = multiscale_.mask_invalid_disparities(disp)

        # Ground truth filtered disparities
        gt_filtered_disp = np.array(
            [
                [np.nan, np.nan, np.nan, -4, -5, -6],
                [-7, -8, -9, -10, -11, -12],
                [-13, -14, -15, -16, np.nan, -18],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [-25, -26, -27, -28, -29, -30],
            ],
            dtype=np.float32,
        )

        np.testing.assert_array_equal(filtered_disp, gt_filtered_disp)


if __name__ == "__main__":
    common.setup_logging()
    unittest.main()
