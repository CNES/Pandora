# pylint: disable=duplicate-code
# type:ignore
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
This module contains functions to test the disparity module with masks.
"""

import unittest

import numpy as np
import xarray as xr
from rasterio import Affine
from tests import common
import pandora.constants as cst
from pandora import disparity
from pandora import matching_cost


class TestDisparityMask(unittest.TestCase):
    """
    TestDisparityMask class allows to test the disparity module with masks
    """

    def setUp(self):
        """
        Method called to prepare the test fixture
        """
        # Create stereo images
        data = np.array(([[1, 2, 4, 6], [2, 4, 1, 6], [6, 7, 8, 10]]), dtype=np.float64)
        self.left = xr.Dataset(
            {"im": (["row", "col"], data)}, coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])}
        )
        self.left.attrs = {
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }
        data = np.array(([[6, 1, 2, 4], [6, 2, 4, 1], [10, 6, 7, 8]]), dtype=np.float64)
        self.right = xr.Dataset(
            {"im": (["row", "col"], data)}, coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])}
        )
        self.right.attrs = {
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

    def test_to_disp_validity_mask(self):
        """
        Test the generated validity mask in the to_disp method
        # If bit 1 == 1 : Invalid pixel : the disparity interval is missing in the right image
        # If bit 2 == 1 : Information: the disparity interval is incomplete (edge reached in the right image)
        """
        # ------ Negative disparities ------
        # Create the left cost volume, with SAD measure window size 1, subpixel 1, disp_min -3 disp_max -1
        matching_cost_plugin = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 1, "subpix": 1}
        )
        cv = matching_cost_plugin.compute_cost_volume(self.left, self.right, -3, -1)

        # Compute the disparity map and validity mask
        disparity_ = disparity.AbstractDisparity(**{"disparity_method": "wta", "invalid_disparity": 0})
        dataset = disparity_.to_disp(cv)
        disparity_.validity_mask(dataset, self.left, self.right, cv)

        # Validity mask ground truth
        gt_mask = np.array(
            [
                [
                    cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    0,
                ],
                [
                    cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    0,
                ],
                [
                    cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    0,
                ],
            ],
            dtype=np.uint16,
        )
        # Check if the calculated disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(dataset["validity_mask"].data, gt_mask)
        # ------ Positive disparities ------
        # Create the left cost volume, with SAD measure window size 1, subpixel 1, disp_min 1 disp_max 2
        cv = matching_cost_plugin.compute_cost_volume(self.left, self.right, 1, 2)
        # Compute the disparity map and validity mask
        dataset = disparity_.to_disp(cv)
        disparity_.validity_mask(dataset, self.left, self.right, cv)
        # Validity mask ground truth
        gt_mask = np.array(
            [
                [0, 0, 1 << 2, cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING],
                [0, 0, 1 << 2, cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING],
                [0, 0, 1 << 2, cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING],
            ],
            dtype=np.uint16,
        )
        # Check if the calculated disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(dataset["validity_mask"].data, gt_mask)

        # ------ Negative and positive disparities ------
        # Create the left cost volume, with SAD measure window size 1, subpixel 1, disp_min -1 disp_max 1
        cv = matching_cost_plugin.compute_cost_volume(self.left, self.right, -1, 1)

        # Compute the disparity map and validity mask
        dataset = disparity_.to_disp(cv)
        disparity_.validity_mask(dataset, self.left, self.right, cv)

        # Validity mask ground truth
        gt_mask = np.array(
            [
                [
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    0,
                    0,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                ],
                [
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    0,
                    0,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                ],
                [
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    0,
                    0,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                ],
            ],
            dtype=np.uint16,
        )
        # Check if the calculated disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(dataset["validity_mask"].data, gt_mask)

        # ------ Variable grids of disparities ------
        # Disp_min and disp_max
        disp_min_grid = np.array([[-3, -2, -3, -1], [-2, -2, -1, -3], [-1, -2, -2, -3]])
        disp_max_grid = np.array([[-1, -1, -2, 0], [0, -1, 0, 0], [0, 0, -1, -1]])

        # Create the left cost volume, with SAD measure window size 1, subpixel 1, disp_min -3 disp_max -1
        matching_cost_plugin = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 1, "subpix": 1}
        )
        dmin, dmax = matching_cost_plugin.dmin_dmax(disp_min_grid, disp_max_grid)
        cv = matching_cost_plugin.compute_cost_volume(self.left, self.right, dmin, dmax)
        matching_cost_plugin.cv_masked(self.left, self.right, cv, disp_min_grid, disp_max_grid)

        # Compute the disparity map and validity mask
        dataset = disparity_.to_disp(cv)
        disparity_.validity_mask(dataset, self.left, self.right, cv)

        # Validity mask ground truth
        gt_mask = np.array(
            [
                [
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                    + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    0,
                ],
                [
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    0,
                ],
                [
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    0,
                ],
            ],
            dtype=np.uint16,
        )
        # Check if the calculated disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(dataset["validity_mask"].data, gt_mask)

    def test_to_disp_validity_mask_with_offset(self):
        """
        Test the generated validity mask in the to_disp method
        # If bit 1 == 1 : Invalid pixel : the disparity interval is missing in the right image
        # If bit 2 == 1 : Information: the disparity interval is incomplete (edge reached in the right image)
        """
        # ------ Negative disparities ------
        # Create the left cost volume, with SAD measure window size 1, subpixel 1, disp_min -3 disp_max -1
        matching_cost_plugin = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 3, "subpix": 1}
        )
        cv = matching_cost_plugin.compute_cost_volume(self.left, self.right, -3, -1)

        # Compute the disparity map and validity mask
        disparity_ = disparity.AbstractDisparity(**{"disparity_method": "wta", "invalid_disparity": 0})
        dataset = disparity_.to_disp(cv)
        disparity_.validity_mask(dataset, self.left, self.right, cv)

        # Validity mask ground truth
        gt_mask = np.array(
            [
                [
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                ],
                [
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                ],
                [
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                ],
            ],
            dtype=np.uint16,
        )
        # Check if the calculated disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(dataset["validity_mask"].data, gt_mask)

        # ------ Positive disparities ------
        # Create the left cost volume, with SAD measure window size 1, subpixel 1, disp_min 1 disp_max 2
        cv = matching_cost_plugin.compute_cost_volume(self.left, self.right, 1, 2)

        # Compute the disparity map and validity mask
        dataset = disparity_.to_disp(cv)
        disparity_.validity_mask(dataset, self.left, self.right, cv)

        # Validity mask ground truth
        gt_mask = np.array(
            [
                [
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                ],
                [
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                ],
                [
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                ],
            ],
            dtype=np.uint16,
        )
        # Check if the calculated disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(dataset["validity_mask"].data, gt_mask)

        # ------ Negative and positive disparities ------
        # Create the left cost volume, with SAD measure window size 1, subpixel 1, disp_min -1 disp_max 1
        cv = matching_cost_plugin.compute_cost_volume(self.left, self.right, -1, 1)

        # Compute the disparity map and validity mask
        dataset = disparity_.to_disp(cv)
        disparity_.validity_mask(dataset, self.left, self.right, cv)

        # Validity mask ground truth
        gt_mask = np.array(
            [
                [
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                ],
                [
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                ],
                [
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                ],
            ],
            dtype=np.uint16,
        )
        # Check if the calculated disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(dataset["validity_mask"].data, gt_mask)

        # ------ Variable grids of disparities ------
        # Disp_min and disp_max
        disp_min_grid = np.array([[-3, -2, -3, -1], [-2, -2, -1, -3], [-1, -2, -2, -3]])
        disp_max_grid = np.array([[-1, -1, -2, 0], [0, -1, 0, 0], [0, 0, -1, -1]])

        # Create the left cost volume, with SAD measure window size 1, subpixel 1, disp_min -3 disp_max -1
        matching_cost_plugin = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 3, "subpix": 1}
        )
        dmin, dmax = matching_cost_plugin.dmin_dmax(disp_min_grid, disp_max_grid)
        cv = matching_cost_plugin.compute_cost_volume(self.left, self.right, dmin, dmax)
        matching_cost_plugin.cv_masked(self.left, self.right, cv, disp_min_grid, disp_max_grid)

        # Compute the disparity map and validity mask
        dataset = disparity_.to_disp(cv)
        disparity_.validity_mask(dataset, self.left, self.right, cv)

        # Validity mask ground truth
        gt_mask = np.array(
            [
                [
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                ],
                [
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                    + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                ],
                [
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                ],
            ],
            dtype=np.uint16,
        )
        # Check if the calculated disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(dataset["validity_mask"].data, gt_mask)

    def test_approximate_right_disparity_validity_mask(self):
        """
        Test the generated validity mask in the right_disparity method
        # If bit 1 == 1 : Invalid pixel : the disparity interval is missing in the right image
        # If bit 2 == 1 : Information: the disparity interval is incomplete (edge reached in the right image)
        """
        # Create the left cost volume, with SAD measure window size 1 and subpixel 1
        matching_cost_plugin = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 1, "subpix": 1}
        )
        # ------ Negative and positive disparities ------
        cv = matching_cost_plugin.compute_cost_volume(self.left, self.right, -2, 1)

        # Validity mask ground truth ( for disparities -1 0 1 2 )
        gt_mask = np.array(
            [
                [
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    0,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                ],
                [
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    0,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                ],
                [
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    0,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                ],
            ],
            dtype=np.uint16,
        )
        # Compute the right disparity map and the validity mask
        disparity_ = disparity.AbstractDisparity(**{"disparity_method": "wta", "invalid_disparity": 0})
        dataset = disparity_.approximate_right_disparity(cv, self.right)

        # Check if the calculated right disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(dataset["validity_mask"].data, gt_mask)

        # ------ Negative disparities ------
        cv = matching_cost_plugin.compute_cost_volume(self.left, self.right, 1, 2)
        # Validity mask ground truth ( for disparities -2 -1 )
        gt_mask = np.array(
            [
                [
                    cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    0,
                    0,
                ],
                [
                    cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    0,
                    0,
                ],
                [
                    cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    0,
                    0,
                ],
            ],
            dtype=np.uint16,
        )

        # Compute the right disparity map and the validity mask
        dataset = disparity_.approximate_right_disparity(cv, self.right)

        # Check if the calculated right disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(dataset["validity_mask"].data, gt_mask)

        # ------ Positive disparities ------
        cv = matching_cost_plugin.compute_cost_volume(self.left, self.right, -2, -1)
        # Validity mask ground truth ( for disparities 1 2 )
        gt_mask = np.array(
            [
                [
                    0,
                    0,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                ],
                [
                    0,
                    0,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                ],
                [
                    0,
                    0,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                ],
            ],
            dtype=np.uint16,
        )

        # Compute the right disparity map and the validity mask
        dataset = disparity_.approximate_right_disparity(cv, self.right)

        # Check if the calculated right disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(dataset["validity_mask"].data, gt_mask)

    @staticmethod
    def test_validity_mask():
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
        data = np.array(([[1, 2, 4, 6], [2, 4, 1, 6], [6, 7, 8, 10]]), dtype=np.float64)
        left_mask = np.array([[2, 1, 1, 1], [1, 2, 4, 1], [5, 1, 1, 2]], dtype=np.uint8)
        left = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], left_mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )
        left.attrs = {
            "valid_pixels": 1,
            "no_data_mask": 2,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

        data = np.array(([[6, 1, 2, 4], [6, 2, 4, 1], [10, 6, 7, 8]]), dtype=np.float64)
        right_mask = np.array([[1, 1, 3, 5], [4, 1, 1, 1], [2, 2, 4, 6]], dtype=np.uint8)

        right = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], right_mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )
        right.attrs = {
            "valid_pixels": 1,
            "no_data_mask": 2,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

        matching_cost_plugin = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 1, "subpix": 1}
        )
        cv = matching_cost_plugin.compute_cost_volume(left, right, -1, 1)

        # Compute the disparity map and validity mask
        disparity_ = disparity.AbstractDisparity(**{"disparity_method": "wta", "invalid_disparity": 0})
        dataset = disparity_.to_disp(cv)
        disparity_.validity_mask(dataset, left, right, cv)

        # Validity mask ground truth
        gt_mask = np.array(
            [
                [
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                    + cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    0,
                    0,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                    + cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_RIGHT,
                ],
                [
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                ],
                [
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                    + cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT
                    + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    0,
                    0,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                    + cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER
                    + cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_RIGHT,
                ],
            ],
            dtype=np.uint16,
        )
        # Check if the calculated validity mask is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(dataset["validity_mask"].data, gt_mask)

        # ---------------------- Test with negative disparity range ----------------------
        cv = matching_cost_plugin.compute_cost_volume(left, right, -2, -1)
        # Compute the disparity map and validity mask
        dataset = disparity_.to_disp(cv)
        disparity_.validity_mask(dataset, left, right, cv)

        # Validity mask ground truth
        gt_mask = np.array(
            [
                [
                    cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING
                    + cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    0,
                    0,
                ],
                [
                    cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER
                    + cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                    + cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_RIGHT,
                    cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT,
                    0,
                ],
                [
                    cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING
                    + cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                    + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                ],
            ],
            dtype=np.uint16,
        )
        # Check if the calculated validity mask is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(dataset["validity_mask"].data, gt_mask)

        # ---------------------- Test with positive disparity range ----------------------
        cv = matching_cost_plugin.compute_cost_volume(left, right, 1, 2)

        # Compute the disparity map and validity mask
        dataset = disparity_.to_disp(cv)
        disparity_.validity_mask(dataset, left, right, cv)

        # Validity mask ground truth
        gt_mask = np.array(
            [
                [
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_RIGHT,
                    cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_RIGHT
                    + cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                ],
                [
                    0,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                    + cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT,
                    cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                ],
                [
                    cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT,
                    cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_RIGHT,
                    cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_RIGHT
                    + cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER
                    + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                ],
            ],
            dtype=np.uint16,
        )
        # Check if the calculated validity mask is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(dataset["validity_mask"].data, gt_mask)

        # ---------------------- Test with positive and negative disparity range and window size = 3----------------
        data = np.array(([[1, 2, 4, 6, 1], [2, 4, 1, 6, 1], [6, 7, 8, 10, 1], [0, 5, 6, 7, 8]]), dtype=np.float64)
        left_mask = np.array([[2, 1, 1, 1, 1], [1, 2, 4, 1, 1], [5, 2, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=np.uint8)
        left = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], left_mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )
        left.attrs = {
            "valid_pixels": 1,
            "no_data_mask": 2,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

        data = np.array(([[6, 1, 2, 4, 1], [6, 2, 4, 1, 6], [10, 6, 7, 8, 1], [5, 6, 7, 8, 0]]), dtype=np.float64)
        right_mask = np.array([[1, 1, 1, 2, 1], [5, 1, 1, 1, 1], [2, 1, 1, 6, 1], [0, 1, 1, 1, 1]], dtype=np.uint8)

        right = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], right_mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )
        right.attrs = {
            "valid_pixels": 1,
            "no_data_mask": 2,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

        matching_cost_plugin = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 3, "subpix": 1}
        )
        cv = matching_cost_plugin.compute_cost_volume(left, right, -1, 1)

        # Compute the disparity map and validity mask
        dataset = disparity_.to_disp(cv)
        disparity_.validity_mask(dataset, left, right, cv)

        # Validity mask ground truth
        gt_mask = np.array(
            [
                [
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                ],
                [
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                    + cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER
                    + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER
                    + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING
                    + cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                    + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                ],
                [
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                    + cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                ],
                [
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                ],
            ],
            dtype=np.uint16,
        )
        # Check if the calculated validity mask is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(dataset["validity_mask"].data, gt_mask)

        # ---------------------- Test with positive and negative disparity range on flag 1 ----------------------
        # Masks convention
        # 1 = valid
        # 0 = no_data

        data = np.ones((10, 10), dtype=np.float64)
        left_mask = np.ones((10, 10), dtype=np.uint8)

        left = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], left_mask)},
            coords={"row": np.arange(5, data.shape[0] + 5), "col": np.arange(4, data.shape[1] + 4)},
        )
        left.attrs = {
            "valid_pixels": 1,
            "no_data_mask": 0,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

        data = np.ones((10, 10), dtype=np.float64)
        right_mask = np.ones((10, 10), dtype=np.uint8)
        right_mask = np.tril(right_mask, -1.5)

        right = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], right_mask)},
            coords={"row": np.arange(5, data.shape[0] + 5), "col": np.arange(4, data.shape[1] + 4)},
        )
        right.attrs = {
            "valid_pixels": 1,
            "no_data_mask": 0,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

        matching_cost_plugin = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 3, "subpix": 1}
        )
        cv = matching_cost_plugin.compute_cost_volume(left, right, -3, 2)

        # Compute the disparity map and validity mask
        dataset = disparity_.to_disp(cv)
        disparity_.validity_mask(dataset, left, right, cv)

        # Validity mask ground truth
        gt_mask = np.array(
            [
                [
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                ],
                [
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                    + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                    + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                    + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                    + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                    + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                ],
                [
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                    + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                    + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                    + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                    + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                    + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                ],
                [
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                    + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                    + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                    + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                    + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                    + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                ],
                [
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    0,
                    cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                    + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                    + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                ],
                [
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    0,
                    0,
                    cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                    + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                    + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                ],
                [
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    0,
                    0,
                    0,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                    + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                    + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                ],
                [
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    0,
                    0,
                    0,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                    + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                ],
                [
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    0,
                    0,
                    0,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                ],
                [
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                ],
            ],
            dtype=np.uint8,
        )
        # Check if the calculated validity mask is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(dataset["validity_mask"].data, gt_mask)
        # ---------------------- Test with posible constant duplication ----------------------
        # Masks convention
        # 1 = valid
        # 2 = no_data

        data = np.array(([[1, 2, 4, 6]]), dtype=np.float64)
        left_mask = np.array([[2, 2, 2, 2]], dtype=np.uint8)
        left = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], left_mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )
        left.attrs = {
            "valid_pixels": 1,
            "no_data_mask": 2,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

        data = np.array(([[6, 1, 2, 4]]), dtype=np.float64)
        right_mask = np.array([[2, 2, 2, 1]], dtype=np.uint8)

        right = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], right_mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )
        right.attrs = {
            "valid_pixels": 1,
            "no_data_mask": 2,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

        matching_cost_plugin = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 1, "subpix": 1}
        )
        cv = matching_cost_plugin.compute_cost_volume(right, left, -1, 1)
        matching_cost_plugin.cv_masked(
            left,
            right,
            cv,
            -1,
            1,
        )
        # Compute the disparity map and validity mask
        disparity_ = disparity.AbstractDisparity(**{"disparity_method": "wta", "invalid_disparity": 0})
        dataset = disparity_.to_disp(cv)
        disparity_.validity_mask(dataset, right, left, cv)
        # All right image pixels are nodata
        gt_valid_mask = np.array(
            [
                [
                    cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                ]
            ]
        )
        # Disp -1 1 has incomplete range on the borders
        gt_valid_mask += np.array(
            [
                [
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    0,
                    0,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                ]
            ]
        )
        # 3 first pixels of left image are no data
        gt_valid_mask += np.array(
            [
                [
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    0,
                ]
            ]
        )
        np.testing.assert_array_equal(dataset["validity_mask"].data, gt_valid_mask)


if __name__ == "__main__":
    common.setup_logging()
    unittest.main()
