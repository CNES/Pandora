# pylint: disable=duplicate-code
# pylint:disable=too-many-lines
# type:ignore
#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
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
from pandora.img_tools import add_disparity
from pandora.criteria import validity_mask


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

        # Add disparity on left image
        add_disparity(self.left, [-3, -1], None)

        # Create the left cost volume, with SAD measure window size 1, subpixel 1, disp_min -3 disp_max -1
        matching_cost_plugin = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 1, "subpix": 1}
        )
        # Allocate cost volume
        grid = matching_cost_plugin.allocate_cost_volume(
            self.left, (self.left["disparity"].sel(band_disp="min"), self.left["disparity"].sel(band_disp="max"))
        )

        # Compute validity mask
        grid = validity_mask(self.left, self.right, grid)

        # Compute cost volume
        cv = matching_cost_plugin.compute_cost_volume(self.left, self.right, grid)

        matching_cost_plugin.cv_masked(
            self.left,
            self.right,
            cv,
            self.left["disparity"].sel(band_disp="min"),
            self.left["disparity"].sel(band_disp="max"),
        )

        # Compute the disparity map
        disparity_ = disparity.AbstractDisparity(**{"disparity_method": "wta", "invalid_disparity": 0})
        dataset = disparity_.to_disp(cv)

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

        # Add disparity on left image
        add_disparity(self.left, [1, 2], None)

        # Allocate cost volume
        grid = matching_cost_plugin.allocate_cost_volume(
            self.left, (self.left["disparity"].sel(band_disp="min"), self.left["disparity"].sel(band_disp="max"))
        )

        # Compute validity mask
        grid = validity_mask(self.left, self.right, grid)

        # Compute cost volume
        cv = matching_cost_plugin.compute_cost_volume(self.left, self.right, grid)

        matching_cost_plugin.cv_masked(
            self.left,
            self.right,
            cv,
            self.left["disparity"].sel(band_disp="min"),
            self.left["disparity"].sel(band_disp="max"),
        )

        # Compute the disparity map
        dataset = disparity_.to_disp(cv)

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

        # Add disparity on left image
        add_disparity(self.left, [-1, 1], None)

        # Allocate cost volume
        grid = matching_cost_plugin.allocate_cost_volume(
            self.left, (self.left["disparity"].sel(band_disp="min"), self.left["disparity"].sel(band_disp="max"))
        )

        # Compute validity mask
        grid = validity_mask(self.left, self.right, grid)

        # Compute cost volume
        cv = matching_cost_plugin.compute_cost_volume(self.left, self.right, grid)

        matching_cost_plugin.cv_masked(
            self.left,
            self.right,
            cv,
            self.left["disparity"].sel(band_disp="min"),
            self.left["disparity"].sel(band_disp="max"),
        )

        # Compute the disparity map
        dataset = disparity_.to_disp(cv)

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

        # Add disparity on left image
        self.left.coords["band_disp"] = ["min", "max"]

        self.left["disparity"] = xr.DataArray(
            np.array(
                [
                    disp_min_grid,
                    disp_max_grid,
                ]
            ),
            dims=["band_disp", "row", "col"],
        )

        self.left.attrs["disparity_source"] = [int(np.nanmin(disp_min_grid)), int(np.nanmax(disp_max_grid))]

        # Create the left cost volume, with SAD measure window size 1, subpixel 1, disp_min -3 disp_max -1
        matching_cost_plugin = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 1, "subpix": 1}
        )
        # Allocate cost volume
        grid = matching_cost_plugin.allocate_cost_volume(
            self.left, (self.left["disparity"].sel(band_disp="min"), self.left["disparity"].sel(band_disp="max"))
        )

        # Compute validity mask
        grid = validity_mask(self.left, self.right, grid)

        # Compute cost volume
        cv = matching_cost_plugin.compute_cost_volume(self.left, self.right, grid)

        matching_cost_plugin.cv_masked(self.left, self.right, cv, disp_min_grid, disp_max_grid)

        # Compute the disparity map
        dataset = disparity_.to_disp(cv)

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

        # Add disparity on left image
        add_disparity(self.left, [-3, -1], None)

        # Create the left cost volume, with SAD measure window size 1, subpixel 1, disp_min -3 disp_max -1
        matching_cost_plugin = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 3, "subpix": 1}
        )
        # Allocate cost volume
        grid = matching_cost_plugin.allocate_cost_volume(
            self.left, (self.left["disparity"].sel(band_disp="min"), self.left["disparity"].sel(band_disp="max"))
        )

        # Compute validity mask
        grid = validity_mask(self.left, self.right, grid)

        # Compute cost volume
        cv = matching_cost_plugin.compute_cost_volume(self.left, self.right, grid)

        matching_cost_plugin.cv_masked(
            self.left,
            self.right,
            cv,
            self.left["disparity"].sel(band_disp="min"),
            self.left["disparity"].sel(band_disp="max"),
        )

        # Compute the disparity map
        disparity_ = disparity.AbstractDisparity(**{"disparity_method": "wta", "invalid_disparity": 0})
        dataset = disparity_.to_disp(cv)

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

        # Add disparity on left image
        add_disparity(self.left, [1, 2], None)

        # Allocate cost volume
        grid = matching_cost_plugin.allocate_cost_volume(
            self.left, (self.left["disparity"].sel(band_disp="min"), self.left["disparity"].sel(band_disp="max"))
        )

        # Compute validity mask
        grid = validity_mask(self.left, self.right, grid)

        # Compute cost volume
        cv = matching_cost_plugin.compute_cost_volume(self.left, self.right, grid)

        matching_cost_plugin.cv_masked(
            self.left,
            self.right,
            cv,
            self.left["disparity"].sel(band_disp="min"),
            self.left["disparity"].sel(band_disp="max"),
        )

        # Compute the disparity map
        dataset = disparity_.to_disp(cv)

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

        # Add disparity on left image
        add_disparity(self.left, [-1, 1], None)

        # Allocate cost volume
        grid = matching_cost_plugin.allocate_cost_volume(
            self.left, (self.left["disparity"].sel(band_disp="min"), self.left["disparity"].sel(band_disp="max"))
        )

        # Compute validity mask
        grid = validity_mask(self.left, self.right, grid)

        # Compute cost volume
        cv = matching_cost_plugin.compute_cost_volume(self.left, self.right, grid)

        matching_cost_plugin.cv_masked(
            self.left,
            self.right,
            cv,
            self.left["disparity"].sel(band_disp="min"),
            self.left["disparity"].sel(band_disp="max"),
        )

        # Compute the disparity map
        dataset = disparity_.to_disp(cv)

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

        # Add disparity on left image
        self.left.coords["band_disp"] = ["min", "max"]

        self.left["disparity"] = xr.DataArray(
            np.array(
                [
                    disp_min_grid,
                    disp_max_grid,
                ]
            ),
            dims=["band_disp", "row", "col"],
        )

        self.left.attrs["disparity_source"] = [int(np.nanmin(disp_min_grid)), int(np.nanmax(disp_max_grid))]

        # Create the left cost volume, with SAD measure window size 1, subpixel 1, disp_min -3 disp_max -1
        matching_cost_plugin = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 3, "subpix": 1}
        )

        # Allocate cost volume
        grid = matching_cost_plugin.allocate_cost_volume(
            self.left, (self.left["disparity"].sel(band_disp="min"), self.left["disparity"].sel(band_disp="max"))
        )

        # Compute validity mask
        grid = validity_mask(self.left, self.right, grid)

        # Compute cost volume
        cv = matching_cost_plugin.compute_cost_volume(self.left, self.right, grid)

        matching_cost_plugin.cv_masked(self.left, self.right, cv, disp_min_grid, disp_max_grid)

        # Compute the disparity map
        dataset = disparity_.to_disp(cv)

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
        # Add disparities on left image
        add_disparity(self.left, [-2, 1], None)

        # Create the left cost volume, with SAD measure window size 1 and subpixel 1
        matching_cost_plugin = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 1, "subpix": 1}
        )
        # ------ Negative and positive disparities ------

        # Allocate cost volume
        grid = matching_cost_plugin.allocate_cost_volume(
            self.left, (self.left["disparity"].sel(band_disp="min"), self.left["disparity"].sel(band_disp="max"))
        )

        # Compute validity mask
        grid = validity_mask(self.left, self.right, grid)

        # Compute cost volume
        cv = matching_cost_plugin.compute_cost_volume(self.left, self.right, grid)

        matching_cost_plugin.cv_masked(
            self.left,
            self.right,
            cv,
            self.left["disparity"].sel(band_disp="min"),
            self.left["disparity"].sel(band_disp="max"),
        )

        # Compute the right disparity map
        disparity_ = disparity.AbstractDisparity(**{"disparity_method": "wta", "invalid_disparity": 0})
        dataset = disparity_.approximate_right_disparity(cv, self.right)

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

        # Check if the calculated right disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(dataset["validity_mask"].data, gt_mask)

        # ------ Negative disparities ------

        # Add disparities on left image
        add_disparity(self.left, [1, 2], None)

        # Allocate cost volume
        grid = matching_cost_plugin.allocate_cost_volume(
            self.left, (self.left["disparity"].sel(band_disp="min"), self.left["disparity"].sel(band_disp="max"))
        )

        # Compute validity mask
        grid = validity_mask(self.left, self.right, grid)

        # Compute cost volume
        cv = matching_cost_plugin.compute_cost_volume(self.left, self.right, grid)

        matching_cost_plugin.cv_masked(
            self.left,
            self.right,
            cv,
            self.left["disparity"].sel(band_disp="min"),
            self.left["disparity"].sel(band_disp="max"),
        )

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

        # Add disparities on left image
        add_disparity(self.left, [-2, -1], None)

        # Allocate cost volume
        grid = matching_cost_plugin.allocate_cost_volume(
            self.left, (self.left["disparity"].sel(band_disp="min"), self.left["disparity"].sel(band_disp="max"))
        )

        # Compute validity mask
        grid = validity_mask(self.left, self.right, grid)

        # Compute cost volume
        cv = matching_cost_plugin.compute_cost_volume(self.left, self.right, grid)

        matching_cost_plugin.cv_masked(
            self.left,
            self.right,
            cv,
            self.left["disparity"].sel(band_disp="min"),
            self.left["disparity"].sel(band_disp="max"),
        )

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


if __name__ == "__main__":
    common.setup_logging()
    unittest.main()
