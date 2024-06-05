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
This module contains functions to test the criteria module.
"""

import numpy as np
import xarray as xr
from rasterio import Affine
import pytest

import pandora.constants as cst
from pandora import matching_cost
from pandora.criteria import (
    validity_mask,
    allocate_left_mask,
    allocate_right_mask,
    mask_invalid_variable_disparity_range,
    mask_border,
    binary_dilation_msk,
)
from pandora.img_tools import add_disparity


class TestCriteria:
    """
    TestCriteria class allows to test the methods in the criteria module
    """

    @pytest.mark.parametrize(
        ["left_data", "left_msk", "left_attrs", "window_size", "gt_dil"],
        [
            pytest.param(
                np.array(([[1, 2, 4, 6], [2, 4, 1, 6], [6, 7, 8, 10]]), dtype=np.float64),
                np.array([[2, 1, 1, 1], [1, 2, 4, 1], [5, 1, 1, 2]], dtype=np.uint8),
                {
                    "valid_pixels": 1,
                    "no_data_mask": 2,
                    "crs": None,
                    "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                },
                1,
                np.array(
                    [[True, False, False, False], [False, True, False, False], [False, False, False, True]],
                ),
                id="Window size = 1",
            ),
            pytest.param(
                np.array(([[1, 2, 4, 6, 1], [2, 4, 1, 6, 1], [6, 7, 8, 10, 1], [0, 5, 6, 7, 8]]), dtype=np.float64),
                np.array([[2, 1, 1, 1, 1], [1, 2, 4, 1, 1], [5, 2, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=np.uint8),
                {
                    "valid_pixels": 1,
                    "no_data_mask": 2,
                    "crs": None,
                    "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                },
                3,
                np.array(
                    [
                        [True, True, True, False, False],
                        [True, True, True, False, False],
                        [True, True, True, False, False],
                        [True, True, True, False, False],
                    ],
                ),
                id="Window size = 3",
            ),
        ],
    )
    def test_binary_dilation_msk(self, left_data, left_msk, left_attrs, window_size, gt_dil):
        """
        Test the binary_dilation_msk function
        """

        # Create left dataset
        left = xr.Dataset(
            {"im": (["row", "col"], left_data), "msk": (["row", "col"], left_msk)},
            coords={"row": np.arange(left_data.shape[0]), "col": np.arange(left_data.shape[1])},
        )

        left.attrs = left_attrs

        dil = binary_dilation_msk(left, window_size)

        # Check if the calculated output of binary_dilation_msk is equal to the ground truth
        # (same shape and all elements equals)
        np.testing.assert_array_equal(dil, gt_dil)

    @pytest.mark.parametrize(
        ["left_data", "left_msk", "left_attrs", "window_size", "gt_mask"],
        [
            pytest.param(
                np.array(([[1, 2, 4, 6, 1], [2, 4, 1, 6, 1], [6, 7, 8, 10, 1], [0, 5, 6, 7, 8]]), dtype=np.float64),
                np.array([[2, 1, 1, 1, 1], [1, 2, 4, 1, 1], [5, 2, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=np.uint8),
                {
                    "valid_pixels": 1,
                    "no_data_mask": 2,
                    "crs": None,
                    "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                },
                3,
                np.array(
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
                            0,
                            0,
                            0,
                            cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                        ],
                        [
                            cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                            0,
                            0,
                            0,
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
                ),
                id="Offset = 2",
            ),
        ],
    )
    def test_mask_border(self, left_data, left_msk, left_attrs, window_size, gt_mask):
        """
        Test the mask_border function
        """
        # Create left dataset
        left = xr.Dataset(
            {"im": (["row", "col"], left_data), "msk": (["row", "col"], left_msk)},
            coords={"row": np.arange(left_data.shape[0]), "col": np.arange(left_data.shape[1])},
        )

        left.attrs = left_attrs

        # Add disparity on left image
        add_disparity(left, [-1, 1], None)

        matching_cost_plugin = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": window_size, "subpix": 1}
        )

        # Allocate cost volume
        grid = matching_cost_plugin.allocate_cost_volume(
            left, (left["disparity"].sel(band_disp="min"), left["disparity"].sel(band_disp="max"))
        )

        # Allocate the validity mask
        grid["validity_mask"] = xr.DataArray(
            np.zeros((grid["cost_volume"].shape[0], grid["cost_volume"].shape[1]), dtype=np.uint16),
            dims=["row", "col"],
        )

        mask_border(grid)

        # Check if the calculated output of mask_border is equal to the ground truth
        # (same shape and all elements equals)
        np.testing.assert_array_equal(grid["validity_mask"].data, gt_mask)

    @pytest.mark.parametrize(
        [
            "left_data",
            "left_msk",
            "left_attrs",
            "right_data",
            "right_msk",
            "right_attrs",
            "disp_min_grid",
            "disp_max_grid",
            "gt_mask",
        ],
        [
            pytest.param(
                np.array(([[1, 2, 4, 6], [2, 4, 1, 6], [6, 7, 8, 10]]), dtype=np.float64),
                np.array([[2, 1, 1, 1], [1, 2, 4, 1], [5, 1, 1, 2]], dtype=np.uint8),
                {
                    "valid_pixels": 1,
                    "no_data_mask": 2,
                    "crs": None,
                    "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                },
                np.array(([[6, 1, 2, 4], [6, 2, 4, 1], [10, 6, 7, 8]]), dtype=np.float64),
                np.array([[1, 1, 3, 5], [4, 1, 1, 1], [2, 2, 4, 6]], dtype=np.uint8),
                {
                    "valid_pixels": 1,
                    "no_data_mask": 2,
                    "crs": None,
                    "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                },
                np.array([[-4, -2, -3, -1], [-2, -2, -1, -3], [-1, -2, -2, -3]]),
                np.array([[1, -1, -2, 0], [0, -1, 0, 0], [0, 0, -1, -1]]),
                np.array(
                    [
                        [
                            cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER
                            + cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                            cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                            cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                            cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                        ],
                        [
                            cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                            + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                            cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER
                            + cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                            cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT
                            + cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                            cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                        ],
                        [
                            cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT
                            + cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                            + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                            cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                            cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                            cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                            + cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                        ],
                    ],
                    dtype=np.uint16,
                ),
                id="Variable grids of disparities",
            ),
        ],
    )

    # pylint: disable=too-many-arguments
    def test_mask_invalid_variable_disparity_range(
        self, left_data, left_msk, left_attrs, right_data, right_msk, right_attrs, disp_min_grid, disp_max_grid, gt_mask
    ):
        """
        Test the mask_invalid_variable_disparity_range function
        """

        # Create left dataset
        left = xr.Dataset(
            {"im": (["row", "col"], left_data), "msk": (["row", "col"], left_msk)},
            coords={"row": np.arange(left_data.shape[0]), "col": np.arange(left_data.shape[1])},
        )

        left.attrs = left_attrs

        # Add disparity on left image
        left.coords["band_disp"] = ["min", "max"]

        left["disparity"] = xr.DataArray(
            np.array(
                [
                    disp_min_grid,
                    disp_max_grid,
                ]
            ),
            dims=["band_disp", "row", "col"],
        )

        left.attrs["disparity_source"] = [int(np.nanmin(disp_min_grid)), int(np.nanmax(disp_max_grid))]

        # Create right dataset
        right = xr.Dataset(
            {"im": (["row", "col"], right_data), "msk": (["row", "col"], right_msk)},
            coords={"row": np.arange(right_data.shape[0]), "col": np.arange(right_data.shape[1])},
        )

        right.attrs = right_attrs

        matching_cost_plugin = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 1, "subpix": 1}
        )

        # Allocate cost volume
        grid = matching_cost_plugin.allocate_cost_volume(
            left, (left["disparity"].sel(band_disp="min"), left["disparity"].sel(band_disp="max"))
        )

        # Compute validity mask
        grid = validity_mask(left, right, grid)

        # Compute cost volume
        cv = matching_cost_plugin.compute_cost_volume(left, right, grid)

        # # We add a value that is equal np.nan for all disparities
        cv["cost_volume"][1, 0, :] = np.nan

        mask_invalid_variable_disparity_range(cv)

        # Check if the calculated output of mask_invalid_variable_disparity_range is equal to the ground truth
        # (same shape and all elements equals)
        np.testing.assert_array_equal(cv["validity_mask"].data, gt_mask)

    @pytest.mark.parametrize(
        [
            "left_data",
            "left_msk",
            "left_attrs",
            "right_data",
            "right_msk",
            "right_attrs",
            "disparity",
            "bit_1",
            "window_size",
            "gt_mask",
        ],
        [
            pytest.param(
                np.array(([[1, 2, 4, 6], [2, 4, 1, 6], [6, 7, 8, 10]]), dtype=np.float64),
                np.array([[2, 1, 1, 1], [1, 2, 4, 1], [5, 1, 1, 2]], dtype=np.uint8),
                {
                    "valid_pixels": 1,
                    "no_data_mask": 2,
                    "crs": None,
                    "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                },
                np.array(([[6, 1, 2, 4], [6, 2, 4, 1], [10, 6, 7, 8]]), dtype=np.float64),
                np.array([[1, 1, 3, 5], [4, 1, 1, 1], [2, 2, 4, 6]], dtype=np.uint8),
                {
                    "valid_pixels": 1,
                    "no_data_mask": 2,
                    "crs": None,
                    "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                },
                [-1, 1],
                ([],),
                1,
                np.array(
                    [
                        [
                            0,
                            0,
                            0,
                            cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_RIGHT,
                        ],
                        [
                            0,
                            0,
                            0,
                            0,
                        ],
                        [
                            cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                            0,
                            0,
                            cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_RIGHT,
                        ],
                    ],
                    dtype=np.uint16,
                ),
                id="Positive and negative disparity range",
            ),
            pytest.param(
                np.array(([[1, 2, 4, 6], [2, 4, 1, 6], [6, 7, 8, 10]]), dtype=np.float64),
                np.array([[2, 1, 1, 1], [1, 2, 4, 1], [5, 1, 1, 2]], dtype=np.uint8),
                {
                    "valid_pixels": 1,
                    "no_data_mask": 2,
                    "crs": None,
                    "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                },
                np.array(([[6, 1, 2, 4], [6, 2, 4, 1], [10, 6, 7, 8]]), dtype=np.float64),
                np.array([[1, 1, 3, 5], [4, 1, 1, 1], [2, 2, 4, 6]], dtype=np.uint8),
                {
                    "valid_pixels": 1,
                    "no_data_mask": 2,
                    "crs": None,
                    "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                },
                [-2, -1],
                ([0],),
                1,
                np.array(
                    [
                        [
                            0,
                            0,
                            0,
                            0,
                        ],
                        [
                            0,
                            cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_RIGHT,
                            0,
                            0,
                        ],
                        [
                            0,
                            cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                            cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                            0,
                        ],
                    ],
                    dtype=np.uint16,
                ),
                id="Negative disparity range",
            ),
            pytest.param(
                np.array(([[1, 2, 4, 6], [2, 4, 1, 6], [6, 7, 8, 10]]), dtype=np.float64),
                np.array([[2, 1, 1, 1], [1, 2, 4, 1], [5, 1, 1, 2]], dtype=np.uint8),
                {
                    "valid_pixels": 1,
                    "no_data_mask": 2,
                    "crs": None,
                    "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                },
                np.array(([[6, 1, 2, 4], [6, 2, 4, 1], [10, 6, 7, 8]]), dtype=np.float64),
                np.array([[1, 1, 3, 5], [4, 1, 1, 1], [2, 2, 4, 6]], dtype=np.uint8),
                {
                    "valid_pixels": 1,
                    "no_data_mask": 2,
                    "crs": None,
                    "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                },
                [1, 2],
                ([3],),
                1,
                np.array(
                    [
                        [
                            0,
                            cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_RIGHT,
                            cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_RIGHT,
                            0,
                        ],
                        [
                            0,
                            0,
                            0,
                            0,
                        ],
                        [
                            0,
                            cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_RIGHT,
                            cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_RIGHT,
                            0,
                        ],
                    ],
                    dtype=np.uint16,
                ),
                id="Positive disparity range",
            ),
            pytest.param(
                np.array(([[1, 2, 4, 6, 1], [2, 4, 1, 6, 1], [6, 7, 8, 10, 1], [0, 5, 6, 7, 8]]), dtype=np.float64),
                np.array([[2, 1, 1, 1, 1], [1, 2, 4, 1, 1], [5, 2, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=np.uint8),
                {
                    "valid_pixels": 1,
                    "no_data_mask": 2,
                    "crs": None,
                    "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                },
                np.array(([[6, 1, 2, 4, 1], [6, 2, 4, 1, 6], [10, 6, 7, 8, 1], [5, 6, 7, 8, 0]]), dtype=np.float64),
                np.array([[1, 1, 1, 2, 1], [5, 1, 1, 1, 1], [2, 1, 1, 6, 1], [0, 1, 1, 1, 1]], dtype=np.uint8),
                {
                    "valid_pixels": 1,
                    "no_data_mask": 2,
                    "crs": None,
                    "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                },
                [-1, 1],
                ([],),
                3,
                np.array(
                    [
                        [
                            0,
                            0,
                            0,
                            cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                            cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                        ],
                        [
                            cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                            cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                            cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                            cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                            cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                        ],
                        [
                            cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                            0,
                            0,
                            0,
                            cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_RIGHT,
                        ],
                        [
                            cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                            0,
                            0,
                            0,
                            0,
                        ],
                    ],
                    dtype=np.uint16,
                ),
                id="Positive and negative disparity range and window size = 3",
            ),
        ],
    )

    # pylint: disable=too-many-arguments
    def test_allocate_right_mask(
        self,
        left_data,
        left_msk,
        left_attrs,
        right_data,
        right_msk,
        right_attrs,
        disparity,
        bit_1,
        window_size,
        gt_mask,
    ):
        """
        Test the allocate_right_mask function
        """

        # Create left dataset
        left = xr.Dataset(
            {"im": (["row", "col"], left_data), "msk": (["row", "col"], left_msk)},
            coords={"row": np.arange(left_data.shape[0]), "col": np.arange(left_data.shape[1])},
        )

        left.attrs = left_attrs

        # Add disparity on left image
        add_disparity(left, disparity, None)

        # Create right dataset
        right = xr.Dataset(
            {"im": (["row", "col"], right_data), "msk": (["row", "col"], right_msk)},
            coords={"row": np.arange(right_data.shape[0]), "col": np.arange(right_data.shape[1])},
        )

        right.attrs = right_attrs

        matching_cost_plugin = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": window_size, "subpix": 1}
        )

        grid = matching_cost_plugin.allocate_cost_volume(
            left, (left["disparity"].sel(band_disp="min"), left["disparity"].sel(band_disp="max"))
        )

        # Allocate the validity mask
        grid["validity_mask"] = xr.DataArray(
            np.zeros((grid["cost_volume"].shape[0], grid["cost_volume"].shape[1]), dtype=np.uint16),
            dims=["row", "col"],
        )

        allocate_right_mask(grid, right, bit_1)

        # Check if the calculated output of allocate_right_mask is equal to the ground truth
        # (same shape and all elements equals)
        np.testing.assert_array_equal(grid["validity_mask"].data, gt_mask)

    @pytest.mark.parametrize(
        ["left_data", "left_msk", "left_attrs", "window_size", "gt_mask"],
        [
            pytest.param(
                np.array(([[1, 2, 4, 6], [2, 4, 1, 6], [6, 7, 8, 10]]), dtype=np.float64),
                np.array([[2, 1, 1, 1], [1, 2, 4, 1], [5, 1, 1, 2]], dtype=np.uint8),
                {
                    "valid_pixels": 1,
                    "no_data_mask": 2,
                    "crs": None,
                    "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                },
                1,
                np.array(
                    [
                        [
                            cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                            0,
                            0,
                            0,
                        ],
                        [
                            0,
                            cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                            cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT,
                            0,
                        ],
                        [
                            cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT,
                            0,
                            0,
                            cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                        ],
                    ],
                    dtype=np.uint16,
                ),
                id="Window size = 1",
            ),
            pytest.param(
                np.array(([[1, 2, 4, 6, 1], [2, 4, 1, 6, 1], [6, 7, 8, 10, 1], [0, 5, 6, 7, 8]]), dtype=np.float64),
                np.array([[2, 1, 1, 1, 1], [1, 2, 4, 1, 1], [5, 2, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=np.uint8),
                {
                    "valid_pixels": 1,
                    "no_data_mask": 2,
                    "crs": None,
                    "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                },
                3,
                np.array(
                    [
                        [
                            cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                            cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                            cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                            0,
                            0,
                        ],
                        [
                            cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                            cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                            cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER + cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT,
                            0,
                            0,
                        ],
                        [
                            cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER + cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT,
                            cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                            cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                            0,
                            0,
                        ],
                        [
                            cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                            cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                            cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                            0,
                            0,
                        ],
                    ],
                    dtype=np.uint16,
                ),
                id="Window size = 3",
            ),
        ],
    )
    def test_allocate_left_mask(self, left_data, left_msk, left_attrs, window_size, gt_mask):
        """
        Test the allocate_left_mask function
        """

        # Create left dataset
        left = xr.Dataset(
            {"im": (["row", "col"], left_data), "msk": (["row", "col"], left_msk)},
            coords={"row": np.arange(left_data.shape[0]), "col": np.arange(left_data.shape[1])},
        )

        left.attrs = left_attrs

        # Add disparity on left image
        add_disparity(left, [-1, 1], None)

        matching_cost_plugin = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": window_size, "subpix": 1}
        )

        # Allocate cost volume
        grid = matching_cost_plugin.allocate_cost_volume(
            left, (left["disparity"].sel(band_disp="min"), left["disparity"].sel(band_disp="max"))
        )

        # Allocate the validity mask
        grid["validity_mask"] = xr.DataArray(
            np.zeros((grid["cost_volume"].shape[0], grid["cost_volume"].shape[1]), dtype=np.uint16),
            dims=["row", "col"],
        )

        allocate_left_mask(grid, left)

        # Check if the calculated output of allocate_left_mask is equal to the ground truth
        # (same shape and all elements equals)
        np.testing.assert_array_equal(grid["validity_mask"].data, gt_mask)

    @pytest.mark.parametrize(
        [
            "left_data",
            "left_msk",
            "left_attrs",
            "right_data",
            "right_msk",
            "right_attrs",
            "disparity",
            "window_size",
            "gt_mask",
        ],
        [
            pytest.param(
                np.array(([[1, 2, 4, 6], [2, 4, 1, 6], [6, 7, 8, 10]]), dtype=np.float64),
                np.array([[2, 1, 1, 1], [1, 2, 4, 1], [5, 1, 1, 2]], dtype=np.uint8),
                {
                    "valid_pixels": 1,
                    "no_data_mask": 2,
                    "crs": None,
                    "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                },
                np.array(([[6, 1, 2, 4], [6, 2, 4, 1], [10, 6, 7, 8]]), dtype=np.float64),
                np.array([[1, 1, 3, 5], [4, 1, 1, 1], [2, 2, 4, 6]], dtype=np.uint8),
                {
                    "valid_pixels": 1,
                    "no_data_mask": 2,
                    "crs": None,
                    "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                },
                [-1, 1],
                1,
                np.array(
                    [
                        [
                            cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                            + cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER
                            + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                            0,
                            0,
                            cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                            + cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_RIGHT
                            + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                        ],
                        [
                            cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                            cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER
                            + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                            cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT
                            + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                            cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                        ],
                        [
                            cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                            + cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT
                            + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                            cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                            cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                            cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                            + cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER
                            + cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_RIGHT
                            + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                        ],
                    ],
                    dtype=np.uint16,
                ),
                id="Positive and negative disparity range",
            ),
            pytest.param(
                np.array(([[1, 2, 4, 6], [2, 4, 1, 6], [6, 7, 8, 10]]), dtype=np.float64),
                np.array([[2, 1, 1, 1], [1, 2, 4, 1], [5, 1, 1, 2]], dtype=np.uint8),
                {
                    "valid_pixels": 1,
                    "no_data_mask": 2,
                    "crs": None,
                    "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                },
                np.array(([[6, 1, 2, 4], [6, 2, 4, 1], [10, 6, 7, 8]]), dtype=np.float64),
                np.array([[1, 1, 3, 5], [4, 1, 1, 1], [2, 2, 4, 6]], dtype=np.uint8),
                {
                    "valid_pixels": 1,
                    "no_data_mask": 2,
                    "crs": None,
                    "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                },
                [-2, -1],
                1,
                np.array(
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
                            + cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_RIGHT
                            + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                            cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT
                            + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                            0,
                        ],
                        [
                            cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING
                            + cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT,
                            cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                            + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                            cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                            cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER
                            + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                        ],
                    ],
                    dtype=np.uint16,
                ),
                id="Negative disparity range",
            ),
            pytest.param(
                np.array(([[1, 2, 4, 6], [2, 4, 1, 6], [6, 7, 8, 10]]), dtype=np.float64),
                np.array([[2, 1, 1, 1], [1, 2, 4, 1], [5, 1, 1, 2]], dtype=np.uint8),
                {
                    "valid_pixels": 1,
                    "no_data_mask": 2,
                    "crs": None,
                    "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                },
                np.array(([[6, 1, 2, 4], [6, 2, 4, 1], [10, 6, 7, 8]]), dtype=np.float64),
                np.array([[1, 1, 3, 5], [4, 1, 1, 1], [2, 2, 4, 6]], dtype=np.uint8),
                {
                    "valid_pixels": 1,
                    "no_data_mask": 2,
                    "crs": None,
                    "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                },
                [1, 2],
                1,
                np.array(
                    [
                        [
                            cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER
                            + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                            cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_RIGHT
                            + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                            cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_RIGHT
                            + cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                            + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                            cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                        ],
                        [
                            0,
                            cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER
                            + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                            cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                            + cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT
                            + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                            cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                        ],
                        [
                            cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT
                            + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                            cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_RIGHT
                            + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                            cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_RIGHT
                            + cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                            + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                            cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER
                            + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                        ],
                    ],
                    dtype=np.uint16,
                ),
                id="Positive disparity range",
            ),
            pytest.param(
                np.array(([[1, 2, 4, 6, 1], [2, 4, 1, 6, 1], [6, 7, 8, 10, 1], [0, 5, 6, 7, 8]]), dtype=np.float64),
                np.array([[2, 1, 1, 1, 1], [1, 2, 4, 1, 1], [5, 2, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=np.uint8),
                {
                    "valid_pixels": 1,
                    "no_data_mask": 2,
                    "crs": None,
                    "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                },
                np.array(([[6, 1, 2, 4, 1], [6, 2, 4, 1, 6], [10, 6, 7, 8, 1], [5, 6, 7, 8, 0]]), dtype=np.float64),
                np.array([[1, 1, 1, 2, 1], [5, 1, 1, 1, 1], [2, 1, 1, 6, 1], [0, 1, 1, 1, 1]], dtype=np.uint8),
                {
                    "valid_pixels": 1,
                    "no_data_mask": 2,
                    "crs": None,
                    "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                },
                [-1, 1],
                3,
                np.array(
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
                            + cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER
                            + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                            cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER
                            + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
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
                ),
                id="Positive and negative disparity range and window size = 3",
            ),
            pytest.param(
                np.ones((10, 10), dtype=np.float64),
                np.ones((10, 10), dtype=np.uint8),
                {
                    "valid_pixels": 1,
                    "no_data_mask": 0,
                    "crs": None,
                    "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                },
                np.ones((10, 10), dtype=np.float64),
                np.tril(np.ones((10, 10), dtype=np.uint8), -1.5),
                {
                    "valid_pixels": 1,
                    "no_data_mask": 0,
                    "crs": None,
                    "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                },
                [-3, 2],
                3,
                np.array(
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
                ),
                id="Positive and negative disparity range on flag 1",
            ),
            pytest.param(
                np.array(([[1, 2, 4, 6]]), dtype=np.float64),
                np.array([[2, 2, 2, 1]], dtype=np.uint8),
                {
                    "valid_pixels": 1,
                    "no_data_mask": 2,
                    "crs": None,
                    "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                },
                np.array(([[6, 1, 2, 4]]), dtype=np.float64),
                np.array([[2, 2, 2, 2]], dtype=np.uint8),
                {
                    "valid_pixels": 1,
                    "no_data_mask": 2,
                    "crs": None,
                    "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                },
                [-1, 1],
                1,
                np.array(
                    [
                        [
                            cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING
                            + cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                            + cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                            cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING
                            + cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                            cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING
                            + cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                            cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING
                            + cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                        ]
                    ]
                ),
                id="Possible constant duplication",
            ),
        ],
    )

    # pylint: disable=too-many-arguments
    def test_validity_mask(
        self, left_data, left_msk, left_attrs, right_data, right_msk, right_attrs, disparity, window_size, gt_mask
    ):
        """
        Test the validity_mask function

        # If bit 0 == 1 : Invalid pixel : the disparity interval is missing in the right image
        # If bit 1 == 1 : Invalid pixel : the disparity interval is missing in the right image
        # If bit 2 == 1 : Information: the disparity interval is incomplete (edge reached in the right image)
        # If bit 6 == 1 : Invalid pixel : invalidated by the validity mask of the left image given as input
        # If bit 7 == 1 : Invalid pixel : right positions invalidated by the mask of the right image given as
        #    input
        """

        # Create left dataset
        left = xr.Dataset(
            {"im": (["row", "col"], left_data), "msk": (["row", "col"], left_msk)},
            coords={"row": np.arange(left_data.shape[0]), "col": np.arange(left_data.shape[1])},
        )

        left.attrs = left_attrs

        # Add disparity on left image
        add_disparity(left, disparity, None)

        # Create right dataset
        right = xr.Dataset(
            {"im": (["row", "col"], right_data), "msk": (["row", "col"], right_msk)},
            coords={"row": np.arange(right_data.shape[0]), "col": np.arange(right_data.shape[1])},
        )

        right.attrs = right_attrs

        matching_cost_plugin = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": window_size, "subpix": 1}
        )

        # Allocate cost volume
        grid = matching_cost_plugin.allocate_cost_volume(
            left, (left["disparity"].sel(band_disp="min"), left["disparity"].sel(band_disp="max"))
        )

        # Compute validity mask
        grid = validity_mask(left, right, grid)

        # Compute cost volume
        cv = matching_cost_plugin.compute_cost_volume(left, right, grid)

        matching_cost_plugin.cv_masked(
            left,
            right,
            cv,
            left["disparity"].sel(band_disp="min"),
            left["disparity"].sel(band_disp="max"),
        )

        # Check if the calculated validity mask is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(cv["validity_mask"].data, gt_mask)
