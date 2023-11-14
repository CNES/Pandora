# type:ignore
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

import numpy as np
import pytest
import xarray as xr
from json_checker import MissKeyCheckerError

import pandora.constants as cst
import pandora.filter as flt
from pandora.descriptors.margins import Margins


class TestMedianFilter:
    """Test MedianFilter"""

    @pytest.fixture()
    def filter_median(self, request):
        return flt.AbstractFilter(**{"filter_method": "median", "filter_size": request.param})

    @pytest.fixture()
    def dataset1(self):
        """Dataset #1"""
        disp = np.array([[5, 6, 7, 8, 9], [6, 85, 1, 36, 5], [5, 9, 23, 12, 2], [6, 1, 9, 2, 4]], dtype=np.float32)

        valid = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE, 0, 0, 0],
                [0, cst.PANDORA_MSK_PIXEL_FILLED_OCCLUSION, 0, 0, 0],
                [0, 0, 0, 0, cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION],
            ],
            dtype=np.uint16,
        )

        return xr.Dataset(
            {"disparity_map": (["row", "col"], disp), "validity_mask": (["row", "col"], valid)},
            coords={"row": np.arange(4), "col": np.arange(5)},
        )

    @pytest.fixture()
    def dataset2(self):
        """Dataset #2"""
        disp = np.array([[7, 8, 4, 5, 5], [5, 9, 4, 3, 8], [5, 2, 7, 2, 2], [6, 1, 9, 2, 4]], dtype=np.float32)

        valid = np.array(
            [
                [
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    0,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    cst.PANDORA_MSK_PIXEL_FILLED_OCCLUSION + cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    0,
                ],
                [
                    cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_RIGHT,
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_OCCLUSION,
                    0,
                    0,
                ],
                [
                    cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT,
                    cst.PANDORA_MSK_PIXEL_MISMATCH,
                    cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                    + cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION,
                    0,
                ],
                [
                    cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_OCCLUSION,
                    cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT,
                    0,
                    cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                ],
            ],
            dtype=np.uint16,
        )

        return xr.Dataset(
            {"disparity_map": (["row", "col"], disp), "validity_mask": (["row", "col"], valid)},
            coords={"row": np.arange(4), "col": np.arange(5)},
        )

    @pytest.fixture()
    def dataset3(self):
        """Dataset #3"""
        disp = np.array([[7, 8, 4, 5, 5], [5, 9, 4, 3, 8], [5, 2, 7, 2, 2], [6, 1, 9, 2, 4]], dtype=np.float32)

        valid = np.array(
            [
                [
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    0,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    cst.PANDORA_MSK_PIXEL_FILLED_OCCLUSION + cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    0,
                ],
                [0, 0, cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION, 0, 0],
                [
                    0,
                    0,
                    0,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                    + cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION,
                    0,
                ],
                [cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_RIGHT, 0, 0, 0, 0],
            ],
            dtype=np.uint16,
        )

        return xr.Dataset(
            {"disparity_map": (["row", "col"], disp), "validity_mask": (["row", "col"], valid)},
            coords={"row": np.arange(4), "col": np.arange(5)},
        )

    @pytest.fixture()
    def dataset4(self):
        """Dataset #4"""
        disp = np.array(
            [[7, 8, 4, 5, 5], [5, 9, 4, 3, 8], [5, 2, 7, 2, 2], [6, 1, 9, 2, 4], [1, 6, 2, 7, 8]], dtype=np.float32
        )

        valid = np.array(
            [
                [
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    0,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    cst.PANDORA_MSK_PIXEL_FILLED_OCCLUSION + cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    0,
                ],
                [0, 0, cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION, 0, 0],
                [
                    0,
                    0,
                    0,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
                    + cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION,
                    0,
                ],
                [cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_RIGHT, 0, 0, 0, 0],
                [
                    cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT,
                    0,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING
                    + cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION,
                    0,
                ],
            ],
            dtype=np.uint16,
        )

        return xr.Dataset(
            {"disparity_map": (["row", "col"], disp), "validity_mask": (["row", "col"], valid)},
            coords={"row": np.arange(5), "col": np.arange(5)},
        )

    @pytest.mark.parametrize(
        [
            "filter_median",
            "disparity_dataset",
            "gt_disp",
        ],
        [
            pytest.param(
                3,
                "dataset1",
                np.array([[5, 6, 7, 8, 9], [6, 6, 9, 8, 5], [5, 6, 9, 5, 2], [6, 1, 9, 2, 4]], dtype=np.float32),
                id="Case1",
            ),
            pytest.param(
                3,
                "dataset2",
                np.array([[7, 8, 4, 5, 5], [5, 9, 4, 3.5, 8], [5, 2, 7, 2, 2], [6, 1, 9, 2, 4]], dtype=np.float32),
                id="Case2",
            ),
            pytest.param(
                3,
                "dataset3",
                np.array([[7, 8, 4, 5, 5], [5, 5, 4, 4, 8], [5, 5, 3, 4, 2], [6, 1, 9, 2, 4]], dtype=np.float32),
                id="Case3",
            ),
            pytest.param(
                5,
                "dataset4",
                np.array(
                    [[7, 8, 4, 5, 5], [5, 9, 4, 3, 8], [5, 2, 5, 2, 2], [6, 1, 9, 2, 4], [1, 6, 2, 7, 8]],
                    dtype=np.float32,
                ),
                id="Case4",
            ),
        ],
        indirect=["filter_median"],
    )
    def test_median_filter(self, request, filter_median, disparity_dataset, gt_disp):
        """
        Test the median method

        """
        disp_dataset = request.getfixturevalue(disparity_dataset)
        # Apply median filter to the disparity map. Median filter is only applied on valid pixels.
        filter_median.filter_disparity(disp_dataset)

        # Check if the calculated disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(disp_dataset["disparity_map"].data, gt_disp)

    @pytest.mark.parametrize(["filter_median", "filter_size"], [(1, 1), (3, 3)], indirect=["filter_median"])
    def test_margins(self, filter_median, filter_size):
        """Check that margin computation is correct."""
        assert filter_median.margins == Margins(filter_size, filter_size, filter_size, filter_size)


class TestBilateralFilter:
    """
    Test BilateralFilter
    """

    @pytest.mark.parametrize(
        ["sigma_space", "row_length", "col_length", "expected"],
        [
            pytest.param(1.0, 5, 5, Margins(4, 4, 4, 4), id="Result should be computed from sigma_space"),
            pytest.param(2.0, 7, 6, Margins(6, 6, 6, 6), id="Result should be lowest image's dimension"),
        ],
    )
    def test_margins(self, sigma_space, row_length, col_length, expected):
        """Check that margin computation is correct."""
        filter_config = {
            "filter_method": "bilateral",
            "sigma_color": 4.0,
            "sigma_space": sigma_space,
            "image_shape": (row_length, col_length),
        }

        filter_ = flt.AbstractFilter(**filter_config)
        assert filter_.margins == expected

    @pytest.mark.parametrize("missing_key", ["filter_method", "image_shape"])
    def test_check_conf_fails_when_is_missing_mandatory_key(self, missing_key):
        """When a mandatory key is missing instanciation should fail."""
        filter_config = {
            "filter_method": "bilateral",
            "sigma_color": 4.0,
            "sigma_space": 6.0,
            "image_shape": (450, 600),
        }
        del filter_config[missing_key]

        with pytest.raises((MissKeyCheckerError, KeyError)):
            flt.AbstractFilter(**filter_config)

    @staticmethod
    def test_gauss_spatial_kernel():
        """
        Test the gauss spatial kernel function

        """

        user_cfg = {
            "filter_method": "bilateral",
            "sigma_color": 4.0,
            "sigma_space": 6.0,
            "image_shape": [5, 5],
        }

        filter_bilateral = flt.AbstractFilter(**user_cfg)

        # Gauss spatial kernel of size (5,5) and sigma_space = 6
        # arr[i, j] = np.sqrt(abs(i - kernel_size // 2) ** 2 + abs(j - kernel_size // 2) ** 2)
        spatial_kernel = np.array(
            [
                [
                    np.sqrt(abs(0 - 2) ** 2 + abs(0 - 2) ** 2),
                    np.sqrt(abs(0 - 2) ** 2 + abs(1 - 2) ** 2),
                    np.sqrt(abs(0 - 2) ** 2 + abs(2 - 2) ** 2),
                    np.sqrt(abs(0 - 2) ** 2 + abs(3 - 2) ** 2),
                    np.sqrt(abs(0 - 2) ** 2 + abs(4 - 2) ** 2),
                ],
                [
                    np.sqrt(abs(1 - 2) ** 2 + abs(0 - 2) ** 2),
                    np.sqrt(abs(1 - 2) ** 2 + abs(1 - 2) ** 2),
                    np.sqrt(abs(1 - 2) ** 2 + abs(2 - 2) ** 2),
                    np.sqrt(abs(1 - 2) ** 2 + abs(3 - 2) ** 2),
                    np.sqrt(abs(1 - 2) ** 2 + abs(4 - 2) ** 2),
                ],
                [
                    np.sqrt(abs(2 - 2) ** 2 + abs(0 - 2) ** 2),
                    np.sqrt(abs(2 - 2) ** 2 + abs(1 - 2) ** 2),
                    np.sqrt(abs(2 - 2) ** 2 + abs(2 - 2) ** 2),
                    np.sqrt(abs(2 - 2) ** 2 + abs(3 - 2) ** 2),
                    np.sqrt(abs(2 - 2) ** 2 + abs(4 - 2) ** 2),
                ],
                [
                    np.sqrt(abs(3 - 2) ** 2 + abs(0 - 2) ** 2),
                    np.sqrt(abs(3 - 2) ** 2 + abs(1 - 2) ** 2),
                    np.sqrt(abs(3 - 2) ** 2 + abs(2 - 2) ** 2),
                    np.sqrt(abs(3 - 2) ** 2 + abs(3 - 2) ** 2),
                    np.sqrt(abs(3 - 2) ** 2 + abs(4 - 2) ** 2),
                ],
                [
                    np.sqrt(abs(4 - 2) ** 2 + abs(0 - 2) ** 2),
                    np.sqrt(abs(4 - 2) ** 2 + abs(1 - 2) ** 2),
                    np.sqrt(abs(4 - 2) ** 2 + abs(2 - 2) ** 2),
                    np.sqrt(abs(4 - 2) ** 2 + abs(3 - 2) ** 2),
                    np.sqrt(abs(4 - 2) ** 2 + abs(4 - 2) ** 2),
                ],
            ],
            dtype=np.float32,
        )
        gauss_spatial_kernel = np.exp(-((spatial_kernel / 6) ** 2) * 0.5) * 1 / (6 * np.sqrt(2 * np.pi))

        gt_gauss_spatial_kernel = filter_bilateral.gauss_spatial_kernel(kernel_size=5, sigma=6)

        np.testing.assert_allclose(gauss_spatial_kernel, gt_gauss_spatial_kernel, rtol=1e-07)

    @staticmethod
    def test_on_valid_pixels():
        """
        Test the bilateral method. Bilateral filter is only applied on valid pixels.


        """

        user_cfg = {
            "filter_method": "bilateral",
            "sigma_color": 4.0,
            "sigma_space": 6.0,
            "image_shape": [5, 5],
        }

        filter_bilateral = flt.AbstractFilter(**user_cfg)

        disp = np.array(
            [[5, 6, 7, 8, 9], [6, 85, 1, 36, 5], [5, 9, 23, 12, 2], [6, 1, 9, 2, 4], [6, 7, 4, 2, 1]], dtype=np.float32
        )

        valid = np.array(
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.uint16
        )

        disp_dataset = xr.Dataset(
            {"disparity_map": (["row", "col"], disp), "validity_mask": (["row", "col"], valid)},
            coords={"row": np.arange(5), "col": np.arange(5)},
        )

        # Window = [[5, 6, 7, 8, 9],
        #         [6, 85, 1, 36, 5],
        #         [5, 9, 23, 12, 2],
        #         [6, 1, 9, 2, 4],
        #         [6, 7, 4, 2, 1]]
        # Window center = 23 and gaussian with sigma color = 4
        gauss_disp_offset = (
            np.array(
                [
                    [
                        np.exp(-(((5 - 23) / 4) ** 2) * 0.5),
                        np.exp(-(((6 - 23) / 4) ** 2) * 0.5),
                        np.exp(-(((7 - 23) / 4) ** 2) * 0.5),
                        np.exp(-(((8 - 23) / 4) ** 2) * 0.5),
                        np.exp(-(((9 - 23) / 4) ** 2) * 0.5),
                    ],
                    [
                        np.exp(-(((6 - 23) / 4) ** 2) * 0.5),
                        np.exp(-(((85 - 23) / 4) ** 2) * 0.5),
                        np.exp(-(((1 - 23) / 4) ** 2) * 0.5),
                        np.exp(-(((36 - 23) / 4) ** 2) * 0.5),
                        np.exp(-(((5 - 23) / 4) ** 2) * 0.5),
                    ],
                    [
                        np.exp(-(((5 - 23) / 4) ** 2) * 0.5),
                        np.exp(-(((9 - 23) / 4) ** 2) * 0.5),
                        np.exp(-(((23 - 23) / 4) ** 2) * 0.5),
                        np.exp(-(((12 - 23) / 4) ** 2) * 0.5),
                        np.exp(-(((2 - 23) / 4) ** 2) * 0.5),
                    ],
                    [
                        np.exp(-(((6 - 23) / 4) ** 2) * 0.5),
                        np.exp(-(((1 - 23) / 4) ** 2) * 0.5),
                        np.exp(-(((9 - 23) / 4) ** 2) * 0.5),
                        np.exp(-(((2 - 23) / 4) ** 2) * 0.5),
                        np.exp(-(((4 - 23) / 4) ** 2) * 0.5),
                    ],
                    [
                        np.exp(-(((6 - 23) / 4) ** 2) * 0.5),
                        np.exp(-(((7 - 23) / 4) ** 2) * 0.5),
                        np.exp(-(((4 - 23) / 4) ** 2) * 0.5),
                        np.exp(-(((2 - 23) / 4) ** 2) * 0.5),
                        np.exp(-(((1 - 23) / 4) ** 2) * 0.5),
                    ],
                ],
                dtype=np.float32,
            )
            * 1
            / (4 * np.sqrt(2 * np.pi))
        )

        # Multiply by already tested gaussian spatial kernel
        gauss_spatial_kernel = filter_bilateral.gauss_spatial_kernel(kernel_size=5, sigma=6)
        weights = np.multiply(gauss_spatial_kernel, gauss_disp_offset)
        # Multiply by its pixel
        pixel_weights = np.multiply(disp, weights)

        filtered_pixel = np.sum(pixel_weights) / np.sum(weights)

        # Filtered disparity map ground truth
        gt_disp = np.array(
            [[5, 6, 7, 8, 9], [6, 85, 1, 36, 5], [5, 9, filtered_pixel, 12, 2], [6, 1, 9, 2, 4], [6, 7, 4, 2, 1]],
            dtype=np.float32,
        )

        # Apply bilateral filter to the disparity map.
        filter_bilateral.filter_disparity(disp_dataset)

        np.testing.assert_allclose(gt_disp, disp_dataset["disparity_map"].data, rtol=1e-07)

    @staticmethod
    def test_with_nans():
        """
        Test the bilateral method with input Nans.

        """
        user_cfg = {
            "filter_method": "bilateral",
            "sigma_color": 4.0,
            "sigma_space": 6.0,
            "image_shape": [5, 5],
        }

        filter_bilateral = flt.AbstractFilter(**user_cfg)

        disp = np.array(
            [[5, 6, 7, 8, 9], [6, 85, np.nan, 36, 5], [5, 9, 23, 12, 2], [6, np.nan, 9, 2, 4], [1, 6, 2, 7, 8]],
            dtype=np.float32,
        )

        valid = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION, 0, 0],
                [0, 0, 0, 0, 0],
                [0, cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.uint16,
        )

        disp_dataset = xr.Dataset(
            {"disparity_map": (["row", "col"], disp), "validity_mask": (["row", "col"], valid)},
            coords={"row": np.arange(5), "col": np.arange(5)},
        )

        # Apply bilateral filter to the disparity map.
        filter_bilateral.filter_disparity(disp_dataset)

        # Bilateral filter should not expand nans
        assert len(np.where(np.isnan(disp_dataset["disparity_map"].data))[0]) == 2

        # Test the bilateral method with input nans. Bilateral filter is only applied on valid pixels.

        disp = np.array(
            [
                [5, 6, np.nan, 8, 9],
                [6, np.nan, 1, 36, 5],
                [5, 9, 23, 12, np.nan],
                [6, np.nan, 9, 2, 4],
                [6, 7, 4, 2, 1],
            ],
            dtype=np.float32,
        )

        valid = np.array(
            [
                [0, 0, cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION, 0, 0],
                [0, cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE, 0, 0, 0],
                [0, cst.PANDORA_MSK_PIXEL_FILLED_OCCLUSION, 0, 0, cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION],
                [0, 0, 0, 0, cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.uint16,
        )

        disp_dataset = xr.Dataset(
            {"disparity_map": (["row", "col"], disp), "validity_mask": (["row", "col"], valid)},
            coords={"row": np.arange(5), "col": np.arange(5)},
        )

        # Window = [[5, 6, np.nan, 8, 9],
        #           [6, np.nan, 1, 36, 5],
        #           [5, 9, 23, 12, np.nan],
        #           [6, np.nan, 9, 2, 4],
        #           [1, 6, 2, 7, 8]]
        # Window center = 23 and gaussian with sigma color = 4
        gauss_disp_offset = (
            np.array(
                [
                    [
                        np.exp(-(((5 - 23) / 4) ** 2) * 0.5),
                        np.exp(-(((6 - 23) / 4) ** 2) * 0.5),
                        np.nan,
                        np.exp(-(((8 - 23) / 4) ** 2) * 0.5),
                        np.exp(-(((9 - 23) / 4) ** 2) * 0.5),
                    ],
                    [
                        np.exp(-(((6 - 23) / 4) ** 2) * 0.5),
                        np.nan,
                        np.exp(-(((1 - 23) / 4) ** 2) * 0.5),
                        np.exp(-(((36 - 23) / 4) ** 2) * 0.5),
                        np.exp(-(((5 - 23) / 4) ** 2) * 0.5),
                    ],
                    [
                        np.exp(-(((5 - 23) / 4) ** 2) * 0.5),
                        np.exp(-(((9 - 23) / 4) ** 2) * 0.5),
                        np.exp(-(((23 - 23) / 4) ** 2) * 0.5),
                        np.exp(-(((12 - 23) / 4) ** 2) * 0.5),
                        np.nan,
                    ],
                    [
                        np.exp(-(((6 - 23) / 4) ** 2) * 0.5),
                        np.nan,
                        np.exp(-(((9 - 23) / 4) ** 2) * 0.5),
                        np.exp(-(((2 - 23) / 4) ** 2) * 0.5),
                        np.exp(-(((4 - 23) / 4) ** 2) * 0.5),
                    ],
                    [
                        np.exp(-(((6 - 23) / 4) ** 2) * 0.5),
                        np.exp(-(((7 - 23) / 4) ** 2) * 0.5),
                        np.exp(-(((4 - 23) / 4) ** 2) * 0.5),
                        np.exp(-(((2 - 23) / 4) ** 2) * 0.5),
                        np.exp(-(((1 - 23) / 4) ** 2) * 0.5),
                    ],
                ],
                dtype=np.float32,
            )
            * 1
            / (4 * np.sqrt(2 * np.pi))
        )

        # Multiply by already tested gaussian spatial kernel
        gauss_spatial_kernel = filter_bilateral.gauss_spatial_kernel(kernel_size=5, sigma=6)
        weights = np.multiply(gauss_spatial_kernel, gauss_disp_offset)

        # Multiply by its pixel
        pixel_weights = np.multiply(disp, weights)

        filtered_pixel = np.nansum(pixel_weights) / np.nansum(weights)
        # Filtered disparity map ground truth
        gt_disp = np.array(
            [
                [5, 6, np.nan, 8, 9],
                [6, np.nan, 1, 36, 5],
                [5, 9, filtered_pixel, 12, np.nan],
                [6, np.nan, 9, 2, 4],
                [6, 7, 4, 2, 1],
            ],
            dtype=np.float32,
        )

        # Apply bilateral filter to the disparity map
        filter_bilateral.filter_disparity(disp_dataset)

        np.testing.assert_allclose(gt_disp, disp_dataset["disparity_map"].data, rtol=1e-07)

    @staticmethod
    def test_with_invalid_center():
        """
        Test the bilateral method with center pixel invalid. Bilateral filter is only applied on valid pixels.

        """

        user_cfg = {
            "filter_method": "bilateral",
            "sigma_color": 4.0,
            "sigma_space": 6.0,
            "image_shape": [5, 5],
        }

        filter_bilateral = flt.AbstractFilter(**user_cfg)
        disp = np.array(
            [[5, 6, 7, 8, 9], [6, 85, 1, 36, 5], [5, 9, 23, 12, 2], [6, 1, 9, 2, 4], [6, 7, 4, 2, 1]], dtype=np.float32
        )

        valid = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE, 0, 0, 0],
                [0, cst.PANDORA_MSK_PIXEL_FILLED_OCCLUSION, cst.PANDORA_MSK_PIXEL_INVALID, 0, 0],
                [0, 0, 0, 0, cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.uint16,
        )

        disp_dataset = xr.Dataset(
            {"disparity_map": (["row", "col"], disp), "validity_mask": (["row", "col"], valid)},
            coords={"row": np.arange(5), "col": np.arange(5)},
        )

        # Filtered disparity must be the same as input since center pixel was invalid

        gt_disp = np.array(
            [[5, 6, 7, 8, 9], [6, 85, 1, 36, 5], [5, 9, 23, 12, 2], [6, 1, 9, 2, 4], [6, 7, 4, 2, 1]], dtype=np.float32
        )

        # Apply bilateral filter to the disparity map
        filter_bilateral.filter_disparity(disp_dataset)

        np.testing.assert_allclose(gt_disp, disp_dataset["disparity_map"].data, rtol=1e-07)
