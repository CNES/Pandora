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
This module contains functions to test the disparity map validation step.
"""

import unittest

import numpy as np
import xarray as xr

import tests.common as common
import pandora.constants as cst
import pandora.validation as validation


class TestValidation(unittest.TestCase):
    """
    TestValidation class allows to test all the methods in the module Validation
    """

    def setUp(self):
        """
        Method called to prepare the test fixture

        """
        # Create left and right disparity map
        self.left = xr.Dataset(
            {
                "disparity_map": (["row", "col"], np.array([[0, -1, 1, -2], [2, 2, -1, 0]], dtype=np.float32)),
                "confidence_measure": (["row", "col", "indicator"], np.full((2, 4, 1), np.nan)),
                "validity_mask": (
                    ["row", "col"],
                    np.array(
                        [[0, 0, 0, cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING], [0, 0, 0, 0]],
                        dtype=np.uint16,
                    ),
                ),
            },
            coords={"row": [0, 1], "col": np.arange(4)},
        )
        self.left.attrs["disp_min"] = -2
        self.left.attrs["disp_max"] = 2
        self.left.attrs["offset_row_col"] = 0

        self.right = xr.Dataset(
            {
                "disparity_map": (["row", "col"], np.array([[0, 2, -1, -1], [1, 1, -2, -1]], dtype=np.float32)),
                "confidence_measure": (["row", "col", "indicator"], np.full((2, 4, 1), np.nan)),
                "validity_mask": (["row", "col"], np.array([[0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint16)),
            },
            coords={"row": [0, 1], "col": np.arange(4)},
        )
        self.right.attrs["disp_min"] = -2
        self.right.attrs["disp_max"] = 2
        self.right.attrs["offset_row_col"] = 0

    def test_cross_checking(self):
        """
        Test the confidence measure and the validity_mask for the cross checking method,
                - If out & MSK_PIXEL_OCCLUSION != 0 : Invalid pixel : occluded pixel
                - If out & MSK_PIXEL_MISMATCH  != 0  : Invalid pixel : mismatched pixel
        """
        # Compute the cross checking confidence measure and validity mask
        validation_matcher = validation.AbstractValidation(
            **{"validation_method": "cross_checking", "cross_checking_threshold": 0.0}
        )

        left = validation_matcher.disparity_checking(self.left, self.right)

        # Confidence measure ground truth
        gt_dist = np.array(
            [
                [[np.nan, 0.0], [np.nan, 1.0], [np.nan, 0.0], [np.nan, np.nan]],
                [[np.nan, 0.0], [np.nan, 1.0], [np.nan, 0.0], [np.nan, 1.0]],
            ],
            dtype=np.float32,
        )

        # Check if the calculated confidence measure is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(left["confidence_measure"].data, gt_dist)

        # validity mask ground truth
        gt_mask = np.array(
            [
                [0, cst.PANDORA_MSK_PIXEL_MISMATCH, 0, cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING],
                [0, cst.PANDORA_MSK_PIXEL_MISMATCH, 0, cst.PANDORA_MSK_PIXEL_OCCLUSION],
            ],
            dtype=np.uint16,
        )

        # Check if the calculated mask is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(left["validity_mask"].data, gt_mask)

    @staticmethod
    def test_distance_lr_rl():
        """
        Test the confidence measure that computes the distance disp LR / disp RL with nodata pixel in images
        """
        # Create left and right disparity map
        left = xr.Dataset(
            {
                "disparity_map": (
                    ["row", "col"],
                    np.array(
                        [[np.nan, np.nan, np.nan, np.nan], [np.nan, 1, -1, np.nan], [np.nan, np.nan, np.nan, np.nan]],
                        dtype=np.float32,
                    ),
                ),
                "confidence_measure": (["row", "col", "indicator"], np.full((3, 4, 1), np.nan)),
                "validity_mask": (
                    ["row", "col"],
                    np.array(
                        [
                            [
                                cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                                cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                                cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                                cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                            ],
                            [
                                cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                                0,
                                0,
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
                    ),
                ),
            },
            coords={"row": [0, 1, 2], "col": [0, 1, 2, 3]},
        )
        left.attrs["disp_min"] = -1
        left.attrs["disp_max"] = 1

        right = xr.Dataset(
            {
                "disparity_map": (
                    ["row", "col"],
                    np.array(
                        [[np.nan, np.nan, np.nan, np.nan], [np.nan, 0, -1, np.nan], [np.nan, np.nan, np.nan, np.nan]],
                        dtype=np.float32,
                    ),
                ),
                "confidence_measure": (["row", "col", "indicator"], np.full((3, 4, 1), np.nan)),
                "validity_mask": (
                    ["row", "col"],
                    np.array(
                        [
                            [
                                cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                                cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                                cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                                cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                            ],
                            [
                                cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                                0,
                                0,
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
                    ),
                ),
            },
            coords={"row": [0, 1, 2], "col": [0, 1, 2, 3]},
        )
        right.attrs["disp_min"] = -1
        right.attrs["disp_max"] = 1

        # Compute the confidence measure
        validation_matcher = validation.AbstractValidation(
            **{"validation_method": "cross_checking", "cross_checking_threshold": 0.0}
        )
        left = validation_matcher.disparity_checking(left, right)

        # Confidence measure ground truth
        gt_dist = np.array(
            [
                [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]],
                [[np.nan, np.nan], [np.nan, 0.0], [np.nan, 1], [np.nan, np.nan]],
                [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]],
            ],
            dtype=np.float32,
        )

        # Check if the calculated confidence measure is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(left["confidence_measure"].data, gt_dist)

    @staticmethod
    def test_cross_checking_float_disparity():
        """
        Test the validity_mask for the cross checking method with floating disparity,
                - If out & MSK_PIXEL_OCCLUSION != 0 : Invalid pixel : occluded pixel
                - If out & MSK_PIXEL_MISMATCH  != 0  : Invalid pixel : mismatched pixel
        """
        # Create left and right disparity map
        left = xr.Dataset(
            {
                "disparity_map": (["row", "col"], np.array([[0, -1.2, 1, -2], [2, 1.8, -1, 0]], dtype=np.float32)),
                "confidence_measure": (["row", "col", "indicator"], np.full((2, 4, 1), np.nan)),
                "validity_mask": (
                    ["row", "col"],
                    np.array(
                        [[0, 0, 0, cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING], [0, 0, 0, 0]],
                        dtype=np.uint16,
                    ),
                ),
            },
            coords={"row": [0, 1], "col": np.arange(4)},
        )
        left.attrs["disp_min"] = -2
        left.attrs["disp_max"] = 2
        left.attrs["offset_row_col"] = 0

        right = xr.Dataset(
            {
                "disparity_map": (["row", "col"], np.array([[0, 2, -1.2, -1], [0.8, 1, -2, -1]], dtype=np.float32)),
                "confidence_measure": (["row", "col", "indicator"], np.full((2, 4, 1), np.nan)),
                "validity_mask": (["row", "col"], np.array([[0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint16)),
            },
            coords={"row": [0, 1], "col": np.arange(4)},
        )
        right.attrs["disp_min"] = -2
        right.attrs["disp_max"] = 2
        right.attrs["offset_row_col"] = 0

        # Compute the cross checking confidence measure and validity mask
        validation_matcher = validation.AbstractValidation(
            **{"validation_method": "cross_checking", "cross_checking_threshold": 0.0}
        )

        left = validation_matcher.disparity_checking(left, right)

        # validity mask ground truth
        gt_mask = np.array(
            [
                [0, cst.PANDORA_MSK_PIXEL_MISMATCH, 0, cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING],
                [0, cst.PANDORA_MSK_PIXEL_MISMATCH, 0, cst.PANDORA_MSK_PIXEL_OCCLUSION],
            ],
            dtype=np.uint16,
        )

        # Check if the calculated mask is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(left["validity_mask"].data, gt_mask)

    @staticmethod
    def test_interpolate_occlusion_mc_cnn():
        """
        Test the disparity interpolation of occlusion
        """
        disp_data = np.array([[0, -1, 1, -2.1], [2, 2, -1.7, 0]], dtype=np.float32)

        msk_data = np.array(
            [
                [
                    cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_OCCLUSION,
                    cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    0,
                ],
                [
                    cst.PANDORA_MSK_PIXEL_OCCLUSION,
                    cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT,
                    0,
                    cst.PANDORA_MSK_PIXEL_OCCLUSION,
                ],
            ],
            dtype=np.uint16,
        )
        # Create left and right disparity map
        left = xr.Dataset(
            {"disparity_map": (["row", "col"], disp_data), "validity_mask": (["row", "col"], msk_data)},
            coords={"row": [0, 1], "col": np.arange(4)},
        )
        left.attrs["disp_min"] = -2
        left.attrs["disp_max"] = 2

        # Interpolate occlusions
        interpolation_matcher = validation.AbstractInterpolation(**{"interpolated_disparity": "mc-cnn"})
        interpolation_matcher.interpolated_disparity(left)

        # validity mask after interpolation
        gt_mask_after_int = np.array(
            [
                [
                    cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    cst.PANDORA_MSK_PIXEL_FILLED_OCCLUSION,
                    cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING,
                    0,
                ],
                [
                    cst.PANDORA_MSK_PIXEL_FILLED_OCCLUSION,
                    cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT,
                    0,
                    cst.PANDORA_MSK_PIXEL_FILLED_OCCLUSION,
                ],
            ],
            dtype=np.uint16,
        )

        # Check if the calculated mask is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(left["validity_mask"].data, gt_mask_after_int)

        # left disparity map after interpolation
        gt_disp_after_int = np.array([[0, -2.1, 1, -2.1], [-1.7, 2, -1.7, -1.7]], dtype=np.float32)

        # Check if the calculated disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(left["disparity_map"].data, gt_disp_after_int)

    @staticmethod
    def test_interpolate_mismatch_mc_cnn():
        """
        Test the disparity interpolation of mismatch
        """
        disp_data = np.array(
            [[0, 1.2, -2, -1, -2], [1, 0, 1, 0, 0], [2, 1, -1, -2, -1], [1, -1, 1, -1, -1.3]], dtype=np.float32
        )

        msk_data = np.array(
            [
                [
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    0,
                    cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION,
                    0,
                ],
                [0, 0, cst.PANDORA_MSK_PIXEL_MISMATCH, 0, 0],
                [
                    0,
                    cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION,
                    cst.PANDORA_MSK_PIXEL_MISMATCH,
                    cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT,
                    cst.PANDORA_MSK_PIXEL_MISMATCH,
                ],
                [0, cst.PANDORA_MSK_PIXEL_MISMATCH, 0, 0, 0],
            ],
            dtype=np.uint16,
        )
        # Create left and right disparity map
        left = xr.Dataset(
            {"disparity_map": (["row", "col"], disp_data), "validity_mask": (["row", "col"], msk_data)},
            coords={"row": np.arange(4), "col": np.arange(5)},
        )
        left.attrs["disp_min"] = -2
        left.attrs["disp_max"] = 2

        # Interpolate mistmatch
        interpolation_matcher = validation.AbstractInterpolation(**{"interpolated_disparity": "mc-cnn"})
        interpolation_matcher.interpolated_disparity(left)

        # validity mask after interpolation
        gt_mask_after_int = np.array(
            [
                [
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    0,
                    cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION,
                    0,
                ],
                [0, 0, cst.PANDORA_MSK_PIXEL_FILLED_MISMATCH, 0, 0],
                [
                    0,
                    (1 << 3),
                    cst.PANDORA_MSK_PIXEL_FILLED_MISMATCH,
                    cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT,
                    cst.PANDORA_MSK_PIXEL_FILLED_MISMATCH,
                ],
                [0, cst.PANDORA_MSK_PIXEL_FILLED_MISMATCH, 0, 0, 0],
            ],
            dtype=np.uint16,
        )

        # Check if the calculated mask is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(left["validity_mask"].data, gt_mask_after_int)

        # left disparity map after interpolation
        gt_disp_after_int = np.array(
            [
                [0, 1.2, -2, -1, -2],
                [1, 0, np.median([1.2, 1, 0, 0, 0, 1, -2, -2, -2, -1, 0, 0, 0, -1, -1.3]), 0, 0],
                [
                    2,
                    1,
                    np.median([1, 1, 1, 1, 1, 0, 1, -2, -1, 0, 0, -1, -1, 1]),
                    -2,
                    np.median([-1, -1, -1, 1, 1, 0, 0, 0, 0, 0]),
                ],
                [1, np.median([1, 1, 1, 2, 1, 1, 1, 0, 1, 1, 1]), 1, -1, -1.3],
            ],
            dtype=np.float32,
        )

        # Check if the calculated disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(left["disparity_map"].data, gt_disp_after_int)

    @staticmethod
    def test_interpolate_occlusion_sgm():
        """
        Test the disparity interpolation of occlusion
        """
        disp_data = np.array(
            [[0, 1.2, -2, -1, -2], [1, 0, 1, 0, 0], [2, 1, -1, -2, -1], [1, -1, 1, -1, -1.3]], dtype=np.float32
        )

        msk_data = np.array(
            [
                [
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    0,
                    cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION,
                    0,
                ],
                [0, 0, cst.PANDORA_MSK_PIXEL_OCCLUSION, 0, 0],
                [
                    0,
                    cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION,
                    cst.PANDORA_MSK_PIXEL_OCCLUSION,
                    cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT,
                    cst.PANDORA_MSK_PIXEL_OCCLUSION,
                ],
                [0, cst.PANDORA_MSK_PIXEL_OCCLUSION, 0, 0, 0],
            ],
            dtype=np.uint16,
        )
        # Create left and right disparity map
        left = xr.Dataset(
            {"disparity_map": (["row", "col"], disp_data), "validity_mask": (["row", "col"], msk_data)},
            coords={"row": np.arange(4), "col": np.arange(5)},
        )
        left.attrs["disp_min"] = -2
        left.attrs["disp_max"] = 2

        # Interpolate occlusion
        interpolation_matcher = validation.AbstractInterpolation(**{"interpolated_disparity": "sgm"})
        interpolation_matcher.interpolated_disparity(left)

        # validity mask after interpolation
        gt_mask_after_int = np.array(
            [
                [
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    0,
                    cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION,
                    0,
                ],
                [0, 0, cst.PANDORA_MSK_PIXEL_FILLED_OCCLUSION, 0, 0],
                [
                    0,
                    cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION,
                    cst.PANDORA_MSK_PIXEL_FILLED_OCCLUSION,
                    cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT,
                    cst.PANDORA_MSK_PIXEL_FILLED_OCCLUSION,
                ],
                [0, cst.PANDORA_MSK_PIXEL_FILLED_OCCLUSION, 0, 0, 0],
            ],
            dtype=np.uint16,
        )

        # Check if the calculated mask is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(left["validity_mask"].data, gt_mask_after_int)

        # left disparity map after interpolation
        gt_disp_after_int = np.array(
            [[0, 1.2, -2, -1, -2], [1, 0, 0, 0, 0], [2, 1, 0, -2, 0], [1, 1, 1, -1, -1.3]], dtype=np.float32
        )

        # Check if the calculated disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(left["disparity_map"].data, gt_disp_after_int)

    @staticmethod
    def test_interpolate_mismatch_sgm():
        """
        Test the disparity interpolation of mismatch
        """
        disp_data = np.array(
            [[0, 1.2, -2, -1, -2], [1, 0, 1, 0, 0], [2, 1, -1, -2, -1], [1, -1, 1, -1, -1.3]], dtype=np.float32
        )

        msk_data = np.array(
            [
                [
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    0,
                    cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION,
                    0,
                ],
                [0, 0, cst.PANDORA_MSK_PIXEL_MISMATCH, 0, 0],
                [
                    0,
                    cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION,
                    cst.PANDORA_MSK_PIXEL_MISMATCH,
                    cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT,
                    cst.PANDORA_MSK_PIXEL_MISMATCH,
                ],
                [0, cst.PANDORA_MSK_PIXEL_MISMATCH, 0, 0, 0],
            ],
            dtype=np.uint16,
        )
        # Create left and right disparity map
        left = xr.Dataset(
            {"disparity_map": (["row", "col"], disp_data), "validity_mask": (["row", "col"], msk_data)},
            coords={"row": np.arange(4), "col": np.arange(5)},
        )
        left.attrs["disp_min"] = -2
        left.attrs["disp_max"] = 2

        # Interpolate mismatch
        interpolation_matcher = validation.AbstractInterpolation(**{"interpolated_disparity": "sgm"})
        interpolation_matcher.interpolated_disparity(left)

        # validity mask after interpolation
        gt_mask_after_int = np.array(
            [
                [
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    0,
                    cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION,
                    0,
                ],
                [0, 0, cst.PANDORA_MSK_PIXEL_FILLED_MISMATCH, 0, 0],
                [
                    0,
                    cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION,
                    cst.PANDORA_MSK_PIXEL_FILLED_MISMATCH,
                    cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT,
                    cst.PANDORA_MSK_PIXEL_FILLED_MISMATCH,
                ],
                [0, cst.PANDORA_MSK_PIXEL_FILLED_MISMATCH, 0, 0, 0],
            ],
            dtype=np.uint16,
        )

        # Check if the calculated mask is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(left["validity_mask"].data, gt_mask_after_int)

        # left disparity map after interpolation
        gt_disp_after_int = np.array(
            [
                [0, 1.2, -2, -1, -2],
                [1, 0, np.median([1.2, -2, -1, 0, 0, 1, 1, -1.3]), 0, 0],
                [2, 1, np.median([-2, 0, -1, -1, 1, 1, 0]), -2, np.median([0, -1.3, -1, 1, 0])],
                [1, np.median([2, 1, 0, 1, 1]), 1, -1, -1.3],
            ],
            dtype=np.float32,
        )

        # Check if the calculated disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(left["disparity_map"].data, gt_disp_after_int)

    @staticmethod
    def test_interpolate_mismatch_and_occlusion_sgm():
        """
        Test the disparity interpolation of mismatch and occlusion
        """
        # Test with mismatched pixel that are direct neighbors of occluded pixels
        disp_data = np.array(
            [[0, 1, -2, -1, -2], [1, 0, 1, 0, 0], [2, 1, -1, -2, -1], [1, -1, 1, -1, -1]], dtype=np.float32
        )

        msk_data = np.array(
            [
                [
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    0,
                    cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION,
                    cst.PANDORA_MSK_PIXEL_OCCLUSION,
                ],
                [0, 0, cst.PANDORA_MSK_PIXEL_MISMATCH, 0, 0],
                [
                    0,
                    cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION,
                    cst.PANDORA_MSK_PIXEL_MISMATCH,
                    cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT,
                    cst.PANDORA_MSK_PIXEL_MISMATCH,
                ],
                [cst.PANDORA_MSK_PIXEL_OCCLUSION, cst.PANDORA_MSK_PIXEL_MISMATCH, 0, 0, 0],
            ],
            dtype=np.uint16,
        )
        # Create left and right disparity map
        left = xr.Dataset(
            {"disparity_map": (["row", "col"], disp_data), "validity_mask": (["row", "col"], msk_data)},
            coords={"row": np.arange(4), "col": np.arange(5)},
        )
        left.attrs["disp_min"] = -2
        left.attrs["disp_max"] = 2

        # Interpolate mismatch
        interpolation_matcher = validation.AbstractInterpolation(**{"interpolated_disparity": "sgm"})
        interpolation_matcher.interpolated_disparity(left)

        # validity mask after interpolation
        gt_mask_after_int = np.array(
            [
                [
                    cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER,
                    cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE,
                    0,
                    cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION,
                    cst.PANDORA_MSK_PIXEL_FILLED_OCCLUSION,
                ],
                [0, 0, cst.PANDORA_MSK_PIXEL_FILLED_MISMATCH, 0, 0],
                [
                    0,
                    cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION,
                    cst.PANDORA_MSK_PIXEL_FILLED_MISMATCH,
                    cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT,
                    cst.PANDORA_MSK_PIXEL_FILLED_MISMATCH,
                ],
                [cst.PANDORA_MSK_PIXEL_FILLED_OCCLUSION, cst.PANDORA_MSK_PIXEL_FILLED_OCCLUSION, 0, 0, 0],
            ],
            dtype=np.uint16,
        )

        # Check if the calculated mask is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(left["validity_mask"].data, gt_mask_after_int)

        # left disparity map after interpolation
        gt_disp_after_int = np.array(
            [
                [0, 1, -2, -1, 0],
                [1, 0, np.median([1, 1, 0, 1, -2, -1, 0, -1]), 0, 0],
                [2, 1, np.median([1, 1, 0, -2, 0, -1]), -2, np.median([-1, -1, 1, 0, 0])],
                [1, 1, 1, -1, -1],
            ],
            dtype=np.float32,
        )

        # Check if the calculated disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(left["disparity_map"].data, gt_disp_after_int)


if __name__ == "__main__":
    common.setup_logging()
    unittest.main()
