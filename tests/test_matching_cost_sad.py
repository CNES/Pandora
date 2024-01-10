# pylint: disable=duplicate-code
# pylint:disable=too-many-lines
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
This module contains functions to test the SAD matching cost step.
"""
import unittest
import pytest

import numpy as np
import xarray as xr
from rasterio import Affine

from pandora import matching_cost
from pandora.img_tools import add_disparity
from pandora.criteria import validity_mask
from tests import common


class TestMatchingCostSAD(unittest.TestCase):
    """
    TestMatchingCost class allows to test all the methods in the
    matching_cost SAD class
    """

    def setUp(self):
        """
        Method called to prepare the test fixture

        """

        self.left, self.right = common.matching_cost_tests_setup()
        self.left.pipe(add_disparity, disparity=[-1, 1], window=None)
        self.left_multiband, self.right_multiband = common.matching_cost_tests_multiband_setup()
        self.left_multiband.pipe(add_disparity, disparity=[-1, 1], window=None)

    def test_sad_cost(self):
        """
        Test the absolute difference method

        """
        # Absolute difference pixel-wise ground truth for the images self.left, self.right
        ad_ground_truth = np.array(
            (
                [0, 0, 0, 1, 1, 1],
                [0, 0, 0, abs(1 - 4), 0, abs(1 - 4)],
                [0, 0, 0, 0, abs(3 - 4), 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            )
        )

        # Computes the ad cost for the whole images
        matching_cost_matcher = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 1, "subpix": 1}
        )

        grid = matching_cost_matcher.allocate_cost_volume(
            self.left, (self.left["disparity"].sel(band_disp="min"), self.left["disparity"].sel(band_disp="max"))
        )
        sad = matching_cost_matcher.compute_cost_volume(img_left=self.left, img_right=self.right, cost_volume=grid)

        # Check if the calculated ad cost is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(sad["cost_volume"].sel(disp=0), ad_ground_truth)

        # Sum of absolute difference pixel-wise ground truth for the images self.left, self.right with window size 5
        sad_ground_truth = np.array(
            (
                [
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, 6.0, 10.0, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                ]
            )
        )

        # Computes the ad cost for the whole images
        matching_cost_matcher = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 5, "subpix": 1}
        )
        grid = matching_cost_matcher.allocate_cost_volume(
            self.left, (self.left["disparity"].sel(band_disp="min"), self.left["disparity"].sel(band_disp="max"))
        )

        # Compute validity mask
        grid = validity_mask(self.left, self.right, grid)

        sad = matching_cost_matcher.compute_cost_volume(img_left=self.left, img_right=self.right, cost_volume=grid)
        matching_cost_matcher.cv_masked(
            self.left,
            self.right,
            sad,
            self.left["disparity"].sel(band_disp="min"),
            self.left["disparity"].sel(band_disp="max"),
        )

        # Check if the calculated ad cost is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(sad["cost_volume"].sel(disp=0), sad_ground_truth)

    def test_sad_cost_multiband(self):
        """
        Test the absolute difference method

        """
        # Absolute difference pixel-wise ground truth for the images self.left, self.right
        ad_ground_truth = np.array(
            (
                [0, 0, 0, 1, 1, 1],
                [0, 0, 0, abs(1 - 4), 0, abs(1 - 4)],
                [0, 0, 0, 0, abs(3 - 4), 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            )
        )

        # Computes the ad cost for the whole images
        matching_cost_matcher = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 1, "subpix": 1, "band": "r"}
        )

        grid = matching_cost_matcher.allocate_cost_volume(
            self.left_multiband,
            (
                self.left_multiband["disparity"].sel(band_disp="min"),
                self.left_multiband["disparity"].sel(band_disp="max"),
            ),
        )
        sad = matching_cost_matcher.compute_cost_volume(
            img_left=self.left_multiband, img_right=self.right_multiband, cost_volume=grid
        )

        # Check if the calculated ad cost is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(sad["cost_volume"].sel(disp=0), ad_ground_truth)

        # Sum of absolute difference pixel-wise ground truth for the images self.left, self.right with window size 5
        sad_ground_truth = np.array(
            (
                [
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, 6.0, 10.0, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                ]
            )
        )
        # Computes the ad cost for the whole images
        matching_cost_matcher = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 5, "subpix": 1, "band": "r"}
        )
        grid = matching_cost_matcher.allocate_cost_volume(
            self.left_multiband,
            (
                self.left_multiband["disparity"].sel(band_disp="min"),
                self.left_multiband["disparity"].sel(band_disp="max"),
            ),
        )

        # Compute validity mask
        grid = validity_mask(self.left, self.right, grid)

        sad = matching_cost_matcher.compute_cost_volume(
            img_left=self.left_multiband, img_right=self.right_multiband, cost_volume=grid
        )
        matching_cost_matcher.cv_masked(
            self.left_multiband,
            self.right_multiband,
            sad,
            self.left_multiband["disparity"].sel(band_disp="min"),
            self.left_multiband["disparity"].sel(band_disp="max"),
        )

        # Compute gt cmax:
        window_size = 5
        sad_cmax_ground_truth = np.max(ad_ground_truth) * window_size**2
        # Check if the calculated ad cost is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(sad["cost_volume"].sel(disp=0), sad_ground_truth)

        # Check if the calculated max cost is equal to the ground truth
        np.testing.assert_array_equal(sad.attrs["cmax"], sad_cmax_ground_truth)

    @staticmethod
    def test_cost_volume():
        """
        Test the cost volume method

        """
        # Create simple images
        data = np.array(([1, 2, 1, 4], [6, 2, 7, 4], [1, 1, 3, 6]), dtype=np.float64)
        left = xr.Dataset(
            {"im": (["row", "col"], data)}, coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])}
        )
        left.attrs["crs"] = None
        left.attrs["transform"] = Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        left.pipe(add_disparity, disparity=[-2, 1], window=None)

        data = np.array(([6, 7, 8, 10], [2, 4, 1, 6], [9, 10, 1, 2]), dtype=np.float64)
        right = xr.Dataset(
            {"im": (["row", "col"], data)}, coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])}
        )
        right.attrs["crs"] = None
        right.attrs["transform"] = Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

        # Cost Volume ground truth for the stereo image simple_stereo_imgs,
        # with disp_min = -2, disp_max = 1, sad measure and subpixel_offset = 0
        ground_truth = np.array(
            [
                [
                    [np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan],
                ],
                [
                    [np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, 48, 35],
                    [np.nan, 40, 43, np.nan],
                    [np.nan, np.nan, np.nan, np.nan],
                ],
                [
                    [np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan],
                ],
            ]
        )

        # Computes the Cost Volume for the stereo image simple_stereo_imgs,
        # with disp_min = -2, disp_max = 1, sad measure, window_size = 3 and subpix = 1
        matching_cost_matcher = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 3, "subpix": 1}
        )

        grid = matching_cost_matcher.allocate_cost_volume(
            left, (left["disparity"].sel(band_disp="min"), left["disparity"].sel(band_disp="max"))
        )

        # Compute validity mask
        grid = validity_mask(left, right, grid)

        cv = matching_cost_matcher.compute_cost_volume(img_left=left, img_right=right, cost_volume=grid)
        matching_cost_matcher.cv_masked(
            left,
            right,
            cv,
            left["disparity"].sel(band_disp="min"),
            left["disparity"].sel(band_disp="max"),
        )

        # Check if the calculated mean is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(cv["cost_volume"].data, ground_truth)

    @staticmethod
    def test_masks_invalid_pixels():
        """
        Test the method masks_invalid_pixels

        """
        # ------------ Test the method with a left mask ( right mask contains valid pixels ) ------------
        # Mask convention
        # cfg['image']['valid_pixels'] = 0
        # cfg['image']['no_data'] = 1
        # invalid_pixels all other values
        data = np.array(([1, 1, 1, 3, 4], [1, 2, 1, 0, 2], [2, 1, 0, 1, 2], [1, 1, 1, 1, 4]), dtype=np.float64)
        mask = np.array(([0, 0, 2, 0, 1], [0, 2, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 2]), dtype=np.int16)
        left = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )
        left.attrs = common.img_attrs
        left.pipe(add_disparity, disparity=[-1, 1], window=None)

        data = np.array(([5, 1, 2, 3, 4], [1, 2, 1, 0, 2], [2, 2, 0, 1, 4], [1, 1, 1, 1, 2]), dtype=np.float64)
        # right mask contains valid pixels
        mask = np.zeros((4, 5), dtype=np.int16)
        right = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )
        right.attrs = common.img_attrs

        matching_cost_ = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 3, "subpix": 1}
        )

        # Compute the cost volume and invalidate pixels if need
        grid = matching_cost_.allocate_cost_volume(
            left, (left["disparity"].sel(band_disp="min"), left["disparity"].sel(band_disp="max"))
        )

        # Compute validity mask
        grid = validity_mask(left, right, grid)

        cv = matching_cost_.compute_cost_volume(img_left=left, img_right=right, cost_volume=grid)
        matching_cost_.cv_masked(
            img_left=left,
            img_right=right,
            cost_volume=cv,
            disp_min=left["disparity"].sel(band_disp="min"),
            disp_max=left["disparity"].sel(band_disp="max"),
        )

        # Cost volume before invalidation
        #  disp       -1    0   1
        # col 1    [[[nan, 6., 8.],
        # col 2      [12., 2., 13.],
        # col 3      [10., 3., nan]],
        # col 1    [[nan, 1., 5.],
        # col 2     [7., 1., 10.],
        # col 3     [11., 4., nan]]], dtype=float32)

        # Cost volume ground truth after invalidation
        cv_ground_truth = np.array(
            [
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                ],
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [12.0, 2.0, 13.0],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                ],
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [7.0, 1.0, 10.0],
                    [11.0, 4.0, np.nan],
                    [np.nan, np.nan, np.nan],
                ],
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                ],
            ],
            dtype=np.float32,
        )

        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(cv["cost_volume"], cv_ground_truth)

        # ------------ Test the method with a right mask ( left mask contains valid pixels ) ------------
        # Mask convention
        # cfg['image']['valid_pixels'] = 0
        # cfg['image']['no_data'] = 1
        # invalid_pixels all other values
        data = np.array(([1, 1, 1, 3, 4], [1, 2, 1, 0, 2], [2, 1, 0, 1, 2], [1, 1, 1, 1, 4]), dtype=np.float64)
        # left mask contains valid pixels
        mask = np.zeros((4, 5), dtype=np.int16)
        left = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )
        left.attrs = common.img_attrs
        left.pipe(add_disparity, disparity=[-1, 1], window=None)

        data = np.array(([5, 1, 2, 3, 4], [1, 2, 1, 0, 2], [2, 2, 0, 1, 4], [1, 1, 1, 1, 2]), dtype=np.float64)
        mask = np.array(([0, 0, 0, 0, 2], [0, 1, 0, 0, 0], [0, 2, 0, 2, 0], [1, 0, 0, 0, 0]), dtype=np.int16)
        right = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )
        right.attrs = common.img_attrs

        matching_cost_ = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 3, "subpix": 1}
        )

        # Compute the cost volume and invalidate pixels if need
        grid = matching_cost_.allocate_cost_volume(
            left, (left["disparity"].sel(band_disp="min"), left["disparity"].sel(band_disp="max"))
        )

        # Compute validity mask
        grid = validity_mask(left, right, grid)

        cv = matching_cost_.compute_cost_volume(img_left=left, img_right=right, cost_volume=grid)
        matching_cost_.cv_masked(
            img_left=left,
            img_right=right,
            cost_volume=cv,
            disp_min=left["disparity"].sel(band_disp="min"),
            disp_max=left["disparity"].sel(band_disp="max"),
        )

        # Cost volume before invalidation
        #  disp       -1    0   1
        # col 1    [[[nan, 6., 8.],
        # col 2      [12., 2., 13.],
        # col 3      [10., 3., nan]],
        # col 1    [[nan, 1., 5.],
        # col 2     [7., 1., 10.],
        # col 3     [11., 4., nan]]], dtype=float32)

        # Cost volume ground truth after invalidation
        cv_ground_truth = np.array(
            [
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                ],
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, 13.0],
                    [np.nan, 3.0, np.nan],
                    [np.nan, np.nan, np.nan],
                ],
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                ],
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                ],
            ],
            dtype=np.float32,
        )

        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(cv["cost_volume"], cv_ground_truth)

        # ------------ Test the method with a left and right mask ------------
        # Mask convention
        # cfg['image']['valid_pixels'] = 0
        # cfg['image']['no_data'] = 1
        # invalid_pixels all other values
        data = np.array(([1, 1, 1, 3, 4], [1, 2, 1, 0, 2], [2, 1, 0, 1, 2], [1, 1, 1, 1, 4]), dtype=np.float64)
        # left mask contains valid pixels
        mask = np.array(([1, 0, 0, 2, 0], [0, 0, 0, 0, 0], [0, 0, 2, 0, 0], [2, 0, 0, 0, 1]), dtype=np.int16)
        left = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )
        left.attrs = common.img_attrs
        left.pipe(add_disparity, disparity=[-1, 1], window=None)

        data = np.array(([5, 1, 2, 3, 4], [1, 2, 1, 0, 2], [2, 2, 0, 1, 4], [1, 1, 1, 1, 2]), dtype=np.float64)
        mask = np.array(([0, 2, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 2, 0], [1, 0, 2, 0, 0]), dtype=np.int16)
        right = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )
        right.attrs = common.img_attrs

        matching_cost_ = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 3, "subpix": 1}
        )

        # Compute the cost volume and invalidate pixels if need
        grid = matching_cost_.allocate_cost_volume(
            left, (left["disparity"].sel(band_disp="min"), left["disparity"].sel(band_disp="max"))
        )

        # Compute validity mask
        grid = validity_mask(left, right, grid)

        cv = matching_cost_.compute_cost_volume(img_left=left, img_right=right, cost_volume=grid)
        matching_cost_.cv_masked(
            img_left=left,
            img_right=right,
            cost_volume=cv,
            disp_min=left["disparity"].sel(band_disp="min"),
            disp_max=left["disparity"].sel(band_disp="max"),
        )

        # Cost volume before invalidation
        #  disp       -1    0   1
        # col 1    [[[nan, 6., 8.],
        # col 2      [12., 2., 13.],
        # col 3      [10., 3., nan]],
        # col 1    [[nan, 1., 5.],
        # col 2     [7., 1., 10.],
        # col 3     [11., 4., nan]]], dtype=float32)

        # Cost volume ground truth after invalidation
        cv_ground_truth = np.array(
            [
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                ],
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [12, 2, np.nan],
                    [10, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                ],
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, 5],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                ],
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                ],
            ],
            dtype=np.float32,
        )

        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(cv["cost_volume"], cv_ground_truth)

        # ------------ Test the method with a left and right mask and window size 5 ------------
        # Mask convention
        # cfg['image']['valid_pixels'] = 0
        # cfg['image']['no_data'] = 1
        # invalid_pixels all other values
        data = np.array(
            (
                [0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 3, 4, 0],
                [0, 1, 2, 1, 0, 2, 0],
                [0, 2, 1, 0, 1, 2, 0],
                [0, 1, 1, 1, 1, 4, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ),
            dtype=np.float64,
        )
        mask = np.array(
            (
                [2, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 2, 0, 0, 0, 0, 0],
                [0, 0, 0, 2, 0, 0, 0],
                [0, 0, 0, 0, 0, 2, 0],
                [1, 0, 0, 0, 0, 0, 2],
            ),
            dtype=np.int16,
        )

        left = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )
        left.attrs = common.img_attrs
        left.pipe(add_disparity, disparity=[-1, 1], window=None)

        data = np.array(
            (
                [0, 0, 0, 0, 0, 0, 0],
                [0, 5, 1, 2, 3, 4, 0],
                [0, 1, 2, 1, 0, 2, 0],
                [0, 2, 2, 0, 1, 4, 0],
                [0, 1, 1, 1, 1, 2, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ),
            dtype=np.float64,
        )
        mask = np.array(
            (
                [1, 0, 0, 0, 0, 0, 2],
                [0, 0, 0, 0, 0, 0, 0],
                [2, 0, 2, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 2],
                [0, 0, 0, 0, 0, 0, 0],
                [2, 0, 0, 0, 0, 0, 1],
            ),
            dtype=np.int16,
        )
        right = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )
        right.attrs = common.img_attrs

        matching_cost_ = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 5, "subpix": 1}
        )

        # Compute the cost volume and invalidate pixels if need
        grid = matching_cost_.allocate_cost_volume(
            left, (left["disparity"].sel(band_disp="min"), left["disparity"].sel(band_disp="max"))
        )

        # Compute validity mask
        grid = validity_mask(left, right, grid)

        cv = matching_cost_.compute_cost_volume(img_left=left, img_right=right, cost_volume=grid)
        matching_cost_.cv_masked(
            img_left=left,
            img_right=right,
            cost_volume=cv,
            disp_min=left["disparity"].sel(band_disp="min"),
            disp_max=left["disparity"].sel(band_disp="max"),
        )

        # Cost volume ground truth after invalidation
        cv_ground_truth = np.array(
            [
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                ],
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                ],
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, 24.0],
                    [np.nan, 10.0, 27.0],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                ],
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [31.0, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                ],
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                ],
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                ],
            ],
            dtype=np.float32,
        )

        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(cv["cost_volume"], cv_ground_truth)

        # ------------ Test the method with a left and right mask with window size 1------------
        # Mask convention
        # cfg['image']['valid_pixels'] = 0
        # cfg['image']['no_data'] = 1
        # invalid_pixels all other values
        data = np.array(([1, 1, 1, 3, 4], [1, 1, 1, 1, 4]), dtype=np.float64)
        # left mask contains valid pixels
        mask = np.array(([1, 0, 0, 2, 0], [2, 0, 0, 0, 1]), dtype=np.int16)
        left = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )
        left.attrs = common.img_attrs
        left.pipe(add_disparity, disparity=[-1, 1], window=None)

        data = np.array(([5, 1, 2, 3, 4], [1, 1, 1, 1, 2]), dtype=np.float64)
        mask = np.array(([0, 2, 0, 0, 1], [1, 0, 2, 0, 0]), dtype=np.int16)
        right = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )
        right.attrs = common.img_attrs

        matching_cost_ = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 1, "subpix": 1}
        )

        # Compute the cost volume and invalidate pixels if need
        grid = matching_cost_.allocate_cost_volume(
            left, (left["disparity"].sel(band_disp="min"), left["disparity"].sel(band_disp="max"))
        )

        # Compute validity mask
        grid = validity_mask(left, right, grid)

        cv = matching_cost_.compute_cost_volume(img_left=left, img_right=right, cost_volume=grid)
        matching_cost_.cv_masked(
            img_left=left,
            img_right=right,
            cost_volume=cv,
            disp_min=left["disparity"].sel(band_disp="min"),
            disp_max=left["disparity"].sel(band_disp="max"),
        )

        # Cost volume ground truth after invalidation
        cv_ground_truth = np.array(
            [
                [
                    [np.nan, np.nan, np.nan],
                    [4, np.nan, 1],
                    [np.nan, 1, 2],
                    [np.nan, np.nan, np.nan],
                    [1, np.nan, np.nan],
                ],
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, 0, np.nan],
                    [0, np.nan, 0],
                    [np.nan, 0, 1],
                    [np.nan, np.nan, np.nan],
                ],
            ],
            dtype=np.float32,
        )

        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(cv["cost_volume"], cv_ground_truth)

    @staticmethod
    def test_masks_invalid_pixels_subpixel():
        """
        Test the method masks_invalid_pixels with subpixel precision

        """
        # ------------ Test the method with a right mask with window size 1 subpixel 2 ------------
        # Mask convention
        # cfg['image']['valid_pixels'] = 0
        # cfg['image']['no_data'] = 1
        # invalid_pixels all other values
        data = np.array(([1, 1, 1, 3, 4], [1, 1, 1, 1, 4]), dtype=np.float64)
        # left mask contains valid pixels
        mask = np.array(([0, 0, 0, 0, 0], [0, 0, 0, 0, 0]), dtype=np.int16)
        left = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )
        left.attrs = common.img_attrs
        left.pipe(add_disparity, disparity=[-1, 1], window=None)

        data = np.array(([5, 1, 2, 3, 4], [1, 1, 1, 1, 2]), dtype=np.float64)
        mask = np.array(([0, 0, 0, 0, 1], [1, 0, 2, 0, 0]), dtype=np.int16)
        right = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )
        right.attrs = common.img_attrs

        matching_cost_ = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 1, "subpix": 2}
        )

        # Compute the cost volume and invalidate pixels if need
        grid = matching_cost_.allocate_cost_volume(
            left, (left["disparity"].sel(band_disp="min"), left["disparity"].sel(band_disp="max"))
        )

        # Compute validity mask
        grid = validity_mask(left, right, grid)

        cv = matching_cost_.compute_cost_volume(img_left=left, img_right=right, cost_volume=grid)
        matching_cost_.cv_masked(
            img_left=left,
            img_right=right,
            cost_volume=cv,
            disp_min=left["disparity"].sel(band_disp="min"),
            disp_max=left["disparity"].sel(band_disp="max"),
        )

        # The cost volume before invalidation
        # <xarray.DataArray 'cost_volume' (row: 2, col: 5, disp: 5)>
        # array([[[nan, nan, 4. , 2. , 0. ],
        #         [4. , 2. , 0. , 0.5, 1. ],
        #         [0. , 0.5, 1. , 1.5, 2. ],
        #         [1. , 0.5, 0. , 0.5, 1. ],
        #         [1. , 0.5, 0. , nan, nan]],
        #
        #        [[nan, nan, 0. , 0. , 0. ],
        #         [0. , 0. , 0. , 0. , 0. ],
        #         [0. , 0. , 0. , 0. , 0. ],
        #         [0. , 0. , 0. , 0.5, 1. ],
        #         [3. , 2.5, 2. , nan, nan]]], dtype=float32)
        # Coordinates:
        #   * row      (row) int64 0 1
        #   * col      (col) int64 0 1 2 3 4
        #   * disp     (disp) float64 -1.0 -0.5 0.0 0.5 1.0

        cv_ground_truth = np.array(
            [
                [
                    [np.nan, np.nan, 4, 2, 0],
                    [4, 2, 0, 0.5, 1],
                    [0, 0.5, 1, 1.5, 2],
                    [1, 0.5, 0, np.nan, np.nan],
                    [1, np.nan, np.nan, np.nan, np.nan],
                ],
                [
                    [np.nan, np.nan, np.nan, np.nan, 0],
                    [np.nan, np.nan, 0, np.nan, np.nan],
                    [0, np.nan, np.nan, np.nan, 0],
                    [np.nan, np.nan, 0, 0.5, 1],
                    [3, 2.5, 2, np.nan, np.nan],
                ],
            ],
            dtype=np.float32,
        )

        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(cv["cost_volume"], cv_ground_truth)

        # ------------ Test the method with a right mask with window size 1 subpixel 4 ------------
        # Mask convention
        # cfg['image']['valid_pixels'] = 5
        # cfg['image']['no_data'] = 7
        # invalid_pixels all other values
        data = np.array(([1, 1, 1], [1, 1, 1]), dtype=np.float64)
        # left mask contains valid pixels
        mask = np.array(([5, 5, 5], [5, 5, 5]), dtype=np.int16)
        left = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )
        left.attrs = {
            "valid_pixels": 5,
            "no_data_mask": 7,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }
        left.pipe(add_disparity, disparity=[-1, 1], window=None)

        data = np.array(([5, 1, 2], [1, 1, 1]), dtype=np.float64)
        mask = np.array(([5, 4, 7], [6, 7, 5]), dtype=np.int16)
        right = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )
        right.attrs = {
            "valid_pixels": 5,
            "no_data_mask": 7,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

        matching_cost_ = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 1, "subpix": 4}
        )

        # Compute the cost volume and invalidate pixels if need
        grid = matching_cost_.allocate_cost_volume(
            left, (left["disparity"].sel(band_disp="min"), left["disparity"].sel(band_disp="max"))
        )

        # Compute validity mask
        grid = validity_mask(left, right, grid)

        cv = matching_cost_.compute_cost_volume(img_left=left, img_right=right, cost_volume=grid)
        matching_cost_.cv_masked(
            img_left=left,
            img_right=right,
            cost_volume=cv,
            disp_min=left["disparity"].sel(band_disp="min"),
            disp_max=left["disparity"].sel(band_disp="max"),
        )

        # The cost volume before invalidation
        # <xarray.DataArray 'cost_volume' (row: 2, col: 5, disp: 5)>
        # array([[[ nan,  nan,  nan,  nan, 4.  , 3.  , 2.  , 1.  , 0.  ],
        #         [4.  , 3.  , 2.  , 1.  , 0.  , 0.25, 0.5 , 0.75, 1.  ],
        #         [0.  , 0.25, 0.5 , 0.75, 1.  ,  nan,  nan,  nan,  nan]],
        #
        #        [[ nan,  nan,  nan,  nan, 0.  , 0.  , 0.  , 0.  , 0.  ],
        #         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
        #         [0.  , 0.  , 0.  , 0.  , 0.  ,  nan,  nan,  nan,  nan]]],
        #       dtype=float32)
        # Coordinates:
        #   * row      (row) int64 0 1
        #   * col      (col) int64 0 1 2
        #   * disp     (disp) float64 -1.0 -0.75 -0.5 -0.25 0.0 0.25 0.5 0.75 1.0

        cv_ground_truth = np.array(
            [
                [
                    [np.nan, np.nan, np.nan, np.nan, 4.0, np.nan, np.nan, np.nan, np.nan],
                    [4.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                ],
                [
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.0],
                    [np.nan, np.nan, np.nan, np.nan, 0.0, np.nan, np.nan, np.nan, np.nan],
                ],
            ],
            dtype=np.float32,
        )

        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(cv["cost_volume"], cv_ground_truth)

        # ------------ Test the method with a left and right mask, window size 3, subpixel 2 ------------
        # Mask convention
        # cfg['image']['valid_pixels'] = 5
        # cfg['image']['no_data'] = 7
        # invalid_pixels all other values
        data = np.array(([1, 1, 1, 3, 4], [1, 2, 1, 0, 2], [2, 1, 0, 1, 2], [1, 1, 1, 1, 4]), dtype=np.float64)
        mask = np.array(([5, 56, 5, 12, 5], [5, 5, 5, 5, 5], [5, 5, 5, 5, 5], [3, 5, 4, 5, 7]), dtype=np.int16)
        left = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )
        left.attrs = {
            "valid_pixels": 5,
            "no_data_mask": 7,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }
        left.pipe(add_disparity, disparity=[-1, 1], window=None)

        data = np.array(([5, 1, 2, 3, 4], [1, 2, 1, 0, 2], [2, 2, 0, 1, 4], [1, 1, 1, 1, 2]), dtype=np.float64)
        mask = np.array(([7, 5, 5, 5, 5], [5, 5, 5, 65, 5], [5, 5, 5, 5, 5], [5, 23, 5, 5, 2]), dtype=np.int16)
        right = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )
        right.attrs = {
            "valid_pixels": 5,
            "no_data_mask": 7,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

        matching_cost_ = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 3, "subpix": 2}
        )

        # Compute the cost volume and invalidate pixels if need
        grid = matching_cost_.allocate_cost_volume(
            left, (left["disparity"].sel(band_disp="min"), left["disparity"].sel(band_disp="max"))
        )

        # Compute validity mask
        grid = validity_mask(left, right, grid)

        cv = matching_cost_.compute_cost_volume(img_left=left, img_right=right, cost_volume=grid)
        matching_cost_.cv_masked(
            img_left=left,
            img_right=right,
            cost_volume=cv,
            disp_min=left["disparity"].sel(band_disp="min"),
            disp_max=left["disparity"].sel(band_disp="max"),
        )

        # Cost volume before invalidation
        # array([[[ nan,  nan,  6. ,  6. ,  8. ],
        #         [12. ,  7. ,  2. ,  6.5, 13. ],
        #         [10. ,  5.5,  3. ,  nan,  nan]],
        #
        #        [[ nan,  nan,  1. ,  2. ,  5. ],
        #         [ 7. ,  4. ,  1. ,  4.5, 10. ],
        #         [11. ,  6.5,  4. ,  nan,  nan]]], dtype=float32)
        # Coordinates:
        #   * row      (row) int64 1 2
        #   * col      (col) int64 1 2 3
        #   * disp     (disp) float64 -1.0 -0.5 0.0 0.5 1.0

        # Cost volume ground truth after invalidation
        cv_ground_truth = np.array(
            [
                [
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                ],
                [
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, 8.0],
                    [np.nan, np.nan, 2.0, np.nan, np.nan],
                    [10.0, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                ],
                [
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, 1.0, 2.0, 5.0],
                    [7.0, 4.0, 1.0, 4.5, 10.0],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                ],
                [
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                ],
            ],
            dtype=np.float32,
        )

        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(cv["cost_volume"], cv_ground_truth)

    @staticmethod
    def test_check_band_sad():
        """
        Test the multiband choice for census measure with wrong matching_cost initialization
        """

        # Initialize multiband data
        data = np.zeros((2, 4, 4))
        data[0, :, :] = np.array(
            (
                [1, 1, 1, 3],
                [1, 3, 2, 5],
                [2, 1, 0, 1],
                [1, 5, 4, 3],
            ),
            dtype=np.float64,
        )
        data[1, :, :] = np.array(
            (
                [2, 3, 4, 6],
                [8, 7, 0, 4],
                [4, 9, 1, 5],
                [6, 5, 2, 1],
            ),
            dtype=np.float64,
        )
        left = xr.Dataset(
            {"im": (["band_im", "row", "col"], data)},
            coords={
                "band_im": np.arange(data.shape[0]),
                "row": np.arange(data.shape[1]),
                "col": np.arange(data.shape[2]),
            },
        )
        left.attrs = common.img_attrs
        left.pipe(add_disparity, disparity=[-1, 1], window=None)

        # Initialize multiband data
        data = np.zeros((2, 4, 4))
        data[0, :, :] = np.array(
            (
                [5, 1, 2, 3],
                [1, 3, 0, 2],
                [2, 3, 5, 0],
                [1, 6, 7, 5],
            ),
            dtype=np.float64,
        )
        data[1, :, :] = np.array(
            (
                [6, 5, 2, 7],
                [8, 7, 6, 5],
                [5, 2, 3, 6],
                [0, 3, 4, 7],
            ),
            dtype=np.float64,
        )
        right = xr.Dataset(
            {"im": (["band_im", "row", "col"], data)},
            coords={
                "band_im": np.arange(data.shape[0]),
                "row": np.arange(data.shape[1]),
                "col": np.arange(data.shape[2]),
            },
        )
        right.attrs = common.img_attrs

        # Initialization of matching_cost plugin with wrong band
        matching_cost_ = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 3, "subpix": 1, "band": "b"}
        )

        grid = matching_cost_.allocate_cost_volume(
            left, (left["disparity"].sel(band_disp="min"), left["disparity"].sel(band_disp="max"))
        )

        # Compute the cost_volume
        with pytest.raises(AttributeError, match="Wrong band instantiate : b not in img_left or img_right"):
            _ = matching_cost_.compute_cost_volume(img_left=left, img_right=right, cost_volume=grid)

        # Initialization of matching_cost plugin with no band
        matching_cost_ = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 3, "subpix": 1}
        )

        grid = matching_cost_.allocate_cost_volume(
            left, (left["disparity"].sel(band_disp="min"), left["disparity"].sel(band_disp="max"))
        )

        # Compute the cost_volume
        with pytest.raises(AttributeError, match="Band must be instantiated in matching cost step"):
            _ = matching_cost_.compute_cost_volume(img_left=left, img_right=right, cost_volume=grid)


if __name__ == "__main__":
    common.setup_logging()
    unittest.main()
