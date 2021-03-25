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
This module contains functions to test the cost volume measure step.
"""
# pylint: disable=too-many-lines

import unittest

import numpy as np
import xarray as xr
from rasterio import Affine

import tests.common as common
import pandora.matching_cost as matching_cost


class TestMatchingCost(unittest.TestCase):
    """
    TestMatchingCost class allows to test all the methods in the class MatchingCost,
    and the plugins pixel_wise, zncc
    """

    def setUp(self):
        """
        Method called to prepare the test fixture

        """
        # Create a stereo object
        data = np.array(
            ([1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 2, 1], [1, 1, 1, 4, 3, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]),
            dtype=np.float64,
        )
        self.left = xr.Dataset(
            {"im": (["row", "col"], data)}, coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])}
        )
        self.left.attrs = {
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

        data = np.array(
            ([1, 1, 1, 2, 2, 2], [1, 1, 1, 4, 2, 4], [1, 1, 1, 4, 4, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]),
            dtype=np.float64,
        )
        self.right = xr.Dataset(
            {"im": (["row", "col"], data)}, coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])}
        )
        self.right.attrs = {
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

    def test_ssd_cost(self):
        """
        Test the sum of squared difference method

        """
        # Squared difference pixel-wise ground truth for the images self.left, self.right, with window_size = 1
        sd_ground_truth = np.array(
            (
                [0, 0, 0, 1, 1, 1],
                [0, 0, 0, (1 - 4) ** 2, 0, (1 - 4) ** 2],
                [0, 0, 0, 0, (3 - 4) ** 2, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            )
        )

        # Computes the sd cost for the whole images
        matching_cost_matcher = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "ssd", "window_size": 1, "subpix": 1}
        )
        ssd = matching_cost_matcher.compute_cost_volume(
            img_left=self.left, img_right=self.right, disp_min=-1, disp_max=1
        )

        # Check if the calculated sd cost is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(ssd["cost_volume"].sel(disp=0), sd_ground_truth)

        # Sum of squared difference pixel-wise ground truth for the images self.left, self.right, with window_size = 5
        ssd_ground_truth = np.array(
            (
                [
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, 12.0, 22.0, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                ]
            )
        )

        # Computes the sd cost for the whole images
        matching_cost_matcher = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "ssd", "window_size": 5, "subpix": 1}
        )
        ssd = matching_cost_matcher.compute_cost_volume(
            img_left=self.left, img_right=self.right, disp_min=-1, disp_max=1
        )
        matching_cost_matcher.cv_masked(self.left, self.right, ssd, -1, 1)

        # Check if the calculated sd cost is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(ssd["cost_volume"].sel(disp=0), ssd_ground_truth)

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
        sad = matching_cost_matcher.compute_cost_volume(
            img_left=self.left, img_right=self.right, disp_min=-1, disp_max=1
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
            **{"matching_cost_method": "sad", "window_size": 5, "subpix": 1}
        )
        sad = matching_cost_matcher.compute_cost_volume(
            img_left=self.left, img_right=self.right, disp_min=-1, disp_max=1
        )
        matching_cost_matcher.cv_masked(self.left, self.right, sad, -1, 1)

        # Check if the calculated ad cost is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(sad["cost_volume"].sel(disp=0), sad_ground_truth)

    @staticmethod
    def test_census_cost():
        """
        Test the census method

        """
        data = np.array(([1, 1, 1, 3], [1, 2, 1, 0], [2, 1, 0, 1], [1, 1, 1, 1]), dtype=np.float64)
        left = xr.Dataset(
            {"im": (["row", "col"], data)}, coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])}
        )
        left.attrs["crs"] = None
        left.attrs["transform"] = Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        data = np.array(([5, 1, 2, 3], [1, 2, 1, 0], [2, 2, 0, 1], [1, 1, 1, 1]), dtype=np.float64)
        right = xr.Dataset(
            {"im": (["row", "col"], data)}, coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])}
        )
        right.attrs["crs"] = None
        right.attrs["transform"] = Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        # census ground truth for the images left, right, window size = 3 and disp = -1
        census_ground_truth_d1 = np.array(
            (
                [np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, 3, np.nan],
                [np.nan, np.nan, 7, np.nan],
                [np.nan, np.nan, np.nan, np.nan],
            )
        )

        # census ground truth for the images left, right, window size = 3 and disp = 0
        census_ground_truth_d2 = np.array(
            (
                [np.nan, np.nan, np.nan, np.nan],
                [np.nan, 1, 2, np.nan],
                [np.nan, 2, 0, np.nan],
                [np.nan, np.nan, np.nan, np.nan],
            )
        )

        # census ground truth for the images left, right, window size = 3 and disp = 1
        census_ground_truth_d3 = np.array(
            (
                [np.nan, np.nan, np.nan, np.nan],
                [np.nan, 4, np.nan, np.nan],
                [np.nan, 5, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan],
            )
        )

        # Computes the census transform for the images with window size = 3
        matching_cost_matcher = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "census", "window_size": 3, "subpix": 1}
        )
        census = matching_cost_matcher.compute_cost_volume(img_left=left, img_right=right, disp_min=-1, disp_max=1)
        matching_cost_matcher.cv_masked(left, right, census, -1, 1)

        # Check if the calculated census cost is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(census["cost_volume"].sel(disp=-1), census_ground_truth_d1)
        np.testing.assert_array_equal(census["cost_volume"].sel(disp=0), census_ground_truth_d2)
        np.testing.assert_array_equal(census["cost_volume"].sel(disp=1), census_ground_truth_d3)

    def test_point_interval(self):
        """
        Test the point interval method

        """
        matching_cost_matcher = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "census", "window_size": 3, "subpix": 1}
        )

        # Using the two images in self.left, self.right,
        # for disparity = 0, the similarity measure will be applied over the whole images
        p_ground_truth_disp = (0, self.left["im"].shape[1])
        q_ground_truth_disp = (0, self.right["im"].shape[1])
        calculated_range = matching_cost_matcher.point_interval(self.left, self.right, 0)

        # Check if the calculated range is equal to the ground truth
        np.testing.assert_array_equal(calculated_range[0], p_ground_truth_disp)
        np.testing.assert_array_equal(calculated_range[1], q_ground_truth_disp)

        # for disparity = -2, the similarity measure will be applied over the range
        #          row=2   row=6        row=0   row=4
        #           1 1 1 1             1 1 1 2
        #           1 1 2 1             1 1 1 4
        #           1 4 3 1             1 1 1 4
        #           1 1 1 1             1 1 1 1
        #           1 1 1 1             1 1 1 1
        p_ground_truth_disp = (2, 6)
        q_ground_truth_disp = (0, 4)
        calculated_range = matching_cost_matcher.point_interval(self.left, self.right, -2)
        # Check if the calculated range is equal to the ground truth
        np.testing.assert_array_equal(calculated_range[0], p_ground_truth_disp)
        np.testing.assert_array_equal(calculated_range[1], q_ground_truth_disp)

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
        cv = matching_cost_matcher.compute_cost_volume(left, right, disp_min=-2, disp_max=1)
        matching_cost_matcher.cv_masked(left, right, cv, -2, 1)

        # Check if the calculated mean is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(cv["cost_volume"].data, ground_truth)

    def test_popcount32b(self):
        """
        Test the popcount32b method

        """
        matching_cost_matcher = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "census", "window_size": 3, "subpix": 1}
        )

        # Count the number of symbols that are different from the zero
        count_ = matching_cost_matcher.popcount32b(0b0001000101000)
        # Check if the calculated count_ is equal to the ground truth 3.
        self.assertEqual(count_, 3)

        # Count the number of symbols that are different from the zero
        count_ = matching_cost_matcher.popcount32b(0b0000000000000000000)
        # Check if the calculated count_ is equal to the ground truth 0.
        self.assertEqual(count_, 0)

    def test_zncc_cost(self):
        """
        Test the zncc_cost method

        """
        # Compute the cost volume for the images self.left, self.right,
        # with zncc measure, disp = -1, 1 window size = 5 and subpix = 1
        matching_cost_matcher = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "zncc", "window_size": 5, "subpix": 1}
        )
        cost_volume_zncc = matching_cost_matcher.compute_cost_volume(self.left, self.right, disp_min=-1, disp_max=1)
        matching_cost_matcher.cv_masked(self.left, self.right, cost_volume_zncc, -1, 1)

        # Ground truth zncc cost for the disparity -1
        row = self.left["im"].data[:, 1:]
        col = self.right["im"].data[:, :5]
        ground_truth = np.array(
            (
                [
                    np.nan,
                    np.nan,
                    np.nan,
                    (np.mean(row * col) - (np.mean(row) * np.mean(col))) / (np.std(row) * np.std(col)),
                    np.nan,
                    np.nan,
                ]
            )
        )

        # Check if the calculated cost volume for the disparity -1 is equal to the ground truth
        np.testing.assert_allclose(cost_volume_zncc["cost_volume"].data[2, :, 0], ground_truth, rtol=1e-05)

        # Ground truth zncc cost for the disparity 1
        row = self.left["im"].data[:, :5]
        col = self.right["im"].data[:, 1:]
        ground_truth = np.array(
            (
                [
                    np.nan,
                    np.nan,
                    (np.mean(row * col) - (np.mean(row) * np.mean(col))) / (np.std(row) * np.std(col)),
                    np.nan,
                    np.nan,
                    np.nan,
                ]
            )
        )
        # Check if the calculated cost volume
        # Check if the calculated cost volume for the disparity 1 is equal to the ground truth
        np.testing.assert_allclose(cost_volume_zncc["cost_volume"].data[2, :, 2], ground_truth, rtol=1e-05)

    @staticmethod
    def test_subpixel_offset():
        """
        Test the cost volume method with 2 subpixel disparity

        """
        # Create a matching_cost object with simple images
        data = np.array(([7, 8, 1, 0, 2], [4, 5, 2, 1, 0], [8, 9, 10, 0, 0]), dtype=np.float64)
        left = xr.Dataset(
            {"im": (["row", "col"], data)}, coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])}
        )
        left.attrs["crs"] = None
        left.attrs["transform"] = Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        data = np.array(([1, 5, 6, 3, 4], [2, 5, 10, 6, 9], [0, 7, 5, 3, 1]), dtype=np.float64)
        right = xr.Dataset(
            {"im": (["row", "col"], data)}, coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])}
        )
        right.attrs["crs"] = None
        right.attrs["transform"] = Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        # Computes the cost volume for disp min -2 disp max 2 and subpix = 2
        matching_cost_matcher = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 3, "subpix": 2}
        )
        cv_zncc_subpixel = matching_cost_matcher.compute_cost_volume(left, right, disp_min=-2, disp_max=2)
        matching_cost_matcher.cv_masked(left, right, cv_zncc_subpixel, -2, 1)
        # Test the disparity range
        disparity_range_compute = cv_zncc_subpixel.coords["disp"].data
        disparity_range_ground_truth = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
        # Check if the calculated disparity range is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(disparity_range_compute, disparity_range_ground_truth)
        # Cost volume ground truth with subpixel precision 0.5

        cost_volume_ground_truth = np.array(
            [
                [
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                ],
                [
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, 39, 32.5, 28, 34.5, 41],
                    [np.nan, np.nan, 49, 41.5, 34, 35.5, 37, np.nan, np.nan],
                    [45, 42.5, 40, 40.5, 41, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                ],
                [
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                ],
            ]
        )

        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(cv_zncc_subpixel["cost_volume"].data, cost_volume_ground_truth)

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

        left.attrs = {
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

        data = np.array(([5, 1, 2, 3, 4], [1, 2, 1, 0, 2], [2, 2, 0, 1, 4], [1, 1, 1, 1, 2]), dtype=np.float64)
        # right mask contains valid pixels
        mask = np.zeros((4, 5), dtype=np.int16)
        right = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )

        right.attrs = {
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

        dmin = -1
        dmax = 1

        matching_cost_ = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 3, "subpix": 1}
        )
        # left_dil, right_dil = matching_cost_.masks_dilatation(left, right, 1, 3, {'valid_pixels': 0, 'no_data': 1})
        # print ('left_dil ', left_dil)
        # exit()
        # Compute the cost volume and invalidate pixels if need

        cv = matching_cost_.compute_cost_volume(img_left=left, img_right=right, disp_min=dmin, disp_max=dmax)
        matching_cost_.cv_masked(img_left=left, img_right=right, cost_volume=cv, disp_min=dmin, disp_max=dmax)
        # Cost volume before invalidation
        #  disp       -1    0   1
        # Row 1
        # col 1    [[[nan, 6., 8.],
        # col 2      [12., 2., 13.],
        # col 3      [10., 3., nan]],
        #
        #  Row 2
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

        left.attrs = {
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

        data = np.array(([5, 1, 2, 3, 4], [1, 2, 1, 0, 2], [2, 2, 0, 1, 4], [1, 1, 1, 1, 2]), dtype=np.float64)
        mask = np.array(([0, 0, 0, 0, 2], [0, 1, 0, 0, 0], [0, 2, 0, 2, 0], [1, 0, 0, 0, 0]), dtype=np.int16)
        right = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )

        right.attrs = {
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

        dmin = -1
        dmax = 1

        matching_cost_ = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 3, "subpix": 1}
        )
        # Compute the cost volume and invalidate pixels if need
        cv = matching_cost_.compute_cost_volume(img_left=left, img_right=right, disp_min=dmin, disp_max=dmax)
        matching_cost_.cv_masked(img_left=left, img_right=right, cost_volume=cv, disp_min=dmin, disp_max=dmax)
        # Cost volume before invalidation
        #  disp       -1    0   1
        # Row 1
        # col 1    [[[nan, 6., 8.],
        # col 2      [12., 2., 13.],
        # col 3      [10., 3., nan]],
        #
        #  Row 2
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

        left.attrs = {
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

        data = np.array(([5, 1, 2, 3, 4], [1, 2, 1, 0, 2], [2, 2, 0, 1, 4], [1, 1, 1, 1, 2]), dtype=np.float64)
        mask = np.array(([0, 2, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 2, 0], [1, 0, 2, 0, 0]), dtype=np.int16)
        right = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )

        right.attrs = {
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

        dmin = -1
        dmax = 1

        matching_cost_ = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 3, "subpix": 1}
        )
        # Compute the cost volume and invalidate pixels if need
        cv = matching_cost_.compute_cost_volume(img_left=left, img_right=right, disp_min=dmin, disp_max=dmax)
        matching_cost_.cv_masked(img_left=left, img_right=right, cost_volume=cv, disp_min=dmin, disp_max=dmax)
        # Cost volume before invalidation
        #  disp       -1    0   1
        # Row 1
        # col 1    [[[nan, 6., 8.],
        # col 2      [12., 2., 13.],
        # col 3      [10., 3., nan]],
        #
        #  Row 2
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
        left.attrs = {
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }
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
        right.attrs = {
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

        dmin = -1
        dmax = 1

        matching_cost_ = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 5, "subpix": 1}
        )
        # Compute the cost volume and invalidate pixels if need
        cv = matching_cost_.compute_cost_volume(img_left=left, img_right=right, disp_min=dmin, disp_max=dmax)
        matching_cost_.cv_masked(img_left=left, img_right=right, cost_volume=cv, disp_min=dmin, disp_max=dmax)

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
        left.attrs = {
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

        data = np.array(([5, 1, 2, 3, 4], [1, 1, 1, 1, 2]), dtype=np.float64)
        mask = np.array(([0, 2, 0, 0, 1], [1, 0, 2, 0, 0]), dtype=np.int16)
        right = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )
        right.attrs = {
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

        dmin = -1
        dmax = 1

        matching_cost_ = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 1, "subpix": 1}
        )
        # Compute the cost volume and invalidate pixels if need
        cv = matching_cost_.compute_cost_volume(img_left=left, img_right=right, disp_min=dmin, disp_max=dmax)
        matching_cost_.cv_masked(img_left=left, img_right=right, cost_volume=cv, disp_min=dmin, disp_max=dmax)

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

        # ------------ Test the method with a left and right mask with window size 3 and ZNCC ------------
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

        left.attrs = {
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

        data = np.array(([5, 1, 2, 3, 4], [1, 2, 1, 0, 2], [2, 2, 0, 1, 4], [1, 1, 1, 1, 2]), dtype=np.float64)
        mask = np.array(([0, 2, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 2, 0], [1, 0, 2, 0, 0]), dtype=np.int16)
        right = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )

        right.attrs = {
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

        dmin = -1
        dmax = 1

        matching_cost_ = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "zncc", "window_size": 3, "subpix": 1}
        )
        # Compute the cost volume and invalidate pixels if need
        cv = matching_cost_.compute_cost_volume(img_left=left, img_right=right, disp_min=dmin, disp_max=dmax)
        matching_cost_.cv_masked(img_left=left, img_right=right, cost_volume=cv, disp_min=dmin, disp_max=dmax)

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
                    [0.02146693953705469, 0.8980265101338747, np.nan],
                    [0.40624999999999994, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                ],
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, 0.2941742027072762],
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

        left.attrs = {
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

        data = np.array(([5, 1, 2, 3, 4], [1, 1, 1, 1, 2]), dtype=np.float64)
        mask = np.array(([0, 0, 0, 0, 1], [1, 0, 2, 0, 0]), dtype=np.int16)
        right = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )

        right.attrs = {
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

        dmin = -1
        dmax = 1

        matching_cost_ = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 1, "subpix": 2}
        )
        # Compute the cost volume and invalidate pixels if need
        cv = matching_cost_.compute_cost_volume(img_left=left, img_right=right, disp_min=dmin, disp_max=dmax)
        matching_cost_.cv_masked(img_left=left, img_right=right, cost_volume=cv, disp_min=dmin, disp_max=dmax)

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

        dmin = -1
        dmax = 1

        matching_cost_ = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 1, "subpix": 4}
        )
        # Compute the cost volume and invalidate pixels if need
        cv = matching_cost_.compute_cost_volume(img_left=left, img_right=right, disp_min=dmin, disp_max=dmax)
        matching_cost_.cv_masked(img_left=left, img_right=right, cost_volume=cv, disp_min=dmin, disp_max=dmax)

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

        dmin = -1
        dmax = 1

        matching_cost_ = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 3, "subpix": 2}
        )
        # Compute the cost volume and invalidate pixels if need
        cv = matching_cost_.compute_cost_volume(img_left=left, img_right=right, disp_min=dmin, disp_max=dmax)
        matching_cost_.cv_masked(img_left=left, img_right=right, cost_volume=cv, disp_min=dmin, disp_max=dmax)

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

        # ------------ Test the method with a left and right mask with window size 3 and census ------------
        # Mask convention
        # cfg['image']['valid_pixels'] = 5
        # cfg['image']['no_data'] = 7
        # invalid_pixels all other values
        data = np.array(([1, 1, 1, 3], [1, 2, 1, 0], [2, 1, 0, 1], [1, 1, 1, 1]), dtype=np.float64)
        mask = np.array(([7, 5, 5, 2], [0, 5, 5, 5], [5, 5, 5, 0], [0, 5, 5, 7]), dtype=np.int16)
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

        data = np.array(([5, 1, 2, 3], [1, 2, 1, 0], [2, 2, 0, 1], [1, 1, 1, 1]), dtype=np.float64)
        mask = np.array(([2, 5, 5, 2], [0, 5, 2, 5], [5, 5, 5, 0], [7, 5, 5, 5]), dtype=np.int16)
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

        dmin = -1
        dmax = 1

        # Cost volume ground truth after invalidation
        census_ground_truth = np.array(
            [
                [
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                ],
                [
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [3.0, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                ],
                [
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, 5.0],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                ],
                [
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                ],
            ],
            dtype=np.float32,
        )

        # Computes the census transform for the images with window size = 3
        matching_cost_matcher = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "census", "window_size": 3, "subpix": 2}
        )
        census = matching_cost_matcher.compute_cost_volume(img_left=left, img_right=right, disp_min=dmin, disp_max=dmax)
        matching_cost_matcher.cv_masked(
            img_left=left, img_right=right, cost_volume=census, disp_min=dmin, disp_max=dmax
        )

        # Check if the calculated census cost is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(census["cost_volume"], census_ground_truth)

        # ------------ Test the method with a left and right mask with window size 3 and ZNCC ------------
        data = np.array(([1, 1, 1, 3, 4], [1, 2, 1, 0, 2], [2, 1, 0, 1, 2], [1, 1, 1, 1, 4]), dtype=np.float64)
        # left mask contains valid pixels
        mask = np.array(([1, 0, 0, 2, 0], [0, 0, 0, 0, 0], [0, 0, 2, 0, 0], [2, 0, 0, 0, 1]), dtype=np.int16)

        left = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )

        left.attrs = {
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

        data = np.array(([5, 1, 2, 3, 4], [1, 2, 1, 0, 2], [2, 2, 0, 1, 4], [1, 1, 1, 1, 2]), dtype=np.float64)
        mask = np.array(([0, 2, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 2, 0], [1, 0, 2, 0, 0]), dtype=np.int16)
        right = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )

        right.attrs = {
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

        dmin = -1
        dmax = 1

        matching_cost_ = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "zncc", "window_size": 3, "subpix": 2}
        )
        # Compute the cost volume and invalidate pixels if need
        cv = matching_cost_.compute_cost_volume(img_left=left, img_right=right, disp_min=dmin, disp_max=dmax)
        matching_cost_.cv_masked(img_left=left, img_right=right, cost_volume=cv, disp_min=dmin, disp_max=dmax)

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
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [0.02146693953705469, 0.5486081, 0.8980265101338747, np.nan, np.nan],
                    [0.40624999999999994, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                ],
                [
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, 0.2941742027072762],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
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
    def test_masks_dilatation():
        """
        Test the method masks_dilatation

        """
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

        # masks_dilatation(self, img_left, img_right, offset_row_col, window_size, subp, cfg)
        matching_cost_ = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 3, "subpix": 4}
        )
        # Compute the dilated / shifted masks
        mask_left, masks_right = matching_cost_.masks_dilatation(img_left=left, img_right=right, window_size=3, subp=4)

        # left mask ground truth
        gt_left = np.array(
            [
                [0, np.nan, 0, np.nan, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, np.nan, np.nan],
                [np.nan, 0, np.nan, np.nan, np.nan],
            ],
            dtype=np.float32,
        )
        gt_left = xr.DataArray(gt_left, coords=[[0, 1, 2, 3], [0, 1, 2, 3, 4]], dims=["row", "col"])

        # Check if the calculated left masks is equal to the ground truth (same dimensions, coordinates and values)
        if not mask_left.equals(gt_left):
            raise ValueError("test_masks_dilatation error : left mask ")

        # right mask ground truth with pixel precision
        gt_right_pixel = np.array(
            [[np.nan, np.nan, 0, 0, 0], [np.nan, np.nan, 0, np.nan, 0], [0, 0, 0, 0, 0], [0, np.nan, 0, 0, np.nan]],
            dtype=np.float32,
        )
        gt_right_pixel = xr.DataArray(gt_right_pixel, coords=[[0, 1, 2, 3], [0, 1, 2, 3, 4]], dims=["row", "col"])

        if not masks_right[0].equals(gt_right_pixel):
            raise ValueError("test_masks_dilatation error : right mask ")

        # right mask ground truth with sub-pixel precision
        gt_right_subpixel = np.array(
            [[np.nan, np.nan, 0, 0], [np.nan, np.nan, np.nan, np.nan], [0, 0, 0, 0], [np.nan, np.nan, 0, np.nan]],
            dtype=np.float32,
        )
        gt_right_subpixel = xr.DataArray(
            gt_right_subpixel, coords=[[0, 1, 2, 3], [0.5, 1.5, 2.5, 3.5]], dims=["row", "col"]
        )

        if not masks_right[1].equals(gt_right_subpixel):
            raise ValueError("test_masks_dilatation error : right shifted mask ")

    def test_cmax(self):
        """
        Test the cmax attribute of the cost volume

        """
        # Test cmax for the census mesure
        matching_cost_matcher = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "census", "window_size": 3, "subpix": 1}
        )
        census_cmax_w3 = matching_cost_matcher.compute_cost_volume(
            img_left=self.left, img_right=self.right, disp_min=-1, disp_max=1
        )
        # Check if the calculated maximal cost is equal to the ground truth
        np.testing.assert_array_equal(census_cmax_w3.attrs["cmax"], 9)
        assert np.nanmax(census_cmax_w3["cost_volume"].data) <= 9

        matching_cost_matcher = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "census", "window_size": 5, "subpix": 1}
        )
        census_cmax_w5 = matching_cost_matcher.compute_cost_volume(
            img_left=self.left, img_right=self.right, disp_min=-1, disp_max=1
        )
        # Check if the calculated maximal cost is equal to the ground truth
        np.testing.assert_array_equal(census_cmax_w5.attrs["cmax"], 25)
        assert np.nanmax(census_cmax_w5["cost_volume"].data) <= 25

        # Test cmax for the sad mesure
        matching_cost_matcher = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 3, "subpix": 1}
        )
        sad_cmax_w3 = matching_cost_matcher.compute_cost_volume(
            img_left=self.left, img_right=self.right, disp_min=-1, disp_max=1
        )
        # Check if the calculated maximal cost is equal to the ground truth
        np.testing.assert_array_equal(sad_cmax_w3.attrs["cmax"], int(abs(4 - 1) * (3 ** 2)))
        assert np.nanmax(sad_cmax_w3["cost_volume"].data) <= int(abs(4 - 1) * (3 ** 2))

        matching_cost_matcher = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 5, "subpix": 1}
        )
        sad_cmax_w5 = matching_cost_matcher.compute_cost_volume(
            img_left=self.left, img_right=self.right, disp_min=-1, disp_max=1
        )
        # Check if the calculated maximal cost is equal to the ground truth
        np.testing.assert_array_equal(sad_cmax_w5.attrs["cmax"], int(abs(4 - 1) * (5 ** 2)))
        assert np.nanmax(sad_cmax_w3["cost_volume"].data) <= int(abs(4 - 1) * (5 ** 2))

        # Test cmax for the ssd mesure
        matching_cost_matcher = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "ssd", "window_size": 3, "subpix": 1}
        )
        ssd_cmax_w3 = matching_cost_matcher.compute_cost_volume(
            img_left=self.left, img_right=self.right, disp_min=-1, disp_max=1
        )
        # Check if the calculated maximal cost is equal to the ground truth
        np.testing.assert_array_equal(ssd_cmax_w3.attrs["cmax"], int(abs(4 - 1) ** 2 * (3 ** 2)))
        assert np.nanmax(sad_cmax_w3["cost_volume"].data) <= int(abs(4 - 1) ** 2 * (3 ** 2))

        matching_cost_matcher = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "ssd", "window_size": 5, "subpix": 1}
        )
        ssd_cmax_w5 = matching_cost_matcher.compute_cost_volume(
            img_left=self.left, img_right=self.right, disp_min=-1, disp_max=1
        )
        # Check if the calculated maximal cost is equal to the ground truth
        np.testing.assert_array_equal(ssd_cmax_w5.attrs["cmax"], int(abs(4 - 1) ** 2 * (5 ** 2)))
        assert np.nanmax(sad_cmax_w3["cost_volume"].data) <= int(abs(4 - 1) ** 2 * (5 ** 2))

        # Test cmax for the zncc mesure
        matching_cost_matcher = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "zncc", "window_size": 3, "subpix": 1}
        )
        zncc_cmax = matching_cost_matcher.compute_cost_volume(
            img_left=self.left, img_right=self.right, disp_min=-1, disp_max=1
        )
        # Check if the calculated maximal cost is equal to the ground truth
        np.testing.assert_array_equal(zncc_cmax.attrs["cmax"], 1)
        assert np.nanmax(zncc_cmax["cost_volume"].data) <= 1

    def test_dmin_dmax(self):
        """
        Test dmin_dmax function which returns the min disparity and the max disparity

        """
        # Load matching_cost plugin
        matching_cost_matcher = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "census", "window_size": 3, "subpix": 1}
        )

        # dmin and dmax values
        dmin_int = -2

        dmin_grid = np.array([[2, 3, 5, 12], [15, 0, -5, -2], [-4, 5, 10, 1]])

        dmax_int = 20

        dmax_grid = np.array([[18, 12, 8, 25], [16, 7, -1, 0], [5, 10, 20, 11]])

        # Case with dmin and dmax are fixed disparities
        gt_fixed_disp = (-2, 20)
        compute_fixed_disp = matching_cost_matcher.dmin_dmax(dmin_int, dmax_int)
        self.assertEqual(gt_fixed_disp, compute_fixed_disp)

        # Case with dmin is a fixed disparity and dmax is a variable disparity
        gt_fixed_var_disp = (-2, 25)
        compute_fixed_var_disp = matching_cost_matcher.dmin_dmax(dmin_int, dmax_grid)
        self.assertEqual(gt_fixed_var_disp, compute_fixed_var_disp)

        # Case with dmin is a variable disparity and dmax is a fixed disparity
        gt_var_fixed_disp = (-5, 20)
        compute_var_fixed_disp = matching_cost_matcher.dmin_dmax(dmin_grid, dmax_int)
        self.assertEqual(gt_var_fixed_disp, compute_var_fixed_disp)

        # Case with dmin and dmax are variable disparities
        gt_variable_disp = (-5, 25)
        compute_var_disp = matching_cost_matcher.dmin_dmax(dmin_grid, dmax_grid)
        self.assertEqual(gt_variable_disp, compute_var_disp)

    @staticmethod
    def test_cv_masked():
        """
        Test cv_masked function which masks with nan, the costs which have been computed with disparities outside
        of the range of variable disparities grid

        """
        # Initialize data
        data = np.array(
            (
                [1, 1, 1, 3, 2, 1, 7, 2, 3, 4, 6],
                [1, 3, 2, 5, 2, 6, 1, 8, 7, 0, 4],
                [2, 1, 0, 1, 7, 9, 5, 4, 9, 1, 5],
                [1, 5, 4, 3, 2, 6, 7, 6, 5, 2, 1],
            ),
            dtype=np.float64,
        )

        left = xr.Dataset(
            {"im": (["row", "col"], data)}, coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])}
        )

        left.attrs = {
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

        data = np.array(
            (
                [5, 1, 2, 3, 4, 7, 9, 6, 5, 2, 7],
                [1, 3, 0, 2, 5, 3, 7, 8, 7, 6, 5],
                [2, 3, 5, 0, 1, 5, 6, 5, 2, 3, 6],
                [1, 6, 7, 5, 3, 2, 1, 0, 3, 4, 7],
            ),
            dtype=np.float64,
        )

        right = xr.Dataset(
            {"im": (["row", "col"], data)}, coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])}
        )

        right.attrs = {
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

        dmin_grid = np.array(
            [
                [-0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0],
                [-0, -8, -8, -5, -8, -4, -6, -7, -9, -8, -0],
                [-0, -9, -8, -4, -6, -5, -7, -8, -9, -7, -0],
                [-0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0],
            ]
        )

        dmax_grid = np.array(
            [
                [-0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0],
                [-0, -2, -1, -1, -5, -1, -2, -6, -4, -3, -0],
                [-0, -3, 0, -2, -2, -2, -3, -5, -5, -4, -0],
                [-0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0],
            ]
        )

        # Initialization of matching_cost plugin
        matching_cost_ = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "census", "window_size": 3, "subpix": 1}
        )

        # compute the min disparity of disp_min and the max disparity of disp_max
        dmin_int, dmax_int = matching_cost_.dmin_dmax(dmin_grid, dmax_grid)

        # ------------ Test the method with disp_min as a grid and disp_max as a grid, subpixel = 1 ------------
        # Compute the cost_volume
        cv = matching_cost_.compute_cost_volume(img_left=left, img_right=right, disp_min=dmin_int, disp_max=dmax_int)

        # Compute the masked cost volume
        matching_cost_.cv_masked(img_left=left, img_right=right, cost_volume=cv, disp_min=dmin_grid, disp_max=dmax_grid)

        # Cost volume ground truth
        gt_cv_masked = np.array(
            [
                [
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                ],
                [
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 5.0, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1.0, 8.0, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, 5.0, 4.0, 3.0, 2.0, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, 7.0, 2.0, 3.0, 6.0, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, 1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, 4.0, 5.0, 6.0, 3.0, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, 6.0, 1.0, 4.0, 7.0, 1.0, 6.0, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                ],
                [
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 5.0, 6.0],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 4.0, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 2.0, 3.0, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, 3.0, 2.0, 7.0, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, 5.0, 4.0, 3.0, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, 4.0, 5.0, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, 3.0, 2.0, 7.0, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, 4.0, 3.0, 3.0, 4.0, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                ],
                [
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                ],
            ],
            dtype=np.float32,
        )

        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(gt_cv_masked, cv["cost_volume"].data)

        # ------------ Test the method with disp_min as a grid and disp_max as a grid, subpixel = 2 ------------
        # Initialization of matching_cost plugin
        matching_cost_ = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "census", "window_size": 3, "subpix": 2}
        )

        # compute the min disparity of disp_min and the max disparity of disp_max
        dmin_int, dmax_int = matching_cost_.dmin_dmax(dmin_grid, dmax_grid)

        # Compute the cost_volume
        cv = matching_cost_.compute_cost_volume(img_left=left, img_right=right, disp_min=dmin_int, disp_max=dmax_int)

        # Compute the masked cost volume
        matching_cost_.cv_masked(img_left=left, img_right=right, cost_volume=cv, disp_min=dmin_grid, disp_max=dmax_grid)

        # Cost volume ground truth
        gt_cv_masked = np.array(
            [
                [
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                ],
                [
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        5.0,
                        np.nan,
                        np.nan,
                    ],
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        1.0,
                        5.0,
                        8.0,
                        np.nan,
                        np.nan,
                    ],
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        5.0,
                        3.0,
                        4.0,
                        4.0,
                        3.0,
                        3.0,
                        2.0,
                        np.nan,
                        np.nan,
                    ],
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        7.0,
                        3.0,
                        2.0,
                        2.0,
                        3.0,
                        5.0,
                        6.0,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        1.0,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        4.0,
                        4.0,
                        5.0,
                        5.0,
                        6.0,
                        4.0,
                        3.0,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    [
                        np.nan,
                        np.nan,
                        6.0,
                        2.0,
                        1.0,
                        1.0,
                        4.0,
                        6.0,
                        7.0,
                        4.0,
                        1.0,
                        2.0,
                        6.0,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                ],
                [
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        5.0,
                        6.0,
                        6.0,
                    ],
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        4.0,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        2.0,
                        3.0,
                        3.0,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        3.0,
                        2.0,
                        2.0,
                        5.0,
                        7.0,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        5.0,
                        4.0,
                        4.0,
                        1.0,
                        3.0,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        4.0,
                        5.0,
                        5.0,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        3.0,
                        2.0,
                        2.0,
                        5.0,
                        7.0,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        4.0,
                        3.0,
                        3.0,
                        2.0,
                        3.0,
                        3.0,
                        4.0,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                ],
                [
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                ],
            ],
            dtype=np.float32,
        )

        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(gt_cv_masked, cv["cost_volume"].data)


if __name__ == "__main__":
    common.setup_logging()
    unittest.main()
