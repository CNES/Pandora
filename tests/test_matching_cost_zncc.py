# pylint: disable=duplicate-code
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
import unittest

import numpy as np
import xarray as xr

from pandora import matching_cost
from tests import common


class TestMatchingCost(unittest.TestCase):
    """
    TestMatchingCost class allows to test all the methods in the class MatchingCost,
    and the plugins pixel_wise, zncc
    """

    def setUp(self):
        """
        Method called to prepare the test fixture

        """

        self.left, self.right = common.matching_cost_tests_setup()

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
        left.attrs = common.img_attrs
        data = np.array(([1, 5, 6, 3, 4], [2, 5, 10, 6, 9], [0, 7, 5, 3, 1]), dtype=np.float64)
        right = xr.Dataset(
            {"im": (["row", "col"], data)}, coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])}
        )
        right.attrs = common.img_attrs
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

        left.attrs = common.img_attrs
        data = np.array(([5, 1, 2, 3, 4], [1, 2, 1, 0, 2], [2, 2, 0, 1, 4], [1, 1, 1, 1, 2]), dtype=np.float64)
        # right mask contains valid pixels
        mask = np.zeros((4, 5), dtype=np.int16)
        right = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )

        right.attrs = common.img_attrs

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

        left.attrs = common.img_attrs

        data = np.array(([5, 1, 2, 3, 4], [1, 2, 1, 0, 2], [2, 2, 0, 1, 4], [1, 1, 1, 1, 2]), dtype=np.float64)
        mask = np.array(([0, 2, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 2, 0], [1, 0, 2, 0, 0]), dtype=np.int16)
        right = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )

        right.attrs = common.img_attrs

        matching_cost_ = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "zncc", "window_size": 3, "subpix": 1}
        )
        # Compute the cost volume and invalidate pixels if need
        cv = matching_cost_.compute_cost_volume(img_left=left, img_right=right, disp_min=-1, disp_max=1)
        matching_cost_.cv_masked(img_left=left, img_right=right, cost_volume=cv, disp_min=-1, disp_max=1)

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

        left.attrs = common.img_attrs

        data = np.array(([5, 1, 2, 3, 4], [1, 1, 1, 1, 2]), dtype=np.float64)
        mask = np.array(([0, 0, 0, 0, 1], [1, 0, 2, 0, 0]), dtype=np.int16)
        right = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )

        right.attrs = common.img_attrs

        dmin = -1
        dmax = 1

        # ------------ Test the method with a left and right mask with window size 3 and ZNCC ------------
        data = np.array(([1, 1, 1, 3, 4], [1, 2, 1, 0, 2], [2, 1, 0, 1, 2], [1, 1, 1, 1, 4]), dtype=np.float64)
        # left mask contains valid pixels
        mask = np.array(([1, 0, 0, 2, 0], [0, 0, 0, 0, 0], [0, 0, 2, 0, 0], [2, 0, 0, 0, 1]), dtype=np.int16)

        left = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )

        left.attrs = common.img_attrs

        data = np.array(([5, 1, 2, 3, 4], [1, 2, 1, 0, 2], [2, 2, 0, 1, 4], [1, 1, 1, 1, 2]), dtype=np.float64)
        mask = np.array(([0, 2, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 2, 0], [1, 0, 2, 0, 0]), dtype=np.int16)
        right = xr.Dataset(
            {"im": (["row", "col"], data), "msk": (["row", "col"], mask)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        )

        right.attrs = common.img_attrs

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


if __name__ == "__main__":
    common.setup_logging()
    unittest.main()
