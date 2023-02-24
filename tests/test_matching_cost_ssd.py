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
This module contains functions to test the SSD matching cost step.
"""
import unittest
import pytest

import numpy as np
import xarray as xr

from pandora import matching_cost
from tests import common


class TestMatchingCostSSD(unittest.TestCase):
    """
    TestMatchingCost class allows to test all the methods in the
    matching_cost SSD class
    """

    def setUp(self):
        """
        Method called to prepare the test fixture

        """

        self.left, self.right = common.matching_cost_tests_setup()
        self.left_multiband, self.right_multiband = common.matching_cost_tests_multiband_setup()

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

    def test_ssd_cost_multiband(self):
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
            **{"matching_cost_method": "ssd", "window_size": 1, "subpix": 1, "band": "r"}
        )
        ssd = matching_cost_matcher.compute_cost_volume(
            img_left=self.left_multiband, img_right=self.right_multiband, disp_min=-1, disp_max=1
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

        # Compute gt cmax:
        window_size = 5
        ssd_cmax_ground_truth = np.nanmax(sd_ground_truth) * window_size**2
        # Computes the sd cost for the whole images
        matching_cost_matcher = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "ssd", "window_size": 5, "subpix": 1, "band": "r"}
        )
        ssd = matching_cost_matcher.compute_cost_volume(
            img_left=self.left_multiband, img_right=self.right_multiband, disp_min=-1, disp_max=1
        )
        matching_cost_matcher.cv_masked(self.left_multiband, self.right_multiband, ssd, -1, 1)

        # Check if the calculated sd cost is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(ssd["cost_volume"].sel(disp=0), ssd_ground_truth)

        # Check if the calculated max cost is equal to the ground truth
        np.testing.assert_array_equal(ssd.attrs["cmax"], ssd_cmax_ground_truth)

    @staticmethod
    def test_check_band_ssd():
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
            {"im": (["band", "row", "col"], data)},
            coords={"band": np.arange(data.shape[0]), "row": np.arange(data.shape[1]), "col": np.arange(data.shape[2])},
        )

        left.attrs = common.img_attrs

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
            {"im": (["band", "row", "col"], data)},
            coords={"band": np.arange(data.shape[0]), "row": np.arange(data.shape[1]), "col": np.arange(data.shape[2])},
        )

        right.attrs = common.img_attrs

        # Initialization of matching_cost plugin with wrong band
        matching_cost_ = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "ssd", "window_size": 3, "subpix": 1, "band": "b"}
        )

        # Compute the cost_volume
        with pytest.raises(SystemExit):
            _ = matching_cost_.compute_cost_volume(img_left=left, img_right=right, disp_min=-1, disp_max=1)

        # Initialization of matching_cost plugin with no band
        matching_cost_ = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "ssd", "window_size": 3, "subpix": 1}
        )

        # Compute the cost_volume
        with pytest.raises(SystemExit):
            _ = matching_cost_.compute_cost_volume(img_left=left, img_right=right, disp_min=-1, disp_max=1)


if __name__ == "__main__":
    common.setup_logging()
    unittest.main()
