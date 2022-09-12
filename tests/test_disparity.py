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
This module contains functions to test the disparity module.
"""

import unittest

import numpy as np
import xarray as xr
from rasterio import Affine

from tests import common
import pandora
from pandora import disparity
from pandora import matching_cost
from pandora.img_tools import read_img
from pandora.state_machine import PandoraMachine


class TestDisparity(unittest.TestCase):
    """
    TestDisparity class allows to test the disparity module
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

    def test_to_disp(self):
        """
        Test the to disp method

        """

        # Create the left cost volume, with SAD measure window size 1, subpixel 1, disp_min -3 disp_max 1
        matching_cost_plugin = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 1, "subpix": 1}
        )
        cv = matching_cost_plugin.compute_cost_volume(self.left, self.right, -3, 1)

        # Disparity map ground truth, for the images described in the setUp method
        gt_disp = np.array([[1, 1, 1, -3], [1, 1, 1, -3], [1, 1, 1, -3]])

        # Compute the disparity
        disparity_ = disparity.AbstractDisparity(**{"disparity_method": "wta", "invalid_disparity": 0})
        disp = disparity_.to_disp(cv)

        # Check if the calculated disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(disp["disparity_map"].data, gt_disp)

        #
        # Test the to_disp method with negative disparity range
        #
        cv = matching_cost_plugin.compute_cost_volume(self.left, self.right, -3, -1)

        # Disparity map ground truth
        gt_disp = np.array([[0, -1, -2, -3], [0, -1, -1, -3], [0, -1, -2, -3]])

        # Compute the disparity
        disp = disparity_.to_disp(cv)

        # Check if the calculated disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(disp["disparity_map"].data, gt_disp)

        #
        # Test the to_disp method with positive disparity range
        #
        cv = matching_cost_plugin.compute_cost_volume(self.left, self.right, 1, 3)

        # Disparity map ground truth
        gt_disp = np.array([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0]])

        # Compute the disparity
        disp = disparity_.to_disp(cv)

        # Check if the calculated disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(disp["disparity_map"].data, gt_disp)

        # Test disp_indices copy
        # Modify the disparity map
        disp["disparity_map"].data[0, 0] = -95
        # Check if the xarray disp_indices is equal to the ground truth disparity map
        np.testing.assert_array_equal(cv["disp_indices"].data, gt_disp)

    def test_to_disp_with_offset(self):
        """
        Test the to disp method with window_size > 1

        """

        # Create the left cost volume, with SAD measure window size 3, subpixel 1, disp_min -3 disp_max 1
        matching_cost_plugin = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 3, "subpix": 1}
        )
        cv = matching_cost_plugin.compute_cost_volume(self.left, self.right, -3, 1)

        # Disparity map ground truth, for the images described in the setUp method
        # Check if gt is full size and border (i.e [offset:-offset] equal to invalid_disparity
        gt_disp = np.array([[-99, -99, -99, -99], [-99, 1, 0, -99], [-99, -99, -99, -99]])

        # Compute the disparity
        disparity_ = disparity.AbstractDisparity(**{"disparity_method": "wta", "invalid_disparity": -99})
        disp = disparity_.to_disp(cv)

        # Check if the calculated disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(disp["disparity_map"].data, gt_disp)

        #
        # Test the to_disp method with negative disparity range
        #
        cv = matching_cost_plugin.compute_cost_volume(self.left, self.right, -3, -1)

        # Disparity map ground truth
        gt_disp = np.array([[-99, -99, -99, -99], [-99, -99, -1, -99], [-99, -99, -99, -99]])

        # Compute the disparity
        disp = disparity_.to_disp(cv)

        # Check if the calculated disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(disp["disparity_map"].data, gt_disp)

        #
        # Test the to_disp method with positive disparity range
        #
        cv = matching_cost_plugin.compute_cost_volume(self.left, self.right, 1, 3)

        # Disparity map ground truth
        gt_disp = np.array([[-99, -99, -99, -99], [-99, 1, -99, -99], [-99, -99, -99, -99]])
        # Compute the disparity
        disp = disparity_.to_disp(cv)

        # Check if the calculated disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(disp["disparity_map"].data, gt_disp)

        # Test disp_indices copy
        # Modify the disparity map
        disp["disparity_map"].data[0, 0] = -95
        # Check if the xarray disp_indices is equal to the ground truth disparity map
        np.testing.assert_array_equal(cv["disp_indices"].data, gt_disp)

    def test_argmin_split(self):
        """
        Test the argmin_split method

        """
        # Create the left cost volume, with SAD measure, window size 1, subpixel 2, disp_min -3 disp_max 1
        matching_cost_plugin = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 1, "subpix": 2}
        )
        cv = matching_cost_plugin.compute_cost_volume(self.left, self.right, -3, 1)
        indices_nan = np.isnan(cv["cost_volume"].data)
        cv["cost_volume"].data[indices_nan] = np.inf

        # ground truth
        gt_disp = np.array([[1.0, 1.0, 1.0, -3.0], [1.0, -0.5, 1.0, -3.0], [1.0, 1.0, -1.5, -3]], dtype=np.float32)

        # Compute the disparity
        disparity_ = disparity.AbstractDisparity(**{"disparity_method": "wta", "invalid_disparity": 0})
        disp = disparity_.argmin_split(cv)

        # Check if the calculated coefficient map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(gt_disp, disp)

    def test_argmax_split(self):
        """
        Test the argmax_split method

        """
        # Create the left cost volume, with ZNCC measure, window size 1, subpixel 2, disp_min -3 disp_max 1
        matching_cost_plugin = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "zncc", "window_size": 1, "subpix": 2}
        )
        cv = matching_cost_plugin.compute_cost_volume(self.left, self.right, -3, 1)
        indices_nan = np.isnan(cv["cost_volume"].data)
        cv["cost_volume"].data[indices_nan] = -np.inf

        # ground truth
        gt_disp = np.array(
            [[0.0, -1.0, -2.0, -3.0], [0.0, -1.0, -2.0, -3.0], [0.0, -1.0, -2.0, -3.0]], dtype=np.float32
        )

        # Compute the disparity
        disparity_ = disparity.AbstractDisparity(**{"disparity_method": "wta", "invalid_disparity": 0})
        disp = disparity_.argmax_split(cv)

        # Check if the calculated coefficient map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(gt_disp, disp)

    def test_coefficient_map(self):
        """
        Test the method coefficient map

        """
        # Create the left cost volume, with SAD measure window size 1, subpixel 1, disp_min -3 disp_max 1
        matching_cost_plugin = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 1, "subpix": 1}
        )
        cv = matching_cost_plugin.compute_cost_volume(self.left, self.right, -3, 1)

        # Compute the disparity
        disparity_ = disparity.AbstractDisparity(**{"disparity_method": "wta", "invalid_disparity": 0})
        disparity_.to_disp(cv)

        # Coefficient map ground truth, for the images described in the setUp method
        gt_coeff = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        # Compute the disparity, and the coefficient map
        coeff = disparity_.coefficient_map(cv)

        # Check if the calculated coefficient map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(coeff.data, gt_coeff)

    def test_approximate_right_disparity(self):
        """
        Test the approximate_right_disparity method

        """
        # Create the left cost volume, with SAD measure window size 3 and subpixel 1
        matching_cost_plugin = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 3, "subpix": 1}
        )
        cv = matching_cost_plugin.compute_cost_volume(self.left, self.right, -2, 1)

        # Right disparity map ground truth, for the images described in the setUp method
        gt_disp = np.array([[0, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 0]])

        # Compute the right disparity map
        disparity_ = disparity.AbstractDisparity(**{"disparity_method": "wta", "invalid_disparity": 0})
        disp_r = disparity_.approximate_right_disparity(cv, self.right)

        # Check if the calculated right disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(disp_r["disparity_map"].data, gt_disp)

    def test_right_disparity_subpixel(self):
        """
        Test the right disparity method, with subpixel disparity

        """
        # Create the left cost volume, with SAD measure window size 3 and subpixel 4
        matching_cost_plugin = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 3, "subpix": 4}
        )
        cv = matching_cost_plugin.compute_cost_volume(self.left, self.right, -2, 1)

        # Right disparity map ground truth
        gt_disp = np.array([[0, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 0]])

        # Compute the right disparity map
        disparity_ = disparity.AbstractDisparity(**{"disparity_method": "wta", "invalid_disparity": 0})
        disp_r = disparity_.approximate_right_disparity(cv, self.right)

        # Check if the calculated right disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(disp_r["disparity_map"].data, gt_disp)

    @staticmethod
    def test_right_disparity_comparaison():
        """
        Test the right disparity method by comparing the right disparity map calculated from scratch with the one
        calculated with the fast method

        """
        # Build the default configuration
        default_cfg = pandora.check_json.default_short_configuration

        pandora_left = read_img("tests/pandora/left.png", no_data=np.nan, mask=None)
        pandora_right = read_img("tests/pandora/right.png", no_data=np.nan, mask=None)

        fast_cfg = {
            "pipeline": {
                "right_disp_map": {"method": "accurate"},
                "matching_cost": {"matching_cost_method": "census"},
                "disparity": {"disparity_method": "wta"},
                "refinement": {"refinement_method": "vfit"},
                "validation": {"validation_method": "cross_checking"},
            }
        }

        pandora_machine_fast = PandoraMachine()
        cfg = pandora.check_json.update_conf(default_cfg, fast_cfg)
        left, right_fast = pandora.run(  # pylint: disable=unused-variable
            pandora_machine_fast, pandora_left, pandora_right, -60, 0, cfg["pipeline"]
        )

        acc_cfg = {
            "pipeline": {
                "right_disp_map": {"method": "accurate"},
                "matching_cost": {"matching_cost_method": "census"},
                "disparity": {"disparity_method": "wta"},
                "refinement": {"refinement_method": "vfit"},
                "validation": {"validation_method": "cross_checking"},
            }
        }

        pandora_machine_acc = PandoraMachine()
        cfg = pandora.check_json.update_conf(default_cfg, acc_cfg)
        left, right_acc = pandora.run(pandora_machine_acc, pandora_left, pandora_right, -60, 0, cfg["pipeline"])
        # Check if the calculated disparity map in fast mode is equal to the disparity map in accurate mode
        np.testing.assert_array_equal(right_fast["disparity_map"].data, right_acc["disparity_map"].data)

        # Check if the calculated coefficient map in fast mode is equal to the coefficient map in accurate mode
        np.testing.assert_array_equal(right_fast["interpolated_coeff"].data, right_acc["interpolated_coeff"].data)


if __name__ == "__main__":
    common.setup_logging()
    unittest.main()
