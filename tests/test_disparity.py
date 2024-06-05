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
This module contains functions to test the disparity module.
"""
import copy
import unittest
from unittest.mock import patch

import numpy as np
import xarray as xr
from rasterio import Affine

from tests import common
import pandora
from pandora import disparity
from pandora import matching_cost
from pandora.img_tools import create_dataset_from_inputs, add_disparity
from pandora.criteria import validity_mask
from pandora.state_machine import PandoraMachine
from pandora.margins.descriptors import NullMargins


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

    def test_margins(self):
        assert isinstance(disparity.AbstractDisparity.margins, NullMargins)

    def test_to_disp(self):
        """
        Test the to disp method

        """
        # Add disparity on left image
        add_disparity(self.left, [-3, 1], None)

        # Create the left cost volume, with SAD measure window size 1, subpixel 1, disp_min -3 disp_max 1
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

        # Add disparity on left image
        add_disparity(self.left, [-3, -1], None)

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

        # Disparity map ground truth
        gt_disp = np.array([[0, -1, -2, -3], [0, -1, -1, -3], [0, -1, -2, -3]])

        # Compute the disparity
        disp = disparity_.to_disp(cv)

        # Check if the calculated disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(disp["disparity_map"].data, gt_disp)

        #
        # Test the to_disp method with positive disparity range
        #

        # Add disparity on left image
        add_disparity(self.left, [1, 3], None)

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

    @patch("pandora.disparity.disparity.extract_disparity_interval_from_cost_volume", return_value=[1])
    def test_to_disp_disparity_interval_data_variable(self, mocked_extract_disparity_interval_from_cost_volume):
        """Test data_variable `disparity_interval` is as expected."""
        disparity_min, disparity_max = -3, 1
        cost_volume_data = np.array(
            [
                [
                    [np.nan, np.nan, np.nan, 5.0, 0.0],
                    [np.nan, np.nan, 4.0, 1.0, 0.0],
                    [np.nan, 2.0, 3.0, 2.0, 0.0],
                    [0.0, 5.0, 4.0, 2.0, np.nan],
                ],
                [
                    [np.nan, np.nan, np.nan, 4.0, 0.0],
                    [np.nan, np.nan, 2.0, 2.0, 0.0],
                    [np.nan, 5.0, 1.0, 3.0, 0.0],
                    [0.0, 4.0, 2.0, 5.0, np.nan],
                ],
                [
                    [np.nan, np.nan, np.nan, 4.0, 0.0],
                    [np.nan, np.nan, 3.0, 1.0, 0.0],
                    [np.nan, 2.0, 2.0, 1.0, 0.0],
                    [0.0, 4.0, 3.0, 2.0, np.nan],
                ],
            ],
            dtype=np.float32,
        )

        cost_volume = xr.Dataset(
            {"cost_volume": (["row", "col", "disp"], cost_volume_data)},
            coords={"row": np.arange(3), "col": np.arange(4), "disp": np.arange(disparity_min, disparity_max + 1)},
            attrs={
                "measure": "sad",
                "subpixel": 1,
                "offset_row_col": 1,
                "window_size": 3,
                "type_measure": "min",
                "cmax": 81,
                "band_correl": None,
                "crs": None,
                "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                "disparity_source": [disparity_min, disparity_max],
            },
        )

        # Compute validity mask
        cost_volume = validity_mask(self.left, self.right, cost_volume)

        # Compute the disparity
        disparity_ = disparity.AbstractDisparity(**{"disparity_method": "wta", "invalid_disparity": 0})
        disp = disparity_.to_disp(cost_volume)

        result = disp["disparity_interval"]

        mocked_extract_disparity_interval_from_cost_volume.assert_called_with(cost_volume)
        assert result == mocked_extract_disparity_interval_from_cost_volume.return_value

    def test_to_disp_with_offset(self):
        """
        Test the to disp method with window_size > 1

        """
        # Add disparity on left image
        add_disparity(self.left, [-3, 1], None)

        # Create the left cost volume, with SAD measure window size 3, subpixel 1, disp_min -3 disp_max 1
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

        # Add disparity on left image
        add_disparity(self.left, [-3, -1], None)

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

        # Disparity map ground truth
        gt_disp = np.array([[-99, -99, -99, -99], [-99, -99, -1, -99], [-99, -99, -99, -99]])

        # Compute the disparity
        disp = disparity_.to_disp(cv)

        # Check if the calculated disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(disp["disparity_map"].data, gt_disp)

        #
        # Test the to_disp method with positive disparity range
        #

        # Add disparity on left image
        add_disparity(self.left, [1, 3], None)

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
        # Add disparity on left image
        add_disparity(self.left, [-3, 1], None)

        # Create the left cost volume, with SAD measure, window size 1, subpixel 2, disp_min -3 disp_max 1
        matching_cost_plugin = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 1, "subpix": 2}
        )
        grid = matching_cost_plugin.allocate_cost_volume(
            self.left, (self.left["disparity"].sel(band_disp="min"), self.left["disparity"].sel(band_disp="max"))
        )
        cv = matching_cost_plugin.compute_cost_volume(self.left, self.right, cost_volume=grid)
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
        # Add disparity on left image
        add_disparity(self.left, [-3, 1], None)

        # Create the left cost volume, with ZNCC measure, window size 1, subpixel 2, disp_min -3 disp_max 1
        matching_cost_plugin = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "zncc", "window_size": 1, "subpix": 2}
        )
        grid = matching_cost_plugin.allocate_cost_volume(
            self.left, (self.left["disparity"].sel(band_disp="min"), self.left["disparity"].sel(band_disp="max"))
        )
        cv = matching_cost_plugin.compute_cost_volume(self.left, self.right, cost_volume=grid)
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
        # Add disparity on left image
        add_disparity(self.left, [-3, 1], None)

        # Create the left cost volume, with SAD measure window size 1, subpixel 1, disp_min -3 disp_max 1
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

        # Compute the disparity
        disparity_ = disparity.AbstractDisparity(**{"disparity_method": "wta", "invalid_disparity": 0})
        disparity_.to_disp(cv)

        # Coefficient map ground truth, for the images described in the setUp method
        gt_coeff = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        # Compute the disparity, and the coefficient map
        coeff = disparity_.coefficient_map(cv)

        # Check if the calculated coefficient map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(coeff.data, gt_coeff)

    def test_approximate_right_disparity_data_variable_disparity_map(self):
        """
        Test `disparity_map` data_values are correct.

        """
        # Add disparity on right image
        disparity_min, disparity_max = -2, 1
        add_disparity(self.right, [disparity_min, disparity_max], window=None)

        # Create the left cost volume, with SAD measure window size 3 and subpixel 1
        cost_volume_data = np.full((3, 4, 4), np.nan, dtype=np.float32)
        cost_volume_data[1, 1, 2] = 23
        cost_volume_data[1, 1, 3] = 0
        cost_volume_data[1, 2, 1] = 24
        cost_volume_data[1, 2, 2] = 19

        # Cost Volume, for the images described in the setUp method
        cv = xr.Dataset(
            {"cost_volume": (["row", "col", "disp"], cost_volume_data)},
            coords={"row": np.arange(3), "col": np.arange(4), "disp": np.arange(disparity_min, disparity_max + 1)},
            attrs={
                "measure": "sad",
                "subpixel": 1,
                "offset_row_col": 1,
                "window_size": 3,
                "type_measure": "min",
                "cmax": 81,
                "band_correl": None,
                "crs": None,
                "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                "disparity_source": self.right.attrs["disparity_source"],
            },
        )

        # Right disparity map ground truth, for the images described in the setUp method
        gt_disp = np.array([[0, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 0]])

        # Compute the right disparity map
        disparity_ = disparity.AbstractDisparity(**{"disparity_method": "wta", "invalid_disparity": 0})
        disp_r = disparity_.approximate_right_disparity(cv, self.right)

        # Check if the calculated right disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(disp_r["disparity_map"].data, gt_disp)

    @patch("pandora.disparity.disparity.extract_disparity_interval_from_cost_volume", return_value=[1])
    def test_approximate_right_disparity_data_variable_disparity_interval(
        self, mocked_extract_disparity_interval_from_cost_volume
    ):
        """
        Test `disparity_interval` data_values are correct.

        """
        disparity_min, disparity_max = -2, 1
        # Create the left cost volume, with SAD measure window size 3 and subpixel 1
        cost_volume_data = np.full((3, 4, 4), np.nan, dtype=np.float32)
        cost_volume_data[1, 1, 2] = 23
        cost_volume_data[1, 1, 3] = 0
        cost_volume_data[1, 2, 1] = 24
        cost_volume_data[1, 2, 2] = 19

        # Cost Volume, for the images described in the setUp method
        cost_volume = xr.Dataset(
            {"cost_volume": (["row", "col", "disp"], cost_volume_data)},
            coords={"row": np.arange(3), "col": np.arange(4), "disp": np.arange(disparity_min, disparity_max + 1)},
            attrs={
                "measure": "sad",
                "subpixel": 1,
                "offset_row_col": 1,
                "window_size": 3,
                "type_measure": "min",
                "cmax": 81,
                "band_correl": None,
                "crs": None,
                "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                "disparity_source": [disparity_min, disparity_max],
            },
        )

        # Compute the right disparity map
        disparity_ = disparity.AbstractDisparity(**{"disparity_method": "wta", "invalid_disparity": 0})
        disp_r = disparity_.approximate_right_disparity(cost_volume, self.right)
        result = disp_r["disparity_interval"]

        mocked_extract_disparity_interval_from_cost_volume.assert_called_with(cost_volume)
        assert result == mocked_extract_disparity_interval_from_cost_volume.return_value

    def test_right_disparity_subpixel(self):
        """
        Test the right disparity method, with subpixel disparity

        """
        # Add disparities on left and right images
        add_disparity(self.left, [-2, 1], None)

        # Create the left cost volume, with SAD measure window size 3 and subpixel 4
        matching_cost_plugin = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 3, "subpix": 4}
        )
        grid = matching_cost_plugin.allocate_cost_volume(
            self.left, (self.left["disparity"].sel(band_disp="min"), self.left["disparity"].sel(band_disp="max"))
        )
        cv = matching_cost_plugin.compute_cost_volume(self.left, self.right, cost_volume=grid)

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
        default_cfg = pandora.check_configuration.default_short_configuration

        input_config = common.input_cfg_basic
        input_config["left"]["nodata"] = np.nan
        input_config["right"]["nodata"] = np.nan
        input_config["right"]["disp"] = [-input_config["left"]["disp"][1], -input_config["left"]["disp"][0]]

        pandora_left = create_dataset_from_inputs(input_config=input_config["left"])
        pandora_right = create_dataset_from_inputs(input_config=input_config["right"])

        fast_cfg = {
            "input": copy.deepcopy(common.input_cfg_basic),
            "pipeline": {
                "matching_cost": {"matching_cost_method": "census"},
                "disparity": {"disparity_method": "wta"},
                "refinement": {"refinement_method": "vfit"},
                "validation": {"validation_method": "cross_checking_accurate"},
            },
        }

        pandora_machine_fast = PandoraMachine()
        cfg = pandora.check_configuration.update_conf(default_cfg, fast_cfg)
        left, right_fast = pandora.run(  # pylint: disable=unused-variable
            pandora_machine_fast, pandora_left, pandora_right, cfg
        )

        acc_cfg = {
            "input": copy.deepcopy(common.input_cfg_basic),
            "pipeline": {
                "matching_cost": {"matching_cost_method": "census"},
                "disparity": {"disparity_method": "wta"},
                "refinement": {"refinement_method": "vfit"},
                "validation": {"validation_method": "cross_checking_accurate"},
            },
        }

        pandora_machine_acc = PandoraMachine()
        cfg = pandora.check_configuration.update_conf(default_cfg, acc_cfg)
        left, right_acc = pandora.run(pandora_machine_acc, pandora_left, pandora_right, cfg)
        # Check if the calculated disparity map in fast mode is equal to the disparity map in accurate mode
        np.testing.assert_array_equal(right_fast["disparity_map"].data, right_acc["disparity_map"].data)

        # Check if the calculated coefficient map in fast mode is equal to the coefficient map in accurate mode
        np.testing.assert_array_equal(right_fast["interpolated_coeff"].data, right_acc["interpolated_coeff"].data)


def test_extract_disparity_interval_from_cost_volume():
    """
    We expect coordinate `disparity_interval` to be an array of the first and the last value of cost volume `disp`
    coordinate.

    """
    disparity_min, disparity_max = -2, 1
    # Create the left cost volume, with SAD measure window size 3 and subpixel 1
    cost_volume_data = np.full((3, 4, 4), np.nan, dtype=np.float32)
    cost_volume_data[1, 1, 2] = 23
    cost_volume_data[1, 1, 3] = 0
    cost_volume_data[1, 2, 1] = 24
    cost_volume_data[1, 2, 2] = 19

    # Cost Volume, for the images described in the setUp method
    cost_volume = xr.Dataset(
        {"cost_volume": (["row", "col", "disp"], cost_volume_data)},
        coords={"row": np.arange(3), "col": np.arange(4), "disp": np.arange(disparity_min, disparity_max + 1)},
        attrs={
            "measure": "sad",
            "subpixel": 1,
            "offset_row_col": 1,
            "window_size": 3,
            "type_measure": "min",
            "cmax": 81,
            "band_correl": None,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            "disparity_source": [disparity_min, disparity_max],
        },
    )

    expected = xr.DataArray([disparity_min, disparity_max], coords=[("disparity", ["min", "max"])])

    result = disparity.extract_disparity_interval_from_cost_volume(cost_volume)

    xr.testing.assert_identical(result, expected)


def test_extract_interval_from_disparity_map():
    """
    We expect the return value to be a tuple of integers with disparity min and max as values.
    """

    disparity_min, disparity_max = -2, 1

    disparity_map = xr.Dataset(
        {
            "disparity_map": xr.DataArray(np.zeros((2, 2)), coords=[("row", range(0, 2)), ("col", range(0, 2))]),
            "disparity_interval": xr.DataArray([disparity_min, disparity_max], coords=[("disparity", ["min", "max"])]),
        },
        attrs={
            "measure": "sad",
            "subpixel": 1,
            "offset_row_col": 1,
            "window_size": 3,
            "type_measure": "min",
            "cmax": 81,
            "band_correl": None,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            "disparity_source": [disparity_min, disparity_max],
        },
    )

    result_min, result_max = disparity.extract_interval_from_disparity_map(disparity_map)

    assert isinstance(result_min, int)
    assert result_min == disparity_min
    assert isinstance(result_max, int)
    assert result_max == disparity_max


def test_extract_disparity_range_from_disparity_map():
    """
    We expect the return value to be a numpy array of evenly spaced values within disparity min and disparity max.
    """

    disparity_min, disparity_max = -2, 1
    expected = np.array([-2, -1, 0, 1])

    disparity_map = xr.Dataset(
        {
            "disparity_map": xr.DataArray(np.zeros((2, 2)), coords=[("row", range(0, 2)), ("col", range(0, 2))]),
            "disparity_interval": xr.DataArray([disparity_min, disparity_max], coords=[("disparity", ["min", "max"])]),
        },
        attrs={
            "measure": "sad",
            "subpixel": 1,
            "offset_row_col": 1,
            "window_size": 3,
            "type_measure": "min",
            "cmax": 81,
            "band_correl": None,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            "disparity_source": [disparity_min, disparity_max],
        },
    )

    result = disparity.extract_disparity_range_from_disparity_map(disparity_map)

    np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    common.setup_logging()
    unittest.main()
