# type:ignore
#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
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
This module contains functions to test the census matching cost step.
"""
import unittest
import pytest

import numpy as np
import xarray as xr
from rasterio import Affine
import json_checker

from pandora import matching_cost
from pandora.img_tools import add_disparity
from pandora.criteria import validity_mask

from tests import common


class TestCensusWindowSize:

    @pytest.mark.parametrize("window_size", [3, 5, 7, 9, 11, 13])
    def test_nominal_window_size(self, window_size):
        result = matching_cost.AbstractMatchingCost(**{"matching_cost_method": "census", "window_size": window_size})

        assert result.cfg["window_size"] == window_size

    @pytest.mark.parametrize("window_size", [-5, -1, 0, 1, 2, 4, 6, 8, 14, 15])
    def test_fails_with_invalid_window_size(self, window_size):
        with pytest.raises(json_checker.core.exceptions.DictCheckerError) as err:
            matching_cost.AbstractMatchingCost(**{"matching_cost_method": "census", "window_size": window_size})
        assert "window_size" in err.value.args[0]


class TestMatchingCostCensus(unittest.TestCase):
    """
    TestMatchingCost class allows to test all the methods in the
    matching_cost Census class
    """

    def setUp(self):
        """
        Method called to prepare the test fixture

        """

        self.left, self.right = common.matching_cost_tests_setup()

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
        left.pipe(add_disparity, disparity=[-1, 1], window=None)

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

        grid = matching_cost_matcher.allocate_cost_volume(
            left, (left["disparity"].sel(band_disp="min"), left["disparity"].sel(band_disp="max"))
        )

        # Compute validity mask
        grid = validity_mask(left, right, grid)

        census = matching_cost_matcher.compute_cost_volume(img_left=left, img_right=right, cost_volume=grid)
        matching_cost_matcher.cv_masked(
            left,
            right,
            census,
            left["disparity"].sel(band_disp="min"),
            left["disparity"].sel(band_disp="max"),
        )

        # Check if the calculated census cost is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(census["cost_volume"].sel(disp=-1), census_ground_truth_d1)
        np.testing.assert_array_equal(census["cost_volume"].sel(disp=0), census_ground_truth_d2)
        np.testing.assert_array_equal(census["cost_volume"].sel(disp=1), census_ground_truth_d3)

    def test_cmax(self):
        """
        Test the cmax attribute of the cost volume

        """
        # Add disparity on left image
        self.left.pipe(add_disparity, disparity=[-1, 1], window=None)

        # Test cmax for the census mesure
        matching_cost_matcher = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "census", "window_size": 3, "subpix": 1}
        )
        grid = matching_cost_matcher.allocate_cost_volume(
            self.left, (self.left["disparity"].sel(band_disp="min"), self.left["disparity"].sel(band_disp="max"))
        )
        census_cmax_w3 = matching_cost_matcher.compute_cost_volume(self.left, self.right, grid)
        # Check if the calculated maximal cost is equal to the ground truth
        np.testing.assert_array_equal(census_cmax_w3.attrs["cmax"], 9)
        assert np.nanmax(census_cmax_w3["cost_volume"].data) <= 9

        matching_cost_matcher = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "census", "window_size": 5, "subpix": 1}
        )
        grid = matching_cost_matcher.allocate_cost_volume(
            self.left, (self.left["disparity"].sel(band_disp="min"), self.left["disparity"].sel(band_disp="max"))
        )
        census_cmax_w5 = matching_cost_matcher.compute_cost_volume(self.left, self.right, grid)
        # Check if the calculated maximal cost is equal to the ground truth
        np.testing.assert_array_equal(census_cmax_w5.attrs["cmax"], 25)
        assert np.nanmax(census_cmax_w5["cost_volume"].data) <= 25

        # Test cmax for the sad mesure
        matching_cost_matcher = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 3, "subpix": 1}
        )
        grid = matching_cost_matcher.allocate_cost_volume(
            self.left, (self.left["disparity"].sel(band_disp="min"), self.left["disparity"].sel(band_disp="max"))
        )
        sad_cmax_w3 = matching_cost_matcher.compute_cost_volume(self.left, self.right, grid)
        # Check if the calculated maximal cost is equal to the ground truth
        np.testing.assert_array_equal(sad_cmax_w3.attrs["cmax"], int(abs(4 - 1) * (3**2)))
        assert np.nanmax(sad_cmax_w3["cost_volume"].data) <= int(abs(4 - 1) * (3**2))

        matching_cost_matcher = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 5, "subpix": 1}
        )
        grid = matching_cost_matcher.allocate_cost_volume(
            self.left, (self.left["disparity"].sel(band_disp="min"), self.left["disparity"].sel(band_disp="max"))
        )
        sad_cmax_w5 = matching_cost_matcher.compute_cost_volume(self.left, self.right, grid)
        # Check if the calculated maximal cost is equal to the ground truth
        np.testing.assert_array_equal(sad_cmax_w5.attrs["cmax"], int(abs(4 - 1) * (5**2)))
        assert np.nanmax(sad_cmax_w3["cost_volume"].data) <= int(abs(4 - 1) * (5**2))

        # Test cmax for the ssd mesure
        matching_cost_matcher = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "ssd", "window_size": 3, "subpix": 1}
        )
        grid = matching_cost_matcher.allocate_cost_volume(
            self.left, (self.left["disparity"].sel(band_disp="min"), self.left["disparity"].sel(band_disp="max"))
        )
        ssd_cmax_w3 = matching_cost_matcher.compute_cost_volume(self.left, self.right, grid)
        # Check if the calculated maximal cost is equal to the ground truth
        np.testing.assert_array_equal(ssd_cmax_w3.attrs["cmax"], int(abs(4 - 1) ** 2 * (3**2)))
        assert np.nanmax(sad_cmax_w3["cost_volume"].data) <= int(abs(4 - 1) ** 2 * (3**2))

        matching_cost_matcher = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "ssd", "window_size": 5, "subpix": 1}
        )
        grid = matching_cost_matcher.allocate_cost_volume(
            self.left, (self.left["disparity"].sel(band_disp="min"), self.left["disparity"].sel(band_disp="max"))
        )
        ssd_cmax_w5 = matching_cost_matcher.compute_cost_volume(self.left, self.right, grid)
        # Check if the calculated maximal cost is equal to the ground truth
        np.testing.assert_array_equal(ssd_cmax_w5.attrs["cmax"], int(abs(4 - 1) ** 2 * (5**2)))
        assert np.nanmax(sad_cmax_w3["cost_volume"].data) <= int(abs(4 - 1) ** 2 * (5**2))

        # Test cmax for the zncc mesure
        matching_cost_matcher = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "zncc", "window_size": 3, "subpix": 1}
        )
        grid = matching_cost_matcher.allocate_cost_volume(
            self.left, (self.left["disparity"].sel(band_disp="min"), self.left["disparity"].sel(band_disp="max"))
        )
        zncc_cmax = matching_cost_matcher.compute_cost_volume(self.left, self.right, grid)
        # Check if the calculated maximal cost is equal to the ground truth
        np.testing.assert_array_equal(zncc_cmax.attrs["cmax"], 1)
        assert np.nanmax(zncc_cmax["cost_volume"].data) <= 1

    def test_get_min_max_from_grid(self):
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
        compute_fixed_disp = matching_cost_matcher.get_min_max_from_grid(dmin_int, dmax_int)
        self.assertEqual(gt_fixed_disp, compute_fixed_disp)

        # Case with dmin is a fixed disparity and dmax is a variable disparity
        gt_fixed_var_disp = (-2, 25)
        compute_fixed_var_disp = matching_cost_matcher.get_min_max_from_grid(dmin_int, dmax_grid)
        self.assertEqual(gt_fixed_var_disp, compute_fixed_var_disp)

        # Case with dmin is a variable disparity and dmax is a fixed disparity
        gt_var_fixed_disp = (-5, 20)
        compute_var_fixed_disp = matching_cost_matcher.get_min_max_from_grid(dmin_grid, dmax_int)
        self.assertEqual(gt_var_fixed_disp, compute_var_fixed_disp)

        # Case with dmin and dmax are variable disparities
        gt_variable_disp = (-5, 25)
        compute_var_disp = matching_cost_matcher.get_min_max_from_grid(dmin_grid, dmax_grid)
        self.assertEqual(gt_variable_disp, compute_var_disp)

    @staticmethod
    def test_check_band_census():
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
            **{"matching_cost_method": "census", "window_size": 3, "subpix": 1, "band": "b"}
        )

        grid = matching_cost_.allocate_cost_volume(
            left, (left["disparity"].sel(band_disp="min"), left["disparity"].sel(band_disp="max"))
        )

        # Compute the cost_volume
        with pytest.raises(AttributeError, match="Wrong band instantiate : b not in img_left or img_right"):
            _ = matching_cost_.compute_cost_volume(left, right, grid)

        # Initialization of matching_cost plugin with no band
        matching_cost_ = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "census", "window_size": 3, "subpix": 1}
        )

        grid = matching_cost_.allocate_cost_volume(
            left, (left["disparity"].sel(band_disp="min"), left["disparity"].sel(band_disp="max"))
        )
        # Compute the cost_volume
        with pytest.raises(AttributeError, match="Band must be instantiated in matching cost step"):
            _ = matching_cost_.compute_cost_volume(left, right, grid)

    @staticmethod
    def test_instantiate_band_with_monoband():
        """
        Test the error when user instantiate band in matching_cost step with a monoband data
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

        left.attrs = common.img_attrs
        left.pipe(add_disparity, disparity=[-1, 1], window=None)

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

        right.attrs = common.img_attrs

        # Initialization of matching_cost plugin with a band
        matching_cost_ = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "census", "window_size": 3, "subpix": 1, "band": "r"}
        )

        grid = matching_cost_.allocate_cost_volume(
            left, (left["disparity"].sel(band_disp="min"), left["disparity"].sel(band_disp="max"))
        )
        # Compute the cost_volume
        with pytest.raises(AttributeError, match="Right dataset is monoband: r band cannot be selected"):
            _ = matching_cost_.compute_cost_volume(left, right, grid)

    @staticmethod
    def test_census_window_sizes():
        """
        Test the census method with bigger window sizes

        """

        def test_census(left_data, right_data, ref_out, window_size):

            left = xr.Dataset(
                {"im": (["row", "col"], left_data)},
                coords={"row": np.arange(left_data.shape[0]), "col": np.arange(left_data.shape[1])},
            )
            left.attrs["crs"] = None
            left.attrs["transform"] = Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
            left.pipe(add_disparity, disparity=[0, 0], window=None)

            right = xr.Dataset(
                {"im": (["row", "col"], right_data)},
                coords={"row": np.arange(right_data.shape[0]), "col": np.arange(right_data.shape[1])},
            )
            right.attrs["crs"] = None
            right.attrs["transform"] = Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

            # Computes the census transform for the images
            matching_cost_matcher = matching_cost.AbstractMatchingCost(
                **{"matching_cost_method": "census", "window_size": window_size, "subpix": 1}
            )

            grid = matching_cost_matcher.allocate_cost_volume(
                left, (left["disparity"].sel(band_disp="min"), left["disparity"].sel(band_disp="max"))
            )

            # Compute validity mask
            grid = validity_mask(left, right, grid)

            census = matching_cost_matcher.compute_cost_volume(img_left=left, img_right=right, cost_volume=grid)
            matching_cost_matcher.cv_masked(
                left,
                right,
                census,
                left["disparity"].sel(band_disp="min"),
                left["disparity"].sel(band_disp="max"),
            )

            # Check if the calculated census cost is equal to the ground truth (same shape and all elements equals)
            np.testing.assert_array_equal(census["cost_volume"].sel(disp=0), ref_out)

        n = np.nan

        # Test for window size 3
        test_census(
            np.array(([2, 0, 2], [0, 1, 0], [2, 0, 2]), dtype=np.float64),
            np.array(([0, 2, 0], [2, 1, 2], [0, 2, 0]), dtype=np.float64),
            np.array(([n, n, n], [n, 8, n], [n, n, n])),
            3,
        )
        # Test a cost of 0
        test_census(
            np.array(([2, 0, 2], [0, 1, 0], [2, 0, 2]), dtype=np.float64),
            np.array(([2, 0, 2], [0, 1, 0], [2, 0, 2]), dtype=np.float64),
            np.array(([n, n, n], [n, 0, n], [n, n, n])),
            3,
        )

        # Test for window size 5
        test_census(
            np.array(
                ([0, 2, 2, 0, 2], [2, 0, 2, 0, 2], [2, 0, 1, 2, 0], [0, 2, 0, 0, 0], [0, 2, 2, 2, 0]), dtype=np.float64
            ),
            np.array(
                ([0, 2, 2, 2, 2], [0, 0, 0, 0, 2], [2, 0, 1, 2, 0], [0, 2, 2, 0, 2], [0, 0, 2, 2, 0]), dtype=np.float64
            ),
            np.array(([n, n, n, n, n], [n, n, n, n, n], [n, n, 6, n, n], [n, n, n, n, n], [n, n, n, n, n])),
            5,
        )

        # Test for window size 7
        test_census(
            np.array(
                (
                    [2, 2, 0, 2, 2, 0, 0],
                    [0, 2, 2, 2, 0, 2, 0],
                    [2, 0, 2, 0, 2, 2, 2],
                    [2, 2, 0, 1, 2, 0, 2],
                    [2, 0, 2, 2, 0, 2, 2],
                    [0, 2, 0, 2, 2, 0, 2],
                    [2, 0, 2, 0, 0, 2, 2],
                ),
                dtype=np.float64,
            ),
            np.array(
                (
                    [0, 2, 2, 0, 2, 2, 0],
                    [0, 2, 2, 2, 0, 2, 0],
                    [0, 2, 2, 2, 0, 0, 0],
                    [2, 2, 0, 1, 2, 0, 2],
                    [2, 0, 2, 2, 0, 0, 0],
                    [0, 2, 2, 0, 0, 2, 0],
                    [2, 2, 0, 2, 0, 2, 0],
                ),
                dtype=np.float64,
            ),
            np.array(
                (
                    [n, n, n, n, n, n, n],
                    [n, n, n, n, n, n, n],
                    [n, n, n, n, n, n, n],
                    [n, n, n, 21, n, n, n],
                    [n, n, n, n, n, n, n],
                    [n, n, n, n, n, n, n],
                    [n, n, n, n, n, n, n],
                )
            ),
            7,
        )

        # Test for window size 9
        test_census(
            np.array(
                (
                    [2, 2, 0, 2, 2, 0, 0, 2, 2],
                    [0, 2, 2, 2, 0, 0, 0, 0, 0],
                    [2, 0, 2, 0, 2, 2, 2, 0, 2],
                    [2, 2, 0, 2, 2, 0, 2, 2, 2],
                    [2, 0, 2, 2, 1, 2, 2, 0, 2],
                    [0, 2, 0, 2, 2, 0, 2, 2, 0],
                    [2, 0, 2, 0, 0, 2, 2, 0, 2],
                    [2, 2, 2, 2, 0, 0, 2, 0, 2],
                    [0, 2, 2, 0, 2, 2, 2, 2, 0],
                ),
                dtype=np.float64,
            ),
            np.array(
                (
                    [2, 0, 0, 2, 2, 2, 0, 0, 2],
                    [2, 0, 0, 0, 2, 2, 2, 2, 2],
                    [2, 2, 2, 0, 2, 0, 2, 2, 2],
                    [2, 0, 0, 2, 2, 2, 2, 0, 2],
                    [2, 2, 2, 2, 1, 0, 2, 2, 2],
                    [0, 0, 0, 2, 2, 2, 2, 0, 0],
                    [2, 2, 2, 0, 0, 0, 2, 2, 2],
                    [2, 0, 2, 2, 0, 2, 2, 2, 2],
                    [0, 0, 2, 0, 2, 0, 2, 0, 0],
                ),
                dtype=np.float64,
            ),
            np.array(
                (
                    [n, n, n, n, n, n, n, n, n],
                    [n, n, n, n, n, n, n, n, n],
                    [n, n, n, n, n, n, n, n, n],
                    [n, n, n, n, n, n, n, n, n],
                    [n, n, n, n, 33, n, n, n, n],
                    [n, n, n, n, n, n, n, n, n],
                    [n, n, n, n, n, n, n, n, n],
                    [n, n, n, n, n, n, n, n, n],
                    [n, n, n, n, n, n, n, n, n],
                )
            ),
            9,
        )

        # Test for window size 11
        test_census(
            np.array(
                (
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                    [2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2],
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                ),
                dtype=np.float64,
            ),
            np.array(
                (
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ),
                dtype=np.float64,
            ),
            np.array(
                (
                    [n, n, n, n, n, n, n, n, n, n, n],
                    [n, n, n, n, n, n, n, n, n, n, n],
                    [n, n, n, n, n, n, n, n, n, n, n],
                    [n, n, n, n, n, n, n, n, n, n, n],
                    [n, n, n, n, n, n, n, n, n, n, n],
                    [n, n, n, n, n, 120, n, n, n, n, n],
                    [n, n, n, n, n, n, n, n, n, n, n],
                    [n, n, n, n, n, n, n, n, n, n, n],
                    [n, n, n, n, n, n, n, n, n, n, n],
                    [n, n, n, n, n, n, n, n, n, n, n],
                    [n, n, n, n, n, n, n, n, n, n, n],
                )
            ),
            11,
        )

        # Test for window size 13
        test_census(
            np.array(
                (
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ),
                dtype=np.float64,
            ),
            np.array(
                (
                    [2, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0],
                    [0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0],
                    [0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 2],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
                    [0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 0, 0, 2],
                    [0, 0, 0, 0, 2, 0, 0, 2, 0, 2, 2, 2, 2],
                    [0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 2],
                    [2, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0],
                ),
                dtype=np.float64,
            ),
            np.array(
                (
                    [n, n, n, n, n, n, n, n, n, n, n, n, n],
                    [n, n, n, n, n, n, n, n, n, n, n, n, n],
                    [n, n, n, n, n, n, n, n, n, n, n, n, n],
                    [n, n, n, n, n, n, n, n, n, n, n, n, n],
                    [n, n, n, n, n, n, n, n, n, n, n, n, n],
                    [n, n, n, n, n, n, n, n, n, n, n, n, n],
                    [n, n, n, n, n, n, 49, n, n, n, n, n, n],
                    [n, n, n, n, n, n, n, n, n, n, n, n, n],
                    [n, n, n, n, n, n, n, n, n, n, n, n, n],
                    [n, n, n, n, n, n, n, n, n, n, n, n, n],
                    [n, n, n, n, n, n, n, n, n, n, n, n, n],
                    [n, n, n, n, n, n, n, n, n, n, n, n, n],
                    [n, n, n, n, n, n, n, n, n, n, n, n, n],
                )
            ),
            13,
        )


if __name__ == "__main__":
    common.setup_logging()
    unittest.main()
