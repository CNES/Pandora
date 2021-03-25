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
This module contains functions to test the cost volume aggregation step.
"""

import unittest

import numpy as np
import xarray as xr

from rasterio import Affine

import tests.common as common
import pandora.aggregation as aggregation
import pandora.aggregation.cbca as cbca
import pandora.matching_cost as matching_cost


class TestAggregation(unittest.TestCase):
    """
    TestAggregation class allows to test all the methods in the class Aggregation,
    and the plugins
    """

    def setUp(self):
        """
        Method called to prepare the test fixture

        """
        data = np.array(([[5, 1, 15, 7, 3], [10, 9, 11, 9, 6], [1, 18, 4, 5, 9]]), dtype=np.float32)
        self.left = xr.Dataset(
            {"im": (["row", "col"], data)}, coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])}
        )
        self.left.attrs = {
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

        data = np.array(([[1, 5, 1, 15, 7], [2, 10, 9, 11, 9], [3, 1, 18, 4, 5]]), dtype=np.float32)
        self.right = xr.Dataset(
            {"im": (["row", "col"], data)}, coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])}
        )
        self.right.attrs = {
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

        # Create the matching cost for the images self.left and self.right, with disp = [-1, 0, 1] and SAD measuress
        row = np.arange(3)
        col = np.arange(5)
        disp = np.array([-1, 0, 1])
        self.cv = xr.Dataset(
            {"cost_volume": (["row", "col", "disp"], np.zeros((3, 5, 3), dtype=np.float32) + np.nan)},
            coords={"row": row, "col": col, "disp": disp},
        )
        self.cv["cost_volume"].loc[:, 1:, -1] = abs(self.left["im"].data[:, 1:] - self.right["im"].data[:, :4])
        self.cv["cost_volume"].loc[:, :, 0] = abs(self.left["im"].data - self.right["im"].data)
        self.cv["cost_volume"].loc[:, :3, 1] = abs(self.left["im"].data[:, :4] - self.right["im"].data[:, 1:])

        self.cv.attrs = {"measure": "sad", "subpixel": 1, "offset_row_col": 0, "cmax": 18}

    def test_compute_cbca_subpixel(self):
        """
        Test the cross-based cost aggregation method with subpixel precision

        """
        matching_cost_matcher = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 1, "subpix": 2}
        )
        sad = matching_cost_matcher.compute_cost_volume(
            img_left=self.left, img_right=self.right, disp_min=-1, disp_max=1
        )
        matching_cost_matcher.cv_masked(self.left, self.right, sad, -1, 1)

        # Computes the cost aggregation with the cross-based cost aggregation method,
        # with cbca_intensity=5 and cbca_distance=3
        cbca_obj = aggregation.AbstractAggregation(
            **{"aggregation_method": "cbca", "cbca_intensity": 5.0, "cbca_distance": 3}
        )

        cbca_obj.cost_volume_aggregation(self.left, self.right, sad)

        # Aggregate cost volume ground truth with the cross-based cost aggregation method for the stereo image
        aggregated_ground_truth = np.array(
            [
                [
                    [np.nan, np.nan, (4 + 4 + 8 + 1) / 4, (2 + 2 + 4 + 0.5 + 1) / 5, 0.0],
                    [
                        (0 + 7 + 10 + 1) / 4,
                        (2 + 12 + 3 + 1.5 + 1) / 5,
                        (4 + 4 + 14 + 8 + 1 + 2) / 6,
                        (2 + 2 + 7 + 4 + 0.5 + 1 + 1) / 7,
                        0.0,
                    ],
                    [
                        (0 + 10 + 6 + 7 + 1 + 0) / 6,
                        (2 + 12 + 1 + 3 + 1.5 + 1 + 4) / 7,
                        (14 + 4 + 8 + 1 + 2 + 2 + 3) / 7,
                        (2 + 7 + 4 + 4 + 0.5 + 1 + 1) / 7,
                        0.0,
                    ],
                    [
                        (10 + 6 + 12 + 1 + 0 + 5) / 6,
                        (12 + 1 + 8 + 3 + 1.5 + 1 + 4 + 6 + 5.5 + 4.5) / 10,
                        (14 + 8 + 4 + 2 + 2 + 3) / 6,
                        (7 + 4 + 0.5 + 1 + 1) / 5,
                        0.0,
                    ],
                    [(6 + 12 + 0 + 5) / 4, (1 + 8 + 1.5 + 1 + 4) / 5, (8 + 4 + 2 + 3 + 2) / 5, np.nan, np.nan],
                ],
                [
                    [np.nan, np.nan, (4 + 4 + 8 + 1 + 2 + 17) / 6, (2 + 2 + 4 + 0.5 + 1 + 1 + 8.5) / 7, 0.0],
                    [
                        (0 + 10 + 7 + 1 + 15 + 3) / 6,
                        (2 + 12 + 3 + 1.5 + 1 + 16 + 5.5) / 7,
                        (4 + 4 + 14 + 8 + 1 + 2 + 2 + 17 + 14) / 9,
                        (2 + 2 + 7 + 4 + 0.5 + 1 + 1 + 1 + 8.5 + 7) / 10,
                        0.0,
                    ],
                    [
                        (0 + 10 + 6 + 7 + 1 + 0 + 15 + 3 + 13) / 9,
                        (2 + 12 + 1 + 3 + 1.5 + 1 + 4 + 16 + 5.5 + 6) / 10,
                        (4 + 14 + 8 + 1 + 2 + 2 + 3 + 17 + 14 + 1) / 10,
                        (2 + 7 + 4 + 4 + 0.5 + 1 + 1 + 8.5 + 7 + 0.5) / 10,
                        0.0,
                    ],
                    [
                        (10 + 6 + 12 + 1 + 0 + 5 + 3 + 13 + 5) / 9,
                        (12 + 1 + 8 + 3 + 1.5 + 1 + 4 + 5.5 + 6 + 4.5) / 10,
                        (14 + 8 + 4 + 2 + 2 + 3 + 14 + 1 + 4) / 9,
                        (7 + 4 + 0.5 + 1 + 1 + 7 + 0.5) / 7,
                        0.0,
                    ],
                    [
                        (6 + 12 + 0 + 5 + 13 + 5) / 6,
                        (1 + 8 + 1.5 + 1 + 4 + 6 + 4.5) / 7,
                        (2 + 8 + 4 + 2 + 3 + 1 + 4) / 7,
                        np.nan,
                        np.nan,
                    ],
                ],
                [
                    [np.nan, np.nan, (2 + 8 + 1 + 17) / 4, (4 + 0.5 + 1 + 1 + 8.5) / 5, 0.0],
                    [
                        (7 + 1 + 15 + 3) / 4,
                        (3 + 1.5 + 1 + 16 + 5.5) / 5,
                        (8 + 1 + 2 + 2 + 17 + 14) / 6,
                        (4 + 0.5 + 1 + 1 + 1 + 8.5 + 7) / 7,
                        0.0,
                    ],
                    [
                        (7 + 1 + 0 + 15 + 3 + 13) / 6,
                        (3 + 1.5 + 1 + 4 + 16 + 5.5 + 6) / 7,
                        (1 + 2 + 2 + 17 + 14 + 1 + 3) / 7,
                        (4 + 0.5 + 1 + 1 + 8.5 + 7 + 0.5) / 7,
                        0.0,
                    ],
                    [
                        (1 + 0 + 5 + 3 + 13 + 5) / 6,
                        (1 + 8 + 3 + 1.5 + 1 + 4 + 5.5 + 6 + 4.5 + 12) / 10,
                        (2 + 2 + 3 + 14 + 1 + 4) / 6,
                        (0.5 + 1 + 1 + 7 + 0.5) / 5,
                        0.0,
                    ],
                    [(0 + 5 + 13 + 5) / 4, (1.5 + 1 + 4 + 6 + 4.5) / 5, (2 + 2 + 3 + 1 + 4) / 5, np.nan, np.nan],
                ],
            ]
        )

        # Check if the calculated standard deviation is equal (upto the desired tolerance of 1e-07) to the ground truth
        np.testing.assert_allclose(sad["cost_volume"].data, aggregated_ground_truth, rtol=1e-07)

    def test_cross_support_region(self):
        """
        Test the method cross support region

        """
        # Computes cross support for the left image, with cbca_intensity=5 and cbca_distance=3
        csr_ = cbca.cross_support(self.left["im"].data, 3, 5.0)

        # Cross support region top arm ground truth for the left image self.left
        csr_ground_truth_top_arm = np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 2, 1]])
        # Check if the calculated Cross support region top arm is equal to the ground truth
        # (same shape and all elements equals)

        np.testing.assert_array_equal(csr_[:, :, 2], csr_ground_truth_top_arm)

        # Cross support region bottom arm ground truth for the left image self.left
        csr_ground_truth_bottom_arm = np.array([[1, 1, 1, 2, 1], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]])
        # Check if the calculated Cross support region bottom arm is equal to the ground truth
        # (same shape and all elements equals)
        np.testing.assert_array_equal(csr_[:, :, 3], csr_ground_truth_bottom_arm)

        # Cross support region left arm ground truth for the left image self.left
        csr_ground_truth_left_arm = np.array([[0, 1, 1, 1, 1], [0, 1, 2, 2, 1], [0, 1, 1, 1, 1]])
        # Check if the calculated Cross support region left arm is equal to the ground truth
        # (same shape and all elements equals)
        np.testing.assert_array_equal(csr_[:, :, 0], csr_ground_truth_left_arm)

        # Cross support region right arm ground truth for the left image self.left
        csr_ground_truth_right_arm = np.array([[1, 1, 1, 1, 0], [2, 2, 1, 1, 0], [1, 1, 1, 1, 0]])
        # Check if the calculated Cross support region right arm is equal to the ground truth
        # (same shape and all elements equals)
        np.testing.assert_array_equal(csr_[:, :, 1], csr_ground_truth_right_arm)

    def test_compute_cbca(self):
        """
        Test the cross-based cost aggregation method

        """
        # Computes the cost aggregation with the cross-based cost aggregation method,
        # with cbca_intensity=5 and cbca_distance=3
        cbca_obj = aggregation.AbstractAggregation(
            **{"aggregation_method": "cbca", "cbca_intensity": 5.0, "cbca_distance": 3}
        )
        cbca_obj.cost_volume_aggregation(self.left, self.right, self.cv)

        # Aggregate cost volume ground truth with the cross-based cost aggregation method for the stereo image

        aggregated_ground_truth = np.array(
            [
                [
                    [np.nan, (4 + 4 + 8 + 1) / 4, 0.0],
                    [(0 + 7 + 10 + 1) / 4, (4 + 4 + 14 + 8 + 1 + 2) / 6, 0.0],
                    [(0 + 10 + 6 + 7 + 1 + 0) / 6, (14 + 4 + 8 + 1 + 2 + 2 + 3) / 7, 0.0],
                    [(10 + 6 + 12 + 1 + 0 + 5) / 6, (14 + 8 + 4 + 2 + 2 + 3) / 6, 0.0],
                    [(6 + 12 + 0 + 5) / 4, (8 + 4 + 2 + 3 + 2) / 5, np.nan],
                ],
                [
                    [np.nan, (4 + 4 + 8 + 1 + 2 + 17) / 6, 0.0],
                    [(0 + 10 + 7 + 1 + 15 + 3) / 6, (4 + 4 + 14 + 8 + 1 + 2 + 2 + 17 + 14) / 9, 0.0],
                    [(0 + 10 + 6 + 7 + 1 + 0 + 15 + 3 + 13) / 9, (4 + 14 + 8 + 1 + 2 + 2 + 3 + 17 + 14 + 1) / 10, 0.0],
                    [(10 + 6 + 12 + 1 + 0 + 5 + 3 + 13 + 5) / 9, (14 + 8 + 4 + 2 + 2 + 3 + 14 + 1 + 4) / 9, 0.0],
                    [(6 + 12 + 0 + 5 + 13 + 5) / 6, (2 + 8 + 4 + 2 + 3 + 1 + 4) / 7, np.nan],
                ],
                [
                    [np.nan, (2 + 8 + 1 + 17) / 4, 0.0],
                    [(7 + 1 + 15 + 3) / 4, (8 + 1 + 2 + 2 + 17 + 14) / 6, 0.0],
                    [(7 + 1 + 0 + 15 + 3 + 13) / 6, (1 + 2 + 2 + 17 + 14 + 1 + 3) / 7, 0.0],
                    [(1 + 0 + 5 + 3 + 13 + 5) / 6, (2 + 2 + 3 + 14 + 1 + 4) / 6, 0.0],
                    [(0 + 5 + 13 + 5) / 4, (2 + 2 + 3 + 1 + 4) / 5, np.nan],
                ],
            ]
        )

        # Check if the calculated standard deviation is equal (upto the desired tolerance of 1e-07) to the ground truth
        np.testing.assert_allclose(self.cv["cost_volume"].data, aggregated_ground_truth, rtol=1e-07)

    def test_cmax(self):
        """
        Test the cmax attribute of the cost volume

        """
        cbca_obj = aggregation.AbstractAggregation(
            **{"aggregation_method": "cbca", "cbca_intensity": 5.0, "cbca_distance": 3}
        )

        cbca_obj.cost_volume_aggregation(self.left, self.right, self.cv)

        # Check if the calculated maximal cost is equal to the ground truth
        assert np.nanmax(self.cv["cost_volume"].data) <= (24 * 18)

    @staticmethod
    def test_compute_cbca_with_invalid_cost():
        """
        Test the cross-based cost aggregation method with invalid cost

        """
        # Invalid pixel in the left image
        # Mask convention
        # cfg['image']['valid_pixels'] = 0
        # cfg['image']['no_data'] = 1
        # --------------- Without invalid / nodata pixels ----------------
        data = np.array(([[5, 1, 15, 7, 3], [10, 9, 11, 9, 6], [1, 18, 4, 5, 9], [5, 1, 15, 7, 3]]), dtype=np.float32)
        mask = np.array(([[0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [3, 0, 0, 0, 0]]))
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

        data = np.array(([[1, 5, 1, 15, 7], [2, 10, 9, 11, 9], [3, 1, 18, 4, 5], [1, 5, 1, 15, 7]]), dtype=np.float32)
        mask = np.array(([[0, 0, 0, 0, 0], [0, 0, 5, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]))
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

        matching_cost_matcher = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 1, "subpix": 1}
        )
        sad = matching_cost_matcher.compute_cost_volume(img_left=left, img_right=right, disp_min=-1, disp_max=1)
        matching_cost_matcher.cv_masked(left, right, sad, -1, 1)

        cbca_obj = aggregation.AbstractAggregation(
            **{"aggregation_method": "cbca", "cbca_intensity": 5.0, "cbca_distance": 3}
        )

        cbca_obj.cost_volume_aggregation(left, right, sad, **{"valid_pixels": 0, "no_data": 1})

        # Aggregate cost volume ground truth with the cross-based cost aggregation method for the stereo image disp = 0
        aggregated_ground_truth = np.array(
            [
                [(4 + 8 + 1) / 3, np.nan, (14 + 8) / 2, (8 + 14 + 4) / 3, (4 + 8 + 3) / 3],
                [(8 + 4 + 1 + 2 + 17) / 5, (8 + 1 + 2 + 17 + 14) / 5, np.nan, np.nan, (8 + 4 + 3 + 4 + 4 + 8) / 6.0],
                [
                    (2 + 8 + 1 + 17) / 4,
                    (8 + 1 + 2 + 17 + 14 + 4 + 14) / 7,
                    (17 + 14 + 4 + 14 + 8) / 5,
                    np.nan,
                    (4 + 3 + 4 + 8) / 4,
                ],
                [np.nan, (4 + 2 + 17 + 14 + 14) / 5, (14 + 17 + 14 + 4 + 8) / 5, (14 + 8 + 4) / 3, (4 + 4 + 8) / 3],
            ]
        )

        # Check if the calculated aggregated cost volume is equal (upto the desired tolerance of 1e-07)
        # to the ground truth
        np.testing.assert_allclose(sad["cost_volume"].data[:, :, 1], aggregated_ground_truth, rtol=1e-07)

    @staticmethod
    def test_compute_cbca_with_offset():
        """
        Test the cross-based cost aggregation method when the window_size is > 1

        """
        data = np.array(([[5, 1, 15, 7, 3], [10, 9, 11, 9, 6], [1, 18, 4, 5, 9], [5, 1, 15, 7, 3]]), dtype=np.float32)
        left = xr.Dataset(
            {"im": (["row", "col"], data)}, coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])}
        )
        left.attrs = {
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

        data = np.array(([[1, 5, 1, 15, 7], [2, 10, 9, 11, 9], [3, 1, 18, 4, 5], [1, 5, 1, 15, 7]]), dtype=np.float32)
        right = xr.Dataset(
            {"im": (["row", "col"], data)}, coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])}
        )
        right.attrs = {
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

        matching_cost_matcher = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 3, "subpix": 1}
        )
        sad = matching_cost_matcher.compute_cost_volume(img_left=left, img_right=right, disp_min=-1, disp_max=1)
        matching_cost_matcher.cv_masked(left, right, sad, -1, 1)

        cbca_obj = aggregation.AbstractAggregation(
            **{"aggregation_method": "cbca", "cbca_intensity": 5.0, "cbca_distance": 3}
        )

        cbca_obj.cost_volume_aggregation(left, right, sad, **{"valid_pixels": 5, "no_data": 7})

        # Aggregate cost volume ground truth with the cross-based cost aggregation method for the stereo image
        aggregated_ground_truth = np.array(
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
                    [np.nan, (66.0 + 63 + 66 + 63) / 4, 0.0],
                    [55.0, (66 + 63 + 52 + 66 + 63 + 52) / 6, 0.0],
                    [55.0, (63 + 63 + 52 + 52) / 4, np.nan],
                    [np.nan, np.nan, np.nan],
                ],
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, (66.0 + 63 + 66 + 63) / 4, 0.0],
                    [55.0, (66 + 63 + 52 + 66 + 63 + 52) / 6, 0.0],
                    [55.0, (63 + 63 + 52 + 52) / 4, np.nan],
                    [np.nan, np.nan, np.nan],
                ],
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                ],
            ]
        )

        # Check if the calculated aggregated cost volume is equal (upto the desired tolerance of 1e-07)
        # to the ground truth
        np.testing.assert_allclose(sad["cost_volume"].data, aggregated_ground_truth, rtol=1e-07)

    @staticmethod
    def test_computes_cross_support():
        """
        Test the method computes_cross_support

        """
        # --------------- Without invalid / nodata pixels ----------------
        data = np.array(([[5, 1, 15, 7, 3], [10, 9, 11, 9, 6], [1, 18, 4, 5, 9]]), dtype=np.float32)
        left = xr.Dataset(
            {"im": (["row", "col"], data)}, coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])}
        )
        left.attrs = {
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

        data = np.array(([[1, 5, 1, 15, 7], [2, 10, 9, 11, 9], [3, 1, 18, 4, 5]]), dtype=np.float32)

        right = xr.Dataset(
            {"im": (["row", "col"], data)}, coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])}
        )
        right.attrs = {
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

        matching_cost_matcher = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 1, "subpix": 1}
        )
        sad = matching_cost_matcher.compute_cost_volume(img_left=left, img_right=right, disp_min=-1, disp_max=1)
        matching_cost_matcher.cv_masked(left, right, sad, -1, 1)

        cbca_obj = aggregation.AbstractAggregation(
            **{"aggregation_method": "cbca", "cbca_intensity": 5.0, "cbca_distance": 3}
        )
        cross_left, cross_right = cbca_obj.computes_cross_supports(left, right, sad)

        # Cross support region top arm ground truth for the left
        top_arm = np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 2, 1]])

        # Cross support region bottom arm ground truth for the left image
        bottom_arm = np.array([[1, 1, 1, 2, 1], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]])

        # Cross support region left arm ground truth for the left image
        left_arm = np.array([[0, 1, 1, 1, 1], [0, 1, 2, 2, 2], [0, 1, 1, 1, 1]])

        # Cross support region right arm ground truth for the left image
        right_arm = np.array([[1, 1, 1, 1, 0], [2, 2, 2, 1, 0], [1, 1, 1, 1, 0]])

        gt_left_arms = np.stack((left_arm, right_arm, top_arm, bottom_arm), axis=-1)
        # Check if the calculated left cross support region is equal to the ground truth
        # (same shape and all elements equals)
        np.testing.assert_array_equal(cross_left, gt_left_arms)

        # Cross support region top arm ground truth for the right image
        top_arm = np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [2, 2, 1, 1, 2]])

        # Cross support region bottom arm ground truth for the right image
        bottom_arm = np.array([[2, 2, 1, 1, 2], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]])

        # Cross support region left arm ground truth for the right image
        left_arm = np.array([[0, 1, 2, 1, 1], [0, 1, 1, 1, 2], [0, 1, 1, 1, 1]])

        # Cross support region right arm ground truth for the right image
        right_arm = np.array([[2, 1, 1, 1, 0], [1, 1, 2, 1, 0], [1, 1, 1, 1, 0]])

        gt_right_arms = np.stack((left_arm, right_arm, top_arm, bottom_arm), axis=-1)

        # Check if the calculated right cross support region is equal to the ground truth
        # (same shape and all elements equals)
        np.testing.assert_array_equal(cross_right[0], gt_right_arms)
        # No subpixel precision
        assert len(cross_right) == 1

        # --------------- With invalid / nodata pixels ----------------
        data = np.array(([[5, 1, 15, 7, 3], [10, 9, 11, 9, 6], [1, 18, 4, 5, 9]]), dtype=np.float32)
        mask = np.array(([[2, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 3, 0, 0, 0]]))
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

        data = np.array(([[1, 5, 1, 15, 7], [2, 10, 9, 11, 9], [3, 1, 18, 4, 5]]), dtype=np.float32)
        mask = np.array(([[0, 0, 0, 0, 0], [0, 1, 0, 3, 0], [0, 0, 0, 0, 0]]))
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

        matching_cost_matcher = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 1, "subpix": 1}
        )
        sad = matching_cost_matcher.compute_cost_volume(img_left=left, img_right=right, disp_min=-1, disp_max=1)
        matching_cost_matcher.cv_masked(left, right, sad, -1, 1)

        cbca_obj = aggregation.AbstractAggregation(
            **{"aggregation_method": "cbca", "cbca_intensity": 6.0, "cbca_distance": 3}
        )
        cross_left, cross_right = cbca_obj.computes_cross_supports(left, right, sad)

        # Cross support region top arm ground truth for the left
        top_arm = np.array([[0, 0, 0, 0, 0], [0, 1, 1, 0, 1], [1, 0, 1, 0, 1]])

        # Cross support region bottom arm ground truth for the left image
        bottom_arm = np.array([[0, 1, 1, 0, 1], [1, 0, 1, 0, 1], [0, 0, 0, 0, 0]])

        # Cross support region left arm ground truth for the left image
        left_arm = np.array([[0, 0, 1, 1, 1], [0, 1, 2, 0, 0], [0, 0, 0, 1, 2]])

        # Cross support region right arm ground truth for the left image
        right_arm = np.array([[0, 1, 1, 1, 0], [2, 1, 0, 0, 0], [0, 0, 2, 1, 0]])

        gt_left_arms = np.stack((left_arm, right_arm, top_arm, bottom_arm), axis=-1)

        # Check if the calculated left cross support region is equal to the ground truth
        # (same shape and all elements equals)
        np.testing.assert_array_equal(cross_left, gt_left_arms)

        # Cross support region top arm ground truth for the right
        top_arm = np.array([[0, 0, 0, 0, 0], [1, 0, 1, 0, 1], [2, 0, 1, 0, 2]])

        # Cross support region bottom arm ground truth for the right image
        bottom_arm = np.array([[2, 0, 1, 0, 2], [1, 0, 1, 0, 1], [0, 0, 0, 0, 0]])

        # Cross support region left arm ground truth for the right image
        left_arm = np.array([[0, 1, 2, 1, 1], [0, 0, 0, 0, 0], [0, 1, 1, 1, 1]])

        # Cross support region right arm ground truth for the right image
        right_arm = np.array([[2, 1, 1, 1, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 0]])

        gt_right_arms = np.stack((left_arm, right_arm, top_arm, bottom_arm), axis=-1)

        # Check if the calculated right cross support region is equal to the ground truth
        # (same shape and all elements equals)
        np.testing.assert_array_equal(cross_right[0], gt_right_arms)
        # No subpixel precision
        assert len(cross_right) == 1

    @staticmethod
    def test_computes_cross_support_with_subpixel():
        """
        Test the method computes_cross_support with subpixel precision

        """
        # --------------- Without invalid / nodata pixels ----------------
        data = np.array(([[5, 1, 15, 7, 3], [10, 9, 11, 9, 6], [1, 18, 4, 5, 9]]), dtype=np.float32)
        left = xr.Dataset(
            {"im": (["row", "col"], data)}, coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])}
        )
        left.attrs = {
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

        data = np.array(([[1, 5, 1, 15, 7], [2, 10, 9, 11, 9], [3, 1, 18, 4, 5]]), dtype=np.float32)

        right = xr.Dataset(
            {"im": (["row", "col"], data)}, coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])}
        )
        right.attrs["valid_pixels"] = 0
        right.attrs["no_data"] = 1
        right.attrs["crs"] = None
        right.attrs["transform"] = Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

        matching_cost_matcher = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 1, "subpix": 2}
        )
        sad = matching_cost_matcher.compute_cost_volume(img_left=left, img_right=right, disp_min=-1, disp_max=1)
        matching_cost_matcher.cv_masked(left, right, sad, -1, 1)

        cbca_obj = aggregation.AbstractAggregation(
            **{"aggregation_method": "cbca", "cbca_intensity": 5.0, "cbca_distance": 3}
        )
        cross_left, cross_right = cbca_obj.computes_cross_supports(left, right, sad)  # pylint: disable=unused-variable

        # Cross support region top arm ground truth for the right shifted image
        top_arm = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 1, 2, 1]])

        # Cross support region bottom arm ground truth for the right shifted image
        bottom_arm = np.array([[2, 1, 2, 1], [1, 1, 1, 1], [0, 0, 0, 0]])

        # Cross support region left arm ground truth for the right shifted image
        left_arm = np.array([[0, 1, 1, 1], [0, 1, 2, 2], [0, 1, 1, 1]])

        # Cross support region right arm ground truth for the right shifted image
        right_arm = np.array([[1, 1, 1, 0], [2, 2, 1, 0], [1, 1, 1, 0]])

        gt_right_arms = np.stack((left_arm, right_arm, top_arm, bottom_arm), axis=-1)

        # Check if the calculated right cross support region is equal to the ground truth
        # (same shape and all elements equals)
        np.testing.assert_array_equal(cross_right[1], gt_right_arms)

        # --------------- With invalid / nodata pixels ----------------
        data = np.array(([[5, 1, 15, 7, 3], [10, 9, 11, 9, 6], [1, 18, 4, 5, 9]]), dtype=np.float32)
        mask = np.array(([[0, 0, 0, 0, 0], [0, 1, 0, 3, 0], [0, 0, 0, 0, 0]]))
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

        data = np.array(([[1, 5, 1, 15, 7], [2, 10, 9, 11, 9], [3, 1, 18, 4, 5]]), dtype=np.float32)
        mask = np.array(([[2, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 3, 0, 0, 0]]))

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

        matching_cost_matcher = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 1, "subpix": 2}
        )
        sad = matching_cost_matcher.compute_cost_volume(img_left=left, img_right=right, disp_min=-1, disp_max=1)
        matching_cost_matcher.cv_masked(left, right, sad, -1, 1)

        cbca_obj = aggregation.AbstractAggregation(
            **{"aggregation_method": "cbca", "cbca_intensity": 6.0, "cbca_distance": 3}
        )
        cross_left, cross_right = cbca_obj.computes_cross_supports(left, right, sad)

        # Cross support region top arm ground truth for the right shifted image
        top_arm = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]])

        # Cross support region bottom arm ground truth for the right shifted image
        bottom_arm = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        # Cross support region left arm ground truth for the right shifted image
        left_arm = np.array([[0, 0, 1, 1], [0, 1, 0, 0], [0, 0, 0, 1]])

        # Cross support region right arm ground truth for the right shifted image
        right_arm = np.array([[0, 1, 1, 0], [1, 0, 0, 0], [0, 0, 1, 0]])

        gt_right_arms = np.stack((left_arm, right_arm, top_arm, bottom_arm), axis=-1)

        # Check if the calculated right cross support region is equal to the ground truth
        # (same shape and all elements equals)
        np.testing.assert_array_equal(cross_right[1], gt_right_arms)

    @staticmethod
    def test_computes_cross_support_with_offset():
        """
        Test the method computes_cross_support with window_size > 1

        """
        # --------------- Without invalid / nodata pixels ----------------
        data = np.array(([[5, 1, 15, 7, 3], [10, 9, 11, 9, 6], [1, 18, 4, 5, 9], [5, 1, 15, 7, 3]]), dtype=np.float32)
        left = xr.Dataset(
            {"im": (["row", "col"], data)}, coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])}
        )
        left.attrs = {
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }

        data = np.array(([[1, 5, 1, 15, 7], [2, 10, 9, 11, 9], [3, 1, 18, 4, 5], [1, 5, 1, 15, 7]]), dtype=np.float32)

        right = xr.Dataset(
            {"im": (["row", "col"], data)}, coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])}
        )
        right.attrs["valid_pixels"] = 0
        right.attrs["no_data"] = 1
        right.attrs["crs"] = None
        right.attrs["transform"] = Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

        matching_cost_matcher = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 3, "subpix": 1}
        )
        sad = matching_cost_matcher.compute_cost_volume(img_left=left, img_right=right, disp_min=-1, disp_max=1)
        matching_cost_matcher.cv_masked(left, right, sad, -1, 1)

        cbca_obj = aggregation.AbstractAggregation(
            **{"aggregation_method": "cbca", "cbca_intensity": 5.0, "cbca_distance": 3}
        )
        cross_left, cross_right = cbca_obj.computes_cross_supports(left, right, sad)

        # Cross support region top arm ground truth for the right shifted image
        top_arm = np.array([[0, 0, 0], [1, 1, 1]])

        # Cross support region bottom arm ground truth for the right shifted image
        bottom_arm = np.array([[1, 1, 1], [0, 0, 0]])

        # Cross support region left arm ground truth for the right shifted image
        left_arm = np.array([[0, 1, 2], [0, 1, 2]])

        # Cross support region right arm ground truth for the right shifted image
        right_arm = np.array([[2, 1, 0], [2, 1, 0]])

        gt_left_arms = np.stack((left_arm, right_arm, top_arm, bottom_arm), axis=-1)
        # Check if the calculated left cross support region is equal to the ground truth
        # (same shape and all elements equals)
        np.testing.assert_array_equal(cross_left, gt_left_arms)

        # Cross support region top arm ground truth for the right image
        top_arm = np.array([[0, 0, 0], [1, 1, 1]])

        # Cross support region bottom arm ground truth for the right image
        bottom_arm = np.array([[1, 1, 1], [0, 0, 0]])

        # Cross support region left arm ground truth for the right image
        left_arm = np.array([[0, 1, 1], [0, 1, 1]])

        # Cross support region right arm ground truth for the right image
        right_arm = np.array([[1, 1, 0], [1, 1, 0]])

        gt_right_arms = np.stack((left_arm, right_arm, top_arm, bottom_arm), axis=-1)

        # Check if the calculated right cross support region is equal to the ground truth
        # (same shape and all elements equals)
        np.testing.assert_array_equal(cross_right[0], gt_right_arms)
        # No subpixel precision
        assert len(cross_right) == 1


if __name__ == "__main__":
    common.setup_logging()
    unittest.main()
