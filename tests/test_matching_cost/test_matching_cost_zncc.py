# pylint: disable=duplicate-code
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
This module contains functions to test the Zncc matching cost step.
"""

# pylint: disable=redefined-outer-name

import unittest
import pytest

import numpy as np
import xarray as xr

from pandora import matching_cost
from pandora.img_tools import add_disparity
from pandora.criteria import validity_mask
from tests import common


class TestMatchingCostZncc(unittest.TestCase):
    """
    TestMatchingCost class allows to test all the methods in the
    matching_cost Zncc class
    """

    def setUp(self):
        """
        Method called to prepare the test fixture

        """

        self.left, self.right = common.matching_cost_tests_setup()
        self.left.pipe(add_disparity, disparity=[-1, 1], window=None)

    def test_zncc_cost(self):
        """
        Test the zncc_cost method

        """
        # Compute the cost volume for the images self.left, self.right,
        # with zncc measure, disp = -1, 1 window size = 5 and subpix = 1
        matching_cost_matcher = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "zncc", "window_size": 5, "subpix": 1}
        )

        grid = matching_cost_matcher.allocate_cost_volume(
            self.left, (self.left["disparity"].sel(band_disp="min"), self.left["disparity"].sel(band_disp="max"))
        )

        # Compute validity mask
        grid = validity_mask(self.left, self.right, grid)

        cost_volume_zncc = matching_cost_matcher.compute_cost_volume(
            img_left=self.left, img_right=self.right, cost_volume=grid
        )
        matching_cost_matcher.cv_masked(
            self.left,
            self.right,
            cost_volume_zncc,
            self.left["disparity"].sel(band_disp="min"),
            self.left["disparity"].sel(band_disp="max"),
        )

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
        left.pipe(add_disparity, disparity=[-2, 2], window=None)

        data = np.array(([1, 5, 6, 3, 4], [2, 5, 10, 6, 9], [0, 7, 5, 3, 1]), dtype=np.float64)
        right = xr.Dataset(
            {"im": (["row", "col"], data)}, coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])}
        )
        right.attrs = common.img_attrs

        # Computes the cost volume for disp min -2 disp max 2 and subpix = 2
        matching_cost_matcher = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "sad", "window_size": 3, "subpix": 2}
        )

        grid = matching_cost_matcher.allocate_cost_volume(
            left, (left["disparity"].sel(band_disp="min"), left["disparity"].sel(band_disp="max"))
        )

        # Compute validity mask
        grid = validity_mask(left, right, grid)

        cv_zncc_subpixel = matching_cost_matcher.compute_cost_volume(img_left=left, img_right=right, cost_volume=grid)
        matching_cost_matcher.cv_masked(
            left,
            right,
            cv_zncc_subpixel,
            left["disparity"].sel(band_disp="min"),
            left["disparity"].sel(band_disp="max"),
        )

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
    def test_check_band_zncc():
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
            **{"matching_cost_method": "zncc", "window_size": 3, "subpix": 1, "band": "b"}
        )

        grid = matching_cost_.allocate_cost_volume(
            left, (left["disparity"].sel(band_disp="min"), left["disparity"].sel(band_disp="max"))
        )

        # Compute the cost_volume
        with pytest.raises(AttributeError, match="Wrong band instantiate : b not in img_left or img_right"):
            _ = matching_cost_.compute_cost_volume(img_left=left, img_right=right, cost_volume=grid)

        # Initialization of matching_cost plugin with no band
        matching_cost_ = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "zncc", "window_size": 3, "subpix": 1}
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


@pytest.fixture()
def images():
    return common.matching_cost_tests_setup()


@pytest.fixture()
def left(images):
    return images[0]


@pytest.fixture()
def right(images):
    return images[1]


@pytest.mark.parametrize(
    ["disp", "p_ground_truth_disp", "q_ground_truth_disp"],
    [
        pytest.param(
            0,  # for disparity = 0, the similarity measure will be applied over the whole images
            (0, 6),  # 6 = left["im"].shape[1]
            (0, 6),  # 6 = right["im"].shape[1]),
            id="Disparity=0",
        ),
        pytest.param(
            -2,
            # for disparity = -2, the similarity measure will be applied over the range
            #          row=2   row=6        row=0   row=4
            #           1 1 1 1             1 1 1 2
            #           1 1 2 1             1 1 1 4
            #           1 4 3 1             1 1 1 4
            #           1 1 1 1             1 1 1 1
            #           1 1 1 1             1 1 1 1
            (2, 6),
            (0, 4),
            id="Disparity=-2",
        ),
        pytest.param(
            2,
            # for disparity = -2, the similarity measure will be applied over the range
            #          row=0   row=4        row=2   row=6
            #           1 1 1 1             1 2 2 2
            #           1 1 1 1             1 4 2 4
            #           1 1 1 4             1 4 4 1
            #           1 1 1 1             1 1 1 1
            #           1 1 1 1             1 1 1 1
            (0, 4),
            (2, 6),
            id="Disparity=2",
        ),
        pytest.param(
            -2.5,  # Test for negative floating disparity
            (3, 6),
            (0, 4),
            id="Disparity=-2.5",
        ),
        pytest.param(
            2.5,  # Test for positive floating disparity
            (0, 3),
            (2, 6),
            id="Disparity=2.5",
        ),
        pytest.param(
            7,  # disparity = 7 outside the image
            (6, 6),
            (6, 6),
            id="Disparity=7",
        ),
        pytest.param(
            -7,  # disparity = -7 outside the image
            (6, 6),
            (6, 6),
            id="Disparity=-7",
        ),
        pytest.param(
            5,  # disparity = 5 --> abs(5) > nb_col - int(window_size/2)*2
            (6, 6),
            (6, 6),
            id="Disparity=5",
        ),
        pytest.param(
            -5,  # disparity = -5 --> abs(5) > nb_col - int(window_size/2)*2
            (6, 6),
            (6, 6),
            id="Disparity=-5",
        ),
    ],
)
def test_point_interval_zncc(disp, p_ground_truth_disp, q_ground_truth_disp, left, right):
    """
    Test the point interval method with zncc similarity measure
    """
    matching_cost_ = matching_cost.AbstractMatchingCost(
        **{"matching_cost_method": "zncc", "window_size": 3, "subpix": 1}
    )

    (point_p, point_q) = matching_cost_.point_interval(left, right, disp)

    # Check if the calculated range is equal to the ground truth
    np.testing.assert_array_equal(point_p, p_ground_truth_disp)
    np.testing.assert_array_equal(point_q, q_ground_truth_disp)
