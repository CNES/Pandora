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
This module contains functions to test the matching cost step.
"""
# pylint with pytest's fixtures compatibility:
# pylint: disable=redefined-outer-name
import numpy as np
import xarray as xr
import pytest

from pandora import matching_cost
from pandora.margins.descriptors import HalfWindowMargins

from tests import common


@pytest.fixture()
def images():
    return common.matching_cost_tests_setup()


@pytest.fixture()
def left(images):
    return images[0]


class TestMatchingCost:
    """
    TestMatchingCost class allows to test the methods in the class MatchingCost
    """

    def test_allocate_numpy_cost_volume(self, left):
        """
        Test the allocate_numpy_cost_volume function
        """

        # compute ground truth cv for allocate
        expected = np.full((5, 6, 5), np.nan, dtype=np.float32)

        # Create matching cost object
        matching_cost_ = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "census", "window_size": 3, "subpix": 1, "band": "b"}
        )

        # Function allocate_numpy_cost_volume
        result = matching_cost_.allocate_numpy_cost_volume(left, [0, 0.5, 1, 1.5, 2])

        # Test that cv and cv_cropped are equal to the ground truth
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        ["cost_volume", "expected", "offset"],
        [
            pytest.param(
                np.full((5, 6, 5), np.nan, dtype=np.float32),
                np.full((5, 6, 5), np.nan, dtype=np.float32),
                0,
                id="With null offset",
            ),
            pytest.param(
                np.full((5, 6, 5), np.nan, dtype=np.float32),
                np.full((5, 2, 1), np.nan, dtype=np.float32),
                2,
                id="With offset",
            ),
        ],
    )
    def test_crop_cost_volume(self, cost_volume, expected, offset):
        """
        Test the crop_cost_volume function
        """
        # Function allocate_numpy_cost_volume
        result = matching_cost.AbstractMatchingCost.crop_cost_volume(cost_volume, offset)  # type:ignore

        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        ["img_left", "user_cfg", "step", "col", "ground_truth_attrs"],
        [
            pytest.param(
                common.matching_cost_tests_setup()[0],
                {},
                1,  # step value
                np.arange(0, 6, 1),  # col coords left image
                {
                    "sampling_interval": 1,
                    "roi_origin": (0, 0),
                    "cv_origin": (0, 0),
                    "col_to_compute": np.array([0, 1, 2, 3, 4, 5]),
                },
                id="no ROI in user_configuration and step = 1",
            ),
            pytest.param(
                common.matching_cost_tests_setup()[0],
                {
                    "ROI": {
                        "col": {"first": 2, "last": 3},
                        "row": {"first": 2, "last": 3},
                        "margins": [2, 2, 2, 2],
                    }
                },
                2,  # step value
                np.arange(0, 6, 2),  # col coords left image
                {
                    "sampling_interval": 2,
                    "roi_origin": (2, 2),
                    "cv_origin": (2, 1),
                    "col_to_compute": np.array([0, 2, 4]),
                },
                id="ROI in user_configuration and margin % step == 0",
            ),
            pytest.param(
                common.matching_cost_tests_setup()[0],
                {
                    "ROI": {
                        "col": {"first": 2, "last": 3},
                        "row": {"first": 2, "last": 3},
                        "margins": [2, 2, 2, 2],
                    }
                },
                3,  # step value
                np.arange(0, 6, 3),  # col coords left image
                {
                    "sampling_interval": 3,
                    "roi_origin": (2, 2),
                    "cv_origin": (2, 0),
                    "col_to_compute": np.array([2, 5]),
                },
                id="ROI in user_configuration and margin % step != 0 with margin < step",
            ),
            pytest.param(
                xr.Dataset(
                    {"im": (["row", "col"], np.ones((8, 8), dtype=np.float64))},
                    coords={"row": np.arange(8), "col": np.arange(8)},
                ).assign_attrs(common.img_attrs),
                {
                    "ROI": {
                        "col": {"first": 3, "last": 3},
                        "row": {"first": 3, "last": 3},
                        "margins": [3, 3, 3, 3],
                    }
                },
                2,  # step value
                np.arange(0, 8, 2),  # col coords left image
                {
                    "sampling_interval": 2,
                    "roi_origin": (3, 3),
                    "cv_origin": (3, 1),
                    "col_to_compute": np.array([1, 3, 5, 7]),
                },
                id="ROI in user_configuration and margin % step != 0 with margin > step",
            ),
        ],
    )
    def test_grid_estimation(self, img_left, user_cfg, step, col, ground_truth_attrs):
        """
        Test the grid_estimation function
        """

        # Create matching cost object
        matching_cost_ = matching_cost.AbstractMatchingCost(**common.basic_pipeline_cfg["matching_cost"])

        # Update step for matching cost
        matching_cost_._step_col = step  # pylint: disable=protected-access

        # Grid estimation
        grid = matching_cost_.grid_estimation(img_left, user_cfg)

        # Create ground truth for output of grid_estimation() function
        c_row = img_left["im"].coords["row"]
        row = np.arange(c_row[0], c_row[-1] + 1)

        ground_truth = xr.Dataset(
            {},
            coords={"row": row, "col": col},
        )

        ground_truth.attrs = img_left.attrs
        ground_truth.attrs.update(ground_truth_attrs)

        print(f"{grid=}")

        xr.testing.assert_identical(grid, ground_truth)

    def test_allocate_cost_volume(self, left):
        """ "
        Test the allocate_cost_volume function
        """

        left.attrs["disparity_source"] = [-2, 2]

        # Create matching cost object
        matching_cost_ = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "census", "window_size": 5, "subpix": 4, "band": "b"}
        )

        dataset_cv = matching_cost_.allocate_costvolume(left, 4, -2, 2, 5, {"metadata": "metadata"})

        c_row = [0, 1, 2, 3, 4]
        c_col = [0, 1, 2, 3, 4, 5]

        # First pixel in the image that is fully computable (aggregation windows are complete)
        row = np.arange(c_row[0], c_row[-1] + 1)  # type: np.ndarray
        col = np.arange(c_col[0], c_col[-1] + 1)  # type: np.ndarray

        disparity_range = np.arange(-2, 2, step=1 / float(4), dtype=np.float64)
        disparity_range = np.append(disparity_range, [2])

        np_data = np.zeros((len(row), len(col), len(disparity_range)), dtype=np.float32)

        ground_truth = xr.Dataset(
            {"cost_volume": (["row", "col", "disp"], np_data)},
            coords={"row": row, "col": col, "disp": disparity_range},
        )

        ground_truth.attrs = {
            "crs": left.attrs["crs"],
            "transform": left.attrs["transform"],
            "window_size": 5,
        }

        assert ground_truth.equals(dataset_cv)

    def test_allocate_empty_cost_volume_with_step(self, left):
        """ "
        Test the allocate_cost_volume function
        """

        left.attrs["disparity_source"] = [-2, 2]

        # Create matching cost object
        matching_cost_ = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "census", "window_size": 5, "subpix": 4, "band": "b"}
        )
        matching_cost_._step_col = 2  # pylint:disable=protected-access

        dataset_cv = matching_cost_.allocate_costvolume(left, 4, -2, 2, 5, {"metadata": "metadata"})

        row = [0, 1, 2, 3, 4]
        col = [0, 2, 4]

        disparity_range = np.arange(-2, 2, step=1 / float(4), dtype=np.float64)
        disparity_range = np.append(disparity_range, [2])

        np_data = np.zeros((len(row), len(col), len(disparity_range)), dtype=np.float32)

        ground_truth = xr.Dataset(
            {"cost_volume": (["row", "col", "disp"], np_data)},
            coords={"row": row, "col": col, "disp": disparity_range},
        )

        ground_truth.attrs = {
            "crs": left.attrs["crs"],
            "transform": left.attrs["transform"],
            "window_size": 5,
        }

        assert ground_truth.equals(dataset_cv)

    @pytest.mark.parametrize(
        ["disparity_min", "disparity_max", "subpix", "expected"],
        [
            [0, 2, 1, [0, 1, 2]],
            [0, 2, 2, [0, 0.5, 1, 1.5, 2]],
            [0, 2, 4, [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]],
        ],
    )
    def test_get_disparity_range(self, disparity_min, disparity_max, subpix, expected):
        """Test get_disparity_range."""
        result = matching_cost.AbstractMatchingCost.get_disparity_range(disparity_min, disparity_max, subpix)
        np.testing.assert_array_equal(result, expected)

    def test_margins(self):
        assert isinstance(matching_cost.AbstractMatchingCost.margins, HalfWindowMargins)

    @pytest.mark.parametrize(
        ["step_col", "value", "expected"],
        [
            pytest.param(2, 2, 2, id="value is a multiple of step"),
            pytest.param(2, 3, 4, id="value is not a multiple of step"),
        ],
    )
    def test_find_nearest_multiple_of_step(self, step_col, value, expected):
        """Test values returned by find_nearest_multiple_of_step."""
        matching_cost_ = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "census", "window_size": 5, "subpix": 4, "band": "b"}
        )
        matching_cost_._step_col = step_col  # pylint:disable=protected-access

        result = matching_cost_.find_nearest_multiple_of_step(value)

        assert result == expected
