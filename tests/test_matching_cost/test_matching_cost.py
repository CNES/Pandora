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
This module contains functions to test the matching cost step.
"""

# pylint with pytest's fixtures compatibility:
# pylint: disable=redefined-outer-name
# pylint: disable=too-many-lines

from typing import NamedTuple

import numpy as np
import xarray as xr
import json_checker
import pytest

from pandora import matching_cost
from pandora.criteria import validity_mask
from pandora.margins.descriptors import HalfWindowMargins
from pandora.img_tools import create_dataset_from_inputs, add_disparity, add_disparity_grid
from tests import common  # pylint: disable=no-name-in-module


@pytest.fixture()
def images():
    return common.matching_cost_tests_setup()


@pytest.fixture()
def left(images):
    return images[0]


@pytest.fixture()
def right(images):
    return images[1]


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

    @pytest.mark.parametrize("matching_cost_method", ["census", "sad", "ssd"])
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
                5,
                (0, 1),
                (5, 6),
                id="Disparity=5",
            ),
            pytest.param(
                -5,
                (5, 6),
                (0, 1),
                id="Disparity=-5",
            ),
        ],
    )
    def test_point_interval(self, matching_cost_method, disp, p_ground_truth_disp, q_ground_truth_disp, left, right):
        """
        Test the point interval method
        """
        matching_cost_ = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": matching_cost_method, "window_size": 3, "subpix": 1}
        )

        (point_p, point_q) = matching_cost_.point_interval(left, right, disp)

        # Check if the calculated range is equal to the ground truth
        np.testing.assert_array_equal(point_p, p_ground_truth_disp)
        np.testing.assert_array_equal(point_q, q_ground_truth_disp)

    @pytest.mark.parametrize(
        ["step", "margin", "img_coordinates", "ground_truth"],
        [
            pytest.param(
                2,
                4,
                np.arange(0, 20, 2),
                np.arange(0, 20, 2),
                id="margin % self._step_col = 0",
            ),
            pytest.param(
                3,
                2,
                np.arange(8, 24, 3),
                np.arange(10, 24, 3),
                id="margin % self._step_col != 0 and margin < self._step_col",
            ),
            pytest.param(
                2,
                3,
                np.arange(7, 24, 2),
                np.arange(8, 24, 2),
                id="margin % self._step_col != 0 and margin > self._step_col",
            ),
        ],
    )
    def test_get_coordinates(self, step, margin, img_coordinates, ground_truth):
        """
        Test the get_coordinates method
        """

        # Create matching cost object
        matching_cost_ = matching_cost.AbstractMatchingCost(**common.basic_pipeline_cfg["matching_cost"])

        # Compute new indexes
        index_to_compute = matching_cost_.get_coordinates(margin, img_coordinates, step)

        np.testing.assert_array_equal(index_to_compute, ground_truth)

    @pytest.fixture()
    def default_image_path(self, memory_tiff_file):
        """
        Create a fake image to test ROI in create_dataset_from_inputs
        """
        imarray = np.array(
            (
                [np.inf, 1, 2, 5, 1, 3, 6, 4, 9, 7, 8],
                [5, 1, 2, 7, 1, 4, 7, 8, 5, 8, 0],
                [1, 2, 0, 3, 0, 4, 0, 6, 7, 4, 9],
                [4, 9, 4, 0, 1, 4, 7, 4, 6, 9, 2],
                [2, 3, 5, 0, 1, 5, 9, 2, 8, 6, 7],
                [1, 2, 4, 5, 2, 6, 7, 7, 3, 7, 0],
                [1, 2, 0, 3, 0, 4, 0, 6, 7, 4, 9],
                [np.inf, 9, 4, 0, 1, 3, 7, 4, 6, 9, 2],
            )
        )

        with memory_tiff_file(imarray) as tiff_file:
            yield tiff_file.name

    @pytest.fixture()
    def default_input_roi(self, default_image_path):
        """
        Create an input configuration to test ROI in create_dataset_from_inputs
        """
        input_config = {
            "left": {
                "img": default_image_path,
                "nodata": -9999,
                "disp": [-60, 0],
            }
        }

        return input_config

    @pytest.mark.parametrize(
        ["step", "ground_truth_attrs"],
        [
            pytest.param(
                1,  # step value
                {
                    "sampling_interval": 1,
                    "col_to_compute": np.arange(0, 11, 1),
                },
                id="no ROI in user_configuration and step = 1",
            ),
            pytest.param(
                3,  # step value
                {
                    "sampling_interval": 3,
                    "col_to_compute": np.arange(0, 11, 3),
                },
                id="no ROI in user_configuration and step = 3",
            ),
            pytest.param(
                12,  # step value
                {
                    "sampling_interval": 12,
                    "col_to_compute": np.arange(0, 11, 12),
                },
                id="no ROI in user_configuration and step > shape.col",
            ),
        ],
    )
    def test_grid_estimation(self, default_input_roi, step, ground_truth_attrs):
        """
        Test the grid_estimation function
        """
        # Input configuration
        input_config = default_input_roi

        # Create dataset with ROI
        img_left = create_dataset_from_inputs(input_config=input_config["left"])

        # Create matching cost object
        matching_cost_ = matching_cost.AbstractMatchingCost(**common.basic_pipeline_cfg["matching_cost"])

        # Update step for matching cost
        matching_cost_._step_col = step  # pylint: disable=protected-access

        # Grid estimation
        grid = matching_cost_.grid_estimation(
            img_left,
            input_config,
            (img_left["disparity"].sel(band_disp="min"), img_left["disparity"].sel(band_disp="max")),
        )

        # Create ground truth for output of grid_estimation() function
        c_row = img_left["im"].coords["row"]
        row = np.arange(c_row.values[0], c_row.values[-1] + 1)

        ground_truth = xr.Dataset(
            {},
            coords={
                "row": row,
                "col": ground_truth_attrs["col_to_compute"],
                "disp": matching_cost_.get_disparity_range(
                    -60, 0, common.basic_pipeline_cfg["matching_cost"]["subpix"]
                ),
            },
        )

        ground_truth.attrs = img_left.attrs
        ground_truth.attrs.update(ground_truth_attrs)

        xr.testing.assert_identical(grid, ground_truth)

    @pytest.mark.parametrize(
        ["roi", "step", "ground_truth_attrs"],
        [
            pytest.param(
                {"col": {"first": 3, "last": 5}, "row": {"first": 3, "last": 5}, "margins": [2, 2, 2, 2]},
                2,  # step value
                {
                    "sampling_interval": 2,
                    "col_to_compute": np.array([1, 3, 5, 7]),
                },
                id="ROI in user_configuration and margin % step == 0",
            ),
            pytest.param(
                {"col": {"first": 3, "last": 5}, "row": {"first": 3, "last": 5}, "margins": [2, 2, 2, 2]},
                3,  # step value
                {
                    "sampling_interval": 3,
                    "col_to_compute": np.array([3, 6]),
                },
                id="ROI in user_configuration and margin % step != 0 with margin < step",
            ),
            pytest.param(
                {"col": {"first": 3, "last": 4}, "row": {"first": 3, "last": 4}, "margins": [3, 3, 3, 3]},
                2,  # step value
                {"sampling_interval": 2, "col_to_compute": np.array([1, 3, 5, 7])},
                id="ROI in user_configuration and margin % step != 0 with margin > step",
            ),
            pytest.param(
                {"col": {"first": 0, "last": 2}, "row": {"first": 3, "last": 5}, "margins": [2, 2, 2, 2]},
                2,  # step value
                {
                    "sampling_interval": 2,
                    "col_to_compute": np.array([0, 2, 4]),
                },
                id="ROI overlap on left side",
            ),
            pytest.param(
                {"col": {"first": 10, "last": 12}, "row": {"first": 3, "last": 5}, "margins": [2, 2, 2, 2]},
                3,  # step value
                {
                    "sampling_interval": 3,
                    "col_to_compute": np.array([10]),
                },
                id="ROI overlap on right side",
            ),
            pytest.param(
                {"col": {"first": 3, "last": 5}, "row": {"first": -1, "last": 5}, "margins": [2, 2, 2, 2]},
                2,  # step value
                {
                    "sampling_interval": 2,
                    "col_to_compute": np.array([1, 3, 5, 7]),
                },
                id="ROI overlap on up side",
            ),
            pytest.param(
                {"col": {"first": 3, "last": 5}, "row": {"first": 7, "last": 11}, "margins": [2, 2, 2, 2]},
                2,  # step value
                {
                    "sampling_interval": 2,
                    "col_to_compute": np.array([1, 3, 5, 7]),
                },
                id="ROI overlap on down side",
            ),
        ],
    )
    def test_grid_estimation_with_roi(self, default_input_roi, roi, step, ground_truth_attrs):
        """
        Test the grid_estimation function with a ROI
        """

        # Input configuration
        input_config = default_input_roi

        # ROI
        input_config["ROI"] = roi

        # Create dataset with ROI
        img_left = create_dataset_from_inputs(input_config=input_config["left"], roi=roi)

        # Create matching cost object
        matching_cost_ = matching_cost.AbstractMatchingCost(**common.basic_pipeline_cfg["matching_cost"])

        # Update step for matching cost
        matching_cost_._step_col = step  # pylint: disable=protected-access

        # Grid estimation
        grid = matching_cost_.grid_estimation(
            img_left,
            input_config,
            (img_left["disparity"].sel(band_disp="min"), img_left["disparity"].sel(band_disp="max")),
        )

        # Create ground truth for output of grid_estimation() function
        c_row = img_left["im"].coords["row"]
        row = np.arange(c_row.values[0], c_row.values[-1] + 1)

        ground_truth = xr.Dataset(
            {},
            coords={
                "row": row,
                "col": ground_truth_attrs["col_to_compute"],
                "disp": matching_cost_.get_disparity_range(
                    -60, 0, common.basic_pipeline_cfg["matching_cost"]["subpix"]
                ),
            },
        )

        ground_truth.attrs = img_left.attrs
        ground_truth.attrs.update(ground_truth_attrs)

        xr.testing.assert_identical(grid, ground_truth)

    @pytest.mark.parametrize("method", ["zncc", "census", "sad", "ssd"])
    @pytest.mark.parametrize(
        ["step", "roi", "grid_expected"],
        [
            pytest.param(
                1,
                None,
                xr.Dataset(
                    {"cost_volume": (["row", "col", "disp"], np.full((5, 6, 2), np.nan, dtype=np.float32))},
                    coords={"row": np.arange(5), "col": np.arange(6), "disp": np.arange(-1, 1)},
                    attrs={},
                ),
                id="method with step=1",
            ),
            pytest.param(
                2,
                None,
                xr.Dataset(
                    {"cost_volume": (["row", "col", "disp"], np.full((5, 3, 2), np.nan, dtype=np.float32))},
                    coords={"row": np.arange(5), "col": np.arange(0, 6, 2), "disp": np.arange(-1, 1)},
                    attrs={},
                ),
                id="method with step=2",
            ),
            pytest.param(
                6,
                None,
                xr.Dataset(
                    {"cost_volume": (["row", "col", "disp"], np.full((5, 1, 2), np.nan, dtype=np.float32))},
                    coords={"row": np.arange(5), "col": np.arange(1), "disp": np.arange(-1, 1)},
                    attrs={},
                ),
                id="method with step=6",
            ),
            pytest.param(
                1,
                {
                    "ROI": {
                        "col": {"first": 3, "last": 3},
                        "row": {"first": 3, "last": 3},
                        "margins": [3, 3, 3, 3],
                    }
                },
                xr.Dataset(
                    {"cost_volume": (["row", "col", "disp"], np.full((5, 6, 2), np.nan, dtype=np.float32))},
                    coords={"row": np.arange(5), "col": np.arange(6), "disp": np.arange(-1, 1)},
                    attrs={},
                ),
                id="method with step=1 and roi",
            ),
            pytest.param(
                2,
                {
                    "ROI": {
                        "col": {"first": 3, "last": 3},
                        "row": {"first": 3, "last": 3},
                        "margins": [3, 3, 3, 3],
                    }
                },
                xr.Dataset(
                    {"cost_volume": (["row", "col", "disp"], np.full((5, 3, 2), np.nan, dtype=np.float32))},
                    coords={"row": np.arange(5), "col": np.arange(1, 6, 2), "disp": np.arange(-1, 1)},
                    attrs={},
                ),
                id="method with step=2 and roi",
            ),
            pytest.param(
                4,
                {
                    "ROI": {
                        "col": {"first": 3, "last": 3},
                        "row": {"first": 3, "last": 3},
                        "margins": [3, 3, 3, 3],
                    }
                },
                xr.Dataset(
                    {"cost_volume": (["row", "col", "disp"], np.full((5, 1, 2), np.nan, dtype=np.float32))},
                    coords={"row": np.arange(5), "col": np.arange(3, 4), "disp": np.arange(-1, 1)},
                    attrs={},
                ),
                id="method with step=4 and roi",
            ),
        ],
    )
    def test_allocate_cost_volume(self, left, step, roi, grid_expected, method):
        """
        Test the allocate_cost_volume function
        """
        cfg_mc = {"matching_cost_method": method}
        left.pipe(add_disparity, disparity=[-1, 0], window=None)

        # Create matching cost object
        matching_cost_ = matching_cost.AbstractMatchingCost(**cfg_mc)
        # Update step for matching cost
        matching_cost_._step_col = step  # pylint: disable=protected-access

        # Allocate an empty cost volume
        grid = matching_cost_.allocate_cost_volume(
            image=left,
            disparity_grids=(left["disparity"].sel(band_disp="min"), left["disparity"].sel(band_disp="max")),
            cfg=roi,
        )

        xr.testing.assert_identical(grid["cost_volume"], grid_expected["cost_volume"])

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

        result = matching_cost_.find_nearest_multiple_of_step(value, step_col)

        assert result == expected


class TestSplineOrder:
    """
    Description : Test spline_order in matching_cost configuration
    """

    def test_nominal_case(self):
        matching_cost.AbstractMatchingCost(**{"matching_cost_method": "zncc", "window_size": 5})

    def test_default_spline_order(self):
        result = matching_cost.AbstractMatchingCost(**{"matching_cost_method": "zncc", "window_size": 5})

        assert result._spline_order == 1  # pylint:disable=protected-access

    def test_fails_with_negative_spline_order(self):
        """
        Description : Test if the spline_order is negative
        """
        with pytest.raises(json_checker.core.exceptions.DictCheckerError) as err:
            matching_cost.AbstractMatchingCost(**{"matching_cost_method": "zncc", "window_size": 5, "spline_order": -2})
        assert "spline_order" in err.value.args[0]

    def test_fails_with_null_spline_order(self):
        """
        Description : Test if the spline_order is null
        """
        with pytest.raises(json_checker.core.exceptions.DictCheckerError) as err:
            matching_cost.AbstractMatchingCost(**{"matching_cost_method": "zncc", "window_size": 5, "spline_order": 0})
        assert "spline_order" in err.value.args[0]

    def test_fails_with_more_than_five(self):
        """
        Description : Test if the spline_order is > 5
        """
        with pytest.raises(json_checker.core.exceptions.DictCheckerError) as err:
            matching_cost.AbstractMatchingCost(**{"matching_cost_method": "zncc", "window_size": 5, "spline_order": 6})
        assert "spline_order" in err.value.args[0]

    def test_fails_with_string_element(self):
        """
        Description : Test fails if the spline_order is a string element
        """
        with pytest.raises(json_checker.core.exceptions.DictCheckerError) as err:
            matching_cost.AbstractMatchingCost(
                **{"matching_cost_method": "zncc", "window_size": 5, "spline_order": "1"}
            )
        assert "spline_order" in err.value.args[0]


def make_image(data, disparity):
    """Make an image with a disparity range."""
    return xr.Dataset(
        data_vars={
            "im": (["row", "col"], data),
        },
        coords={
            "row": np.arange(data.shape[0]),
            "col": np.arange(data.shape[1]),
        },
        attrs=common.img_attrs,
    ).pipe(add_disparity, disparity, window=None)


class CvMaskedParameters(NamedTuple):
    matching_cost_instance: matching_cost.AbstractMatchingCost
    left_image: xr.Dataset
    right_image: xr.Dataset
    cost_volume: xr.Dataset


@pytest.fixture(params=["census", "sad", "ssd", "zncc"])
def matching_cost_method(request):
    return request.param


@pytest.fixture()
def make_cv_masked_parameters(matching_cost_method, request) -> CvMaskedParameters:
    """Instantiate a matching_cost and compute cost volume"""
    cfg = {**request.param["cfg"], "matching_cost_method": matching_cost_method}
    left_image = request.getfixturevalue(request.param["left_image"])
    right_image = request.getfixturevalue(request.param["right_image"])
    left_mask = request.param.get("left_mask")
    right_mask = request.param.get("right_mask")
    if left_mask is not None:
        left_image = left_image.assign(msk=(["row", "col"], left_mask))
    if right_mask is not None:
        right_image = right_image.assign(msk=(["row", "col"], right_mask))

    matching_cost_instance = matching_cost.AbstractMatchingCost(**cfg)

    grid = matching_cost_instance.allocate_cost_volume(
        left_image,
        (
            left_image["disparity"].sel(band_disp="min"),
            left_image["disparity"].sel(band_disp="max"),
        ),
    )
    grid = validity_mask(left_image, right_image, grid)
    cost_volume = matching_cost_instance.compute_cost_volume(
        img_left=left_image, img_right=right_image, cost_volume=grid
    )
    return CvMaskedParameters(matching_cost_instance, left_image, right_image, cost_volume)


class TestCvMasked:
    """Test cv_masked."""

    @pytest.fixture()
    def left_4x5(self):
        """left_4x5"""
        data = np.array(
            (
                [1, 1, 1, 3, 4],
                [1, 2, 1, 0, 2],
                [2, 1, 0, 1, 2],
                [1, 1, 1, 1, 4],
            ),
            dtype=np.float64,
        )
        return make_image(data, disparity=[-1, 1])

    @pytest.fixture()
    def right_4x5(self):
        """right_4x5"""
        data = np.array(
            (
                [5, 1, 2, 3, 4],
                [1, 2, 1, 0, 2],
                [2, 2, 0, 1, 4],
                [1, 1, 1, 1, 2],
            ),
            dtype=np.float64,
        )
        return make_image(data, disparity=[-1, 1])

    @pytest.fixture()
    def left_subpixellic_4x5(self):
        """left_subpixellic_4x5"""
        data = np.array(
            (
                [1, 1, 1, 3, 4],
                [1, 2, 1, 0, 2],
                [2, 1, 0, 1, 2],
                [1, 1, 1, 1, 4],
            ),
            dtype=np.float64,
        )
        image = make_image(data, disparity=[-1, 1])
        image.attrs.update({"valid_pixels": 5, "no_data_mask": 7})
        return image

    @pytest.fixture()
    def right_subpixellic_4x5(self):
        """right_subpixellic_4x5"""
        data = np.array(
            (
                [5, 1, 2, 3, 4],
                [1, 2, 1, 0, 2],
                [2, 2, 0, 1, 4],
                [1, 1, 1, 1, 2],
            ),
            dtype=np.float64,
        )
        image = make_image(data, disparity=[-1, 1])
        image.attrs.update({"valid_pixels": 5, "no_data_mask": 7})
        return image

    @pytest.fixture()
    def left_6x7(self):
        """left_6x7"""
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
        return make_image(data, disparity=[-1, 1])

    @pytest.fixture()
    def right_6x7(self):
        """right_6x7"""
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
        return make_image(data, disparity=[-1, 1])

    @pytest.mark.parametrize(
        ["make_cv_masked_parameters", "expected_nan_mask"],
        # Mask convention:
        # cfg['image']['valid_pixels'] = 0
        # cfg['image']['no_data'] = 1
        # invalid_pixels all other values
        #
        # expected_nan_mask:
        # Cost volumes have shape (row, col, disp) but it is easier to read it with shape (disp, row, col) to compare
        # with mask of shape (row, col). Thus, we write it with shape (disp, row, col) then we move axis to be able
        # to compare with true cost volume of shape (row, col, disp).
        [
            pytest.param(
                {
                    "cfg": {"window_size": 3, "subpix": 1},
                    "left_image": "left_4x5",
                    "right_image": "right_4x5",
                    "left_mask": np.array(
                        (
                            [0, 0, 2, 0, 1],
                            [0, 2, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 2],
                        ),
                        dtype=np.int16,
                    ),
                    "right_mask": np.zeros((4, 5), dtype=np.int16),
                },
                np.moveaxis(
                    np.array(
                        [
                            [
                                [True, True, True, True, True],
                                [True, True, False, True, True],
                                [True, True, False, False, True],
                                [True, True, True, True, True],
                            ],
                            [
                                [True, True, True, True, True],
                                [True, True, False, True, True],
                                [True, True, False, False, True],
                                [True, True, True, True, True],
                            ],
                            [
                                [True, True, True, True, True],
                                [True, True, False, True, True],
                                [True, True, False, True, True],
                                [True, True, True, True, True],
                            ],
                        ]
                    ),
                    0,
                    -1,
                ),
                id="Invalids present on left only",
            ),
            pytest.param(
                {
                    "cfg": {"window_size": 3, "subpix": 1},
                    "left_image": "left_4x5",
                    "right_image": "right_4x5",
                    "left_mask": np.zeros((4, 5), dtype=np.int16),
                    "right_mask": np.array(
                        (
                            [0, 0, 0, 0, 2],
                            [0, 1, 0, 0, 0],
                            [0, 2, 0, 2, 0],
                            [1, 0, 0, 0, 0],
                        ),
                        dtype=np.int16,
                    ),
                },
                np.moveaxis(
                    np.array(
                        [
                            [
                                [True, True, True, True, True],
                                [True, True, True, True, True],
                                [True, True, True, True, True],
                                [True, True, True, True, True],
                            ],
                            [
                                [True, True, True, True, True],
                                [True, True, True, False, True],
                                [True, True, True, True, True],
                                [True, True, True, True, True],
                            ],
                            [
                                [True, True, True, True, True],
                                [True, True, False, True, True],
                                [True, True, True, True, True],
                                [True, True, True, True, True],
                            ],
                        ]
                    ),
                    0,
                    -1,
                ),
                id="Invalids present on right only",
            ),
            pytest.param(
                {
                    "cfg": {"window_size": 3, "subpix": 1},
                    "left_image": "left_4x5",
                    "right_image": "right_4x5",
                    "left_mask": np.array(
                        (
                            [1, 0, 0, 2, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 2, 0, 0],
                            [2, 0, 0, 0, 1],
                        ),
                        dtype=np.int16,
                    ),
                    "right_mask": np.array(
                        (
                            [0, 2, 0, 0, 1],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 2, 0],
                            [1, 0, 2, 0, 0],
                        ),
                        dtype=np.int16,
                    ),
                },
                np.moveaxis(
                    np.array(
                        [
                            [
                                [True, True, True, True, True],
                                [True, True, False, False, True],
                                [True, True, True, True, True],
                                [True, True, True, True, True],
                            ],
                            [
                                [True, True, True, True, True],
                                [True, True, False, True, True],
                                [True, True, True, True, True],
                                [True, True, True, True, True],
                            ],
                            [
                                [True, True, True, True, True],
                                [True, True, True, True, True],
                                [True, False, True, True, True],
                                [True, True, True, True, True],
                            ],
                        ]
                    ),
                    0,
                    -1,
                ),
                id="Invalids present on both sides",
            ),
            pytest.param(
                {
                    "cfg": {"window_size": 5, "subpix": 1},
                    "left_image": "left_6x7",
                    "right_image": "right_6x7",
                    "left_mask": np.array(
                        (
                            [2, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 2, 0, 0, 0, 0, 0],
                            [0, 0, 0, 2, 0, 0, 0],
                            [0, 0, 0, 0, 0, 2, 0],
                            [1, 0, 0, 0, 0, 0, 2],
                        ),
                        dtype=np.int16,
                    ),
                    "right_mask": np.array(
                        (
                            [1, 0, 0, 0, 0, 0, 2],
                            [0, 0, 0, 0, 0, 0, 0],
                            [2, 0, 2, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 2],
                            [0, 0, 0, 0, 0, 0, 0],
                            [2, 0, 0, 0, 0, 0, 1],
                        ),
                        dtype=np.int16,
                    ),
                },
                np.moveaxis(
                    np.array(
                        [
                            [
                                [True, True, True, True, True, True, True],
                                [True, True, True, True, True, True, True],
                                [True, True, True, True, True, True, True],
                                [True, True, True, True, False, True, True],
                                [True, True, True, True, True, True, True],
                                [True, True, True, True, True, True, True],
                            ],
                            [
                                [True, True, True, True, True, True, True],
                                [True, True, True, True, True, True, True],
                                [True, True, True, False, True, True, True],
                                [True, True, True, True, True, True, True],
                                [True, True, True, True, True, True, True],
                                [True, True, True, True, True, True, True],
                            ],
                            [
                                [True, True, True, True, True, True, True],
                                [True, True, True, True, True, True, True],
                                [True, True, False, False, True, True, True],
                                [True, True, True, True, True, True, True],
                                [True, True, True, True, True, True, True],
                                [True, True, True, True, True, True, True],
                            ],
                        ]
                    ),
                    0,
                    -1,
                ),
                id="Invalids present on both sides with window size of 5",
            ),
        ],
        indirect=["make_cv_masked_parameters"],
    )
    def test_pixellic(self, make_cv_masked_parameters, expected_nan_mask):
        """Test cost volume NaNs positions with subpix at 1."""
        matching_cost_instance = make_cv_masked_parameters.matching_cost_instance
        left_image = make_cv_masked_parameters.left_image
        right_image = make_cv_masked_parameters.right_image
        cost_volume = make_cv_masked_parameters.cost_volume

        matching_cost_instance.cv_masked(
            img_left=left_image,
            img_right=right_image,
            cost_volume=cost_volume,
            disp_min=left_image["disparity"].sel(band_disp="min"),
            disp_max=left_image["disparity"].sel(band_disp="max"),
        )

        np.testing.assert_array_equal(np.isnan(cost_volume["cost_volume"]), expected_nan_mask)

    @pytest.mark.parametrize(
        ["make_cv_masked_parameters", "expected_nan_mask"],
        # Mask convention:
        # cfg['image']['valid_pixels'] = 0
        # cfg['image']['no_data'] = 1
        # invalid_pixels all other values
        #
        # expected_nan_mask:
        #
        # Cost volumes have shape (row, col, disp) but it is easier to read it with shape (disp, row, col) to compare
        # with mask of shape (row, col). Thus, we write it with shape (disp, row, col) then we move axis to be able to
        # compare with true cost volume of shape (row, col, disp).
        [
            pytest.param(
                {
                    "cfg": {"window_size": 3, "subpix": 2},
                    "left_image": "left_subpixellic_4x5",
                    "right_image": "right_subpixellic_4x5",
                    "left_mask": np.array(
                        (
                            [5, 56, 5, 12, 5],
                            [5, 5, 5, 5, 5],
                            [5, 5, 5, 5, 5],
                            [3, 5, 4, 5, 7],
                        ),
                        dtype=np.int16,
                    ),
                    "right_mask": np.array(
                        (
                            [7, 5, 5, 5, 5],
                            [5, 5, 5, 65, 5],
                            [5, 5, 5, 5, 5],
                            [5, 23, 5, 5, 2],
                        ),
                        dtype=np.int16,
                    ),
                },
                np.moveaxis(
                    np.array(
                        [
                            [
                                [True, True, True, True, True],
                                [True, True, True, False, True],
                                [True, True, False, True, True],
                                [True, True, True, True, True],
                            ],
                            [
                                [True, True, True, True, True],
                                [True, True, True, True, True],
                                [True, True, False, True, True],
                                [True, True, True, True, True],
                            ],
                            [
                                [True, True, True, True, True],
                                [True, True, False, True, True],
                                [True, False, False, True, True],
                                [True, True, True, True, True],
                            ],
                            [
                                [True, True, True, True, True],
                                [True, True, True, True, True],
                                [True, False, False, True, True],
                                [True, True, True, True, True],
                            ],
                            [
                                [True, True, True, True, True],
                                [True, False, True, True, True],
                                [True, False, False, True, True],
                                [True, True, True, True, True],
                            ],
                        ]
                    ),
                    0,
                    -1,
                ),
                id="Subpix: 2",
            ),
        ],
        indirect=["make_cv_masked_parameters"],
    )
    def test_subpixellic(self, make_cv_masked_parameters, expected_nan_mask):
        """Test cost volume NaNs positions with subpix greater than 1."""
        matching_cost_instance = make_cv_masked_parameters.matching_cost_instance
        left_image = make_cv_masked_parameters.left_image
        right_image = make_cv_masked_parameters.right_image
        cost_volume = make_cv_masked_parameters.cost_volume

        matching_cost_instance.cv_masked(
            img_left=left_image,
            img_right=right_image,
            cost_volume=cost_volume,
            disp_min=left_image["disparity"].sel(band_disp="min"),
            disp_max=left_image["disparity"].sel(band_disp="max"),
        )

        np.testing.assert_array_equal(np.isnan(cost_volume["cost_volume"]), expected_nan_mask)


class TestCvMaskedWithWindowSizeOf1:
    """Test cv_masked with methods that works with this window size (i.e.: not `census`)."""

    @pytest.fixture()
    def left_subpixellic_2x3(self):
        """left_subpixellic_2x3"""
        data = np.array(
            (
                [1, 1, 1],
                [1, 1, 1],
            ),
            dtype=np.float64,
        )
        image = make_image(data, disparity=[-1, 1])
        image.attrs.update({"valid_pixels": 5, "no_data_mask": 7})
        return image

    @pytest.fixture()
    def right_subpixellic_2x3(self):
        """right_subpixellic_2x3"""
        data = np.array(
            (
                [5, 1, 2],
                [1, 1, 1],
            ),
            dtype=np.float64,
        )
        image = make_image(data, disparity=[-1, 1])
        image.attrs.update({"valid_pixels": 5, "no_data_mask": 7})
        return image

    @pytest.fixture()
    def left_2x5(self):
        """left_2x5"""
        data = np.array(
            (
                [1, 1, 1, 3, 4],
                [1, 2, 1, 0, 2],
            ),
            dtype=np.float64,
        )
        return make_image(data, disparity=[-1, 1])

    @pytest.fixture()
    def right_2x5(self):
        """right_2x5"""
        data = np.array(
            (
                [5, 1, 2, 3, 4],
                [1, 2, 1, 0, 2],
            ),
            dtype=np.float64,
        )
        return make_image(data, disparity=[-1, 1])

    @pytest.fixture()
    def left_subpixellic_2x5(self):
        """left_subpixellic_2x5"""
        data = np.array(
            (
                [1, 1, 1, 3, 4],
                [1, 1, 1, 1, 4],
            ),
            dtype=np.float64,
        )
        return make_image(data, disparity=[-1, 1])

    @pytest.fixture()
    def right_subpixellic_2x5(self):
        """right_subpixellic_2x5"""
        data = np.array(
            (
                [5, 1, 2, 3, 4],
                [1, 1, 1, 1, 2],
            ),
            dtype=np.float64,
        )
        return make_image(data, disparity=[-1, 1])

    @pytest.fixture(params=["sad", "ssd", "zncc"])
    def matching_cost_method(self, request):
        return request.param

    @pytest.mark.parametrize(
        ["make_cv_masked_parameters", "expected_nan_mask"],
        # Mask convention:
        # cfg['image']['valid_pixels'] = 0
        # cfg['image']['no_data'] = 1
        # invalid_pixels all other values
        #
        # expected_nan_mask:
        # Cost volumes have shape (row, col, disp) but it is easier to read it with shape (disp, row, col) to compare
        # with mask of shape (row, col). Thus, we write it with shape (disp, row, col) then we move axis to be able
        # to compare with true cost volume of shape (row, col, disp).
        [
            pytest.param(
                {
                    "cfg": {"window_size": 1, "subpix": 1},
                    "left_image": "left_2x5",
                    "right_image": "right_2x5",
                    "left_mask": np.array(
                        (
                            [1, 0, 0, 2, 0],
                            [2, 0, 0, 0, 1],
                        ),
                        dtype=np.int16,
                    ),
                    "right_mask": np.array(
                        (
                            [0, 2, 0, 0, 1],
                            [1, 0, 2, 0, 0],
                        ),
                        dtype=np.int16,
                    ),
                },
                np.moveaxis(
                    np.array(
                        [
                            [
                                [True, False, True, True, False],
                                [True, True, False, True, True],
                            ],
                            [
                                [True, True, False, True, True],
                                [True, False, True, False, True],
                            ],
                            [
                                [True, False, False, True, True],
                                [True, True, False, False, True],
                            ],
                        ]
                    ),
                    0,
                    -1,
                ),
                id="Invalids present on both sides",
            )
        ],
        indirect=["make_cv_masked_parameters"],
    )
    def test_pixellic(self, make_cv_masked_parameters, expected_nan_mask):
        """Test cost volume NaNs positions with subpix at 1."""
        matching_cost_instance = make_cv_masked_parameters.matching_cost_instance
        left_image = make_cv_masked_parameters.left_image
        right_image = make_cv_masked_parameters.right_image
        cost_volume = make_cv_masked_parameters.cost_volume

        matching_cost_instance.cv_masked(
            img_left=left_image,
            img_right=right_image,
            cost_volume=cost_volume,
            disp_min=left_image["disparity"].sel(band_disp="min"),
            disp_max=left_image["disparity"].sel(band_disp="max"),
        )

        np.testing.assert_array_equal(np.isnan(cost_volume["cost_volume"]), expected_nan_mask)

    @pytest.mark.parametrize(
        ["make_cv_masked_parameters", "expected_nan_mask"],
        # expected_nan_mask:
        # Cost volumes have shape (row, col, disp) but it is easier to read it with shape (disp, row, col) to compare
        # with mask of shape (row, col). Thus, we write it with shape (disp, row, col) then we move axis to be able
        # to compare with true cost volume of shape (row, col, disp).
        [
            pytest.param(
                # Mask convention
                # cfg['image']['valid_pixels'] = 5
                # cfg['image']['no_data'] = 7
                # invalid_pixels all other values
                {
                    "cfg": {"window_size": 1, "subpix": 2},
                    "left_image": "left_subpixellic_2x5",
                    "right_image": "right_subpixellic_2x5",
                    "left_mask": np.zeros((2, 5), dtype=np.int16),
                    "right_mask": np.array(
                        (
                            [0, 0, 0, 0, 1],
                            [1, 0, 2, 0, 0],
                        ),
                        dtype=np.int16,
                    ),
                },
                np.moveaxis(
                    np.array(
                        [
                            [
                                [True, False, False, False, False],
                                [True, True, False, True, False],
                            ],
                            [
                                [True, False, False, False, True],
                                [True, True, True, True, False],
                            ],
                            [
                                [False, False, False, False, True],
                                [True, False, True, False, False],
                            ],
                            [
                                [False, False, False, True, True],
                                [True, True, True, False, True],
                            ],
                            [
                                [False, False, False, True, True],
                                [False, True, False, False, True],
                            ],
                        ]
                    ),
                    0,
                    -1,
                ),
                id="Subpix: 2",
            ),
            pytest.param(
                # Mask convention:
                # cfg['image']['valid_pixels'] = 0
                # cfg['image']['no_data'] = 1
                # invalid_pixels all other values
                #
                {
                    "cfg": {"window_size": 1, "subpix": 4},
                    "left_image": "left_subpixellic_2x3",
                    "right_image": "right_subpixellic_2x3",
                    "left_mask": np.array(
                        (
                            [5, 5, 5],
                            [5, 5, 5],
                        ),
                        dtype=np.int16,
                    ),
                    "right_mask": np.array(
                        (
                            [5, 4, 7],
                            [6, 7, 5],
                        ),
                        dtype=np.int16,
                    ),
                },
                np.moveaxis(
                    np.array(
                        [
                            [
                                [True, False, True],
                                [True, True, True],
                            ],
                            [
                                [True, True, True],
                                [True, True, True],
                            ],
                            [
                                [True, True, True],
                                [True, True, True],
                            ],
                            [
                                [True, True, True],
                                [True, True, True],
                            ],
                            [
                                [False, True, True],
                                [True, True, False],
                            ],
                            [
                                [True, True, True],
                                [True, True, True],
                            ],
                            [
                                [True, True, True],
                                [True, True, True],
                            ],
                            [
                                [True, True, True],
                                [True, True, True],
                            ],
                            [
                                [True, True, True],
                                [True, False, True],
                            ],
                        ]
                    ),
                    0,
                    -1,
                ),
                id="Subpix: 4",
            ),
        ],
        indirect=["make_cv_masked_parameters"],
    )
    def test_subpixellic(self, make_cv_masked_parameters, expected_nan_mask):
        """Test cost volume NaNs positions."""
        matching_cost_instance = make_cv_masked_parameters.matching_cost_instance
        left_image = make_cv_masked_parameters.left_image
        right_image = make_cv_masked_parameters.right_image
        cost_volume = make_cv_masked_parameters.cost_volume

        matching_cost_instance.cv_masked(
            img_left=left_image,
            img_right=right_image,
            cost_volume=cost_volume,
            disp_min=left_image["disparity"].sel(band_disp="min"),
            disp_max=left_image["disparity"].sel(band_disp="max"),
        )

        np.testing.assert_array_equal(np.isnan(cost_volume["cost_volume"]), expected_nan_mask)


@pytest.mark.filterwarnings("ignore:Dataset has no geotransform")
class TestCvMaskedWithGrid:
    """Test cv_masked with disparity grids instead of disparity ranges."""

    @staticmethod
    def make_image(data, disparity):
        """Make an image with a disparity grid."""
        return xr.Dataset(
            data_vars={
                "im": (["row", "col"], data),
            },
            coords={
                "row": np.arange(data.shape[0]),
                "col": np.arange(data.shape[1]),
            },
            attrs=common.img_attrs,
        ).pipe(add_disparity_grid, disparity)

    @pytest.fixture()
    def disparity_grid_4x11(self):
        """disparity grid 4x11"""
        dmin_grid = [
            [-0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0],
            [-0, -8, -8, -5, -8, -4, -6, -7, -9, -8, -0],
            [-0, -9, -8, -4, -6, -5, -7, -8, -9, -7, -0],
            [-0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0],
        ]

        dmax_grid = [
            [-0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0],
            [-0, -2, -1, -1, -5, -1, -2, -6, -4, -3, -0],
            [-0, -3, 0, -2, -2, -2, -3, -5, -5, -4, -0],
            [-0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0],
        ]

        return xr.DataArray([dmin_grid, dmax_grid], dims=["band_disp", "row", "col"])

    @pytest.fixture()
    def left_with_disparity_grid_4x11(self, disparity_grid_4x11):
        """left with disparity grid 4x11"""
        data = np.array(
            (
                [1, 1, 1, 3, 2, 1, 7, 2, 3, 4, 6],
                [1, 3, 2, 5, 2, 6, 1, 8, 7, 0, 4],
                [2, 1, 0, 1, 7, 9, 5, 4, 9, 1, 5],
                [1, 5, 4, 3, 2, 6, 7, 6, 5, 2, 1],
            ),
            dtype=np.float64,
        )
        return self.make_image(data, disparity_grid_4x11)

    @pytest.fixture()
    def right_without_disparity_4x11(self):
        """right without disparity 4x11"""
        data = np.array(
            (
                [5, 1, 2, 3, 4, 7, 9, 6, 5, 2, 7],
                [1, 3, 0, 2, 5, 3, 7, 8, 7, 6, 5],
                [2, 3, 5, 0, 1, 5, 6, 5, 2, 3, 6],
                [1, 6, 7, 5, 3, 2, 1, 0, 3, 4, 7],
            ),
            dtype=np.float64,
        )
        return make_image(data, None)

    @pytest.fixture()
    def disparity_grid_4x11_for_subpixellic(self):
        """disparity grid 4x11 for subpixellic"""
        dmin_grid = [
            [-0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0],
            [-0, -8, -8, -5, -8, -4, -6, -7, -9, -8, -0],
            [-0, -9, -8, -4, -6, -5, -7, -8, -9, -7, -0],
            [-0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0],
        ]

        dmax_grid = [
            [-0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0],
            [-0, -2, -1, -1, -5, -1, -2, -6, -4, -3, -0],
            [-0, -3, 0, -2, -2, -2, -3, -5, -5, -4, -0],
            [-0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0],
        ]

        return xr.DataArray([dmin_grid, dmax_grid], dims=["band_disp", "row", "col"])

    @pytest.fixture()
    def left_with_disparity_grid_4x11_for_subpixellic(self, disparity_grid_4x11_for_subpixellic):
        """left with disparity grid 4x11 for subpixellic"""
        data = np.array(
            (
                [1, 1, 1, 3, 2, 1, 7, 2, 3, 4, 6],
                [1, 3, 2, 5, 2, 6, 1, 8, 7, 0, 4],
                [2, 1, 0, 1, 7, 9, 5, 4, 9, 1, 5],
                [1, 5, 4, 3, 2, 6, 7, 6, 5, 2, 1],
            ),
            dtype=np.float64,
        )
        return self.make_image(data, disparity_grid_4x11_for_subpixellic)

    @pytest.mark.parametrize(
        ["make_cv_masked_parameters", "expected_nan_mask"],
        # expected_nan_mask:
        # Cost volumes have shape (row, col, disp) but it is easier to read it with shape (disp, row, col) to compare
        # with image of shape (row, col). Thus, we write it with shape (disp, row, col) then we move axis to be able
        # to compare with true cost volume of shape (row, col, disp).
        [
            pytest.param(
                {
                    "cfg": {"window_size": 3, "subpix": 1},
                    "left_image": "left_with_disparity_grid_4x11",
                    "right_image": "right_without_disparity_4x11",
                },
                np.moveaxis(
                    np.array(
                        [
                            [
                                [True, True, True, True, True, True, True, True, True, True, True],
                                [True, True, True, True, True, True, True, True, True, True, True],
                                [True, True, True, True, True, True, True, True, True, True, True],
                                [True, True, True, True, True, True, True, True, True, True, True],
                            ],
                            [
                                [True, True, True, True, True, True, True, True, True, True, True],
                                [True, True, True, True, True, True, True, True, True, False, True],
                                [True, True, True, True, True, True, True, True, True, True, True],
                                [True, True, True, True, True, True, True, True, True, True, True],
                            ],
                            [
                                [True, True, True, True, True, True, True, True, True, True, True],
                                [True, True, True, True, True, True, True, True, False, False, True],
                                [True, True, True, True, True, True, True, True, False, False, True],
                                [True, True, True, True, True, True, True, True, True, True, True],
                            ],
                            [
                                [True, True, True, True, True, True, True, True, True, True, True],
                                [True, True, True, True, True, True, True, False, False, False, True],
                                [True, True, True, True, True, True, True, False, False, False, True],
                                [True, True, True, True, True, True, True, True, True, True, True],
                            ],
                            [
                                [True, True, True, True, True, True, True, True, True, True, True],
                                [True, True, True, True, True, True, False, True, False, False, True],
                                [True, True, True, True, True, True, False, False, False, False, True],
                                [True, True, True, True, True, True, True, True, True, True, True],
                            ],
                            [
                                [True, True, True, True, True, True, True, True, True, True, True],
                                [True, True, True, True, True, False, False, True, False, False, True],
                                [True, True, True, True, True, False, False, True, True, False, True],
                                [True, True, True, True, True, True, True, True, True, True, True],
                            ],
                            [
                                [True, True, True, True, True, True, True, True, True, True, True],
                                [True, True, True, True, True, False, False, True, True, False, True],
                                [True, True, True, True, False, False, False, True, True, True, True],
                                [True, True, True, True, True, True, True, True, True, True, True],
                            ],
                            [
                                [True, True, True, True, True, True, True, True, True, True, True],
                                [True, True, True, False, True, False, False, True, True, True, True],
                                [True, True, True, False, False, False, True, True, True, True, True],
                                [True, True, True, True, True, True, True, True, True, True, True],
                            ],
                            [
                                [True, True, True, True, True, True, True, True, True, True, True],
                                [True, True, False, False, True, False, True, True, True, True, True],
                                [True, True, False, True, True, True, True, True, True, True, True],
                                [True, True, True, True, True, True, True, True, True, True, True],
                            ],
                            [
                                [True, True, True, True, True, True, True, True, True, True, True],
                                [True, True, True, True, True, True, True, True, True, True, True],
                                [True, True, False, True, True, True, True, True, True, True, True],
                                [True, True, True, True, True, True, True, True, True, True, True],
                            ],
                        ]
                    ),
                    0,
                    -1,
                ),
                id="subpix: 1",
            ),
        ],
        indirect=["make_cv_masked_parameters"],
    )
    def test_pixellic(self, make_cv_masked_parameters, expected_nan_mask):
        """Test cost volume NaNs positions with subpix at 1 and a disparity grid."""
        matching_cost_instance = make_cv_masked_parameters.matching_cost_instance
        left_image = make_cv_masked_parameters.left_image
        right_image = make_cv_masked_parameters.right_image
        cost_volume = make_cv_masked_parameters.cost_volume

        matching_cost_instance.cv_masked(
            img_left=left_image,
            img_right=right_image,
            cost_volume=cost_volume,
            disp_min=left_image["disparity"].sel(band_disp="min"),
            disp_max=left_image["disparity"].sel(band_disp="max"),
        )

        np.testing.assert_array_equal(np.isnan(cost_volume["cost_volume"]), expected_nan_mask)

    @pytest.mark.parametrize(
        ["make_cv_masked_parameters", "expected_nan_mask"],
        # expected_nan_mask:
        # Cost volumes have shape (row, col, disp) but it is easier to read it with shape (disp, row, col) to compare
        # with image of shape (row, col). Thus, we write it with shape (disp, row, col) then we move axis to be able
        # to compare with true cost volume of shape (row, col, disp).
        [
            pytest.param(
                {
                    "cfg": {"window_size": 3, "subpix": 2},
                    "left_image": "left_with_disparity_grid_4x11_for_subpixellic",
                    "right_image": "right_without_disparity_4x11",
                },
                np.moveaxis(
                    np.array(
                        [
                            [
                                [True, True, True, True, True, True, True, True, True, True, True],
                                [True, True, True, True, True, True, True, True, True, True, True],
                                [True, True, True, True, True, True, True, True, True, True, True],
                                [True, True, True, True, True, True, True, True, True, True, True],
                            ],
                            [
                                [True, True, True, True, True, True, True, True, True, True, True],
                                [True, True, True, True, True, True, True, True, True, True, True],
                                [True, True, True, True, True, True, True, True, True, True, True],
                                [True, True, True, True, True, True, True, True, True, True, True],
                            ],
                            [
                                [True, True, True, True, True, True, True, True, True, True, True],
                                [True, True, True, True, True, True, True, True, True, False, True],
                                [True, True, True, True, True, True, True, True, True, True, True],
                                [True, True, True, True, True, True, True, True, True, True, True],
                            ],
                            [
                                [True, True, True, True, True, True, True, True, True, True, True],
                                [True, True, True, True, True, True, True, True, True, False, True],
                                [True, True, True, True, True, True, True, True, True, True, True],
                                [True, True, True, True, True, True, True, True, True, True, True],
                            ],
                            [
                                [True, True, True, True, True, True, True, True, True, True, True],
                                [True, True, True, True, True, True, True, True, False, False, True],
                                [True, True, True, True, True, True, True, True, False, False, True],
                                [True, True, True, True, True, True, True, True, True, True, True],
                            ],
                            [
                                [True, True, True, True, True, True, True, True, True, True, True],
                                [True, True, True, True, True, True, True, True, False, False, True],
                                [True, True, True, True, True, True, True, True, False, False, True],
                                [True, True, True, True, True, True, True, True, True, True, True],
                            ],
                            [
                                [True, True, True, True, True, True, True, True, True, True, True],
                                [True, True, True, True, True, True, True, False, False, False, True],
                                [True, True, True, True, True, True, True, False, False, False, True],
                                [True, True, True, True, True, True, True, True, True, True, True],
                            ],
                            [
                                [True, True, True, True, True, True, True, True, True, True, True],
                                [True, True, True, True, True, True, True, True, False, False, True],
                                [True, True, True, True, True, True, True, False, False, False, True],
                                [True, True, True, True, True, True, True, True, True, True, True],
                            ],
                            [
                                [True, True, True, True, True, True, True, True, True, True, True],
                                [True, True, True, True, True, True, False, True, False, False, True],
                                [True, True, True, True, True, True, False, False, False, False, True],
                                [True, True, True, True, True, True, True, True, True, True, True],
                            ],
                            [
                                [True, True, True, True, True, True, True, True, True, True, True],
                                [True, True, True, True, True, True, False, True, False, False, True],
                                [True, True, True, True, True, True, False, True, True, False, True],
                                [True, True, True, True, True, True, True, True, True, True, True],
                            ],
                            [
                                [True, True, True, True, True, True, True, True, True, True, True],
                                [True, True, True, True, True, False, False, True, False, False, True],
                                [True, True, True, True, True, False, False, True, True, False, True],
                                [True, True, True, True, True, True, True, True, True, True, True],
                            ],
                            [
                                [True, True, True, True, True, True, True, True, True, True, True],
                                [True, True, True, True, True, False, False, True, True, False, True],
                                [True, True, True, True, True, False, False, True, True, True, True],
                                [True, True, True, True, True, True, True, True, True, True, True],
                            ],
                            [
                                [True, True, True, True, True, True, True, True, True, True, True],
                                [True, True, True, True, True, False, False, True, True, False, True],
                                [True, True, True, True, False, False, False, True, True, True, True],
                                [True, True, True, True, True, True, True, True, True, True, True],
                            ],
                            [
                                [True, True, True, True, True, True, True, True, True, True, True],
                                [True, True, True, True, True, False, False, True, True, True, True],
                                [True, True, True, True, False, False, True, True, True, True, True],
                                [True, True, True, True, True, True, True, True, True, True, True],
                            ],
                            [
                                [True, True, True, True, True, True, True, True, True, True, True],
                                [True, True, True, False, True, False, False, True, True, True, True],
                                [True, True, True, False, False, False, True, True, True, True, True],
                                [True, True, True, True, True, True, True, True, True, True, True],
                            ],
                            [
                                [True, True, True, True, True, True, True, True, True, True, True],
                                [True, True, True, False, True, False, True, True, True, True, True],
                                [True, True, True, True, True, True, True, True, True, True, True],
                                [True, True, True, True, True, True, True, True, True, True, True],
                            ],
                            [
                                [True, True, True, True, True, True, True, True, True, True, True],
                                [True, True, False, False, True, False, True, True, True, True, True],
                                [True, True, False, True, True, True, True, True, True, True, True],
                                [True, True, True, True, True, True, True, True, True, True, True],
                            ],
                            [
                                [True, True, True, True, True, True, True, True, True, True, True],
                                [True, True, True, True, True, True, True, True, True, True, True],
                                [True, True, False, True, True, True, True, True, True, True, True],
                                [True, True, True, True, True, True, True, True, True, True, True],
                            ],
                            [
                                [True, True, True, True, True, True, True, True, True, True, True],
                                [True, True, True, True, True, True, True, True, True, True, True],
                                [True, True, False, True, True, True, True, True, True, True, True],
                                [True, True, True, True, True, True, True, True, True, True, True],
                            ],
                        ]
                    ),
                    0,
                    -1,
                ),
                id="subpix: 2",
            ),
        ],
        indirect=["make_cv_masked_parameters"],
    )
    def test_subpixellic(self, make_cv_masked_parameters, expected_nan_mask):
        """Test cost volume NaNs positions with subpix at 1 and a disparity grid."""
        matching_cost_instance = make_cv_masked_parameters.matching_cost_instance
        left_image = make_cv_masked_parameters.left_image
        right_image = make_cv_masked_parameters.right_image
        cost_volume = make_cv_masked_parameters.cost_volume

        matching_cost_instance.cv_masked(
            img_left=left_image,
            img_right=right_image,
            cost_volume=cost_volume,
            disp_min=left_image["disparity"].sel(band_disp="min"),
            disp_max=left_image["disparity"].sel(band_disp="max"),
        )

        np.testing.assert_array_equal(np.isnan(cost_volume["cost_volume"]), expected_nan_mask)
