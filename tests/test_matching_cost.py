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
# pylint: disable=too-many-lines

from typing import NamedTuple, Union

import numpy as np
import xarray as xr
import pytest

from pandora import matching_cost
from pandora.criteria import validity_mask
from pandora.margins.descriptors import HalfWindowMargins
from pandora.img_tools import create_dataset_from_inputs, add_disparity, add_disparity_grid
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
        row = np.arange(c_row[0], c_row[-1] + 1)

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
                {
                    "sampling_interval": 2,
                    "col_to_compute": np.array([1, 3, 5, 7]),
                },
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
        row = np.arange(c_row[0], c_row[-1] + 1)

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
        matching_cost_._step_col = step_col  # pylint:disable=protected-access

        result = matching_cost_.find_nearest_multiple_of_step(value)

        assert result == expected


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


class MaskColumnIntervalParameters(NamedTuple):
    """ "Tuple with all parameters for tests in TestMaskColumnInterval"""

    matching_cost_instance: matching_cost.AbstractMatchingCost
    cost_volume: xr.Dataset
    coord_mask_left: np.ndarray
    coord_mask_right: np.ndarray
    disp: Union[int, float]


class TestMaskColumnInterval:
    """Test mask_column_interval."""

    def make_image(self, data, disparity, first_row_coord=0, first_col_coord=0):
        """Make an image with a disparity range."""
        return xr.Dataset(
            data_vars={
                "im": (["row", "col"], data),
            },
            coords={
                "row": np.arange(first_row_coord, first_row_coord + data.shape[0]),
                "col": np.arange(first_col_coord, first_col_coord + data.shape[1]),
            },
            attrs=common.img_attrs,
        ).pipe(add_disparity, disparity, window=None)

    @pytest.fixture()
    def make_mask_column_interval_parameters(self, matching_cost_method, request) -> MaskColumnIntervalParameters:
        """Instantiate a matching_cost, compute cost volume and masks dilatation"""
        cfg = {**request.param["cfg"], "matching_cost_method": matching_cost_method, "window_size": 3}
        left_image = request.getfixturevalue(request.param["left_image"])
        right_image = request.getfixturevalue(request.param["right_image"])
        left_mask = request.param.get("left_mask")
        right_mask = request.param.get("right_mask")
        disp = request.param.get("disp_tested")
        step = request.param.get("step")
        if left_mask is not None:
            left_image = left_image.assign(msk=(["row", "col"], left_mask))
        if right_mask is not None:
            right_image = right_image.assign(msk=(["row", "col"], right_mask))

        matching_cost_instance = matching_cost.AbstractMatchingCost(**cfg)

        if step is not None:
            matching_cost_instance._step_col = step  # pylint: disable=protected-access

        grid = matching_cost_instance.allocate_cost_volume(
            left_image,
            (
                left_image["disparity"].sel(band_disp="min"),
                left_image["disparity"].sel(band_disp="max"),
            ),
            cfg,
        )
        grid = validity_mask(left_image, right_image, grid)
        cost_volume = matching_cost_instance.compute_cost_volume(
            img_left=left_image, img_right=right_image, cost_volume=grid
        )
        mask_left, mask_right = matching_cost_instance.masks_dilatation(
            img_left=left_image,
            img_right=right_image,
            window_size=matching_cost_instance._window_size,  # pylint: disable=protected-access
            subp=matching_cost_instance._subpix,  # pylint: disable=protected-access
        )
        i_right = int((disp % 1) * matching_cost_instance._subpix)  # pylint: disable=protected-access
        i_mask_right = min(1, i_right)
        return MaskColumnIntervalParameters(
            matching_cost_instance,
            cost_volume,
            mask_left.coords["col"].data,
            mask_right[i_mask_right].coords["col"].data,
            disp,
        )

    @pytest.fixture()
    def left_img(self):
        """left img"""
        data = np.array(
            (
                [1, 1, 1, 3, 4],
                [1, 2, 1, 0, 2],
                [2, 1, 0, 1, 2],
                [1, 1, 1, 1, 4],
            ),
            dtype=np.float64,
        )
        return self.make_image(data, [-1, 1])

    @pytest.fixture()
    def right_img(self):
        """right img"""
        data = np.array(
            (
                [5, 1, 2, 3, 4],
                [1, 2, 1, 0, 2],
                [2, 2, 0, 1, 4],
                [1, 1, 1, 1, 2],
            ),
            dtype=np.float64,
        )
        return self.make_image(data, None)

    @pytest.mark.parametrize(
        ["make_mask_column_interval_parameters", "ground_truth"],
        [
            pytest.param(
                {"cfg": {"subpix": 1}, "left_image": "left_img", "right_image": "right_img", "disp_tested": -1},
                (np.arange(1, 5), np.arange(4)),
                id="negative disparity without mask",
            ),
            pytest.param(
                {"cfg": {"subpix": 1}, "left_image": "left_img", "right_image": "right_img", "disp_tested": 0},
                (np.arange(5), np.arange(5)),
                id="null disparity without mask",
            ),
            pytest.param(
                {"cfg": {"subpix": 1}, "left_image": "left_img", "right_image": "right_img", "disp_tested": 1},
                (np.arange(4), np.arange(1, 5)),
                id="positive disparity without mask",
            ),
            pytest.param(
                {
                    "cfg": {"subpix": 1},
                    "left_image": "left_img",
                    "right_image": "right_img",
                    "disp_tested": 1,
                    "step": 2,
                },
                (np.arange(0, 4, 2), np.arange(1, 5, 2)),
                id="positive disparity with step",
            ),
            pytest.param(
                {
                    "cfg": {"subpix": 1},
                    "left_image": "left_img",
                    "right_image": "right_img",
                    "right_mask": np.array(
                        ([0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]),
                        dtype=np.int16,
                    ),
                    "disp_tested": 1,
                },
                (np.arange(4), np.arange(1, 5)),
                id="positive disparity with right mask",
            ),
            pytest.param(
                {
                    "cfg": {"subpix": 1},
                    "left_image": "left_img",
                    "right_image": "right_img",
                    "left_mask": np.array(
                        ([0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]),
                        dtype=np.int16,
                    ),
                    "disp_tested": 1,
                },
                (np.arange(4), np.arange(1, 5)),
                id="positive disparity with left mask",
            ),
            pytest.param(
                {"cfg": {"subpix": 1}, "left_image": "left_img", "right_image": "right_img", "disp_tested": 5},
                (np.array([]), np.array([])),
                id="disparity (positive) higher than the number of columns without mask",
            ),
            pytest.param(
                {"cfg": {"subpix": 1}, "left_image": "left_img", "right_image": "right_img", "disp_tested": -5},
                (np.array([]), np.array([])),
                id="disparity (negative) higher than the number of columns without mask",
            ),
        ],
        indirect=["make_mask_column_interval_parameters"],
    )
    def test_pixellic(self, make_mask_column_interval_parameters, ground_truth):
        """Check mask_column_interval with different disparity"""
        matching_cost_instance = make_mask_column_interval_parameters.matching_cost_instance
        cost_volume = make_mask_column_interval_parameters.cost_volume
        coord_mask_left = make_mask_column_interval_parameters.coord_mask_left
        coord_mask_right = make_mask_column_interval_parameters.coord_mask_right
        disp = make_mask_column_interval_parameters.disp

        result = matching_cost_instance.mask_column_interval(cost_volume, coord_mask_left, coord_mask_right, disp)
        np.testing.assert_array_equal(result, ground_truth)

        # check index exist
        left_mask_index = set(coord_mask_left)
        right_mask_index = set(coord_mask_right)
        cost_volume_index = set(cost_volume.coords["col"].data)
        assert set(result[0]).issubset(left_mask_index)
        assert set(result[1]).issubset(right_mask_index)
        assert set(result[0]).issubset(cost_volume_index)

    @pytest.mark.parametrize(
        ["make_mask_column_interval_parameters", "ground_truth"],
        [
            pytest.param(
                {"cfg": {"subpix": 2}, "left_image": "left_img", "right_image": "right_img", "disp_tested": -1},
                (np.arange(1, 5), np.arange(4)),
                id="disparity=-1 & subpix=2 without mask",
            ),
            pytest.param(
                {"cfg": {"subpix": 4}, "left_image": "left_img", "right_image": "right_img", "disp_tested": -1},
                (np.arange(1, 5), np.arange(4)),
                id="disparity=-1 & subpix=4 without mask",
            ),
            pytest.param(
                {
                    "cfg": {"subpix": 2},
                    "left_image": "left_img",
                    "right_image": "right_img",
                    "right_mask": np.array(
                        ([0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]),
                        dtype=np.int16,
                    ),
                    "disp_tested": -1,
                },
                (np.arange(1, 5), np.arange(4)),
                id="disparity=-1 & subpix=2 with mask",
            ),
            pytest.param(
                {"cfg": {"subpix": 2}, "left_image": "left_img", "right_image": "right_img", "disp_tested": -0.5},
                (np.arange(1, 5), np.arange(0.5, 4.5)),
                id="disparity=-0.5 & subpix=2 without mask",
            ),
            pytest.param(
                {"cfg": {"subpix": 4}, "left_image": "left_img", "right_image": "right_img", "disp_tested": -0.5},
                (np.arange(1, 5), np.arange(0.5, 4.5)),
                id="disparity=-0.5 & subpix=4 without mask",
            ),
            pytest.param(
                {
                    "cfg": {"subpix": 2},
                    "left_image": "left_img",
                    "right_image": "right_img",
                    "right_mask": np.array(
                        ([0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]),
                        dtype=np.int16,
                    ),
                    "disp_tested": -0.5,
                },
                (np.arange(1, 5), np.arange(0.5, 4.5)),
                id="disparity=-0.5 & subpix= with mask",
            ),
            pytest.param(
                {"cfg": {"subpix": 4}, "left_image": "left_img", "right_image": "right_img", "disp_tested": -0.75},
                (np.arange(1, 5), np.arange(0.5, 4.5)),
                id="disparity=-0.75 & subpix=4 without mask",
            ),
            pytest.param(
                {"cfg": {"subpix": 4}, "left_image": "left_img", "right_image": "right_img", "disp_tested": -0.25},
                (np.arange(1, 5), np.arange(0.5, 4.5)),
                id="disparity=-0.25 & subpix=4 without mask",
            ),
            pytest.param(
                {
                    "cfg": {"subpix": 4},
                    "left_image": "left_img",
                    "right_image": "right_img",
                    "right_mask": np.array(
                        ([0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]),
                        dtype=np.int16,
                    ),
                    "disp_tested": -0.25,
                },
                (np.arange(1, 5), np.arange(0.5, 4.5)),
                id="disparity=-0.25 & subpix=4 with mask",
            ),
            pytest.param(
                {"cfg": {"subpix": 2}, "left_image": "left_img", "right_image": "right_img", "disp_tested": 0},
                (np.arange(5), np.arange(5)),
                id="disparity=0 & subpix=2 without mask",
            ),
            pytest.param(
                {"cfg": {"subpix": 4}, "left_image": "left_img", "right_image": "right_img", "disp_tested": 0},
                (np.arange(5), np.arange(5)),
                id="disparity=0 & subpix=4 without mask",
            ),
            pytest.param(
                {"cfg": {"subpix": 2}, "left_image": "left_img", "right_image": "right_img", "disp_tested": 0.5},
                (np.arange(4), np.arange(0.5, 4.5)),
                id="disparity=0.5 & subpix=2 without mask",
            ),
            pytest.param(
                {"cfg": {"subpix": 4}, "left_image": "left_img", "right_image": "right_img", "disp_tested": 0.5},
                (np.arange(4), np.arange(0.5, 4.5)),
                id="disparity=0.5 & subpix=4 without mask",
            ),
            pytest.param(
                {"cfg": {"subpix": 4}, "left_image": "left_img", "right_image": "right_img", "disp_tested": 0.75},
                (np.arange(4), np.arange(0.5, 4.5)),
                id="disparity=0.75 & subpix=4 without mask",
            ),
            pytest.param(
                {"cfg": {"subpix": 4}, "left_image": "left_img", "right_image": "right_img", "disp_tested": 0.25},
                (np.arange(4), np.arange(0.5, 4.5)),
                id="disparity=0.25 & subpix=4 without mask",
            ),
            pytest.param(
                {"cfg": {"subpix": 2}, "left_image": "left_img", "right_image": "right_img", "disp_tested": 1},
                (np.arange(4), np.arange(1, 5)),
                id="disparity=1 & subpix=2 without mask",
            ),
            pytest.param(
                {"cfg": {"subpix": 1}, "left_image": "left_img", "right_image": "right_img", "disp_tested": 5},
                (np.array([]), np.array([])),
                id="disparity (positive) higher than the number of columns without mask",
            ),
            pytest.param(
                {"cfg": {"subpix": 1}, "left_image": "left_img", "right_image": "right_img", "disp_tested": -5},
                (np.array([]), np.array([])),
                id="disparity (negative) higher than the number of columns without mask",
            ),
        ],
        indirect=["make_mask_column_interval_parameters"],
    )
    def test_subpixellic(self, make_mask_column_interval_parameters, ground_truth):
        """Check mask_column_interval with different disparity and subpixellic"""
        matching_cost_instance = make_mask_column_interval_parameters.matching_cost_instance
        cost_volume = make_mask_column_interval_parameters.cost_volume
        coord_mask_right = make_mask_column_interval_parameters.coord_mask_right
        coord_mask_left = make_mask_column_interval_parameters.coord_mask_left
        disp = make_mask_column_interval_parameters.disp

        result = matching_cost_instance.mask_column_interval(cost_volume, coord_mask_left, coord_mask_right, disp)
        np.testing.assert_array_equal(result, ground_truth)

        # check index exist
        left_mask_index = set(coord_mask_left)
        right_mask_index = set(coord_mask_right)
        cost_volume_index = set(cost_volume.coords["col"].data)
        assert set(result[0]).issubset(left_mask_index)
        assert set(result[1]).issubset(right_mask_index)
        assert set(result[0]).issubset(cost_volume_index)

    @pytest.fixture()
    def left_roi_with_3_margins(self):
        """left roi with a (3, 3, 3, 3) margins
        user_roi = 1, 1, 3
                   2, 1, 0
                   1, 0, 1
        first_row_coord=0
        first_col_coord=1
        """
        data = np.array(
            (
                [9, 9, 9, 9, 9, 9, 9, 9, 9],
                [9, 9, 9, 9, 9, 9, 9, 9, 9],
                [9, 9, 9, 9, 9, 9, 9, 9, 9],
                [7, 6, 1, 1, 1, 3, 4, 7, 6],
                [8, 7, 1, 2, 1, 0, 2, 8, 7],
                [9, 8, 2, 1, 0, 1, 2, 9, 8],
                [8, 7, 1, 1, 1, 1, 4, 8, 7],
                [9, 9, 9, 9, 9, 9, 9, 9, 9],
                [9, 9, 9, 9, 9, 9, 9, 9, 9],
            ),
            dtype=np.float64,
        )
        return self.make_image(data, [-1, 1], first_row_coord=0, first_col_coord=1)

    @pytest.fixture()
    def right_roi_with_3_margins(self):
        """right roi with a (3, 3, 3, 3) margins
        user_roi = 1, 2, 3
                   2, 1, 0
                   2, 0, 1
        first_row_coord=0
        first_col_coord=1
        """
        data = np.array(
            (
                [9, 9, 9, 9, 9, 9, 9, 9, 9],
                [9, 9, 9, 9, 9, 9, 9, 9, 9],
                [9, 9, 9, 9, 9, 9, 9, 9, 9],
                [7, 6, 5, 1, 2, 3, 4, 7, 6],
                [8, 7, 1, 2, 1, 0, 2, 8, 7],
                [9, 8, 2, 2, 0, 1, 4, 9, 8],
                [8, 7, 1, 1, 1, 1, 2, 8, 7],
                [9, 9, 9, 9, 9, 9, 9, 9, 9],
                [9, 9, 9, 9, 9, 9, 9, 9, 9],
            ),
            dtype=np.float64,
        )
        return self.make_image(data, None, first_row_coord=0, first_col_coord=1)

    @pytest.mark.parametrize(
        ["disp", "step", "ground_truth"],
        [
            pytest.param(-3, 1, (np.arange(4, 10, 1), np.arange(1, 7, 1)), id="disparity=-3 & step=1"),
            pytest.param(-2, 1, (np.arange(3, 10, 1), np.arange(1, 8, 1)), id="disparity=-2 & step=1"),
            pytest.param(-1, 1, (np.arange(2, 10, 1), np.arange(1, 9, 1)), id="disparity=-1 & step=1"),
            pytest.param(0, 1, (np.arange(1, 10, 1), np.arange(1, 10, 1)), id="disparity=0 & step=1"),
            pytest.param(1, 1, (np.arange(1, 9, 1), np.arange(2, 10, 1)), id="disparity=1 & step=1"),
            pytest.param(2, 1, (np.arange(1, 8, 1), np.arange(3, 10, 1)), id="disparity=2 & step=1"),
            pytest.param(3, 1, (np.arange(1, 7, 1), np.arange(4, 10, 1)), id="disparity=3 & step=1"),
            pytest.param(-7, 2, (np.arange(8, 10, 2), np.arange(1, 2, 2)), id="disparity=-7 & step=2"),
            pytest.param(-6, 2, (np.arange(8, 10, 2), np.arange(2, 3, 2)), id="disparity=-6 & step=2"),
            pytest.param(-5, 2, (np.arange(6, 10, 2), np.arange(1, 4, 2)), id="disparity=-5 & step=2"),
            pytest.param(-4, 2, (np.arange(6, 10, 2), np.arange(2, 5, 2)), id="disparity=-4 & step=2"),
            pytest.param(-3, 2, (np.arange(4, 10, 2), np.arange(1, 6, 2)), id="disparity=-3 & step=2"),
            pytest.param(-2, 2, (np.arange(4, 10, 2), np.arange(2, 8, 2)), id="disparity=-2 & step=2"),
            pytest.param(-1, 2, (np.arange(2, 10, 2), np.arange(1, 9, 2)), id="disparity=-1 & step=2"),
            pytest.param(0, 2, (np.arange(2, 9, 2), np.arange(2, 9, 2)), id="disparity=0 & step=2"),
            pytest.param(1, 2, (np.arange(2, 9, 2), np.arange(3, 10, 2)), id="disparity=1 & step=2"),
            pytest.param(2, 2, (np.arange(2, 7, 2), np.arange(4, 10, 2)), id="disparity=2 & step=2"),
            pytest.param(3, 2, (np.arange(2, 7, 2), np.arange(5, 10, 2)), id="disparity=3 & step=2"),
            pytest.param(4, 2, (np.arange(2, 6, 2), np.arange(6, 9, 2)), id="disparity=4 & step=2"),
            pytest.param(5, 2, (np.arange(2, 6, 2), np.arange(7, 10, 2)), id="disparity=5 & step=2"),
            pytest.param(6, 2, (np.arange(2, 4, 2), np.arange(8, 10, 2)), id="disparity=6 & step=2"),
            pytest.param(7, 2, (np.arange(2, 4, 2), np.arange(9, 10, 2)), id="disparity=7 & step=2"),
            pytest.param(-6, 3, (np.arange(7, 10, 3), np.arange(1, 4, 3)), id="disparity=-6 & step=3"),
            pytest.param(-5, 3, (np.arange(7, 10, 3), np.arange(2, 4, 3)), id="disparity=-5 & step=3"),
            pytest.param(-4, 3, (np.arange(7, 10, 3), np.arange(3, 5, 3)), id="disparity=-4 & step=3"),
            pytest.param(-3, 3, (np.arange(4, 10, 3), np.arange(1, 6, 3)), id="disparity=-3 & step=3"),
            pytest.param(-2, 3, (np.arange(4, 10, 3), np.arange(2, 8, 3)), id="disparity=-2 & step=3"),
            pytest.param(-1, 3, (np.arange(4, 10, 3), np.arange(3, 9, 3)), id="disparity=-1 & step=3"),
            pytest.param(0, 3, (np.arange(1, 9, 3), np.arange(1, 9, 3)), id="disparity=0 & step=3"),
            pytest.param(1, 3, (np.arange(1, 9, 3), np.arange(2, 10, 3)), id="disparity=1 & step=3"),
            pytest.param(2, 3, (np.arange(1, 9, 3), np.arange(3, 10, 3)), id="disparity=2 & step=3"),
            pytest.param(3, 3, (np.arange(1, 7, 3), np.arange(4, 10, 3)), id="disparity=3 & step=3"),
            pytest.param(4, 3, (np.arange(1, 6, 3), np.arange(5, 9, 3)), id="disparity=4 & step=3"),
            pytest.param(5, 3, (np.arange(1, 6, 3), np.arange(6, 10, 3)), id="disparity=5 & step=3"),
            pytest.param(6, 3, (np.arange(1, 4, 3), np.arange(7, 10, 3)), id="disparity=6 & step=3"),
            pytest.param(-7, 4, (np.arange(8, 10, 4), np.arange(1, 2, 4)), id="disparity=-7 & step=4"),
            pytest.param(-6, 4, (np.arange(8, 10, 4), np.arange(2, 3, 4)), id="disparity=-6 & step=4"),
            pytest.param(-5, 4, (np.arange(8, 10, 4), np.arange(3, 4, 4)), id="disparity=-5 & step=4"),
            pytest.param(-4, 4, (np.arange(8, 10, 4), np.arange(4, 5, 4)), id="disparity=-4 & step=4"),
            pytest.param(-3, 4, (np.arange(4, 10, 4), np.arange(1, 6, 4)), id="disparity=-3 & step=4"),
            pytest.param(-2, 4, (np.arange(4, 10, 4), np.arange(2, 8, 4)), id="disparity=-2 & step=4"),
            pytest.param(-1, 4, (np.arange(4, 10, 4), np.arange(3, 9, 4)), id="disparity=-1 & step=4"),
            pytest.param(0, 4, (np.arange(4, 9, 4), np.arange(4, 9, 4)), id="disparity=0 & step=4"),
            pytest.param(1, 4, (np.arange(4, 9, 4), np.arange(5, 10, 4)), id="disparity=1 & step=4"),
            pytest.param(2, 4, (np.arange(4, 7, 4), np.arange(6, 10, 4)), id="disparity=2 & step=4"),
            pytest.param(3, 4, (np.arange(4, 7, 4), np.arange(7, 10, 4)), id="disparity=3 & step=4"),
            pytest.param(4, 4, (np.arange(4, 6, 4), np.arange(8, 9, 4)), id="disparity=4 & step=4"),
            pytest.param(5, 4, (np.arange(4, 6, 4), np.arange(9, 10, 4)), id="disparity=5 & step=4"),
        ],
    )
    def test_roi_pixellic(self, disp, step, ground_truth, left_roi_with_3_margins, right_roi_with_3_margins):
        """Check mask_column_interval with different disparity and subpixellic with ROI"""
        # Create instance of matching_cost and update protected parameters
        cfg = {"subpix": 1, "matching_cost_method": "zncc", "window_size": 3}
        matching_cost_instance = matching_cost.AbstractMatchingCost(**cfg)
        matching_cost_instance._step_col = step  # pylint: disable=protected-access

        # Compute cost volume
        cfg.update({"ROI": {"col": {"first": 3, "last": 5}, "row": {"first": 3, "last": 5}, "margins": (3, 3, 3, 3)}})
        grid = matching_cost_instance.allocate_cost_volume(
            left_roi_with_3_margins,
            (
                left_roi_with_3_margins["disparity"].sel(band_disp="min"),
                left_roi_with_3_margins["disparity"].sel(band_disp="max"),
            ),
            cfg,
        )
        grid = validity_mask(left_roi_with_3_margins, right_roi_with_3_margins, grid)
        cost_volume = matching_cost_instance.compute_cost_volume(
            img_left=left_roi_with_3_margins, img_right=right_roi_with_3_margins, cost_volume=grid
        )
        mask_left, mask_right = matching_cost_instance.masks_dilatation(
            img_left=left_roi_with_3_margins,
            img_right=right_roi_with_3_margins,
            window_size=matching_cost_instance._window_size,  # pylint: disable=protected-access
            subp=matching_cost_instance._subpix,  # pylint: disable=protected-access
        )

        # Test mask_column_interval
        i_right = int((disp % 1) * matching_cost_instance._subpix)  # pylint: disable=protected-access
        i_mask_right = min(1, i_right)

        result = matching_cost_instance.mask_column_interval(
            cost_volume=cost_volume,
            coord_mask_left=mask_left.coords["col"].data,
            coord_mask_right=mask_right[i_mask_right].coords["col"].data,
            disp=disp,
        )
        np.testing.assert_array_equal(result, ground_truth)

        # check index exist
        left_mask_index = set(mask_left.coords["col"].data)
        right_mask_index = set(mask_right[i_mask_right].coords["col"].data)
        cost_volume_index = set(cost_volume.coords["col"].data)
        assert set(result[0]).issubset(left_mask_index)
        assert set(result[1]).issubset(right_mask_index)
        assert set(result[0]).issubset(cost_volume_index)

    @pytest.mark.parametrize(
        ["disp", "step", "subpix", "ground_truth"],
        [
            pytest.param(
                -0.5, 2, 2, (np.arange(2, 10, 2), np.arange(1.5, 9, 2)), id="disparity=-0.5 & step=2 & subpix=2"
            ),
            pytest.param(
                -0.75, 2, 4, (np.arange(2, 10, 2), np.arange(1.5, 9, 2)), id="disparity=-0.75 & step=2 & subpix=4"
            ),
            pytest.param(
                -0.25, 2, 4, (np.arange(2, 10, 2), np.arange(1.5, 9, 2)), id="disparity=-0.25 & step=2 & subpix=4"
            ),
            pytest.param(
                -1.5, 2, 2, (np.arange(4, 10, 2), np.arange(2.5, 8, 2)), id="disparity=-1.5 & step=2 & subpix=2"
            ),
            pytest.param(
                -1.75, 2, 4, (np.arange(4, 10, 2), np.arange(2.5, 8, 2)), id="disparity=-1.75 & step=2 & subpix=4"
            ),
            pytest.param(
                -1.25, 2, 4, (np.arange(4, 10, 2), np.arange(2.5, 8, 2)), id="disparity=-1.25 & step=2 & subpix=4"
            ),
            pytest.param(
                0.5, 2, 2, (np.arange(2, 10, 2), np.arange(2.5, 10, 2)), id="disparity=0.5 & step=2 & subpix=2"
            ),
            pytest.param(
                0.75, 2, 4, (np.arange(2, 10, 2), np.arange(2.5, 10, 2)), id="disparity=0.75 & step=2 & subpix=4"
            ),
            pytest.param(
                0.25, 2, 4, (np.arange(2, 10, 2), np.arange(2.5, 10, 2)), id="disparity=0.25 & step=2 & subpix=4"
            ),
            pytest.param(1.5, 2, 2, (np.arange(2, 8, 2), np.arange(3.5, 8, 2)), id="disparity=1.5 & step=2 & subpix=2"),
            pytest.param(
                1.75, 2, 4, (np.arange(2, 8, 2), np.arange(3.5, 8, 2)), id="disparity=1.75 & step=2 & subpix=4"
            ),
            pytest.param(
                1.25, 2, 4, (np.arange(2, 8, 2), np.arange(3.5, 8, 2)), id="disparity=1.25 & step=2 & subpix=4"
            ),
            pytest.param(
                -0.5, 3, 2, (np.arange(4, 10, 3), np.arange(3.5, 9, 3)), id="disparity=-0.5 & step=3 & subpix=2"
            ),
            pytest.param(
                -0.75, 3, 4, (np.arange(4, 10, 3), np.arange(3.5, 9, 3)), id="disparity=-0.75 & step=3 & subpix=4"
            ),
            pytest.param(
                -0.25, 3, 4, (np.arange(4, 10, 3), np.arange(3.5, 9, 3)), id="disparity=-0.25 & step=3 & subpix=4"
            ),
            pytest.param(1.5, 3, 2, (np.arange(1, 8, 3), np.arange(2.5, 9, 3)), id="disparity=1.5 & step=3 & subpix=2"),
            pytest.param(
                1.75, 3, 4, (np.arange(1, 8, 3), np.arange(2.5, 9, 3)), id="disparity=1.75 & step=3 & subpix=4"
            ),
            pytest.param(
                1.25, 3, 4, (np.arange(1, 8, 3), np.arange(2.5, 9, 3)), id="disparity=1.25 & step=3 & subpix=4"
            ),
            pytest.param(
                -1.5, 4, 2, (np.arange(4, 10, 4), np.arange(2.5, 8, 4)), id="disparity=-1.5 & step=4 & subpix=2"
            ),
            pytest.param(
                -1.75, 4, 4, (np.arange(4, 10, 4), np.arange(2.5, 8, 4)), id="disparity=-1.75 & step=4 & subpix=4"
            ),
            pytest.param(
                -1.25, 4, 4, (np.arange(4, 10, 4), np.arange(2.5, 8, 4)), id="disparity=-1.25 & step=4 & subpix=4"
            ),
            pytest.param(
                0.5, 4, 2, (np.arange(4, 10, 4), np.arange(4.5, 10, 4)), id="disparity=0.5 & step=4 & subpix=2"
            ),
            pytest.param(
                0.75, 4, 4, (np.arange(4, 10, 4), np.arange(4.5, 10, 4)), id="disparity=0.75 & step=4 & subpix=4"
            ),
            pytest.param(
                0.25, 4, 4, (np.arange(4, 10, 4), np.arange(4.5, 10, 4)), id="disparity=0.25 & step=4 & subpix=4"
            ),
        ],
    )
    def test_roi_subpixellic(self, disp, step, subpix, ground_truth, left_roi_with_3_margins, right_roi_with_3_margins):
        """Check mask_column_interval with different disparity and subpixellic with ROI"""
        # Create instance of matching_cost and update protected parameters
        cfg = {"subpix": subpix, "matching_cost_method": "zncc", "window_size": 3}
        matching_cost_instance = matching_cost.AbstractMatchingCost(**cfg)
        matching_cost_instance._step_col = step  # pylint: disable=protected-access

        # Compute cost volume
        cfg.update({"ROI": {"col": {"first": 3, "last": 5}, "row": {"first": 3, "last": 5}, "margins": (3, 3, 3, 3)}})
        grid = matching_cost_instance.allocate_cost_volume(
            left_roi_with_3_margins,
            (
                left_roi_with_3_margins["disparity"].sel(band_disp="min"),
                left_roi_with_3_margins["disparity"].sel(band_disp="max"),
            ),
            cfg,
        )
        grid = validity_mask(left_roi_with_3_margins, right_roi_with_3_margins, grid)
        cost_volume = matching_cost_instance.compute_cost_volume(
            img_left=left_roi_with_3_margins, img_right=right_roi_with_3_margins, cost_volume=grid
        )
        mask_left, mask_right = matching_cost_instance.masks_dilatation(
            img_left=left_roi_with_3_margins,
            img_right=right_roi_with_3_margins,
            window_size=matching_cost_instance._window_size,  # pylint: disable=protected-access
            subp=matching_cost_instance._subpix,  # pylint: disable=protected-access
        )

        # Test mask_column_interval
        i_right = int((disp % 1) * matching_cost_instance._subpix)  # pylint: disable=protected-access
        i_mask_right = min(1, i_right)

        result = matching_cost_instance.mask_column_interval(
            cost_volume=cost_volume,
            coord_mask_left=mask_left.coords["col"].data,
            coord_mask_right=mask_right[i_mask_right].coords["col"].data,
            disp=disp,
        )
        np.testing.assert_array_equal(result, ground_truth)

        # check index exist
        left_mask_index = set(mask_left.coords["col"].data)
        right_mask_index = set(mask_right[i_mask_right].coords["col"].data)
        cost_volume_index = set(cost_volume.coords["col"].data)
        assert set(result[0]).issubset(left_mask_index)
        assert set(result[1]).issubset(right_mask_index)
        assert set(result[0]).issubset(cost_volume_index)


class TestFindNearestColumn:
    """Test find_nearest_column."""

    @pytest.mark.parametrize(
        ["value", "index", "expected"],
        [
            pytest.param(2, [1, 3, 5], 3),
            pytest.param(3, [1, 3, 5], 3),
            pytest.param(0, [1, 3, 5], 1),
            pytest.param(6, [1, 3, 5], 5),
        ],
    )
    def test_increase_value(self, value, index, expected):
        """the goal is to test in the case of a growth search, the user enters the '+' operator"""
        # Create instance of matching_cost
        cfg = {"matching_cost_method": "zncc", "window_size": 3}
        matching_cost_instance = matching_cost.AbstractMatchingCost(**cfg)

        assert matching_cost_instance.find_nearest_column(value, index) == expected

    @pytest.mark.parametrize(
        ["value", "index", "expected"],
        [
            pytest.param(2, [1, 3, 5], 1),
            pytest.param(3, [1, 3, 5], 3),
            pytest.param(0, [1, 3, 5], 1),
            pytest.param(6, [1, 3, 5], 5),
        ],
    )
    def test_decrease_value(self, value, index, expected):
        """the goal is to test in the case of a descending search, the user enters the '-' operator"""
        # Create instance of matching_cost
        cfg = {"matching_cost_method": "zncc", "window_size": 3}
        matching_cost_instance = matching_cost.AbstractMatchingCost(**cfg)

        assert matching_cost_instance.find_nearest_column(value, index, "-") == expected
