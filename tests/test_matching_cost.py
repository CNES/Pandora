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

from skimage.io import imsave
from pandora import matching_cost
from pandora.margins.descriptors import HalfWindowMargins
from pandora.img_tools import create_dataset_from_inputs, add_disparity

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
    def default_image_path(self, tmp_path):
        """
        Create a fake image to test ROI in create_dataset_from_inputs
        """
        image_path = tmp_path / "left_img.tif"
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

        imsave(image_path, imarray, plugin="tifffile", photometric="MINISBLACK")

        return image_path

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
