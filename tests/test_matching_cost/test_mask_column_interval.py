# type:ignore
#!/usr/bin/env python
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
This module contains functions to test the cv_masked method.
"""

# pylint: disable=redefined-outer-name

from typing import NamedTuple, Union

import numpy as np
import xarray as xr
import pytest

from pandora import matching_cost
from pandora.criteria import validity_mask
from pandora.img_tools import add_disparity
from tests import common


@pytest.fixture(params=["census", "sad", "ssd", "zncc"])
def matching_cost_method(request):
    return request.param


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
                id="disparity=-1 without mask",
            ),
            pytest.param(
                {"cfg": {"subpix": 1}, "left_image": "left_img", "right_image": "right_img", "disp_tested": 0},
                (np.arange(5), np.arange(5)),
                id="disparity=0 without mask",
            ),
            pytest.param(
                {"cfg": {"subpix": 1}, "left_image": "left_img", "right_image": "right_img", "disp_tested": 1},
                (np.arange(4), np.arange(1, 5)),
                id="disparity=1 without mask",
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
                id="disparity=1 with step=2",
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
                id="disparity=1 with right mask",
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
                id="disparity=1 with left mask",
            ),
            pytest.param(
                {"cfg": {"subpix": 1}, "left_image": "left_img", "right_image": "right_img", "disp_tested": 4},
                (np.array([0]), np.array([4])),
                id="disparity=4, upper limit",
            ),
            pytest.param(
                {"cfg": {"subpix": 1}, "left_image": "left_img", "right_image": "right_img", "disp_tested": -4},
                (np.array([4]), np.array([0])),
                id="disparity=-4, bottom limit",
            ),
            pytest.param(
                {"cfg": {"subpix": 1}, "left_image": "left_img", "right_image": "right_img", "disp_tested": 5},
                (np.array([]), np.array([])),
                id="disparity (positive) higher than the number of columns",
            ),
            pytest.param(
                {"cfg": {"subpix": 1}, "left_image": "left_img", "right_image": "right_img", "disp_tested": -5},
                (np.array([]), np.array([])),
                id="disparity (negative) higher than the number of columns",
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
            pytest.param(
                -10,
                2,
                (np.array([]), np.array([])),
                id="disparity (negative) higher than the number of columns & step=2",
            ),
            pytest.param(
                -10,
                3,
                (np.array([]), np.array([])),
                id="disparity (negative) higher than the number of columns & step=3",
            ),
            pytest.param(
                10,
                2,
                (np.array([]), np.array([])),
                id="disparity (positive) higher than the number of columns & step=2",
            ),
            pytest.param(
                10,
                3,
                (np.array([]), np.array([])),
                id="disparity (positive) higher than the number of columns & step=3",
            ),
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
            pytest.param(
                -10.5,
                2,
                2,
                (np.array([]), np.array([])),
                id="disparity (negative) higher than the number of columns & step=2",
            ),
            pytest.param(
                -10.25,
                3,
                4,
                (np.array([]), np.array([])),
                id="disparity (negative) higher than the number of columns & step=3",
            ),
            pytest.param(
                10.5,
                2,
                2,
                (np.array([]), np.array([])),
                id="disparity (positive) higher than the number of columns & step=2",
            ),
            pytest.param(
                10.25,
                3,
                4,
                (np.array([]), np.array([])),
                id="disparity (positive) higher than the number of columns & step=3",
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
        ["value", "index", "subpix", "expected"],
        [
            pytest.param(2, [1, 3, 5], 1, 3),
            pytest.param(2, [1, 3, 5], 2, 3),
            pytest.param(2, [1, 3, 5], 4, 3),
            pytest.param(3, [1, 3, 5], 1, 3),
            pytest.param(0, [1, 3, 5], 1, 1),
            pytest.param(0, [1, 3, 5], 2, 1),
            pytest.param(0, [1, 3, 5], 4, 1),
            pytest.param(6, [1, 3, 5], 1, 5),
            pytest.param(6, [1, 3, 5], 2, 5),
            pytest.param(6, [1, 3, 5], 4, 5),
        ],
    )
    def test_increase_value(self, value, index, subpix, expected):
        """the goal is to test in the case of a growth search, the user enters the '+' operator"""
        # Create instance of matching_cost
        cfg = {"matching_cost_method": "zncc", "window_size": 3, "subpix": subpix}
        matching_cost_instance = matching_cost.AbstractMatchingCost(**cfg)

        assert matching_cost_instance.find_nearest_column(value, index) == expected

    @pytest.mark.parametrize(
        ["value", "index", "subpix", "expected"],
        [
            pytest.param(2, [1, 3, 5], 1, 1),
            pytest.param(2, [1, 3, 5], 2, 1),
            pytest.param(2, [1, 3, 5], 4, 1),
            pytest.param(3, [1, 3, 5], 1, 3),
            pytest.param(0, [1, 3, 5], 1, 1),
            pytest.param(0, [1, 3, 5], 2, 1),
            pytest.param(0, [1, 3, 5], 4, 1),
            pytest.param(6, [1, 3, 5], 1, 5),
            pytest.param(6, [1, 3, 5], 2, 5),
            pytest.param(6, [1, 3, 5], 4, 5),
        ],
    )
    def test_decrease_value(self, value, index, subpix, expected):
        """the goal is to test in the case of a descending search, the user enters the '-' operator"""
        # Create instance of matching_cost
        cfg = {"matching_cost_method": "zncc", "window_size": 3, "subpix": subpix}
        matching_cost_instance = matching_cost.AbstractMatchingCost(**cfg)

        assert matching_cost_instance.find_nearest_column(value, index, "-") == expected
