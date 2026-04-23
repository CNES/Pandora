# type: ignore
#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2026 Centre National d'Etudes Spatiales (CNES).
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
This module contains functions to test the matching cost's cv_masked function.
"""

# pylint with pytest's fixtures compatibility:
# pylint: disable=redefined-outer-name
# pylint: disable=too-many-lines

from typing import NamedTuple

import numpy as np
import xarray as xr
import pytest

from pandora import matching_cost
from pandora.criteria import validity_mask
from pandora.img_tools import add_disparity, add_disparity_grid
from tests import common  # pylint: disable=no-name-in-module


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
