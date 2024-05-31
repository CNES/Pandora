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

"""
Set of fixtures available to all confidence tests.
"""

import numpy as np
import pytest
import xarray as xr
from rasterio import Affine

from pandora.img_tools import add_disparity, add_disparity_grid


@pytest.fixture()
def create_img_for_confidence():
    """
    Fixture containing left and right images for confidence tests
    """

    # Create left and right images
    left_im = np.array([[2, 5, 3, 1], [5, 3, 2, 1], [4, 2, 3, 2], [4, 5, 3, 2]], dtype=np.float32)

    mask_ = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]], dtype=np.int16)

    left_im = xr.Dataset(
        {"im": (["row", "col"], left_im), "msk": (["row", "col"], mask_)},
        coords={"row": np.arange(left_im.shape[0]), "col": np.arange(left_im.shape[1])},
    )
    # Add image conf to the image dataset

    left_im.attrs = {
        "no_data_img": 0,
        "valid_pixels": 0,
        "no_data_mask": 1,
        "crs": None,
        "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    }
    left_im.pipe(add_disparity, disparity=[-1, 1], window=None)

    right_im = np.array([[1, 2, 1, 2], [2, 3, 5, 3], [0, 2, 4, 2], [5, 3, 1, 4]], dtype=np.float32)

    mask_ = np.full((4, 4), 0, dtype=np.int16)

    right_im = xr.Dataset(
        {"im": (["row", "col"], right_im), "msk": (["row", "col"], mask_)},
        coords={"row": np.arange(right_im.shape[0]), "col": np.arange(right_im.shape[1])},
    )
    # Add image conf to the image dataset
    right_im.attrs = {
        "no_data_img": 0,
        "valid_pixels": 0,
        "no_data_mask": 1,
        "crs": None,
        "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    }
    right_im.pipe(add_disparity, disparity=[-1, 1], window=None)

    return left_im, right_im


@pytest.fixture()
def create_grids_and_disparity_range_with_variable_disparities():
    """
    Fixture containing grids and disparity range for tests with variable disparities
    """
    grids = np.array(
        [
            [[-1, 0, -1, 0], [0, -1, 0, -1], [0, 0, 0, -1], [-1, -1, -1, -1]],
            [[1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 0], [0, 0, 0, 1]],
        ],
        dtype=np.int64,
    )
    disparity_range = np.array([-1, 0, 1], dtype=np.float32)

    return grids, disparity_range


@pytest.fixture()
def create_cv_for_variable_disparities():
    """
    Fixture containing cv for tests with variable disparities
    """
    cv_ = np.array(
        [
            [[np.nan, 1, 3, 2], [4, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [np.nan, 1, 3, 2]],
            [
                [5, np.nan, np.nan, np.nan],
                [6.2, np.nan, np.nan, np.nan],
                [0, np.nan, 0, 0],
                [5, np.nan, np.nan, np.nan],
            ],
            [[np.nan, 2, 4, 5], [np.nan, 5, 0, 1], [0, 0, 2, np.nan], [np.nan, 2, 4, 5]],
        ],
        dtype=np.float32,
    )
    cv_ = np.rollaxis(cv_, 0, 3)

    return cv_


@pytest.fixture()
def create_images(create_grids_and_disparity_range_with_variable_disparities):
    """Make images with a disparity grid."""

    grids, _ = create_grids_and_disparity_range_with_variable_disparities

    disparity = xr.DataArray(grids, dims=["band_disp", "row", "col"])
    data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, 1, 2], [3, 4, 5, 6]], dtype=np.float64)
    left = xr.Dataset(
        data_vars={
            "im": (["row", "col"], data),
        },
        coords={
            "row": np.arange(data.shape[0]),
            "col": np.arange(data.shape[1]),
        },
        attrs={
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        },
    ).pipe(add_disparity_grid, disparity)

    data = np.array([[7, 1, 5, 9], [8, 2, 6, 0], [9, 3, 7, 1], [0, 4, 8, 2]], dtype=np.float64)
    right = xr.Dataset(
        data_vars={
            "im": (["row", "col"], data),
        },
        coords={
            "row": np.arange(data.shape[0]),
            "col": np.arange(data.shape[1]),
        },
        attrs={
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        },
    ).pipe(add_disparity, disparity=disparity, window=None)

    return left, right, grids
