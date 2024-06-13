# type:ignore
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
This module contains functions to test the confidence module for std_intensity.
"""

import numpy as np
import xarray as xr
from rasterio import Affine

import pandora.cost_volume_confidence as confidence
from pandora import matching_cost
from pandora.criteria import validity_mask
from pandora.img_tools import add_disparity


def test_std_intensity():
    """
    Test the confidence measure std_intensity
    """
    # Create a stereo object
    left_data = np.array(
        ([1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 2, 1], [1, 1, 1, 4, 3, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]),
        dtype=np.float32,
    )
    left = xr.Dataset(
        {"im": (["row", "col"], left_data)},
        coords={"row": np.arange(left_data.shape[0]), "col": np.arange(left_data.shape[1])},
    )
    left.attrs = {
        "no_data_img": 0,
        "valid_pixels": 0,
        "no_data_mask": 1,
        "crs": None,
        "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    }
    left.pipe(add_disparity, disparity=[-2, 1], window=None)

    right_data = np.array(
        ([1, 1, 1, 2, 2, 2], [1, 1, 1, 4, 2, 4], [1, 1, 1, 4, 4, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]),
        dtype=np.float64,
    )
    right = xr.Dataset(
        {"im": (["row", "col"], right_data)},
        coords={"row": np.arange(right_data.shape[0]), "col": np.arange(right_data.shape[1])},
    )
    right.attrs = {
        "no_data_img": 0,
        "valid_pixels": 0,
        "no_data_mask": 1,
        "crs": None,
        "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    }

    # create matching_cost object
    stereo_matcher = matching_cost.AbstractMatchingCost(
        **{"matching_cost_method": "sad", "window_size": 3, "subpix": 1}
    )

    # Compute bright standard deviation inside a window of size 3 and create the confidence measure
    std_bright_ground_truth = np.array(
        [
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, 0.0, np.sqrt(8 / 9), np.sqrt(10 / 9), np.sqrt(10 / 9), np.nan],
            [np.nan, 0.0, np.sqrt(8 / 9), np.sqrt(10 / 9), np.sqrt(10 / 9), np.nan],
            [np.nan, 0.0, np.sqrt(8 / 9), np.sqrt(92 / 81), np.sqrt(92 / 81), np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        ],
        dtype=np.float32,
    )

    std_bright_ground_truth = std_bright_ground_truth.reshape((5, 6, 1))

    # compute with compute_cost_volume
    grid = stereo_matcher.allocate_cost_volume(
        left, (left["disparity"].sel(band_disp="min"), left["disparity"].sel(band_disp="max"))
    )

    # Compute validity mask
    grid = validity_mask(left, right, grid)

    cv = stereo_matcher.compute_cost_volume(img_left=left, img_right=right, cost_volume=grid)
    stereo_matcher.cv_masked(
        left,
        right,
        cv,
        left["disparity"].sel(band_disp="min"),
        left["disparity"].sel(band_disp="max"),
    )

    std_intensity = confidence.AbstractCostVolumeConfidence(**{"confidence_method": "std_intensity"})

    # Compute confidence prediction
    _, cv_with_intensity = std_intensity.confidence_prediction(None, left, right, cv)

    # Check if the calculated confidence_measure is equal to the ground truth (same shape and all elements equals)
    assert np.sum(cv_with_intensity.coords["indicator"].data != ["confidence_from_intensity_std"]) == 0
    np.testing.assert_array_equal(cv_with_intensity["confidence_measure"].data, std_bright_ground_truth)


def test_std_intensity_multiband():
    """
    Test the confidence measure std_intensity with multiband input images
    """
    # Create a stereo object
    left_data = np.array(
        [
            [[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 2, 1], [0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0]],
            [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 2, 1], [1, 1, 1, 4, 3, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]],
        ],
        dtype=np.float32,
    )
    left = xr.Dataset(
        {"im": (["band_im", "row", "col"], left_data)},
        coords={
            "band_im": ["r", "g"],
            "row": np.arange(left_data.shape[1]),
            "col": np.arange(left_data.shape[2]),
        },
    )
    left.attrs = {
        "no_data_img": 0,
        "valid_pixels": 0,
        "no_data_mask": 1,
        "crs": None,
        "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    }
    left.pipe(add_disparity, disparity=[-2, 1], window=None)

    right_data = np.array(
        [
            [[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 2, 1], [0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0]],
            [[1, 1, 1, 2, 2, 2], [1, 1, 1, 4, 2, 4], [1, 1, 1, 4, 4, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]],
        ],
        dtype=np.float64,
    )
    right = xr.Dataset(
        {"im": (["band_im", "row", "col"], right_data)},
        coords={
            "band_im": ["r", "g"],
            "row": np.arange(right_data.shape[1]),
            "col": np.arange(right_data.shape[2]),
        },
    )

    right.attrs = {
        "no_data_img": 0,
        "valid_pixels": 0,
        "no_data_mask": 1,
        "crs": None,
        "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    }

    # create matching_cost object
    stereo_matcher = matching_cost.AbstractMatchingCost(
        **{"matching_cost_method": "sad", "window_size": 3, "subpix": 1, "band": "g"}
    )

    # Compute bright standard deviation inside a window of size 3 and create the confidence measure
    std_bright_ground_truth = np.array(
        [
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, 0.0, np.sqrt(8 / 9), np.sqrt(10 / 9), np.sqrt(10 / 9), np.nan],
            [np.nan, 0.0, np.sqrt(8 / 9), np.sqrt(10 / 9), np.sqrt(10 / 9), np.nan],
            [np.nan, 0.0, np.sqrt(8 / 9), np.sqrt(92 / 81), np.sqrt(92 / 81), np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        ],
        dtype=np.float32,
    )

    std_bright_ground_truth = std_bright_ground_truth.reshape((5, 6, 1))

    # compute with compute_cost_volume
    grid = stereo_matcher.allocate_cost_volume(
        left, (left["disparity"].sel(band_disp="min"), left["disparity"].sel(band_disp="max"))
    )

    # Compute validity mask
    grid = validity_mask(left, right, grid)

    cv = stereo_matcher.compute_cost_volume(img_left=left, img_right=right, cost_volume=grid)
    stereo_matcher.cv_masked(
        left,
        right,
        cv,
        left["disparity"].sel(band_disp="min"),
        left["disparity"].sel(band_disp="max"),
    )

    std_intensity = confidence.AbstractCostVolumeConfidence(**{"confidence_method": "std_intensity"})

    # Compute confidence prediction
    _, cv_with_intensity = std_intensity.confidence_prediction(None, left, right, cv)

    # Check if the calculated confidence_measure is equal to the ground truth (same shape and all elements equals)
    assert np.sum(cv_with_intensity.coords["indicator"].data != ["confidence_from_intensity_std"]) == 0
    np.testing.assert_array_equal(cv_with_intensity["confidence_measure"].data, std_bright_ground_truth)
