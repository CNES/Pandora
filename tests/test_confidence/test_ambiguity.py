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
This module contains functions to test the confidence module with ambiguity.
"""

import numpy as np
import xarray as xr

import pandora.cost_volume_confidence as confidence
from pandora import img_tools


def test_ambiguity(create_img_for_confidence):
    """
    Test the ambiguity method

    """

    left_im, right_im = create_img_for_confidence

    cv_ = np.array(
        [[[np.nan, 1, 3], [4, 1, 1], [1.2, 1, 2]], [[5, np.nan, np.nan], [6.2, np.nan, np.nan], [0, np.nan, 0]]],
        dtype=np.float32,
    )

    cv_ = xr.Dataset(
        {"cost_volume": (["row", "col", "disp"], cv_)}, coords={"row": [0, 1], "col": [0, 1, 2], "disp": [-1, 0, 1]}
    )

    cv_.attrs["type_measure"] = "min"

    ambiguity_ = confidence.AbstractCostVolumeConfidence(
        **{"confidence_method": "ambiguity", "eta_max": 0.2, "eta_step": 0.1}
    )

    # Apply median filter to the disparity map. Median filter is only applied on valid pixels.
    ambiguity_.confidence_prediction(None, left_im, right_im, cv_)

    # Ambiguity integral not normalized
    _ = np.array([[4.0, 4.0, 3.0], [6.0, 6.0, 6.0]])
    # Normalized ambiguity
    amb_int = np.array([[(4.0 - 3.05) / (6.0 - 3.05), (4.0 - 3.05) / (6.0 - 3.05), 0], [1.0, 1.0, 1.0]])
    # Ambiguity to confidence measure
    ambiguity_ground_truth = 1 - amb_int

    # Check if the calculated ambiguity is equal to the ground truth (same shape and all elements equals)
    np.testing.assert_allclose(cv_["confidence_measure"].data[:, :, 0], ambiguity_ground_truth, rtol=1e-06)


def test_ambiguity_without_normalization(create_img_for_confidence):
    """
    Test the ambiguity method
    """

    left_im, right_im = create_img_for_confidence

    cv_ = np.array(
        [[[np.nan, 1, 3], [4, 1, 1], [1.2, 1, 2]], [[5, np.nan, np.nan], [6.2, np.nan, np.nan], [0, np.nan, 0]]],
        dtype=np.float32,
    )

    cv_ = xr.Dataset(
        {"cost_volume": (["row", "col", "disp"], cv_)}, coords={"row": [0, 1], "col": [0, 1, 2], "disp": [-1, 0, 1]}
    )

    cv_.attrs["type_measure"] = "min"

    ambiguity_ = confidence.AbstractCostVolumeConfidence(
        **{"confidence_method": "ambiguity", "eta_max": 0.2, "eta_step": 0.1, "normalization": False}
    )

    # Apply median filter to the disparity map. Median filter is only applied on valid pixels.
    ambiguity_.confidence_prediction(None, left_im, right_im, cv_)

    # Ambiguity integral not normalized
    amb_int_not_normalized = np.array([[4.0, 4.0, 3.0], [6.0, 6.0, 6.0]])
    # Ambiguity to confidence measure
    ambiguity_ground_truth = 1 - amb_int_not_normalized

    # Check if the calculated ambiguity is equal to the ground truth (same shape and all elements equals)
    np.testing.assert_allclose(cv_["confidence_measure"].data[:, :, 0], ambiguity_ground_truth, rtol=1e-06)


def test_compute_ambiguity_and_sampled_ambiguity():
    """
    Test ambiguity and sampled ambiguity

    """
    cv_ = np.array(
        [
            [[np.nan, 1, 3], [4, 1, 1], [np.nan, np.nan, np.nan]],
            [[5, np.nan, np.nan], [6.2, np.nan, np.nan], [0, np.nan, 0]],
        ],
        dtype=np.float32,
    )

    grids = np.array([-1 * np.ones((2, 3)), np.ones((2, 3))], dtype="int64")
    disparity_range = np.array([-1, 0, 1], dtype="float32")

    ambiguity_ = confidence.AbstractCostVolumeConfidence(
        **{"confidence_method": "ambiguity", "eta_max": 0.2, "eta_step": 0.1}
    )

    etas = np.arange(0.0, 0.2, 0.1)
    nbr_etas = etas.shape[0]

    amb, sampled_amb = ambiguity_.compute_ambiguity_and_sampled_ambiguity(cv_, etas, nbr_etas, grids, disparity_range)

    # Ambiguity integral
    gt_amb_int = np.array([[4.0, 4.0, 6.0], [6.0, 6.0, 6.0]])

    # Sampled ambiguity
    gt_sam_amb = np.array([[[2, 2], [2, 2], [3, 3]], [[3, 3], [3, 3], [3, 3]]], dtype=np.float32)

    # Check if the calculated ambiguity is equal to the ground truth (same shape and all elements equals)
    np.testing.assert_allclose(amb, gt_amb_int, rtol=1e-06)
    np.testing.assert_allclose(sampled_amb, gt_sam_amb, rtol=1e-06)


def test_compute_ambiguity_with_variable_disparity(
    create_grids_and_disparity_range_with_variable_disparities,
    create_cv_for_variable_disparities,
):
    """
    Test ambiguity with variable disparity interval
    """

    grids, disparity_range = create_grids_and_disparity_range_with_variable_disparities

    cv_ = create_cv_for_variable_disparities

    ambiguity_ = confidence.AbstractCostVolumeConfidence(
        **{"confidence_method": "ambiguity", "eta_max": 0.2, "eta_step": 0.1}
    )

    etas = np.arange(0.0, 0.2, 0.1)
    nbr_etas = etas.shape[0]

    amb = ambiguity_.compute_ambiguity(cv_, etas, nbr_etas, grids, disparity_range)

    # Ambiguity integral
    gt_amb_int = np.array([[6.0, 4.0, 4.0, 4.0], [4.0, 4.0, 4.0, 6.0], [4.0, 4.0, 2.0, 4.0], [4.0, 4.0, 4.0, 4.0]])

    # Check if the calculated ambiguity is equal to the ground truth (same shape and all elements equals)
    np.testing.assert_allclose(amb, gt_amb_int, rtol=1e-06)


def test_compute_compute_ambiguity_and_sampled_ambiguity_with_variable_disparity(
    create_grids_and_disparity_range_with_variable_disparities,
    create_cv_for_variable_disparities,
):
    """
    Test ambiguity with variable disparity interval
    """

    grids, disparity_range = create_grids_and_disparity_range_with_variable_disparities

    cv_ = create_cv_for_variable_disparities

    ambiguity_ = confidence.AbstractCostVolumeConfidence(
        **{"confidence_method": "ambiguity", "eta_max": 0.2, "eta_step": 0.1}
    )

    etas = np.arange(0.0, 0.2, 0.1)
    nbr_etas = etas.shape[0]

    amb, amb_sampl = ambiguity_.compute_ambiguity_and_sampled_ambiguity(cv_, etas, nbr_etas, grids, disparity_range)

    # Ambiguity integral
    gt_amb_int = np.array([[6.0, 4.0, 4.0, 4.0], [4.0, 4.0, 4.0, 6.0], [4.0, 4.0, 2.0, 4.0], [4.0, 4.0, 4.0, 4.0]])

    gt_sampl_amb = np.array(
        [
            [[3.0, 3.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0]],
            [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [3.0, 3.0]],
            [[2.0, 2.0], [2.0, 2.0], [1.0, 1.0], [2.0, 2.0]],
            [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0]],
        ]
    )

    # Check if the calculated ambiguity is equal to the ground truth (same shape and all elements equals)
    np.testing.assert_allclose(amb, gt_amb_int, rtol=1e-06)
    np.testing.assert_allclose(amb_sampl, gt_sampl_amb, rtol=1e-06)


def test_normalize_with_extremum(create_img_for_confidence):
    """
    test normalize_with_extremum function
    """

    # create datas
    left_im, _ = create_img_for_confidence

    # Add tiles disparity
    left_im.attrs["disp_min"] = 0
    left_im.attrs["disp_max"] = 1

    # Add global disparity
    left_im = img_tools.add_global_disparity(left_im, -2, 2)

    ambiguity_ = confidence.AbstractCostVolumeConfidence(
        **{"confidence_method": "ambiguity", "eta_max": 0.2, "eta_step": 0.1}
    )
    ambiguity = np.ones((4, 4))

    # normalize_with_extremum function to test
    amb_test = ambiguity_.normalize_with_extremum(ambiguity, left_im, ambiguity_._nbr_etas)

    # create ground truth
    nbr_etas = np.arange(0.0, 0.2, 0.1).shape[0]
    amb_vt = np.copy(ambiguity) / ((2 - (-2)) * nbr_etas)

    np.testing.assert_array_equal(amb_test, amb_vt)


def test_perfect_case_min(
    create_grids_and_disparity_range_with_variable_disparities, create_cv_for_variable_disparities
):
    """
    Test a perfect case for min matching cost functions
    """

    grids, disparity_range = create_grids_and_disparity_range_with_variable_disparities

    cv_ = create_cv_for_variable_disparities

    value_min = 0.1
    ind_min = np.nanargmin(cv_[1, 1, :])
    cv_[1, 1, :] = np.full(3, 24)
    cv_[1, 1, ind_min] = value_min

    ambiguity_ = confidence.AbstractCostVolumeConfidence(
        **{"confidence_method": "ambiguity", "eta_max": 0.2, "eta_step": 0.1}
    )

    amb = ambiguity_.compute_ambiguity(cv_, ambiguity_._etas, ambiguity_._nbr_etas, grids, disparity_range)

    ambiguity = ambiguity_.normalize_with_percentile(amb)

    confidence_measure = 1 - ambiguity

    np.testing.assert_almost_equal(1.0, confidence_measure[1, 1])


def test_perfect_case_max(
    create_grids_and_disparity_range_with_variable_disparities, create_cv_for_variable_disparities
):
    """
    Test a perfect case for max matching cost functions
    """
    grids, disparity_range = create_grids_and_disparity_range_with_variable_disparities

    cv_ = create_cv_for_variable_disparities

    value_max = 20
    ind_max = np.nanargmax(cv_[1, 1, :])
    cv_[1, 1, :] = np.full(3, -30)
    cv_[1, 1, ind_max] = value_max

    # cv_ is win max measure, we need to revert it
    cv_ *= -1

    ambiguity_ = confidence.AbstractCostVolumeConfidence(
        **{"confidence_method": "ambiguity", "eta_max": 0.2, "eta_step": 0.1}
    )

    amb = ambiguity_.compute_ambiguity(cv_, ambiguity_._etas, ambiguity_._nbr_etas, grids, disparity_range)

    ambiguity = ambiguity_.normalize_with_percentile(amb)

    confidence_measure = 1 - ambiguity

    np.testing.assert_almost_equal(1.0, confidence_measure[1, 1])
