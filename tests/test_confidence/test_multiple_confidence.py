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
This module contains functions to test the confidence module with mixed measure.
"""

import numpy as np
import xarray as xr
from rasterio import Affine

import pandora
from pandora.img_tools import add_disparity
from pandora.state_machine import PandoraMachine


def test_ambiguity_std_full_pipeline(create_img_for_confidence):
    """
    Test the ambiguity and std_intensity methods using the pandora run method

    """
    left_im, right_im = create_img_for_confidence

    user_cfg = {
        "input": {"disp_left": (-1, 1)},
        "pipeline": {
            "matching_cost": {"matching_cost_method": "sad", "window_size": 1, "subpix": 1},
            "cost_volume_confidence": {"confidence_method": "std_intensity"},
            "cost_volume_confidence.2": {"confidence_method": "ambiguity", "eta_max": 0.3, "eta_step": 0.25},
            "disparity": {"disparity_method": "wta"},
            "filter": {"filter_method": "median"},
        },
    }
    pandora_machine = PandoraMachine()

    # Update the user configuration with default values
    cfg = pandora.check_configuration.update_conf(pandora.check_configuration.default_short_configuration, user_cfg)

    # Run the pandora pipeline
    left, _ = pandora.run(pandora_machine, left_im, right_im, cfg)

    assert (
        np.sum(left.coords["indicator"].data != ["confidence_from_intensity_std", "confidence_from_ambiguity.2"]) == 0
    )

    # ----- Check ambiguity results ------

    # Cost volume               Normalized cost volume
    # [[[nan  1.  0.]           [[[ nan 0.25 0.  ]
    #   [ 4.  3.  4.]            [1.   0.75 1.  ]
    #   [ 1.  2.  1.]            [0.25 0.5  0.25]
    #   [ 0.  1. nan]]           [0.   0.25  nan]]
    #  [[nan  3.  2.]           [[ nan 0.75 0.5 ]
    #   [nan nan nan]            [ nan  nan  nan]
    #   [ 1.  3.  1.]            [0.25 0.75 0.25]
    #   [ 4.  2. nan]]           [1.   0.5   nan]]
    #  [[nan  4.  2.]           [[ nan 1.   0.5 ]
    #   [ 2.  0.  2.]            [0.5  0.   0.5 ]
    #   [ 1.  1.  1.]            [0.25 0.25 0.25]
    #   [ 2.  0. nan]]           [0.5  0.    nan]]
    #  [[nan  1.  1.]           [[ nan 0.25 0.25]
    #   [ 0.  2.  4.]            [0.   0.5  1.  ]
    #   [ 0.  2.  1.]            [0.   0.5  0.25]
    #   [nan nan nan]]]          [ nan  nan  nan]]]
    #
    # Ambiguity integral not normalized
    amb_int = np.array([[5.0, 4.0, 5.0, 5.0], [5.0, 6.0, 4.0, 4.0], [4.0, 2.0, 6.0, 4.0], [6.0, 2.0, 3.0, 6.0]])
    # Normalized ambiguity
    amb_int = np.array(
        [
            [(5.0 - 2) / 4.0, (4.0 - 2) / 4.0, (5.0 - 2) / 4.0, (5.0 - 2) / 4.0],
            [(5.0 - 2) / 4.0, (6.0 - 2) / 4.0, (4.0 - 2) / 4.0, (4.0 - 2) / 4.0],
            [(4.0 - 2) / 4.0, (2.0 - 2) / 4.0, (6.0 - 2) / 4.0, (4.0 - 2) / 4.0],
            [(6.0 - 2) / 4.0, (2.0 - 2) / 4.0, (3.0 - 2) / 4.0, (6.0 - 2) / 4.0],
        ]
    )
    # Ambiguity to confidence measure
    ambiguity_ground_truth = 1 - amb_int

    # Check if the calculated ambiguity is equal to the ground truth (same shape and all elements equals)
    np.testing.assert_allclose(left["confidence_measure"].data[:, :, 1], ambiguity_ground_truth, rtol=1e-06)

    # ----- Check std_intensity results ------
    std_intensity_gt = np.array(
        [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    )
    # Check if the calculated std_intensity is equal to the ground truth (same shape and all elements equals)
    np.testing.assert_array_equal(left["confidence_measure"].data[:, :, 0], std_intensity_gt)


def test_non_normalized_ambiguity_std_full_pipeline(create_img_for_confidence):
    """
    Test the non normalized ambiguity and std_intensity methods using the pandora run method

    """
    left_im, right_im = create_img_for_confidence

    user_cfg = {
        "input": {"disp_left": (-1, 1)},
        "pipeline": {
            "matching_cost": {"matching_cost_method": "sad", "window_size": 1, "subpix": 1},
            "cost_volume_confidence": {"confidence_method": "std_intensity"},
            "cost_volume_confidence.2": {
                "confidence_method": "ambiguity",
                "eta_max": 0.3,
                "eta_step": 0.25,
                "normalization": False,
            },
            "disparity": {"disparity_method": "wta"},
            "filter": {"filter_method": "median"},
        },
    }
    pandora_machine = PandoraMachine()

    # Update the user configuration with default values
    cfg = pandora.check_configuration.update_conf(pandora.check_configuration.default_short_configuration, user_cfg)

    # Run the pandora pipeline
    left, _ = pandora.run(pandora_machine, left_im, right_im, cfg)

    assert (
        np.sum(left.coords["indicator"].data != ["confidence_from_intensity_std", "confidence_from_ambiguity.2"]) == 0
    )

    # ----- Check ambiguity results ------

    # Cost volume               Normalized cost volume
    # [[[nan  1.  0.]           [[[ nan 0.25 0.  ]
    #   [ 4.  3.  4.]            [1.   0.75 1.  ]
    #   [ 1.  2.  1.]            [0.25 0.5  0.25]
    #   [ 0.  1. nan]]           [0.   0.25  nan]]
    #  [[nan  3.  2.]           [[ nan 0.75 0.5 ]
    #   [nan nan nan]            [ nan  nan  nan]
    #   [ 1.  3.  1.]            [0.25 0.75 0.25]
    #   [ 4.  2. nan]]           [1.   0.5   nan]]
    #  [[nan  4.  2.]           [[ nan 1.   0.5 ]
    #   [ 2.  0.  2.]            [0.5  0.   0.5 ]
    #   [ 1.  1.  1.]            [0.25 0.25 0.25]
    #   [ 2.  0. nan]]           [0.5  0.    nan]]
    #  [[nan  1.  1.]           [[ nan 0.25 0.25]
    #   [ 0.  2.  4.]            [0.   0.5  1.  ]
    #   [ 0.  2.  1.]            [0.   0.5  0.25]
    #   [nan nan nan]]]          [ nan  nan  nan]]]
    #
    # Ambiguity integral not normalized
    amb_int_not_normalized = np.array(
        [[5.0, 4.0, 5.0, 5.0], [5.0, 6.0, 4.0, 4.0], [4.0, 2.0, 6.0, 4.0], [6.0, 2.0, 3.0, 6.0]]
    )

    # Ambiguity to confidence measure
    ambiguity_ground_truth = 1 - amb_int_not_normalized

    # Check if the calculated ambiguity is equal to the ground truth (same shape and all elements equals)
    np.testing.assert_allclose(left["confidence_measure"].data[:, :, 1], ambiguity_ground_truth, rtol=1e-06)

    # ----- Check std_intensity results ------
    std_intensity_gt = np.array(
        [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    )
    # Check if the calculated std_intensity is equal to the ground truth (same shape and all elements equals)
    np.testing.assert_array_equal(left["confidence_measure"].data[:, :, 0], std_intensity_gt)
