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
This module contains functions to test the confidence module.
"""

import unittest

import numpy as np
import xarray as xr
from rasterio import Affine

import pandora
import pandora.cost_volume_confidence as confidence
from pandora import matching_cost
from pandora.state_machine import PandoraMachine


class TestConfidence(unittest.TestCase):
    """
    TestConfidence class allows to test the confidence module
    """

    def setUp(self):
        """
        Method called to prepare the test fixture

        """

    @staticmethod
    def test_ambiguity():
        """
        Test the ambiguity method

        """
        cv_ = np.array(
            [[[np.nan, 1, 3], [4, 1, 1], [1.2, 1, 2]], [[5, np.nan, np.nan], [6.2, np.nan, np.nan], [0, np.nan, 0]]],
            dtype=np.float32,
        )

        cv_ = xr.Dataset(
            {"cost_volume": (["row", "col", "disp"], cv_)}, coords={"row": [0, 1], "col": [0, 1, 2], "disp": [-1, 0, 1]}
        )

        ambiguity_ = confidence.AbstractCostVolumeConfidence(
            **{"confidence_method": "ambiguity", "eta_max": 0.2, "eta_step": 0.1}
        )

        # Apply median filter to the disparity map. Median filter is only applied on valid pixels.
        ambiguity_.confidence_prediction(None, None, None, cv_)

        # Ambiguity integral not normalized
        amb_int = np.array([[4.0, 4.0, 3.0], [6.0, 6.0, 6.0]])  # pylint: disable=unused-variable
        # Normalized ambiguity
        amb_int = np.array([[(4.0 - 3.05) / (6.0 - 3.05), (4.0 - 3.05) / (6.0 - 3.05), 0], [1.0, 1.0, 1.0]])
        #  Ambiguity to confidence measure
        ambiguity_ground_truth = 1 - amb_int

        # Check if the calculated ambiguity is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_allclose(cv_["confidence_measure"].data[:, :, 0], ambiguity_ground_truth, rtol=1e-06)

    @staticmethod
    def test_ambiguity_std_full_pipeline():
        """
        Test the ambiguity and std_intensity methods using the pandora run method

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

        right_im = np.array([[1, 2, 1, 2], [2, 3, 5, 3], [0, 2, 4, 2], [5, 3, 1, 4]], dtype=np.float32)

        mask_ = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int16)

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

        user_cfg = {
            "input": {"disp_min": -1, "disp_max": 1},
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
        cfg = pandora.check_json.update_conf(pandora.check_json.default_short_configuration, user_cfg)

        # Run the pandora pipeline
        left, _ = pandora.run(pandora_machine, left_im, right_im, -1, 1, cfg["pipeline"])

        assert np.sum(left.coords["indicator"].data != ["stereo_pandora_intensityStd", "ambiguity_confidence"]) == 0

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
        #  Ambiguity to confidence measure
        ambiguity_ground_truth = 1 - amb_int

        # Check if the calculated ambiguity is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_allclose(left["confidence_measure"].data[:, :, 1], ambiguity_ground_truth, rtol=1e-06)

        # ----- Check std_intensity results ------
        std_intensity_gt = np.array(
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        )
        # Check if the calculated std_intensity is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(left["confidence_measure"].data[:, :, 0], std_intensity_gt)

    @staticmethod
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

        # load plugins
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
        cv = stereo_matcher.compute_cost_volume(left, right, disp_min=-2, disp_max=1)
        stereo_matcher.cv_masked(left, right, cv, -2, 1)

        std_intensity = confidence.AbstractCostVolumeConfidence(**{"confidence_method": "std_intensity"})

        # Apply median filter to the disparity map. Median filter is only applied on valid pixels.
        _, cv_with_intensity = std_intensity.confidence_prediction(None, left, right, cv)

        # Check if the calculated confidence_measure is equal to the ground truth (same shape and all elements equals)
        assert np.sum(cv_with_intensity.coords["indicator"].data != ["stereo_pandora_intensityStd"]) == 0
        np.testing.assert_array_equal(cv_with_intensity["confidence_measure"].data, std_bright_ground_truth)

    @staticmethod
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

        ambiguity_ = confidence.AbstractCostVolumeConfidence(
            **{"confidence_method": "ambiguity", "eta_max": 0.2, "eta_step": 0.1}
        )

        amb, sampled_amb = ambiguity_.compute_ambiguity_and_sampled_ambiguity(cv_, 0.0, 0.2, 0.1)

        # Ambiguity integral
        gt_amb_int = np.array([[4.0, 4.0, 6.0], [6.0, 6.0, 6.0]])

        # Sampled ambiguity
        gt_sam_amb = np.array([[[2, 2], [2, 2], [3, 3]], [[3, 3], [3, 3], [3, 3]]], dtype=np.float32)

        # Check if the calculated ambiguity is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_allclose(amb, gt_amb_int, rtol=1e-06)
        np.testing.assert_allclose(sampled_amb, gt_sam_amb, rtol=1e-06)

    @staticmethod
    def test_compute_risk():
        """
        Test the compute_risk method

        """
        risk_ = confidence.AbstractCostVolumeConfidence(
            **{"confidence_method": "risk", "eta_max": 0.5, "eta_step": 0.3}
        )
        cv_ = np.array(
            [
                [
                    [39, 28, 28, 34.5],
                    [49, 34, 41.5, 34],
                    [np.nan, np.nan, np.nan, np.nan],
                ]
            ],
            dtype=np.float32,
        )

        sampled_ambiguity = np.array([[[2.0, 2.0], [2.0, 2.0], [4.0, 4.0]]], dtype=np.float32)
        max_cost = 49
        min_cost = 28
        normalized_min_cost = [  # pylint:disable=unused-variable
            (28 - min_cost) / (max_cost - min_cost),
            (34 - min_cost) / (max_cost - min_cost),
            np.nan,
        ]
        normalized_cv = [  # pylint:disable=unused-variable
            [
                (39 - min_cost) / (max_cost - min_cost),
                (28.01 - min_cost) / (max_cost - min_cost),
                (28 - min_cost) / (max_cost - min_cost),
                (34.5 - min_cost) / (max_cost - min_cost),
            ],
            [
                (49 - min_cost) / (max_cost - min_cost),
                (41.5 - min_cost) / (max_cost - min_cost),
                (34.1 - min_cost) / (max_cost - min_cost),
                (34 - min_cost) / (max_cost - min_cost),
            ],
            [np.nan, np.nan, np.nan, np.nan],
        ]

        # invalidate similarity values outside of [min;min+eta[
        masked_normalized_cv = [  # pylint:disable=unused-variable
            [np.nan, (28.01 - min_cost) / (max_cost - min_cost), (28 - min_cost) / (max_cost - min_cost), np.nan],
            [np.nan, (28.01 - min_cost) / (max_cost - min_cost), (28 - min_cost) / (max_cost - min_cost), np.nan],
            [np.nan, (34.1 - min_cost) / (max_cost - min_cost), np.nan, (34 - min_cost) / (max_cost - min_cost)],
            [np.nan, (34.1 - min_cost) / (max_cost - min_cost), np.nan, (34 - min_cost) / (max_cost - min_cost)],
            [np.nan, np.nan, np.nan, np.nan],
        ]

        disparities = [  # pylint:disable=unused-variable
            [np.nan, 1, 2, np.nan],
            [np.nan, 1, 2, np.nan],
            [np.nan, 1, np.nan, 3],
            [np.nan, 1, np.nan, 3],
            [np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan],
        ]
        # Risk mas is defined as risk(p,k) = mean(max(di) - min(di)) for di in [cmin(p);cmin(p)+kŋ[
        gt_risk_max = [[((2 - 1) + (2 - 1)) / 2, ((3 - 1) + (3 - 1)) / 2, np.nan]]
        # Risk min is defined as mean( (1+risk(p,k)) - amb(p,k) )
        gt_risk_min = [
            [
                (((1 + 1) - sampled_ambiguity[0][0][0]) + ((1 + 1) - sampled_ambiguity[0][0][1])) / 2,
                (((1 + 2) - sampled_ambiguity[0][1][0]) + ((1 + 2) - sampled_ambiguity[0][1][1])) / 2,
                np.nan,
            ]
        ]

        # Compute risk
        risk_max, risk_min = risk_.compute_risk(cv_, sampled_ambiguity, 0.0, 0.5, 0.3)

        # Check if the calculated risks are equal to the ground truth (same shape and all elements equals)
        np.testing.assert_allclose(gt_risk_max, risk_max, rtol=1e-06)
        np.testing.assert_allclose(gt_risk_min, risk_min, rtol=1e-06)

    @staticmethod
    def test_compute_risk_and_sampled_risk():
        """
        Test the compute_risk_and_sampled_risk method

        """
        risk_ = confidence.AbstractCostVolumeConfidence(
            **{"confidence_method": "risk", "eta_max": 0.5, "eta_step": 0.3}
        )
        cv_ = np.array(
            [
                [
                    [39, 28, 28, 34.5],
                    [49, 34, 41.5, 34],
                    [np.nan, np.nan, np.nan, np.nan],
                ]
            ],
            dtype=np.float32,
        )

        sampled_ambiguity = np.array([[[2.0, 2.0], [2.0, 2.0], [4.0, 4.0]]], dtype=np.float32)
        max_cost = 49
        min_cost = 28
        normalized_min_cost = [  # pylint:disable=unused-variable
            (28 - min_cost) / (max_cost - min_cost),
            (34 - min_cost) / (max_cost - min_cost),
            np.nan,
        ]
        normalized_cv = [  # pylint:disable=unused-variable
            [
                (39 - min_cost) / (max_cost - min_cost),
                (28.01 - min_cost) / (max_cost - min_cost),
                (28 - min_cost) / (max_cost - min_cost),
                (34.5 - min_cost) / (max_cost - min_cost),
            ],
            [
                (49 - min_cost) / (max_cost - min_cost),
                (41.5 - min_cost) / (max_cost - min_cost),
                (34.1 - min_cost) / (max_cost - min_cost),
                (34 - min_cost) / (max_cost - min_cost),
            ],
            [np.nan, np.nan, np.nan, np.nan],
        ]

        # invalidate similarity values outside of [min;min+eta[
        masked_normalized_cv = [  # pylint:disable=unused-variable
            [np.nan, (28.01 - min_cost) / (max_cost - min_cost), (28 - min_cost) / (max_cost - min_cost), np.nan],
            [np.nan, (28.01 - min_cost) / (max_cost - min_cost), (28 - min_cost) / (max_cost - min_cost), np.nan],
            [np.nan, (34.1 - min_cost) / (max_cost - min_cost), np.nan, (34 - min_cost) / (max_cost - min_cost)],
            [np.nan, (34.1 - min_cost) / (max_cost - min_cost), np.nan, (34 - min_cost) / (max_cost - min_cost)],
            [np.nan, np.nan, np.nan, np.nan],
        ]

        disparities = [  # pylint:disable=unused-variable
            [np.nan, 1, 2, np.nan],
            [np.nan, 1, 2, np.nan],
            [np.nan, 1, np.nan, 3],
            [np.nan, 1, np.nan, 3],
            [np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan],
        ]
        # Risk max is defined as risk(p,k) = mean(max(di) - min(di)) for di in [cmin(p);cmin(p)+kŋ[
        gt_risk_max = [[((2 - 1) + (2 - 1)) / 2, ((3 - 1) + (3 - 1)) / 2, np.nan]]
        # Risk min is defined as mean( (1+risk(p,k)) - amb(p,k) )
        gt_risk_min = [
            [
                (((1 + 1) - sampled_ambiguity[0][0][0]) + ((1 + 1) - sampled_ambiguity[0][0][1])) / 2,
                (((1 + 2) - sampled_ambiguity[0][1][0]) + ((1 + 2) - sampled_ambiguity[0][1][1])) / 2,
                np.nan,
            ]
        ]

        gt_sampled_risk_max = [[[(2 - 1), (2 - 1)], [(3 - 1), (3 - 1)], [np.nan, np.nan]]]
        gt_sampled_risk_min = [
            [
                [(1 + 1) - sampled_ambiguity[0][0][0], (1 + 1) - sampled_ambiguity[0][0][1]],
                [(1 + 2) - sampled_ambiguity[0][1][0], (1 + 2) - sampled_ambiguity[0][1][1]],
                [np.nan, np.nan],
            ]
        ]

        # Compute risk
        risk_max, risk_min, sampled_risk_max, sampled_risk_min = risk_.compute_risk_and_sampled_risk(
            cv_, sampled_ambiguity, 0.0, 0.5, 0.3
        )

        # Check if the calculated risks are equal to the ground truth (same shape and all elements equals)
        np.testing.assert_allclose(gt_risk_max, risk_max, rtol=1e-06)
        np.testing.assert_allclose(gt_risk_min, risk_min, rtol=1e-06)
        # Check if the calculated sampled risks are equal to the ground truth (same shape and all elements equals)
        np.testing.assert_allclose(gt_sampled_risk_max, sampled_risk_max, rtol=1e-06)
        np.testing.assert_allclose(gt_sampled_risk_min, sampled_risk_min, rtol=1e-06)


if __name__ == "__main__":
    unittest.main()
