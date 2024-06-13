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
This module contains functions to test the confidence module with risk.
"""

import numpy as np

import pandora.cost_volume_confidence as confidence
from pandora import matching_cost
from pandora.criteria import validity_mask


def test_compute_risk():
    """
    Test the compute_risk method

    """
    risk_ = confidence.AbstractCostVolumeConfidence(**{"confidence_method": "risk", "eta_max": 0.5, "eta_step": 0.3})
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
    # normalized_min_cost
    _ = [
        (28 - min_cost) / (max_cost - min_cost),
        (34 - min_cost) / (max_cost - min_cost),
        np.nan,
    ]
    # normalized_cv
    _ = [
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

    # invalidate similarity values outside [min;min+eta[
    # masked_normalized_cv
    _ = [
        [np.nan, (28.01 - min_cost) / (max_cost - min_cost), (28 - min_cost) / (max_cost - min_cost), np.nan],
        [np.nan, (28.01 - min_cost) / (max_cost - min_cost), (28 - min_cost) / (max_cost - min_cost), np.nan],
        [np.nan, (34.1 - min_cost) / (max_cost - min_cost), np.nan, (34 - min_cost) / (max_cost - min_cost)],
        [np.nan, (34.1 - min_cost) / (max_cost - min_cost), np.nan, (34 - min_cost) / (max_cost - min_cost)],
        [np.nan, np.nan, np.nan, np.nan],
    ]

    # disparities
    _ = [
        [np.nan, 1, 2, np.nan],
        [np.nan, 1, 2, np.nan],
        [np.nan, 1, np.nan, 3],
        [np.nan, 1, np.nan, 3],
        [np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan],
    ]
    # Risk mas is defined as risk(p,k) = mean(max(di) - min(di)) for di in [cmin(p);cmin(p)+kŋ[
    gt_risk_max = np.array([[((2 - 1) + (2 - 1)) / 2, ((3 - 1) + (3 - 1)) / 2, np.nan]])
    # Risk min is defined as mean( (1+risk(p,k)) - amb(p,k) )
    gt_risk_min = np.array(
        [
            [
                (((1 + 1) - sampled_ambiguity[0][0][0]) + ((1 + 1) - sampled_ambiguity[0][0][1])) / 2,
                (((1 + 2) - sampled_ambiguity[0][1][0]) + ((1 + 2) - sampled_ambiguity[0][1][1])) / 2,
                np.nan,
            ]
        ]
    )

    grids = np.array([-1 * np.ones((3, 4)), np.ones((3, 4))], dtype="int64")
    disparity_range = np.array([-1, 0, 1], dtype="float32")

    etas = np.arange(0.0, 0.5, 0.3)
    nbr_etas = etas.shape[0]

    # Compute risk
    risk_max, risk_min = risk_.compute_risk(cv_, sampled_ambiguity, etas, nbr_etas, grids, disparity_range)

    # Check if the calculated risks are equal to the ground truth (same shape and all elements equals)
    np.testing.assert_allclose(gt_risk_max, risk_max, rtol=1e-06)
    np.testing.assert_allclose(gt_risk_min, risk_min, rtol=1e-06)


def test_compute_risk_with_subpix(create_images):
    """
    Test the compute_risk method with subpixel disparity interval, non regression test
    """
    left, right, grids = create_images

    disparity_range = np.array([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)

    cfg = {"matching_cost_method": "ssd", "window_size": 1, "subpix": 2}
    matching_cost_instance = matching_cost.AbstractMatchingCost(**cfg)

    grid = matching_cost_instance.allocate_cost_volume(
        left, (left["disparity"].sel(band_disp="min"), left["disparity"].sel(band_disp="max")), cfg
    )
    grid = validity_mask(left, right, grid)
    _ = matching_cost_instance.compute_cost_volume(img_left=left, img_right=right, cost_volume=grid)

    # window_size = 1
    cv = np.array(
        [
            [
                [np.nan, np.nan, 36.0, 9.0, 0.0],
                [25.0, 4.0, 1.0, 1.0, 9.0],
                [4.0, 0.0, 4.0, 16.0, 36.0],
                [1.0, 9.0, 25.0, np.nan, np.nan],
            ],
            [
                [np.nan, np.nan, 9.0, 0.0, 9.0],
                [4.0, 1.0, 16.0, 4.0, 0.0],
                [25.0, 9.0, 1.0, 16.0, 49.0],
                [4.0, 25.0, 64.0, np.nan, np.nan],
            ],
            [
                [np.nan, np.nan, 0.0, 9.0, 36.0],
                [81.0, 36.0, 9.0, 25.0, 49.0],
                [4.0, 16.0, 36.0, 9.0, 0.0],
                [25.0, 4.0, 1.0, np.nan, np.nan],
            ],
            [
                [np.nan, np.nan, 9.0, 1.0, 1.0],
                [16.0, 4.0, 0.0, 4.0, 16.0],
                [1.0, 1.0, 9.0, 0.0, 9.0],
                [4.0, 1.0, 16.0, np.nan, np.nan],
            ],
        ],
        dtype=np.float32,
    )

    ambiguity_ = confidence.AbstractCostVolumeConfidence(**{"confidence_method": "ambiguity"})

    etas = np.arange(0.0, 0.7, 0.01).astype(np.float64)
    nbr_etas = etas.shape[0]

    _, sampled_ambiguity = ambiguity_.compute_ambiguity_and_sampled_ambiguity(
        cv, etas, nbr_etas, grids, disparity_range
    )

    # Compute risk
    risk_ = confidence.AbstractCostVolumeConfidence(**{"confidence_method": "risk", "eta_max": 0.7, "eta_step": 0.01})

    risk_max, risk_min = risk_.compute_risk(cv, sampled_ambiguity, etas, nbr_etas, grids, disparity_range)

    gt_risk_max = np.array(
        [
            [4.0, 3.3714285, 2.9285715, 4.0],
            [3.8285713, 3.8428571, 2.3, 4.0],
            [3.1857142, 1.5142857, 3.7142856, 3.5142858],
            [4.0, 3.2857144, 3.7428572, 3.942857],
        ],
        dtype=np.float32,
    )
    gt_risk_min = np.array(
        [
            [0.8142857, 0.0, 0.0, 1.5714285],
            [2.1714287, 0.3, 0.0, 1.3714286],
            [2.0, 0.0, 0.8857143, 0.0],
            [0.14285715, 0.0, 0.14285715, 0.27142859],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(gt_risk_max, risk_max, rtol=1e-06)
    np.testing.assert_allclose(gt_risk_min, risk_min, rtol=1e-06)


def test_compute_risk_with_variable_disparity(
    create_grids_and_disparity_range_with_variable_disparities, create_cv_for_variable_disparities
):
    """
    Test the compute_risk method with variable disparity interval
    """

    grids, disparity_range = create_grids_and_disparity_range_with_variable_disparities

    cv_ = create_cv_for_variable_disparities

    amb_sampl = np.array(
        [
            [[3.0, 3.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0]],
            [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [3.0, 3.0]],
            [[2.0, 2.0], [2.0, 2.0], [1.0, 1.0], [2.0, 2.0]],
            [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0]],
        ],
        dtype=np.float32,
    )

    risk_ = confidence.AbstractCostVolumeConfidence(**{"confidence_method": "risk", "eta_max": 0.2, "eta_step": 0.1})

    gt_risk_max = np.array(
        [[2.0, 1.5, 1.5, 1.0], [2.0, 1.0, 1.5, 2.0], [2.0, 2.0, 1.0, 2.0], [2.0, 1.5, 1.5, 1.0]], dtype=np.float32
    )
    gt_risk_min = np.array(
        [[0.0, 0.5, 0.5, 0.0], [1.0, 0.0, 0.5, 0.0], [1.0, 1.0, 1.0, 1.0], [1.0, 0.5, 0.5, 0.0]], dtype=np.float32
    )

    etas = np.arange(0.0, 0.5, 0.3)
    nbr_etas = etas.shape[0]

    # Compute risk
    risk_max, risk_min = risk_.compute_risk(cv_, amb_sampl, etas, nbr_etas, grids, disparity_range)

    # Check if the calculated risks are equal to the ground truth (same shape and all elements equals)
    np.testing.assert_allclose(gt_risk_max, risk_max, rtol=1e-06)
    np.testing.assert_allclose(gt_risk_min, risk_min, rtol=1e-06)


def test_compute_risk_and_sampled_risk():
    """
    Test the compute_risk_and_sampled_risk method

    """
    risk_ = confidence.AbstractCostVolumeConfidence(**{"confidence_method": "risk", "eta_max": 0.5, "eta_step": 0.3})
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
    # normalized_min_cost
    _ = [
        (28 - min_cost) / (max_cost - min_cost),
        (34 - min_cost) / (max_cost - min_cost),
        np.nan,
    ]
    # normalized_cv
    _ = [
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

    # invalidate similarity values outside [min;min+eta[
    # masked_normalized_cv
    _ = [
        [np.nan, (28.01 - min_cost) / (max_cost - min_cost), (28 - min_cost) / (max_cost - min_cost), np.nan],
        [np.nan, (28.01 - min_cost) / (max_cost - min_cost), (28 - min_cost) / (max_cost - min_cost), np.nan],
        [np.nan, (34.1 - min_cost) / (max_cost - min_cost), np.nan, (34 - min_cost) / (max_cost - min_cost)],
        [np.nan, (34.1 - min_cost) / (max_cost - min_cost), np.nan, (34 - min_cost) / (max_cost - min_cost)],
        [np.nan, np.nan, np.nan, np.nan],
    ]

    # disparities
    _ = [
        [np.nan, 1, 2, np.nan],
        [np.nan, 1, 2, np.nan],
        [np.nan, 1, np.nan, 3],
        [np.nan, 1, np.nan, 3],
        [np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan],
    ]
    # Risk max is defined as risk(p,k) = mean(max(di) - min(di)) for di in [cmin(p);cmin(p)+kŋ[
    gt_risk_max = np.array([[((2 - 1) + (2 - 1)) / 2, ((3 - 1) + (3 - 1)) / 2, np.nan]])
    # Risk min is defined as mean( (1+risk(p,k)) - amb(p,k) )
    gt_risk_min = np.array(
        [
            [
                (((1 + 1) - sampled_ambiguity[0][0][0]) + ((1 + 1) - sampled_ambiguity[0][0][1])) / 2,
                (((1 + 2) - sampled_ambiguity[0][1][0]) + ((1 + 2) - sampled_ambiguity[0][1][1])) / 2,
                np.nan,
            ]
        ]
    )

    gt_sampled_risk_max = np.array([[[(2 - 1), (2 - 1)], [(3 - 1), (3 - 1)], [np.nan, np.nan]]])
    gt_sampled_risk_min = np.array(
        [
            [
                [(1 + 1) - sampled_ambiguity[0][0][0], (1 + 1) - sampled_ambiguity[0][0][1]],
                [(1 + 2) - sampled_ambiguity[0][1][0], (1 + 2) - sampled_ambiguity[0][1][1]],
                [np.nan, np.nan],
            ]
        ]
    )

    grids = np.array([-1 * np.ones((3, 4)), np.ones((3, 4))], dtype="int64")
    disparity_range = np.array([-1, 0, 1], dtype="float32")

    etas = np.arange(0.0, 0.5, 0.3)
    nbr_etas = etas.shape[0]

    # Compute risk
    risk_max, risk_min, sampled_risk_max, sampled_risk_min = risk_.compute_risk_and_sampled_risk(
        cv_, sampled_ambiguity, etas, nbr_etas, grids, disparity_range
    )

    # Check if the calculated risks are equal to the ground truth (same shape and all elements equals)
    np.testing.assert_allclose(gt_risk_max, risk_max, rtol=1e-06)
    np.testing.assert_allclose(gt_risk_min, risk_min, rtol=1e-06)
    # Check if the calculated sampled risks are equal to the ground truth (same shape and all elements equals)
    np.testing.assert_allclose(gt_sampled_risk_max, sampled_risk_max, rtol=1e-06)
    np.testing.assert_allclose(gt_sampled_risk_min, sampled_risk_min, rtol=1e-06)


def test_compute_risk_and_sampled_risk_with_variable_disparity(
    create_grids_and_disparity_range_with_variable_disparities, create_cv_for_variable_disparities
):
    """
    Test the compute_risk_and_sampled_risk method with variable disparity interval

    """
    grids, disparity_range = create_grids_and_disparity_range_with_variable_disparities

    cv_ = create_cv_for_variable_disparities

    amb_sampl = np.array(
        [
            [[3.0, 3.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0]],
            [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [3.0, 3.0]],
            [[2.0, 2.0], [2.0, 2.0], [1.0, 1.0], [2.0, 2.0]],
            [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0]],
        ],
        dtype=np.float32,
    )

    risk_ = confidence.AbstractCostVolumeConfidence(**{"confidence_method": "risk", "eta_max": 0.2, "eta_step": 0.1})

    etas = np.arange(0.0, 0.5, 0.3)
    nbr_etas = etas.shape[0]

    # Compute risk
    risk_max, risk_min, sampled_risk_max, sampled_risk_min = risk_.compute_risk_and_sampled_risk(
        cv_, amb_sampl, etas, nbr_etas, grids, disparity_range
    )

    gt_risk_max = np.array(
        [[2.0, 1.5, 1.5, 1.0], [2.0, 1.0, 1.5, 2.0], [2.0, 2.0, 1.0, 2.0], [2.0, 1.5, 1.5, 1.0]], dtype=np.float32
    )

    gt_risk_min = np.array(
        [[0.0, 0.5, 0.5, 0.0], [1.0, 0.0, 0.5, 0.0], [1.0, 1.0, 1.0, 1.0], [1.0, 0.5, 0.5, 0.0]], dtype=np.float32
    )

    gt_sampled_risk_max = np.array(
        [
            [[2.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 1.0]],
            [[2.0, 2.0], [1.0, 1.0], [1.0, 2.0], [2.0, 2.0]],
            [[2.0, 2.0], [2.0, 2.0], [1.0, 1.0], [2.0, 2.0]],
            [[2.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 1.0]],
        ],
        dtype=np.float32,
    )
    gt_sampled_risk_min = np.array(
        [
            [[0.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 0.0]],
            [[1.0, 1.0], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
            [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            [[1.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 0.0]],
        ],
        dtype=np.float32,
    )

    # Check if the calculated risks are equal to the ground truth (same shape and all elements equals)
    np.testing.assert_allclose(gt_risk_max, risk_max, rtol=1e-06)
    np.testing.assert_allclose(gt_risk_min, risk_min, rtol=1e-06)
    # Check if the calculated sampled risks are equal to the ground truth (same shape and all elements equals)
    np.testing.assert_allclose(gt_sampled_risk_max, sampled_risk_max, rtol=1e-06)
    np.testing.assert_allclose(gt_sampled_risk_min, sampled_risk_min, rtol=1e-06)
