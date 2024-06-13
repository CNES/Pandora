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
This module contains functions to test the confidence module with interval bounds.
"""

import numpy as np

import pandora
from pandora.state_machine import PandoraMachine


def test_interval_bounds(create_img_for_confidence):
    """
    Test the interval bounds method using the pandora run method

    """

    left_im, right_im = create_img_for_confidence

    user_cfg = {
        "input": {"left": {"disp_min": -1, "disp_max": 1}},
        "pipeline": {
            "matching_cost": {"matching_cost_method": "sad", "window_size": 1, "subpix": 1},
            "cost_volume_confidence": {"confidence_method": "interval_bounds", "possibility_threshold": 0.7},
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
        np.sum(
            left.coords["indicator"].data
            != [
                "confidence_from_interval_bounds_inf",
                "confidence_from_interval_bounds_sup",
            ]
        )
        == 0
    )

    # ----- Check interval results ------

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
    #
    #
    # Possibility
    # [[[ nan, 0.75, 1.  ],
    #   [0.75, 1.  , 0.75],
    #   [1.  , 0.75, 1.  ],
    #   [1.  , 0.75,  nan]],
    #  [[ nan, 0.75, 1.  ],
    #   [ nan,  nan,  nan],
    #   [1.  , 0.5 , 1.  ],
    #   [0.5 , 1.  ,  nan]],
    #  [[ nan, 0.5 , 1.  ],
    #   [0.5 , 1.  , 0.5 ],
    #   [1.  , 1.  , 1.  ],
    #   [0.5 , 1.  ,  nan]],
    #  [[ nan, 1.  , 1.  ],
    #   [1.  , 0.5 , 0.  ],
    #   [1.  , 0.5 , 0.75],
    #   [ nan,  nan,  nan]]]

    # Infimum and supremum bound of the interval confidence
    # If the interval bound has a possibility of 1,
    # the intervals are extended by one (refinement)
    inf_bound_gt = np.array(
        [[0, -1, -1, -1], [0, np.nan, -1, -1], [0, -1, -1, -1], [-1, -1, -1, np.nan]], dtype=np.float32
    )
    sup_bound_gt = np.array([[1, 1, 1, 0], [1, np.nan, 1, 1], [1, 1, 1, 1], [1, 0, 1, np.nan]], dtype=np.float32)

    # Check if the calculated intervals is equal to the ground truth (same shape and all elements equals)
    np.testing.assert_allclose(left["confidence_measure"].data[:, :, 0], inf_bound_gt, rtol=1e-06)
    np.testing.assert_allclose(left["confidence_measure"].data[:, :, 1], sup_bound_gt, rtol=1e-06)
