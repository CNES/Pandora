# pylint: disable=duplicate-code
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
This module contains functions to test the cost volume measure step.
"""
import unittest

import numpy as np

from pandora import matching_cost
from tests import common


class TestMatchingCost(unittest.TestCase):
    """
    TestMatchingCost class allows to test all the methods in the class MatchingCost,
    and the plugins pixel_wise, zncc
    """

    def setUp(self):
        """
        Method called to prepare the test fixture

        """

        self.left, self.right = common.matching_cost_tests_setup()

    def test_ssd_cost(self):
        """
        Test the sum of squared difference method

        """
        # Squared difference pixel-wise ground truth for the images self.left, self.right, with window_size = 1
        sd_ground_truth = np.array(
            (
                [0, 0, 0, 1, 1, 1],
                [0, 0, 0, (1 - 4) ** 2, 0, (1 - 4) ** 2],
                [0, 0, 0, 0, (3 - 4) ** 2, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            )
        )

        # Computes the sd cost for the whole images
        matching_cost_matcher = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "ssd", "window_size": 1, "subpix": 1}
        )
        ssd = matching_cost_matcher.compute_cost_volume(
            img_left=self.left, img_right=self.right, disp_min=-1, disp_max=1
        )

        # Check if the calculated sd cost is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(ssd["cost_volume"].sel(disp=0), sd_ground_truth)

        # Sum of squared difference pixel-wise ground truth for the images self.left, self.right, with window_size = 5
        ssd_ground_truth = np.array(
            (
                [
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, 12.0, 22.0, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                ]
            )
        )

        # Computes the sd cost for the whole images
        matching_cost_matcher = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "ssd", "window_size": 5, "subpix": 1}
        )
        ssd = matching_cost_matcher.compute_cost_volume(
            img_left=self.left, img_right=self.right, disp_min=-1, disp_max=1
        )
        matching_cost_matcher.cv_masked(self.left, self.right, ssd, -1, 1)

        # Check if the calculated sd cost is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(ssd["cost_volume"].sel(disp=0), ssd_ground_truth)


if __name__ == "__main__":
    common.setup_logging()
    unittest.main()
