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
This module contains functions to test the matching cost step.
"""
import unittest
import numpy as np

from pandora import matching_cost

from tests import common


class TestMatchingCost(unittest.TestCase):
    """
    TestMatchingCost class allows to test the methods in the class MatchingCost
    """

    def setUp(self):
        """
        Method called to prepare the test fixture

        """

        self.left, self.right = common.matching_cost_tests_setup()

    def test_allocate_numpy_cost_volume(self):
        """
        Test the allocate_numpy_cost_volume function
        """

        # compute ground truth cv for allocate
        gt_cv = np.zeros((5, 6, 5), dtype=np.float32)
        gt_cv += np.nan

        gt_cv_cropped = np.zeros((5, 2, 1), dtype=np.float32)
        gt_cv_cropped += np.nan

        # Create matching cost object
        matching_cost_ = matching_cost.AbstractMatchingCost(
            **{"matching_cost_method": "census", "window_size": 3, "subpix": 1, "band": "b"}
        )

        # Function allocate_numpy_cost_volume
        cv, cv_cropped = matching_cost_.allocate_numpy_cost_volume(self.left, [0, 0.5, 1, 1.5, 2], 2)

        # Test that cv and cv_cropped are equal to the ground truth
        np.testing.assert_array_equal(cv, gt_cv)
        np.testing.assert_array_equal(cv_cropped, gt_cv_cropped)


if __name__ == "__main__":
    common.setup_logging()
    unittest.main()
