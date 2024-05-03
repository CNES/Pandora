#
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
This module contains functional tests for the matching cost step.
"""

import pytest
import pandora

import pandora.matching_cost
from pandora.state_machine import PandoraMachine
from pandora.img_tools import create_dataset_from_inputs


class TestFunctionalMatchingCost:
    """
    TestFunctionalMatchingCost class allows to test the matching cost step
    """

    @pytest.fixture()
    def user_cfg(self, correct_input_cfg, correct_pipeline_cfg):
        """
        Return user configuration
        """

        user_cfg = {**correct_input_cfg, **correct_pipeline_cfg}

        return user_cfg

    @pytest.mark.parametrize("matching_cost_method", ["census", "sad", "ssd", "zncc"])
    @pytest.mark.parametrize("subpix", [1, 2, 4])
    @pytest.mark.parametrize(
        ["disp", "window_size"],
        [
            pytest.param(
                [-60, 0],
                5,
                id="Negative disparity inside the image and window_size=5",
            ),
            pytest.param(
                [0, 60],
                5,
                id="Positive disparity inside the image and window_size=5",
            ),
            pytest.param(
                [-452, -445],
                5,
                id="Negative disparity outside the image and window_size=5",
            ),
            pytest.param(
                [445, 452],
                5,
                id="Positive disparity outside the image and window_size=5",
            ),
            pytest.param(
                [445, 448],
                5,
                id="Positive disparity > nb_col - (int(window_size / 2) * 2) and window_size=5",
            ),
            pytest.param(
                [-448, -445],
                5,
                id="Negative disparity > nb_col - (int(window_size / 2) * 2) and window_size=5",
            ),
            pytest.param(
                [-60, 0],
                3,
                id="Negative disparity inside the image and window_size=3",
            ),
            pytest.param(
                [0, 60],
                3,
                id="Positive disparity inside the image and window_size=3",
            ),
            pytest.param(
                [-452, -445],
                3,
                id="Negative disparity outside the image and window_size=3",
            ),
            pytest.param(
                [445, 452],
                3,
                id="Positive disparity outside the image and window_size=3",
            ),
            pytest.param(
                [445, 449],
                3,
                id="Positive disparity > nb_col - (int(window_size / 2) * 2) and window_size=3",
            ),
            pytest.param(
                [-449, -445],
                3,
                id="Negative disparity > nb_col - (int(window_size / 2) * 2) and window_size=3",
            ),
        ],
    )
    def test_functional_matching_cost(self, user_cfg):
        """
        Test matching cost step
        """
        pandora_machine = PandoraMachine()

        img_left = create_dataset_from_inputs(input_config=user_cfg["input"]["left"])
        img_right = create_dataset_from_inputs(input_config=user_cfg["input"]["right"])

        pandora_machine.run_prepare(user_cfg, img_left, img_right)
        left_disparity, right_disparity = pandora.run(  # pylint: disable=unused-variable
            pandora_machine, img_left, img_right, user_cfg
        )
