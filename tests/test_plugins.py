# type:ignore
#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2020 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of PANDORA
#
#     https://github.com/CNES/Pandora
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
This module contains functions to test the error message of Pandora
 when a plugin step is present on the pipeline, but the plugin has not
 been installed.
"""


import unittest

import numpy as np

import pandora
from pandora.img_tools import read_img, rasterio_open
from pandora.state_machine import PandoraMachine


class TestPandora(unittest.TestCase):
    """
    TestPandora class allows to test the pandora pipeline
    """

    def setUp(self):
        """
        Method called to prepare the test fixture

        """

        self.left = read_img("tests/pandora/left.png", no_data=np.nan, mask=None)
        self.right = read_img("tests/pandora/right.png", no_data=np.nan, mask=None)
        self.disp_left = rasterio_open("tests/pandora/disp_left.tif").read(1)
        self.disp_right = rasterio_open("tests/pandora/disp_right.tif").read(1)
        self.occlusion = rasterio_open("tests/pandora/occlusion.png").read(1)

    def test_run_with_semantic_segmentation(self):
        """
        Test the run method error message with semantic segmentation step when the plugin is not installed
        """

        pipeline_cfg = {
            "right_disp_map": {"method": "accurate"},
            "matching_cost": {"matching_cost_method": "zncc", "window_size": 5, "subpix": 2},
            "semantic_segmentation": {"segmentation_method": "ARNN"},
            "disparity": {"disparity_method": "wta", "invalid_disparity": -9999},
            "refinement": {"refinement_method": "vfit"},
            "filter": {"filter_method": "median", "filter_size": 3},
            "validation": {"validation_method": "cross_checking", "cross_checking_threshold": 1.0},
        }
        user_cfg = {"pipeline": pipeline_cfg}

        pandora_machine = PandoraMachine()
        # Update the user configuration with default values
        cfg = pandora.check_json.update_conf(pandora.check_json.default_short_configuration, user_cfg)

        # Run the pandora pipeline
        with self.assertRaises(KeyError) as error:
            pandora.run(pandora_machine, self.left, self.right, -60, 0, cfg["pipeline"])
        self.assertEqual(str(error.exception), "'No semantic segmentation method named ARNN supported'")

    def test_run_with_sgm_optimization(self):
        """
        Test the run method error message with sgm optimization step when the plugin is not installed

        """

        pipeline_cfg = {
            "right_disp_map": {"method": "accurate"},
            "matching_cost": {"matching_cost_method": "zncc", "window_size": 5, "subpix": 2},
            "optimization": {
                "optimization_method": "sgm",
                "penalty": {"penalty_method": "sgm_penalty", "P1": 8, "P2": 32, "p2_method": "constant"},
            },
            "disparity": {"disparity_method": "wta", "invalid_disparity": -9999},
            "refinement": {"refinement_method": "vfit"},
            "filter": {"filter_method": "median", "filter_size": 3},
            "validation": {"validation_method": "cross_checking", "cross_checking_threshold": 1.0},
        }
        user_cfg = {"pipeline": pipeline_cfg}

        pandora_machine = PandoraMachine()

        # Update the user configuration with default values
        cfg = pandora.check_json.update_conf(pandora.check_json.default_short_configuration, user_cfg)

        # Run the pandora pipeline
        with self.assertRaises(KeyError) as error:
            pandora.run(pandora_machine, self.left, self.right, -60, 0, cfg["pipeline"])
        self.assertEqual(str(error.exception), "'No optimization method named sgm supported'")

    def test_run_with_mc_cnn_matching_cost(self):
        """
        Test the run method error message with mc_cnn matching cost step when the plugin is not installed

        """

        pipeline_cfg = {
            "right_disp_map": {"method": "accurate"},
            "matching_cost": {"matching_cost_method": "mc_cnn", "window_size": 11, "subpix": 1},
            "disparity": {"disparity_method": "wta", "invalid_disparity": -9999},
            "refinement": {"refinement_method": "vfit"},
            "filter": {"filter_method": "median", "filter_size": 3},
            "validation": {"validation_method": "cross_checking", "cross_checking_threshold": 1.0},
        }
        user_cfg = {"pipeline": pipeline_cfg}
        pandora_machine = PandoraMachine()

        # Update the user configuration with default values
        cfg = pandora.check_json.update_conf(pandora.check_json.default_short_configuration, user_cfg)

        # Run the pandora pipeline
        with self.assertRaises(KeyError) as error:
            pandora.run(pandora_machine, self.left, self.right, -60, 0, cfg["pipeline"])
        self.assertEqual(str(error.exception), "'No matching cost method named mc_cnn supported'")
