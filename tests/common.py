#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2020 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of PANDORA
#
#     https://github.com/CNES/Pandora
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
This module contains common functions present in Pandora's tests.
"""

import os
import logging
import json


def setup_logging(
    path="logging.json",
    default_level=logging.WARNING,
):
    """
    Setup the logging configuration

    :param path: path to the configuration file
    :type path: string
    :param default_level: default level
    :type default_level: logging level
    """
    if os.path.exists(path):
        with open(path, "rt") as file_:
            config = json.load(file_)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


basic_pipeline_cfg = {
    "right_disp_map": {"method": "none"},
    "matching_cost": {"matching_cost_method": "zncc", "window_size": 5, "subpix": 2},
    "disparity": {"disparity_method": "wta", "invalid_disparity": -9999},
    "refinement": {"refinement_method": "vfit"},
    "filter": {"filter_method": "median", "filter_size": 3},
}

validation_pipeline_cfg = {
    "right_disp_map": {"method": "accurate"},
    "matching_cost": {"matching_cost_method": "zncc", "window_size": 5, "subpix": 2},
    "cost_volume_confidence": {"confidence_method": "std_intensity"},
    "disparity": {"disparity_method": "wta", "invalid_disparity": -9999},
    "refinement": {"refinement_method": "vfit"},
    "filter": {"filter_method": "median", "filter_size": 3},
    "validation": {"validation_method": "cross_checking", "cross_checking_threshold": 1.0},
}

multiscale_pipeline_cfg = {
    "right_disp_map": {"method": "none"},
    "matching_cost": {"matching_cost_method": "zncc", "window_size": 5, "subpix": 2},
    "disparity": {"disparity_method": "wta", "invalid_disparity": -9999},
    "refinement": {"refinement_method": "vfit"},
    "filter": {"filter_method": "median", "filter_size": 3},
    "multiscale": {"multiscale_method": "fixed_zoom_pyramid", "num_scales": 2, "scale_factor": 2, "marge": 1},
}

input_cfg_basic = {
    "img_left": "tests/pandora/left.png",
    "img_right": "tests/pandora/right.png",
    "disp_min": -60,
    "disp_max": 0,
}

input_cfg_left_grids = {
    "img_left": "tests/pandora/left.png",
    "img_right": "tests/pandora/right.png",
    "disp_min": "tests/pandora/disp_min_grid.tif",
    "disp_max": "tests/pandora/disp_max_grid.tif",
}

input_cfg_left_right_grids = {
    "img_left": "tests/pandora/left.png",
    "img_right": "tests/pandora/right.png",
    "disp_min": "tests/pandora/disp_min_grid.tif",
    "disp_max": "tests/pandora/disp_max_grid.tif",
    "disp_min_right": "tests/pandora/right_disp_min_grid.tif",
    "disp_max_right": "tests/pandora/right_disp_max_grid.tif",
}
