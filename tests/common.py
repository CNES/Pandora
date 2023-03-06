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
from typing import Tuple
import numpy as np
import xarray as xr
from rasterio import Affine


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
        with open(path, "rt", encoding="utf-8") as file_:
            config = json.load(file_)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


def matching_cost_tests_setup() -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Setup the matching_cost_tests data

    :return: left, right datasets
    :rtype: Tuple[xr.Dataset]
    """
    # Create a stereo object
    data = np.array(
        ([1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 2, 1], [1, 1, 1, 4, 3, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]),
        dtype=np.float64,
    )
    left = xr.Dataset(
        {"im": (["row", "col"], data)}, coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])}
    )
    left.attrs = {"valid_pixels": 0, "no_data_mask": 1, "crs": None, "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)}

    data = np.array(
        ([1, 1, 1, 2, 2, 2], [1, 1, 1, 4, 2, 4], [1, 1, 1, 4, 4, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]),
        dtype=np.float64,
    )
    right = xr.Dataset(
        {"im": (["row", "col"], data)}, coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])}
    )
    right.attrs = {"valid_pixels": 0, "no_data_mask": 1, "crs": None, "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)}

    return left, right


def matching_cost_tests_multiband_setup() -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Setup the matching_cost_tests data

    :return: left, right datasets
    :rtype: Tuple[xr.Dataset]
    """
    # Create a multiband stereo object
    # Initialize multiband data
    data = np.zeros((2, 5, 6))
    data[0, :, :] = np.array(
        (
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 2, 1],
            [1, 1, 1, 4, 3, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ),
        dtype=np.float64,
    )

    data[1, :, :] = np.array(
        (
            [1, 1, 1, 1, 1, 1],
            [1, 2, 1, 1, 1, 1],
            [4, 3, 1, 1, 1, 1],
            [5, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ),
        dtype=np.float64,
    )

    left = xr.Dataset(
        {"im": (["band", "row", "col"], data)},
        coords={"band": ["r", "g"], "row": np.arange(data.shape[1]), "col": np.arange(data.shape[2])},
    )

    left.attrs = {"valid_pixels": 0, "no_data_mask": 1, "crs": None, "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)}
    # initialize right data
    data = np.zeros((2, 5, 6))
    data[0, :, :] = np.array(
        (
            [1, 1, 1, 2, 2, 2],
            [1, 1, 1, 4, 2, 4],
            [1, 1, 1, 4, 4, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ),
        dtype=np.float64,
    )

    data[1, :, :] = np.array(
        (
            [1, 1, 1, 1, 1, 1],
            [2, 2, 2, 1, 1, 1],
            [4, 2, 4, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ),
        dtype=np.float64,
    )

    right = xr.Dataset(
        {"im": (["band", "row", "col"], data)},
        coords={"band": ["r", "g"], "row": np.arange(data.shape[1]), "col": np.arange(data.shape[2])},
    )
    right.attrs = {
        "valid_pixels": 0,
        "no_data_mask": 1,
        "crs": None,
        "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    }

    return left, right


basic_pipeline_cfg = {
    "right_disp_map": {"method": "none"},
    "matching_cost": {"matching_cost_method": "zncc", "window_size": 5, "subpix": 2, "band": None},
    "disparity": {"disparity_method": "wta", "invalid_disparity": -9999},
    "refinement": {"refinement_method": "vfit"},
    "filter": {"filter_method": "median", "filter_size": 3},
}

validation_pipeline_cfg = {
    "right_disp_map": {"method": "accurate"},
    "matching_cost": {"matching_cost_method": "zncc", "window_size": 5, "subpix": 2, "band": None},
    "cost_volume_confidence": {"confidence_method": "std_intensity"},
    "disparity": {"disparity_method": "wta", "invalid_disparity": -9999},
    "refinement": {"refinement_method": "vfit"},
    "filter": {"filter_method": "median", "filter_size": 3},
    "validation": {"validation_method": "cross_checking", "cross_checking_threshold": 1.0},
}

multiscale_pipeline_cfg = {
    "right_disp_map": {"method": "none"},
    "matching_cost": {"matching_cost_method": "zncc", "window_size": 5, "subpix": 2, "band": None},
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

input_multiband_cfg = {
    "img_left": "tests/pandora/left_rgb.tif",
    "img_right": "tests/pandora/right_rgb.tif",
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

# Image common attributes for matching_cost_tests
img_attrs = {
    "valid_pixels": 0,
    "no_data_mask": 1,
    "crs": None,
    "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
}
