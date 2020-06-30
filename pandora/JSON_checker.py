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
This module contains functions allowing to check the configuration given to Pandora pipeline.
"""

import json
from json_checker import Checker, And, Or
import rasterio
import numpy as np
import sys
from typing import Dict, List
import logging

import copy
from collections.abc import Mapping
from . import stereo
from . import optimization
from . import aggregation
from . import filter
from . import validation
from . import refinement


def gdal_can_open_mandatory(f: str) -> bool:
    """
    Test if file f can be open by gdal

    :param f: File to test
    :type f: string
    :returns: True if rasterio can open file and False otherwise
    :rtype: bool
    """

    try:
        rasterio.open(f)
        return True
    except Exception as e:
        logging.warning("Impossible to read file {}: {}".format(f, e))
        return False


def gdal_can_open(f: str) -> bool:
    """
    Test if file f can be open by gdal

    :param f: File to test
    :type f: string
    :returns: True if rasterio can open file and False otherwise
    :rtype: bool
    """

    if f == 'none' or f is None:
        return True
    else:
        return gdal_can_open_mandatory(f)


def check_images(img_ref: str, img_sec: str, msk_ref: str, msk_sec: str) -> None:
    """
    Check the images

    :param img_ref: path to the reference image
    :type img_ref: string
    :param img_sec: path to the secondary image
    :type img_sec: string
    :param msk_ref: path to the mask of the reference image
    :type msk_ref: string
    :param msk_sec: path to the mask of the secondary image
    :type msk_sec: string
    """
    # verify that the images have 1 channel
    ref_ = rasterio.open(img_ref)
    if ref_.count != 1:
        logging.error('The input images must be 1-channel grayscale images')
        sys.exit(1)
    sec_ = rasterio.open(img_sec)
    if sec_.count != 1:
        logging.error('The input images must be 1-channel grayscale images')
        sys.exit(1)

    # verify that the images have the same size
    if (ref_.width != sec_.width) or \
            (ref_.height != sec_.height):
        logging.error('Images must have the same size')
        sys.exit(1)

    # verify that image and mask have the same size
    if msk_ref is not None:
        msk_ = rasterio.open(msk_ref)
        if (ref_.width != msk_.width) or \
                (ref_.height != msk_.height):
            logging.error('Image and masks must have the same size')
            sys.exit(1)

    # verify that image and mask have the same size
    if msk_sec is not None:
        msk_ = rasterio.open(msk_sec)
        # verify that the image and mask have the same size
        if (sec_.width != msk_.width) or \
                (sec_.height != msk_.height):
            logging.error('Image and masks must have the same size')
            sys.exit(1)


def check_disparity(disp_min: int, disp_max: int) -> None:
    """
    Check the disparity

    :param disp_min: minimal disparity
    :type disp_min: int
    :param disp_max: maximal disparity
    :type disp_max: int
    """

    # verify the disparity
    if abs(disp_min) + abs(disp_max) == 0:
        logging.error('Disparity range must be greater than 0')
        sys.exit(1)


def get_config_pipeline(user_cfg: Dict[str, dict]) -> Dict[str, dict]:
    """
    Get the pipeline configuration

    :param user_cfg: user configuration
    :type user_cfg: dict
    :return cfg: partial configuration
    :rtype cfg: dict
    """

    cfg = {}

    if 'invalid_disparity' in user_cfg:
        cfg['invalid_disparity'] = user_cfg['invalid_disparity']
    if 'stereo' in user_cfg:
        cfg['stereo'] = user_cfg['stereo']
    if 'aggregation' in user_cfg:
        cfg['aggregation'] = user_cfg['aggregation']
    if 'optimization' in user_cfg:
        cfg['optimization'] = user_cfg['optimization']
    if 'refinement' in user_cfg:
        cfg['refinement'] = user_cfg['refinement']
    if 'filter' in user_cfg:
        cfg['filter'] = user_cfg['filter']
    if 'validation' in user_cfg:
        cfg['validation'] = user_cfg['validation']

    return cfg


def get_config_input(user_cfg: Dict[str, dict]) -> Dict[str, dict]:
    """
    Get the input configuration

    :param user_cfg: user configuration
    :type user_cfg: dict
    :return cfg: partial configuration
    :rtype cfg: dict
    """

    cfg = {}

    if 'input' in user_cfg:
        cfg['input'] = user_cfg['input']

    return cfg


def get_config_image(user_cfg: Dict[str, dict]) -> Dict[str, dict]:
    """
    Get the image configuration

    :param user_cfg: user configuration
    :type user_cfg: dict
    :return cfg: partial configuration
    :rtype cfg: dict
    """

    cfg = {}

    if 'image' in user_cfg:
        cfg['image'] = user_cfg['image']

    return cfg


def check_pipeline_section(user_cfg: Dict[str, dict]) -> Dict[str, dict]:
    """
    Complete and check if the pipeline dictionary is correct

    :param user_cfg: user configuration
    :type user_cfg: dict
    :return cfg: global configuration
    :rtype cfg: dict
    """
    # Add missing steps and inputs defaults values in user_cfg
    cfg = update_conf(default_short_configuration_pipeline, user_cfg)

    # Initialize the plugins
    stereo_ = stereo.AbstractStereo(**cfg["stereo"])
    aggregation_ = aggregation.AbstractAggregation(**cfg["aggregation"])
    optimization_ = optimization.AbstractOptimization(**cfg["optimization"])
    refinement_ = refinement.AbstractRefinement(**cfg["refinement"])
    filter_ = filter.AbstractFilter(**cfg["filter"])
    validation_ = validation.AbstractValidation(**cfg["validation"])

    # Load configuration steps
    cfg_stereo = {'stereo': stereo_.cfg}
    cfg_aggregation = {'aggregation': aggregation_.cfg}
    cfg_optimization = {'optimization': optimization_.cfg}
    cfg_refinement = {'refinement': refinement_.cfg}
    cfg_filter = {'filter': filter_.cfg}
    cfg_validation = {'validation': validation_.cfg}

    # Update the configuration with steps configuration
    cfg = update_conf(cfg, cfg_stereo)
    cfg = update_conf(cfg, cfg_aggregation)
    cfg = update_conf(cfg, cfg_optimization)
    cfg = update_conf(cfg, cfg_refinement)
    cfg = update_conf(cfg, cfg_filter)
    cfg = update_conf(cfg, cfg_validation)

    configuration_schema = {
        "invalid_disparity": Or(int, lambda x: np.isnan(x)),
        "stereo": dict,
        "aggregation": dict,
        "optimization": dict,
        "refinement": dict,
        "filter": dict,
        "validation": dict
    }

    checker = Checker(configuration_schema)
    checker.validate(cfg)

    return cfg


def check_image_section(user_cfg: Dict[str, dict]) -> Dict[str, dict]:
    """
    Complete and check if the dictionary is correct

    :param user_cfg: user configuration
    :type user_cfg: dict
    :return cfg: global configuration
    :rtype cfg: dict
    """
    # Add missing steps and inputs defaults values in user_cfg
    cfg = update_conf(default_short_configuration_image, user_cfg)

    # check schema
    configuration_schema = {
        "image": image_configuration_schema
    }

    checker = Checker(configuration_schema)
    checker.validate(cfg)

    return cfg


def check_input_section(user_cfg: Dict[str, dict]) -> Dict[str, dict]:
    """
    Complete and check if the dictionary is correct

    :param user_cfg: user configuration
    :type user_cfg: dict
    :return cfg: global configuration
    :rtype cfg: dict
    """
    # Add missing steps and inputs defaults values in user_cfg
    cfg = update_conf(default_short_configuration_input, user_cfg)

    # check schema
    configuration_schema = {
        "input": input_configuration_schema
    }

    checker = Checker(configuration_schema)
    checker.validate(cfg)

    # custom checking
    check_disparity(cfg['input']['disp_min'], cfg['input']['disp_max'])
    check_images(cfg['input']['img_ref'], cfg['input']['img_sec'], cfg['input']['ref_mask'],
                 cfg['input']['sec_mask'])

    return cfg


def check_conf(user_cfg: Dict[str, dict]) -> Dict[str, dict]:
    """
    Complete and check if the dictionary is correct

    :param user_cfg: user configuration
    :type user_cfg: dict
    :return cfg: global configuration
    :rtype cfg: dict
    """

    # check pipeline
    user_cfg_pipeline = get_config_pipeline(user_cfg)
    cfg_pipeline = check_pipeline_section(user_cfg_pipeline)

    # check image
    user_cfg_image = get_config_image(user_cfg)
    cfg_image = check_image_section(user_cfg_image)

    # check input
    user_cfg_input = get_config_input(user_cfg)
    cfg_input = check_input_section(user_cfg_input)

    # concatenate updated config
    cfg = concat_conf([cfg_image, cfg_input, cfg_pipeline])

    return cfg


def concat_conf(cfg_list: List[Dict[str, dict]]) -> Dict[str, dict]:
    """
    Concatenate dictionaries

    :param cfg_list: list of configurations
    :type cfg_list: List of dict
    :return cfg: global configuration
    :rtype cfg: dict
    """
    # concatenate updated config
    cfg = {}
    for c in cfg_list:
        cfg.update(c)

    return cfg


input_configuration_schema = {
    "img_ref": And(str, gdal_can_open_mandatory),
    "img_sec": And(str, gdal_can_open_mandatory),
    "ref_mask": And(Or(str, lambda x: x is None), gdal_can_open),
    "sec_mask": And(Or(str, lambda x: x is None), gdal_can_open),
    "disp_min": int,
    "disp_max": int
}

image_configuration_schema = {
    "nodata1": Or(int, lambda x: np.isnan(x)),
    "nodata2": Or(int, lambda x: np.isnan(x)),
    "valid_pixels": int,
    "no_data": int
}

default_short_configuration_image = {
    "image": {
        "nodata1": 0,
        "nodata2": 0,
        "valid_pixels": 0,
        "no_data": 1
    }
}

default_short_configuration_input = {
    "input": {
        "ref_mask": None,
        "sec_mask": None
    }
}

default_short_configuration_pipeline = {
    "invalid_disparity": -9999,
    "stereo": {
        "stereo_method": "ssd"
    },
    "aggregation": {
        "aggregation_method": "none"
    },
    "optimization": {
        "optimization_method": "none"
    },
    "refinement": {
        "refinement_method": "none"
    },
    "filter": {
        "filter_method": "none"
    },
    "validation": {
        "validation_method": "none"
    }
}


default_short_configuration = concat_conf([default_short_configuration_image, default_short_configuration_input,
                                           default_short_configuration_pipeline])


def read_config_file(config_file: str) -> Dict[str, dict]:
    """
    Read a json configuration file

    :param config_file: path to a json file containing the algorithm parameters
    :type config_file: string
    :return user_cfg: configuration dictionary
    :rtype: dict
    """
    with open(config_file, 'r') as f:
        user_cfg = json.load(f)
    return user_cfg


def update_conf(def_cfg: Dict[str, dict], user_cfg: Dict[str, dict]) -> Dict[str, dict]:
    """
    Update the default configuration with the user configuration,

    :param def_cfg: default configuration
    :type def_cfg: dict
    :param user_cfg: user configuration
    :type user_cfg: dict
    :return: the user and default configuration
    :rtype: dict
    """
    config = copy.deepcopy(def_cfg)
    for key, value in user_cfg.items():
        if isinstance(value, Mapping):
            config[key] = update_conf(config.get(key, {}), value)
        else:
            if value == "np.nan":
                value = np.nan
            config[key] = value
    return config


def is_method(s: str, methods: List[str]) -> bool:
    """
    Test if s is a method in methods

    :param s: String to test
    :type s: string
    :param methods: list of available methods
    :type methods: list of strings
    :returns: True if s a method and False otherwise
    :rtype: bool
    """

    if s in methods:
        return True
    else:
        logging.error("{} is not in available methods : ".format(s) + ', '.join(methods))
        return False
