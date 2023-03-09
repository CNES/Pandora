# pylint: disable=missing-module-docstring
#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2020 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of PANDORA
#
#     https://github.com/CNES/Pandora_pandora
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
This module contains functions allowing to check the configuration given to Pandora pipeline.
"""

import copy
import json
import logging
import sys
from collections.abc import Mapping
from typing import Dict, Union, List, Tuple

import numpy as np
from json_checker import Checker, Or, And

from pandora.state_machine import PandoraMachine
from pandora.img_tools import rasterio_open

from pandora import multiscale


def rasterio_can_open_mandatory(file_: str) -> bool:
    """
    Test if file can be open by rasterio

    :param file_: File to test
    :type file_: string
    :returns: True if rasterio can open file and False otherwise
    :rtype: bool
    """

    try:
        rasterio_open(file_)
        return True
    except Exception as exc:
        logging.warning("Impossible to read file %: %", file_, exc)
        return False


def rasterio_can_open(file_: str) -> bool:
    """
    Test if file can be open by rasterio

    :param file_: File to test
    :type file_: string
    :returns: True if rasterio can open file and False otherwise
    :rtype: bool
    """

    if file_ == "none" or file_ is None:
        return True

    return rasterio_can_open_mandatory(file_)


def check_images(img_left: str, img_right: str, msk_left: str, msk_right: str) -> None:
    """
    Check the images

    :param img_left: path to the left image
    :type img_left: string
    :param img_right: path to the right image
    :type img_right: string
    :param msk_left: path to the mask of the left image
    :type msk_left: string
    :param msk_right: path to the mask of the right image
    :type msk_right: string
    :return: None
    """
    left_ = rasterio_open(img_left)
    right_ = rasterio_open(img_right)

    # verify that the images have the same size
    if (left_.width != right_.width) or (left_.height != right_.height):
        logging.error("Images must have the same size")
        sys.exit(1)

    # verify that image and mask have the same size
    if msk_left is not None:
        msk_ = rasterio_open(msk_left)
        if (left_.width != msk_.width) or (left_.height != msk_.height):
            logging.error("Image and masks must have the same size")
            sys.exit(1)

    # verify that image and mask have the same size
    if msk_right is not None:
        msk_ = rasterio_open(msk_right)
        # verify that the image and mask have the same size
        if (right_.width != msk_.width) or (right_.height != msk_.height):
            logging.error("Image and masks must have the same size")
            sys.exit(1)


def check_band_names(img: str) -> None:
    """
    Check that band names have the correct format

    :param img: path to the image
    :type img: string
    :return: None
    """

    # open image
    img_ds = rasterio_open(img)
    img_array = img_ds.read()

    # check that the image have the band names
    if img_array.shape[0] != 1:
        if not img_ds.descriptions:
            logging.error("Image is missing band names metadata")
            sys.exit(1)
        if not all(isinstance(band, str) for band in list(img_ds.descriptions)):
            logging.error("Band value must be str")
            sys.exit(1)


def check_band_pipeline(img: str, step: str, bands: Union[None, str, List[str], Dict]) -> None:
    """
    Check coherence band parameter between pipeline step and image dataset

    :param img: path to the image
    :type img: str
    :param step: pipeline step
    :type step: str
    :param bands: band names
    :type bands: None, str, List[str] or Dict
    :return: None
    """
    # open images
    img_ds = rasterio_open(img)
    # If no bands are given, then the input image shall be monoband
    if not bands:
        if img_ds.count != 1:
            logging.error("Missing band instantiate on %s step : input image is multiband", step)
            sys.exit(1)
    # check that the image have the band names
    elif isinstance(bands, dict):
        for _, band in bands.items():
            if not band in list(img_ds.descriptions):
                logging.error("Wrong band instantiate on %s step: %s not in input image", step, band)
                sys.exit(1)
    else:
        for band in bands:
            if not band in list(img_ds.descriptions):
                logging.error("Wrong band instantiate %s step: %s not in input image", step, band)
                sys.exit(1)


def check_disparities(
    disp_min: Union[int, str, None],
    disp_max: Union[int, str, None],
    img_left: str,
) -> None:
    """
    Check left and right disparities.

    :param disp_min: minimal disparity
    :type disp_min: int or str or None
    :param disp_max: maximal disparity
    :type disp_max: int or str or None
    :param img_left: path to the left image
    :type img_left: str
    :return: None
    """
    # --- Check left disparities
    # left disparity are integers
    if isinstance(disp_min, int) and isinstance(disp_max, int):
        if disp_max < disp_min:
            logging.error("Disp_max must be bigger than Disp_min")
            sys.exit(1)

    # left disparity are grids
    elif (isinstance(disp_min, str)) and (isinstance(disp_max, str)):
        # Load an image to compare the grid size
        img_left_ = rasterio_open(img_left)

        disp_min_ = rasterio_open(disp_min)
        dmin = disp_min_.read(1)
        disp_max_ = rasterio_open(disp_max)
        dmax = disp_max_.read(1)

        # check that disparity grids is a 1-channel grid
        if (disp_min_.count != 1) or (disp_max_.count != 1):
            logging.error("Disparity grids must be a 1-channel grid")
            sys.exit(1)

        # check that disp_min has the same size as the image
        if (
            (disp_min_.width != img_left_.width)
            or (disp_min_.height != img_left_.height)
            or (disp_max_.width != img_left_.width)
            or (disp_max_.height != img_left_.height)
        ):
            logging.error("Disparity grids and image must have the same size")
            sys.exit(1)

        if (dmax < dmin).any():
            logging.error("Disp_max must be bigger than Disp_min")
            sys.exit(1)


def get_config_pipeline(user_cfg: Dict[str, dict]) -> Dict[str, dict]:
    """
    Get the pipeline configuration

    :param user_cfg: user configuration
    :type user_cfg: dict
    :return: cfg: partial configuration
    :rtype: cfg: dict
    """
    cfg = {}

    if "pipeline" in user_cfg:
        cfg["pipeline"] = user_cfg["pipeline"]

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

    if "input" in user_cfg:
        cfg["input"] = user_cfg["input"]

    return cfg


def memory_consumption_estimation(
    user_pipeline_cfg: Dict[str, dict],
    user_input: Union[Dict[str, dict], Tuple[str, int, int]],
    pandora_machine: PandoraMachine,
    checked_cfg_flag: bool = False,
) -> Union[Tuple[float, float], None]:
    """
    Return the approximate memory consumption for a given pipeline in GiB.

    :param user_pipeline_cfg: user pipeline configuration
    :type user_pipeline_cfg: dict
    :param user_input: user input configuration, may be given as a dict or directly as img_path, disp_min, disp_max.
    :type user_input: dict or Tuple[str, int, int]
    :param pandora_machine: instance of PandoraMachine
    :type pandora_machine: PandoraMachine object
    :param checked_cfg_flag: Flag for checking pipeline
    :type checked_cfg_flag: bool
    :return: minimum and maximum memory consumption
    :rtype: Tuple[float, float]
    """
    # If the input configuration is given as a dict
    if isinstance(user_input, dict):
        dmin = user_input["input"]["disp_min"]
        dmax = user_input["input"]["disp_max"]
        img_path = user_input["input"]["img_left"]
    else:
        img_path, dmin, dmax = user_input
    # Read input image
    img = rasterio_open(img_path)
    # Obtain cost volume size
    cv_size = img.width * img.height * np.abs(dmax - dmin)
    if checked_cfg_flag:
        # Obtain pipeline cfg
        pipeline_cfg = user_pipeline_cfg["pipeline"]
    else:
        # First, check if the configuration is valid
        checked_cfg = check_pipeline_section(user_pipeline_cfg, pandora_machine)
        # Obtain pipeline cfg
        pipeline_cfg = checked_cfg["pipeline"]

    for function_info in MEMORY_CONSUMPTION_LIST:
        # [ step, step"_method", subclass, m_line, n_line] being m_line and n_line the values of the line defining
        # function's consumption as y = mx + n, where x is the size of the cost volume and y the consumption in MiB
        if function_info[0] in pipeline_cfg:  # if step in the pipeline
            if function_info[2] in pipeline_cfg[function_info[0]][function_info[1]]:  # if subclass in the pipeline
                # Use m and n to compute memory consumption
                m_line = function_info[3]
                n_line = function_info[4]
                # Obtain memory consumption with a variable of +-10% and convert from MiB to GiB
                minmem = ((cv_size * m_line + n_line) * (1 - 0.1)) / 1024
                maxmem = ((cv_size * m_line + n_line) * (1 + 0.1)) / 1024

                logging.debug(
                    "Estimated maximum memory consumption between "  # pylint:disable=consider-using-f-string
                    "{:.2f} GiB and {:.2f} GiB".format(minmem, maxmem)
                )
                return minmem, maxmem
    return None


def check_pipeline_section(user_cfg: Dict[str, dict], pandora_machine: PandoraMachine) -> Dict[str, dict]:
    """
    Check if the pipeline is correct by
    - Checking the sequence of steps according to the machine transitions
    - Checking parameters, define in dictionary, of each Pandora step

    :param user_cfg: pipeline user configuration
    :type user_cfg: dict
    :param pandora_machine: instance of PandoraMachine
    :type pandora_machine: PandoraMachine object
    :return: cfg: pipeline configuration
    :rtype: cfg: dict
    """
    # Check if user configuration pipeline is compatible with transitions/states of pandora machine.
    cfg = update_conf(default_short_configuration_pipeline, user_cfg)
    pandora_machine.check_conf(cfg["pipeline"])

    cfg = update_conf(cfg, pandora_machine.pipeline_cfg)

    configuration_schema = {"pipeline": dict}

    checker = Checker(configuration_schema)
    checker.validate(cfg)

    return cfg


def check_input_section(user_cfg: Dict[str, dict]) -> Dict[str, dict]:
    """
    Complete and check if the dictionary is correct

    :param user_cfg: user configuration
    :type user_cfg: dict
    :return: cfg: global configuration
    :rtype: cfg: dict
    """
    # Add missing steps and inputs defaults values in user_cfg
    cfg = update_conf(default_short_configuration_input, user_cfg)

    # Disparity can be integer type, or string type (path to the disparity grid)
    # If the left disparity is string type, right disparity must be string type or none
    # if the left disparity is integer type, right disparity must be none
    if isinstance(cfg["input"]["disp_min"], int):
        input_configuration_schema.update(input_configuration_schema_integer_disparity)
    else:
        if isinstance(cfg["input"]["disp_min_right"], str):
            input_configuration_schema.update(input_configuration_schema_left_disparity_grids_right_grids)
        else:
            input_configuration_schema.update(input_configuration_schema_left_disparity_grids_right_none)

    # check schema
    configuration_schema = {"input": input_configuration_schema}

    checker = Checker(configuration_schema)
    checker.validate(cfg)

    # custom checking

    # check left disparities
    check_disparities(
        cfg["input"]["disp_min"],
        cfg["input"]["disp_max"],
        cfg["input"]["img_left"],
    )
    # check right disparities
    check_disparities(
        cfg["input"]["disp_min_right"],
        cfg["input"]["disp_max_right"],
        cfg["input"]["img_left"],
    )

    check_band_names(cfg["input"]["img_left"])
    check_band_names(cfg["input"]["img_right"])

    check_images(
        cfg["input"]["img_left"],
        cfg["input"]["img_right"],
        cfg["input"]["left_mask"],
        cfg["input"]["right_mask"],
    )

    return cfg


def check_conf(user_cfg: Dict[str, dict], pandora_machine: PandoraMachine) -> dict:
    """
    Complete and check if the dictionary is correct

    :param user_cfg: user configuration
    :type user_cfg: dict
    :param pandora_machine: instance of PandoraMachine
    :type pandora_machine: PandoraMachine
    :return: cfg: global configuration
    :rtype: cfg: dict
    """

    # check input
    user_cfg_input = get_config_input(user_cfg)
    cfg_input = check_input_section(user_cfg_input)

    # check pipeline
    user_cfg_pipeline = get_config_pipeline(user_cfg)
    cfg_pipeline = check_pipeline_section(user_cfg_pipeline, pandora_machine)

    # Check band for the matching_cost step
    check_band_pipeline(
        cfg_input["input"]["img_left"],
        cfg_pipeline["pipeline"]["matching_cost"]["matching_cost_method"],
        cfg_pipeline["pipeline"]["matching_cost"]["band"],
    )
    check_band_pipeline(
        cfg_input["input"]["img_right"],
        cfg_pipeline["pipeline"]["matching_cost"]["matching_cost_method"],
        cfg_pipeline["pipeline"]["matching_cost"]["band"],
    )

    # If validation is present, right disparity map computation must be activated
    if cfg_pipeline["pipeline"]["right_disp_map"]["method"] != "accurate":
        right_disp_computation = False
        if "validation" in cfg_pipeline["pipeline"]:
            logging.error('For cross-checking, right_disp_map must be set to "accurate"')
            sys.exit(1)
    else:
        right_disp_computation = True

    # If semantic_segmentation is present, check that the RGB band are present in left image
    if "semantic_segmentation" in cfg_pipeline["pipeline"]:
        check_band_pipeline(
            cfg_input["input"]["img_left"],
            cfg_pipeline["pipeline"]["semantic_segmentation"]["segmentation_method"],
            cfg_pipeline["pipeline"]["semantic_segmentation"]["RGB_bands"],
        )
        # If semantic_segmentation and right_disp_computation is present,
        # check that the RGB band are present in the right image
        if right_disp_computation:
            check_band_pipeline(
                cfg_input["input"]["img_right"],
                cfg_pipeline["pipeline"]["semantic_segmentation"]["segmentation_method"],
                cfg_pipeline["pipeline"]["semantic_segmentation"]["RGB_bands"],
            )

    # If left disparities are grids of disparity and the right disparities are none, the cross-checking
    # method cannot be used
    if (
        (isinstance(cfg_input["input"]["disp_min"], str))
        and (cfg_input["input"]["disp_min_right"] is None)
        and ("validation" in cfg_pipeline["pipeline"])
    ):
        logging.error(
            "The cross-checking step cannot be processed if disp_min, disp_max are paths to the left "
            "disparity grids and disp_right_min, disp_right_max are none."
        )
        sys.exit(1)

    if (
        isinstance(cfg_input["input"]["disp_min"], str)
        or isinstance(cfg_input["input"]["disp_min_right"], str)
        or (isinstance(cfg_input["input"]["disp_max"], str))
        or isinstance(cfg_input["input"]["disp_max_right"], str)
    ) and ("multiscale" in cfg_pipeline["pipeline"]):
        logging.error("Multiscale processing does not accept input disparity grids.")
        sys.exit(1)

    # If segmentation or classif geometric_prior is needed for the optimization step,
    # check that it is present on the input config
    if "optimization" in cfg_pipeline["pipeline"]:
        if "geometric_prior" in cfg_pipeline["pipeline"]["optimization"]:
            source = cfg_pipeline["pipeline"]["optimization"]["geometric_prior"]["source"]
            if source in ["classif", "segm"]:
                if not cfg_input["input"]["left_" + source]:
                    logging.error(
                        "For performing the 3SGM optimization step in the pipeline, left %s must be present.", source
                    )
                    sys.exit(1)
                # If right_disp_computation is to be done and 3SGM optimization is present on the pipeline,
                # then both left and right segmentations/classifications must be present
                if right_disp_computation and not cfg_input["input"]["right_" + source]:
                    logging.error(
                        "For performing right disparity computation with 3SGM optimization in the pipeline,"
                        " both left and right %s must be present.",
                        source,
                    )
                    sys.exit(1)

    # concatenate updated config
    cfg = concat_conf([cfg_input, cfg_pipeline])

    return cfg


def concat_conf(cfg_list: List[Dict[str, dict]]) -> Dict[str, dict]:
    """
    Concatenate dictionaries

    :param cfg_list: list of configurations
    :type cfg_list: List of dict
    :return: cfg: global configuration
    :rtype: cfg: dict
    """
    # concatenate updated config
    cfg = {}
    for conf in cfg_list:
        cfg.update(conf)

    return cfg


def read_multiscale_params(cfg: Dict[str, dict]) -> Tuple[int, int]:
    """
    Returns the multiscale parameters

    :param cfg: configuration
    :type cfg: dict
    :return:
        - num_scales: number of scales
        - scale_factor: factor by which each coarser layer is downsampled
    :rtype: tuple(int, int )
    """

    if "multiscale" in cfg:
        # Multiscale processing in conf
        multiscale_ = multiscale.AbstractMultiscale(**cfg["multiscale"])  # type: ignore

        num_scales = multiscale_.cfg["num_scales"]
        scale_factor = multiscale_.cfg["scale_factor"]
    else:
        # No multiscale selected
        num_scales = 1
        scale_factor = 1
    return num_scales, scale_factor


input_configuration_schema = {
    "img_left": And(str, rasterio_can_open_mandatory),
    "img_right": And(str, rasterio_can_open_mandatory),
    "nodata_left": Or(int, lambda input: np.isnan(input)),
    "nodata_right": Or(int, lambda input: np.isnan(input)),
    "left_mask": And(Or(str, lambda input: input is None), rasterio_can_open),
    "right_mask": And(Or(str, lambda input: input is None), rasterio_can_open),
    "left_classif": And(Or(str, lambda x: x is None), rasterio_can_open),
    "right_classif": And(Or(str, lambda x: x is None), rasterio_can_open),
    "left_segm": And(Or(str, lambda x: x is None), rasterio_can_open),
    "right_segm": And(Or(str, lambda x: x is None), rasterio_can_open),
}

# Input configuration when disparity is integer
input_configuration_schema_integer_disparity = {
    "disp_min": int,
    "disp_max": int,
    "disp_min_right": (lambda input: input is None),
    "disp_max_right": (lambda input: input is None),
}

# Input configuration when left disparity is a grid, and right not provided
input_configuration_schema_left_disparity_grids_right_none = {
    "disp_min": And(str, rasterio_can_open),
    "disp_max": And(str, rasterio_can_open),
    "disp_min_right": (lambda input: input is None),
    "disp_max_right": (lambda input: input is None),
}

# Input configuration when left and right disparity are grids
input_configuration_schema_left_disparity_grids_right_grids = {
    "disp_min": And(str, rasterio_can_open),
    "disp_max": And(str, rasterio_can_open),
    "disp_min_right": And(str, rasterio_can_open),
    "disp_max_right": And(str, rasterio_can_open),
}

default_short_configuration_input = {
    "input": {
        "nodata_left": -9999,
        "nodata_right": -9999,
        "left_mask": None,
        "right_mask": None,
        "left_classif": None,
        "right_classif": None,
        "left_segm": None,
        "right_segm": None,
        "disp_min_right": None,
        "disp_max_right": None,
    }
}

# Memory consumption of the most consuming or used functions, defined as
# [ step, step"_method", subclass, m, n] being m and n the values of the line defining function's consumption
# as y = mx + n, where x is the size of the cost volume and y the consumption in MiB
MEMORY_CONSUMPTION_LIST = [
    ["matching_cost", "matching_cost_method", "mc_cnn", 1.57e-05, 265],
    ["optimization", "optimization_method", "sgm", 1.26e-05, 237],
    ["aggregation", "aggregation_method", "cbca", 1.65e-05, 221],
    ["matching_cost", "matching_cost_method", "sad", 1.14e-05, 236],
    ["matching_cost", "matching_cost_method", "ssd", 1.14e-05, 236],
    ["disparity", "disparity_method", "wta", 8.68e-06, 243],
    ["cost_volume_confidence", "confidence_method", "ambiguity", 7.68e-06, 273],
    ["cost_volume_confidence", "confidence_method", "std_intensity", 7.68e-06, 273],
    ["validation", "interpolated_disparity", "sgm", 7.88e-06, 263],
    ["validation", "interpolated_disparity", "mc_cnn", 7.88e-06, 263],
    ["matching_cost", "matching_cost_method", "census", 7.77e-06, 223],
    ["filter", "filter_method", "bilateral", 7.77e-06, 259],
    ["matching_cost", "matching_cost_method", "zncc", 7.69e-06, 254],
]


default_short_configuration_pipeline = {"pipeline": {"right_disp_map": {"method": "none"}}}

default_short_configuration = concat_conf([default_short_configuration_input, default_short_configuration_pipeline])


def read_config_file(config_file: str) -> Dict[str, dict]:
    """
    Read a json configuration file

    :param config_file: path to a json file containing the algorithm parameters
    :type config_file: string
    :return user_cfg: configuration dictionary
    :rtype: dict
    """
    with open(config_file, "r") as file_:  # pylint: disable=unspecified-encoding
        user_cfg = json.load(file_)
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
            if value == "NaN":
                value = np.nan
            elif value == "inf":
                value = np.inf
            elif value == "-inf":
                value = -np.inf
            config[key] = value
    return config
