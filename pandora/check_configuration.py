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
from __future__ import annotations

import copy
import json
import logging
import sys
from collections.abc import Mapping
from os import PathLike
from typing import Dict, Union, List, Tuple
import xarray as xr
import rasterio

import numpy as np
from json_checker import Checker, Or, And

from pandora.state_machine import PandoraMachine
from pandora.img_tools import rasterio_open, get_metadata
from pandora.common import split_inputs

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


def check_shape(dataset: xr.Dataset, ref: str, test: str) -> None:
    """
    Check if two data_vars are the same dimensions

    :param dataset: dataset
    :type dataset: xr.Dataset
    :param ref: the reference image
    :type str: name of image
    :param test: the tested image
    :type str: name of image
    """
    # check only the rows and columns, the last two elements of the shape
    if dataset[ref].data.shape[-2:] != dataset[test].data.shape[-2:]:
        logging.error("%s and %s must have the same shape", ref, test)
        sys.exit(1)


def check_attributes(dataset: xr.Dataset, attribute_list: set) -> None:
    """
    Check if attributes are in the dataset

    :param dataset: dataset
    :type dataset: xr.Dataset
    :param attribute: the atribute to test
    :type set: list of attribute names
    """
    attribute = attribute_list - set(dataset.attrs)
    if attribute:
        logging.error("User must provide the % attribute(s)", attribute)
        sys.exit(1)


def check_dataset(dataset: xr.Dataset) -> None:
    """
    Check if input dataset is correct

    :param dataset: dataset
    :type dataset: xr.Dataset
    """

    # Check image
    if "im" not in dataset:
        logging.error("User must provide an image im")
        sys.exit(1)

    # Check band in "band_im" coordinates
    check_band_names(dataset)

    # Check not empty image (all nan values)
    if np.isnan(dataset["im"].data).all():
        logging.error("Image contains lony nan values")
        sys.exit(1)

    # Check disparities
    if "disparity" in dataset:
        check_disparities_from_dataset(dataset["disparity"])

    # Check others data_vars : mask, classif and segm
    for data_var in filter(lambda i: i != "im", dataset):
        check_shape(dataset=dataset, ref="im", test=str(data_var))

    # Check attributes
    mandatory_attributes = {"no_data_img", "valid_pixels", "no_data_mask", "crs", "transform"}
    check_attributes(dataset=dataset, attribute_list=mandatory_attributes)


def check_datasets(left: xr.Dataset, right: xr.Dataset) -> None:
    """
    Check that left and right datasets are correct

    :param left: dataset
    :type dataset: xr.Dataset
    :param right: dataset
    :type dataset: xr.Dataset
    """

    # Check the dataset content
    check_dataset(left)
    check_dataset(right)

    # Check disparities at least on the left
    if not "disparity" in left:
        logging.error("left dataset must have disparity DataArray")
        sys.exit(1)

    # Check shape
    # check only the rows and columns, the last two elements of the shape
    if left["im"].data.shape[-2:] != right["im"].data.shape[-2:]:
        logging.error("left and right datasets must have the same shape")
        sys.exit(1)


def check_image_dimension(img1: rasterio.io.DatasetReader, img2: rasterio.io.DatasetReader) -> None:
    """
    Check width and height are the same between two images

    :param img1: image DatasetReader with width and height
    :type img1: rasterio.io.DatasetReader
    :param img2: image DatasetReader with width and height
    :type img2: rasterio.io.DatasetReader
    :return: None
    """
    if (img1.width != img2.width) or (img1.height != img2.height):
        logging.error("Images must have the same size")
        sys.exit(1)


def check_images(user_cfg: Dict[str, dict]) -> None:
    """
    Check the images

    :param user_cfg: user configuration
    :type user_cfg: dict
    :return: None
    """

    left_ = rasterio_open(user_cfg["left"]["img"])
    right_ = rasterio_open(user_cfg["right"]["img"])

    # verify that the images left and right have the same size
    check_image_dimension(left_, right_)

    # verify others images
    images = ["mask", "classif", "segm"]
    for img in images:
        if img in user_cfg["left"] and user_cfg["left"][img] is not None:
            check_image_dimension(left_, rasterio_open(user_cfg["left"][img]))
        if img in user_cfg["right"] and user_cfg["right"][img] is not None:
            check_image_dimension(right_, rasterio_open(user_cfg["right"][img]))


def check_band_names(img: str | xr.Dataset) -> None:
    """
    Check that band names have the correct format

    :param img: path to the image
    :type img: string
    :return: None
    """

    bands = []
    if isinstance(img, str):
        # open image
        img_ds = rasterio_open(img)
        # check that the image have the band names
        if img_ds.count != 1:
            if not img_ds.descriptions:
                logging.error("Image is missing band names metadata")
                sys.exit(1)
            bands = list(img_ds.descriptions)
            logging.info("Image has not band")
    else:  # img is a dataset
        if "band_im" in img.coords:
            bands = list(img.coords["band_im"].data)

    # Check type
    if not all(isinstance(band, str) for band in bands):
        logging.error("Band value must be str")
        sys.exit(1)


def check_disparities_from_input(
    disparity: list[int] | str | None,
    img_left: str,
) -> None:
    """
    Check disparities from user configuration

    :param disparity: disparity to check if list it is a list of two values: min and max.
    :type disparity:  list[int] | str | None
    :param img_left: path to the left image
    :type img_left: str
    :return: None
    """
    # disparities are integers
    if isinstance(disparity, list) and disparity[1] < disparity[0]:
        logging.error("Disp_max must be bigger than Disp_min")
        sys.exit(1)

    # disparites are grids
    if isinstance(disparity, str):
        # Load an image to compare the grid size
        img_left_ = rasterio_open(img_left)

        disparity_reader = rasterio_open(disparity)

        # check that disparity grids is a 2-channel grid
        if disparity_reader.count != 2:
            logging.error("Disparity grids must be a 2-channel grid")
            sys.exit(1)

        # check that disp_min has the same size as the image
        if (disparity_reader.width != img_left_.width) or (disparity_reader.height != img_left_.height):
            logging.error("Disparity grids and image must have the same size")
            sys.exit(1)

        if (disparity_reader.read(1) > disparity_reader.read(2)).any():
            logging.error("Disp_max must be bigger than Disp_min")
            sys.exit(1)


def check_disparities_from_dataset(disparity: xr.DataArray) -> None:
    """
    Check disparities with this format

    disparity: 3D (band_disp, row, col) xarray.DataArray float32
    and band_disp = (min, max)

    :param disparity: disparity to check
    :type disparity:  xr.DataArray
    :return: None
    """
    if "band_disp" not in disparity.coords:
        logging.error("Disparity xr.Dataset must have a band_disp coordinate")
        sys.exit(1)
    band_disp = disparity.coords["band_disp"].data
    if not {"min", "max"}.issubset(band_disp):
        logging.error("Disparity xr.Dataset must have a band_disp coordinate with min and max band")
        sys.exit(1)
    if (disparity.sel(band_disp="min").data > disparity.sel(band_disp="max").data).any():
        logging.error("Disp_max grid must be bigger than Disp_min grid for each pixel")
        sys.exit(1)


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


def get_config_pipeline(user_cfg: Dict[str, dict]) -> Dict[str, dict]:
    """
    Get the pipeline configuration

    :param user_cfg: user configuration
    :type user_cfg: dict
    :return cfg: partial configuration
    :rtype cfg: dict
    """

    cfg = {}

    if "pipeline" in user_cfg:
        cfg["pipeline"] = user_cfg["pipeline"]

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
        disparity_interval = user_input["input"]["disp_left"]
        img_path = user_input["input"]["img_left"]
    else:
        img_path, *disparity_interval = user_input
        # Since only the size is to be used for the memory computation, the same path is set on left and right
        input_cfg = {"disparity_interval": disparity_interval, "img_left": img_path, "img_right": img_path}
        user_input = {"input": input_cfg}

    # Read input image
    img = rasterio_open(img_path)
    # Obtain cost volume size
    dmin, dmax = disparity_interval
    cv_size = img.width * img.height * np.abs(dmax - dmin)
    if checked_cfg_flag:
        # Obtain pipeline cfg
        pipeline_cfg = user_pipeline_cfg["pipeline"]
    else:
        # First, check if the configuration is valid
        cfg = {"pipeline": user_pipeline_cfg["pipeline"], "input": user_input["input"]}
        img_left_metadata = get_metadata(cfg["input"]["img_left"], disparity_interval)
        img_right_metadata = get_metadata(cfg["input"]["img_right"], None, None)
        checked_cfg = check_pipeline_section(cfg, img_left_metadata, img_right_metadata, pandora_machine)
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


def check_pipeline_section(
    user_cfg: Dict[str, dict], img_left: xr.Dataset, img_right: xr.Dataset, pandora_machine: PandoraMachine
) -> Dict[str, dict]:
    """
    Check if the pipeline is correct by
    - Checking the sequence of steps according to the machine transitions
    - Checking parameters, define in dictionary, of each Pandora step

    :param user_cfg: pipeline user configuration
    :type user_cfg: dict
    :param img_left: image left with metadata
    :type  img_left: xarray.Dataset
    :param img_right: image right with metadata
    :type  img_right: xarray.Dataset
    :param pandora_machine: instance of PandoraMachine
    :type pandora_machine: PandoraMachine object
    :return: cfg: pipeline configuration
    :rtype: cfg: dict
    """
    # Check if user configuration pipeline is compatible with transitions/states of pandora machine.
    cfg = update_conf(default_short_configuration_pipeline, user_cfg)
    pandora_machine.check_conf(cfg, img_left, img_right)

    cfg = update_conf(cfg, pandora_machine.pipeline_cfg)

    configuration_schema = {"pipeline": dict}

    checker = Checker(configuration_schema)
    # We select only the pipeline section for the checker
    pipeline_cfg = {"pipeline": cfg["pipeline"]}
    checker.validate(pipeline_cfg)

    return pipeline_cfg


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
    if isinstance(cfg["input"]["disp_left"], list):
        base_input_configuration_schema = input_configuration_schema_integer_disparity
    elif isinstance(cfg["input"]["disp_right"], str):
        base_input_configuration_schema = input_configuration_schema_left_disparity_grids_right_grids
    else:
        base_input_configuration_schema = input_configuration_schema_left_disparity_grids_right_none

    input_configuration_schema.update(base_input_configuration_schema)
    # check schema
    configuration_schema = {"input": input_configuration_schema}

    checker = Checker(configuration_schema)
    checker.validate(cfg)

    # custom checking

    # check left disparities
    check_disparities_from_input(
        cfg["input"]["disp_left"],
        cfg["input"]["img_left"],
    )
    # check right disparities
    check_disparities_from_input(
        cfg["input"]["disp_right"],
        cfg["input"]["img_right"],
    )
    # check bands
    check_band_names(cfg["input"]["img_left"])
    check_band_names(cfg["input"]["img_right"])

    check_images(split_inputs(cfg["input"]))

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

    # read metadata from left and right images
    img_left_metadata = get_metadata(
        cfg_input["input"]["img_left"],
        cfg_input["input"]["disp_left"],
        cfg_input["input"]["left_classif"],
        cfg_input["input"]["left_segm"],
    )
    img_right_metadata = get_metadata(
        cfg_input["input"]["img_right"],
        cfg_input["input"]["disp_right"],
        cfg_input["input"]["right_classif"],
        cfg_input["input"]["right_segm"],
    )

    # check pipeline
    user_cfg_pipeline = get_config_pipeline(user_cfg)
    cfg_pipeline = check_pipeline_section(user_cfg_pipeline, img_left_metadata, img_right_metadata, pandora_machine)

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
    "disp_left": [int, int],
    "disp_right": (lambda input: input is None),
}

# Input configuration when left disparity is a grid, and right not provided
input_configuration_schema_left_disparity_grids_right_none = {
    "disp_left": And(str, rasterio_can_open),
    "disp_right": (lambda input: input is None),
}

# Input configuration when left and right disparity are grids
input_configuration_schema_left_disparity_grids_right_grids = {
    "disp_left": And(str, rasterio_can_open),
    "disp_right": And(str, rasterio_can_open),
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
        "disp_right": None,
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


default_short_configuration_pipeline: dict = {"pipeline": {}}

default_short_configuration = concat_conf([default_short_configuration_input, default_short_configuration_pipeline])


def read_config_file(config_file: PathLike | str) -> Dict[str, dict]:
    """
    Read a json configuration file

    :param config_file: path to a json file containing the algorithm parameters
    :type config_file: PathLike | string
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
