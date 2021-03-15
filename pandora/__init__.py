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
This module contains functions to run Pandora pipeline.
"""

import logging
import logging.config
from typing import Dict, Tuple, Union

import xarray as xr
import numpy as np
from pkg_resources import iter_entry_points

from . import common
from .img_tools import read_img, read_disp
from .check_json import check_conf, read_config_file, read_multiscale_params
from .state_machine import PandoraMachine


# pylint: disable=too-many-arguments
def run(
    pandora_machine: PandoraMachine,
    img_left: xr.Dataset,
    img_right: xr.Dataset,
    disp_min: Union[np.array, int],
    disp_max: Union[np.array, int],
    cfg: Dict[str, dict],
    disp_min_right: Union[None, np.array] = None,
    disp_max_right: Union[None, np.array] = None,
) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Run the pandora pipeline

    :param pandora_machine: instance of PandoraMachine
    :type pandora_machine: PandoraMachine
    :param img_left: left Dataset image containing :

            - im : 2D (row, col) xarray.DataArray
            - msk (optional): 2D (row, col) xarray.DataArray
    :type img_left: xarray.Dataset
    :param img_right: right Dataset image containing :

            - im : 2D (row, col) xarray.DataArray
            - msk (optional): 2D (row, col) xarray.DataArray
    :type img_right: xarray.Dataset
    :param disp_min: minimal disparity
    :type disp_min: int or np.array
    :param disp_max: maximal disparity
    :type disp_max: int or np.array
    :param cfg: pipeline configuration
    :type cfg: Dict[str, dict]
    :param disp_min_right: minimal disparity of the right image
    :type disp_min_right: np.array or None
    :param disp_max_right: maximal disparity of the right image
    :type disp_max_right: np.array or None
    :return: Two xarray.Dataset :


            - left : the left dataset, which contains the variables :
                - disparity_map : the disparity map in the geometry of the left image 2D DataArray (row, col)
                - confidence_measure : the confidence measure in the geometry of the left image 3D DataArray \
                    (row, col, indicator)
                - validity_mask : the validity mask in the geometry of the left image 2D DataArray (row, col)

            - right : the right dataset. If there is no validation step, the right Dataset will be empty.If a \
            validation step is configured, the dataset will contain the variables :
                - disparity_map : the disparity map in the geometry of the right image 2D DataArray (row, col)
                - confidence_measure : the confidence measure in the geometry of the left image 3D DataArray \
                    (row, col, indicator)
                - validity_mask : the validity mask in the geometry of the left image 2D DataArray (row, col)

    :rtype: tuple (xarray.Dataset, xarray.Dataset)
    """

    # Retrieve the multiscale parameter from the conf. If no multiscale was defined, num_scales=0 and scale_factor=1
    num_scales, scale_factor = read_multiscale_params(cfg)

    # Prepare the machine before running, including the image pyramid creation
    pandora_machine.run_prepare(
        cfg,
        img_left,
        img_right,
        disp_min,
        disp_max,
        scale_factor,
        num_scales,
        disp_min_right,
        disp_max_right,
    )

    # For each scale we run the whole pipeline, the scales will be executed from coarse to fine,
    # and the machine will store the necessary parameters for the following scale
    for scale in range(pandora_machine.num_scales):  # pylint:disable=unused-variable
        # Trigger the machine step by step
        for elem in list(cfg)[1:]:
            pandora_machine.run(elem, cfg)
            # If the machine gets to the begin state, pass to next scale
            if pandora_machine.state == "begin":
                break

    # Stop the machine which returns to its initial state
    pandora_machine.run_exit()

    return pandora_machine.left_disparity, pandora_machine.right_disparity


def setup_logging(verbose: bool) -> None:
    """
    Setup the logging configuration

    :param verbose: verbose mode
    :type verbose: bool
    :return: None
    """
    if verbose:
        logging.basicConfig(format="[%(asctime)s][%(levelname)s] %(message)s", level=logging.INFO)
    else:
        logging.basicConfig(format="[%(asctime)s][%(levelname)s] %(message)s", level=logging.ERROR)


def import_plugin() -> None:
    """
    Load all the registered entry points
    :return: None
    """
    for entry_point in iter_entry_points(group="pandora.plugin"):
        entry_point.load()


def main(cfg_path: str, output: str, verbose: bool) -> None:
    """
    Check config file and run pandora framework accordingly

    :param cfg_path: path to the json configuration file
    :type cfg_path: string
    :param output: Path to output directory
    :type output: string
    :param verbose: verbose mode
    :type verbose: bool
    :return: None
    """

    # Read the user configuration file
    user_cfg = read_config_file(cfg_path)

    # Import pandora plugins
    import_plugin()

    # Instantiate pandora state machine
    pandora_machine = PandoraMachine()
    # check the configuration
    cfg = check_conf(user_cfg, pandora_machine)

    # setup the logging configuration
    setup_logging(verbose)

    # Read images and masks
    img_left = read_img(
        cfg["input"]["img_left"],
        no_data=cfg["input"]["nodata_left"],
        mask=cfg["input"]["left_mask"],
        classif=cfg["input"]["left_classif"],
        segm=cfg["input"]["left_segm"],
    )
    img_right = read_img(
        cfg["input"]["img_right"],
        no_data=cfg["input"]["nodata_right"],
        mask=cfg["input"]["right_mask"],
        classif=cfg["input"]["right_classif"],
        segm=cfg["input"]["right_segm"],
    )

    # Read range of disparities
    disp_min = read_disp(cfg["input"]["disp_min"])
    disp_max = read_disp(cfg["input"]["disp_max"])
    disp_min_right = read_disp(cfg["input"]["disp_min_right"])
    disp_max_right = read_disp(cfg["input"]["disp_max_right"])

    # Run the Pandora pipeline
    left, right = run(
        pandora_machine,
        img_left,
        img_right,
        disp_min,
        disp_max,
        cfg["pipeline"],
        disp_min_right,
        disp_max_right,
    )

    # Save the left and right DataArray in tiff files
    common.save_results(left, right, output)

    # Save the configuration
    common.save_config(output, cfg)
