#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
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
from __future__ import annotations

import logging
import logging.config
import sys
from os import PathLike
from typing import Dict, Tuple

import xarray as xr

from . import common
from .check_configuration import check_conf, check_datasets, read_config_file, read_multiscale_params
from .img_tools import create_dataset_from_inputs
from .state_machine import PandoraMachine

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points


# pylint: disable=too-many-arguments
def run(
    pandora_machine: PandoraMachine,
    img_left: xr.Dataset,
    img_right: xr.Dataset,
    cfg: Dict[str, dict],
) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Run the pandora pipeline

    :param pandora_machine: instance of PandoraMachine
    :type pandora_machine: PandoraMachine
    :param img_left: left Dataset image containing :

            - im: 2D (row, col) or 3D (band_im, row, col) xarray.DataArray float32
            - disparity (optional): 3D (disp, row, col) xarray.DataArray float32
            - msk (optional): 2D (row, col) xarray.DataArray int16
            - classif (optional): 3D (band_classif, row, col) xarray.DataArray int16
            - segm (optional): 2D (row, col) xarray.DataArray int16
    :type img_left: xarray.Dataset
    :param img_right: right Dataset image containing :

            - im: 2D (row, col) or 3D (band_im, row, col) xarray.DataArray float32
            - disparity (optional): 3D (disp, row, col) xarray.DataArray float32
            - msk (optional): 2D (row, col) xarray.DataArray int16
            - classif (optional): 3D (band_classif, row, col) xarray.DataArray int16
            - segm (optional): 2D (row, col) xarray.DataArray int16
    :type img_right: xarray.Dataset
    :param cfg: pipeline configuration
    :type cfg: Dict[str, dict]
    :return: Two xarray.Dataset :


            - left : the left dataset, which contains the variables :
                - disparity_map : the disparity map in the geometry of the left image 2D DataArray (row, col)
                - confidence_measure : the confidence measure in the geometry of the left image 3D DataArray \
                    (row, col, indicator)
                - validity_mask : the validity mask in the geometry of the left image 2D DataArray (row, col)
                - classif_mask : information about a classification
                - segm_mask : information about a segmentation

            - right : the right dataset. If there is no validation step, the right Dataset will be empty.If a \
            validation step is configured, the dataset will contain the variables :
                - disparity_map : the disparity map in the geometry of the right image 2D DataArray (row, col)
                - confidence_measure : the confidence measure in the geometry of the left image 3D DataArray \
                    (row, col, indicator)
                - validity_mask : the validity mask in the geometry of the left image 2D DataArray (row, col)
                - classif_mask : information about a classification
                - segm_mask : information about a segmentation

    :rtype: tuple (xarray.Dataset, xarray.Dataset)
    """

    # Retrieve the multiscale parameter from the conf. If no multiscale was defined, num_scales=0 and scale_factor=1
    num_scales, scale_factor = read_multiscale_params(img_left, img_right, cfg)

    # Prepare the machine before running, including the image pyramid creation
    pandora_machine.run_prepare(cfg, img_left, img_right, scale_factor, num_scales)

    # For each scale we run the whole pipeline, the scales will be executed from coarse to fine,
    # and the machine will store the necessary parameters for the following scale
    for _ in range(pandora_machine.num_scales):
        # Trigger the machine step by step
        for elem in list(cfg["pipeline"]):
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
    for entry_point in entry_points(group="pandora.plugin"):
        entry_point.load()


def main(cfg_path: PathLike | str, output: str, verbose: bool) -> None:
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
    img_left = create_dataset_from_inputs(input_config=cfg["input"]["left"])
    # If cfg["input"]["right"]["disp"] is None then "disp_right_min" = - "disp_left_max"
    # and "disp_right_max" = - "disp_left_min"
    if cfg["input"]["right"]["disp"] is None and not isinstance(cfg["input"]["left"]["disp"], str):
        cfg["input"]["right"]["disp"] = [-cfg["input"]["left"]["disp"][1], -cfg["input"]["left"]["disp"][0]]
    img_right = create_dataset_from_inputs(input_config=cfg["input"]["right"])

    # Check datasets: shape, format and content
    check_datasets(img_left, img_right)

    # Run the Pandora pipeline
    left, right = run(pandora_machine, img_left, img_right, cfg)

    # Save the left and right DataArray in tiff files
    common.save_results(left, right, output)

    # Update cfg with margins
    cfg["margins"] = pandora_machine.margins.to_dict()
    # Save the configuration
    common.save_config(output, cfg)
