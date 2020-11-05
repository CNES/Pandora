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
This module contains functions to run Pandora pipeline.
"""

import logging
import logging.config
from typing import Dict, Tuple, Union

import numpy as np
import xarray as xr
from pkg_resources import iter_entry_points

from . import aggregation
from . import common
from . import disparity
from . import filter #pylint:disable=redefined-builtin
from . import optimization
from . import refinement
from . import stereo
from . import validation
from .img_tools import read_img, read_disp
from .json_checker import check_conf, read_config_file
from .state_machine import PandoraMachine


# pylint: disable=too-many-arguments
def run(pandora_machine: PandoraMachine, img_left: xr.Dataset, img_right: xr.Dataset, disp_min: Union[int, np.ndarray],
        disp_max: Union[int, np.ndarray], cfg: Dict[str, dict], disp_min_right: Union[None, int, np.ndarray] = None,
        disp_max_right: Union[None, int, np.ndarray] = None) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Run the pandora pipeline

    :param pandora_machine: instance of PandoraMachine
    :type pandora_machine: PandoraMachine
    :param img_left: left Dataset image
    :type img_left:
        xarray.Dataset containing :
            - im : 2D (row, col) xarray.DataArray
            - msk (optional): 2D (row, col) xarray.DataArray
    :param img_right: right Dataset image
    :type img_right:
        xarray.Dataset containing :
            - im : 2D (row, col) xarray.DataArray
            - msk (optional): 2D (row, col) xarray.DataArray
    :param disp_min: minimal disparity
    :type disp_min: int or np.ndarray
    :param disp_max: maximal disparity
    :type disp_max: int or np.ndarray
    :param cfg: configuration
    :type cfg: dict
    :param disp_min_right: minimal disparity of the right image
    :type disp_min_right: None, int or np.ndarray
    :param disp_max_right: maximal disparity of the right image
    :type disp_max_right: None, int or np.ndarray
    :return:
        Two xarray.Dataset :
            - left : the left dataset, which contains the variables :
                - disparity_map : the disparity map in the geometry of the left image 2D DataArray (row, col)
                - confidence_measure : the confidence measure in the geometry of the left image 3D DataArray
                    (row, col, indicator)
                - validity_mask : the validity mask in the geometry of the left image 2D DataArray (row, col)

            - right : the right dataset. If there is no validation step, the right Dataset will be empty.
                If a validation step is configured, the dataset will contain the variables :
                - disparity_map : the disparity map in the geometry of the right image 2D DataArray (row, col)
                - confidence_measure : the confidence measure in the geometry of the left image 3D DataArray
                    (row, col, indicator)
                - validity_mask : the validity mask in the geometry of the left image 2D DataArray (row, col)

    :rtype: tuple (xarray.Dataset, xarray.Dataset)
    """
    # Prepare machine before running
    pandora_machine.run_prepare(cfg, img_left, img_right, disp_min, disp_max, disp_min_right, disp_max_right)

    # Trigger the machine step by step
    # Warning: first element of cfg dictionary is not a transition. It contains information about the way to
    # compute right disparity map.
    for elem in list(cfg)[1:]:
        pandora_machine.run(elem, cfg)

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
        logging.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=logging.INFO)
    else:
        logging.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=logging.ERROR)


def import_plugin() -> None:
    """
    Load all the registered entry points
    :return: None
    """
    for entry_point in iter_entry_points(group='pandora.plugin'):
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
    img_left = read_img(cfg['input']['img_left'], no_data=cfg['image']['nodata1'], cfg=cfg['image'],
                        mask=cfg['input']['left_mask'])
    img_right = read_img(cfg['input']['img_right'], no_data=cfg['image']['nodata2'], cfg=cfg['image'],
                         mask=cfg['input']['right_mask'])

    # Read range of disparities
    disp_min = read_disp(cfg['input']['disp_min'])
    disp_max = read_disp(cfg['input']['disp_max'])
    disp_min_right = read_disp(cfg['input']['disp_min_right'])
    disp_max_right = read_disp(cfg['input']['disp_max_right'])

    # Run the Pandora pipeline
    left, right = run(pandora_machine, img_left, img_right, disp_min, disp_max, cfg['pipeline'], disp_min_right,
                      disp_max_right)

    # Save the left and right DataArray in tiff files
    common.save_results(left, right, output)

    # Save the configuration
    common.save_config(output, cfg)
