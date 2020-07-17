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
import sys
import xarray as xr
from pkg_resources import iter_entry_points

from . import common
from . import stereo
from . import aggregation
from . import filter
from . import disparity
from . import validation
from . import refinement
from . import optimization
from .img_tools import read_img, read_disp
from .JSON_checker import check_conf, read_config_file
from typing import Dict, Tuple, Union
import numpy as np


def run(img_ref: xr.Dataset, img_sec: xr.Dataset, disp_min: Union[int, np.ndarray], disp_max: Union[int, np.ndarray],
        cfg: Dict[str, dict], disp_min_sec: Union[None, int, np.ndarray] = None,
        disp_max_sec: Union[None, int, np.ndarray] = None) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Run the pandora pipeline

    :param img_ref: reference Dataset image
    :type img_ref:
        xarray.Dataset containing :
            - im : 2D (row, col) xarray.DataArray
            - msk (optional): 2D (row, col) xarray.DataArray
    :param img_sec: secondary Dataset image
    :type img_sec:
        xarray.Dataset containing :
            - im : 2D (row, col) xarray.DataArray
            - msk (optional): 2D (row, col) xarray.DataArray
    :param disp_min: minimal disparity
    :type disp_min: int or np.ndarray
    :param disp_max: maximal disparity
    :type disp_max: int or np.ndarray
    :param cfg: configuration
    :type cfg: dict
    :param disp_min_sec: minimal disparity of the secondary image
    :type disp_min_sec: None, int or np.ndarray
    :param disp_max_sec: maximal disparity of the secondary image
    :type disp_max_sec: None, int or np.ndarray
    :return:
        Two xarray.Dataset :
            - ref : the reference dataset, which contains the variables :
                - disparity_map : the disparity map in the geometry of the reference image 2D DataArray (row, col)
                - confidence_measure : the confidence measure in the geometry of the reference image 3D DataArray (row, col, indicator)
                - validity_mask : the validity mask in the geometry of the reference image 2D DataArray (row, col)

            - sec : the secondary dataset. If there is no validation step, the secondary Dataset will be empty.
                If a validation step is configured, the dataset will contain the variables :
                - disparity_map : the disparity map in the geometry of the secondary image 2D DataArray (row, col)
                - confidence_measure : the confidence measure in the geometry of the reference image 3D DataArray (row, col, indicator)
                - validity_mask : the validity mask in the geometry of the reference image 2D DataArray (row, col)

    :rtype: tuple (xarray.Dataset, xarray.Dataset)
    """
    # Initializes the plugins
    stereo_ = stereo.AbstractStereo(**cfg['stereo'])
    aggregation_ = aggregation.AbstractAggregation(**cfg['aggregation'])
    optimization_ = optimization.AbstractOptimization(**cfg['optimization'])
    filter_ = filter.AbstractFilter(**cfg['filter'])
    refinement_ = refinement.AbstractRefinement(**cfg['refinement'])
    validation_ = validation.AbstractValidation(**cfg['validation'])
    interpolate_ = validation.AbstractInterpolation(**cfg['validation'])

    # Run the pandora pipeline

    # Matching cost computation
    logging.info('Matching cost computation...')
    # Compute the global minimum and maximum of disparity
    dmin_min, dmax_max = stereo_.dmin_dmax(disp_min, disp_max)
    # Compute the cost volume
    cv = stereo_.compute_cost_volume(img_ref, img_sec, dmin_min, dmax_max, **cfg['image'])
    # Masking the costs computed for disparities outside of the variable disparities range
    cv = stereo_.cv_masked(img_ref, img_sec, cv, disp_min, disp_max,  **cfg['image'])

    # Cost (support) aggregation
    logging.info('Cost aggregation...')
    cv = aggregation_.cost_volume_aggregation(img_ref, img_sec, cv, **cfg['image'])

    # Cost optimization

    logging.info('Cost optimization...')
    cv = optimization_.optimize_cv(cv, img_ref, img_sec)

    # Disparity computation and validity mask
    logging.info('Disparity computation...')
    ref = disparity.to_disp(cv, cfg['invalid_disparity'], img_ref, img_sec)
    ref = disparity.validity_mask(ref, img_ref, img_sec, cv, **cfg['image'])

    # Subpixel disparity refinement
    logging.info('Subpixel refinement...')
    cv, ref = refinement_.subpixel_refinement(cv, ref, img_ref, img_sec)

    # Disparity filter
    logging.info('Disparity filtering...')
    ref = filter_.filter_disparity(ref, img_ref, img_sec, cv)

    sec = xr.Dataset()
    if cfg['validation']['validation_method'] == 'cross_checking':

        logging.info('Computing the right disparity map with the accurate method...')

        # Computes the secondary disparity if it is not provided
        if disp_min_sec is None:
            disp_min_sec = -disp_max
            disp_max_sec = -disp_min

        dmin_min_sec, dmax_max_sec = stereo_.dmin_dmax(disp_min_sec, disp_max_sec)
        cv_right = stereo_.compute_cost_volume(img_sec, img_ref, dmin_min_sec, dmax_max_sec, **cfg['image'])
        cv_right = stereo_.cv_masked(img_sec, img_ref, cv_right, disp_min_sec, disp_max_sec, **cfg['image'])
        cv_right = aggregation_.cost_volume_aggregation(img_sec, img_ref, cv_right, **cfg['image'])
        cv_right = optimization_.optimize_cv(cv_right, img_sec, img_ref)
        sec = disparity.to_disp(cv_right, cfg['invalid_disparity'], img_sec, img_ref)
        sec = disparity.validity_mask(sec, img_sec, img_ref, cv_right, **cfg['image'])
        cv_right, sec = refinement_.subpixel_refinement(cv_right, sec, img_sec, img_ref)

        sec = filter_.filter_disparity(sec, img_sec, img_ref, cv_right)

        # Computes the validation mask
        ref = validation_.disparity_checking(ref, sec, img_ref, img_sec, cv)
        sec = validation_.disparity_checking(sec, ref, img_sec, img_ref, cv_right)

        # Interpolated mismatch and occlusions
        ref = interpolate_.interpolated_disparity(ref, img_ref, img_sec, cv)
        sec = interpolate_.interpolated_disparity(sec, img_sec, img_ref, cv_right)

        if cfg['validation']['filter_interpolated_disparities']:
            ref = filter_.filter_disparity(ref, img_ref, img_sec, cv)
            sec = filter_.filter_disparity(sec, img_sec, img_ref, cv)

        # Resize the output products : add rows and columns that have been truncated
        sec = disparity.resize(sec, cfg['invalid_disparity'])

    # Resize the output products : add rows and columns that have been truncated
    ref = disparity.resize(ref, cfg['invalid_disparity'])

    return ref, sec


def setup_logging(verbose: bool) -> None:
    """
    Setup the logging configuration

    :param verbose: verbose mode
    :type verbose: bool
    """
    if verbose:
        logging.basicConfig(format="[%(asctime)s][%(levelname)s] %(message)s", level=logging.INFO)
    else:
        logging.basicConfig(format="[%(asctime)s][%(levelname)s] %(message)s", level=logging.ERROR)


def import_plugin() -> None:
    """
    Load all the registered entry points

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
    """

    # Read the user configuration file
    user_cfg = read_config_file(cfg_path)

    # Import pandora plugins
    import_plugin()

    # check the configuration
    cfg = check_conf(user_cfg)

    # setup the logging configuration
    setup_logging(verbose)

    # Read images and masks
    img_ref = read_img(cfg['input']['img_ref'], no_data=cfg['image']['nodata1'], cfg=cfg['image'],
                       mask=cfg['input']['ref_mask'])
    img_sec = read_img(cfg['input']['img_sec'], no_data=cfg['image']['nodata2'], cfg=cfg['image'],
                       mask=cfg['input']['sec_mask'])

    # Read range of disparities
    disp_min = read_disp(cfg['input']['disp_min'])
    disp_max = read_disp(cfg['input']['disp_max'])
    disp_min_sec = read_disp(cfg['input']['disp_min_sec'])
    disp_max_sec = read_disp(cfg['input']['disp_max_sec'])

    # Run the Pandora pipeline
    ref, sec = run(img_ref, img_sec, disp_min, disp_max, cfg, disp_min_sec, disp_max_sec)

    # Save the reference and secondary DataArray in tiff files
    common.save_results(ref, sec, output)

    # Save the configuration
    common.save_config(output, cfg)
