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
This module contains functions associated to the disparity denoiser filter used to filter the disparity map.
"""
import logging
from typing import Dict, Union

import xarray as xr
from json_checker import Checker, And

from . import filter  # pylint: disable=redefined-builtin


@filter.AbstractFilter.register_subclass("disparity_denoiser")
class DisparityDenoiser(filter.AbstractFilter):
    """
    DisparityDenoiser class allows to perform the filtering step
    """

    # Default configuration, do not change these values
    _FILTER_SIZE = 11
    _SIGMA_EUCLIDIAN = 4.0
    _SIGMA_COLOR = 100.0
    _SIGMA_PLANAR = 12.0
    _SIGMA_GRAD = 1.5

    def __init__(self, *args, cfg: Dict, **kwargs):  # pylint:disable=unused-argument
        """
        :param cfg: optional configuration, {'filterSize': value,  'sigmaEuclidian' : value,
        'sigmaColor' : value, 'sigmaPlanar' : value, 'sigmaGrad': value}
        :type cfg: dict
        """
        self.cfg = self.check_conf(cfg)
        self._filter_size = int(self.cfg["filter_size"])
        self._sigma_euclidian = float(self.cfg["sigma_euclidian"])
        self._sigma_color = float(self.cfg["sigma_color"])
        self._sigma_planar = float(self.cfg["sigma_planar"])
        self._sigma_grad = float(self.cfg["sigma_grad"])

    def check_conf(self, cfg: Dict) -> Dict[str, Union[str, float]]:
        """
        Add default values to the dictionary if there are missing elements and check if the dictionary is correct

        :param cfg: filter configuration
        :type cfg: dict
        :return cfg: filter configuration updated
        :rtype: dict
        """
        # Give the default value if the required element is not in the configuration
        if "filter_size" not in cfg:
            cfg["filter_size"] = self._FILTER_SIZE
        if "sigma_euclidian" not in cfg:
            cfg["sigma_euclidian"] = self._SIGMA_EUCLIDIAN
        if "sigma_color" not in cfg:
            cfg["sigma_color"] = self._SIGMA_COLOR
        if "sigma_planar" not in cfg:
            cfg["sigma_planar"] = self._SIGMA_PLANAR
        if "sigma_grad" not in cfg:
            cfg["sigma_grad"] = self._SIGMA_GRAD

        schema = {
            "filter_method": And(str, lambda input: "disparity_denoiser"),
            "filter_size": And(int, lambda input: input > 0),
            "sigma_euclidian": And(float, lambda input: input > 0),
            "sigma_color": And(float, lambda input: input > 0),
            "sigma_planar": And(float, lambda input: input > 0),
            "sigma_grad": And(float, lambda input: input > 0),
        }

        checker = Checker(schema)
        checker.validate(cfg)
        return cfg

    def desc(self):
        """
        Describes the filtering method
        """
        print("Disparity denoiser filter description")

    def filter_disparity(
        self,
        disp: xr.Dataset,
        img_left: xr.Dataset = None,
        img_right: xr.Dataset = None,
        cv: xr.Dataset = None,
    ) -> None:
        """
        Apply disparity denoiser filter.

        :param disp: the disparity map dataset  with the variables :

                - disparity_map 2D xarray.DataArray (row, col)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
                - validity_mask 2D xarray.DataArray (row, col)
        :type disp: xarray.Dataset
        :param img_left: left Dataset image
        :type img_left: xarray.Dataset
        :param img_right: right Dataset image
        :type img_right: xarray.Dataset
        :param cv: cost volume dataset
        :type cv: xarray.Dataset
        :return: None
        """

        logging.warning("This method is under development")
