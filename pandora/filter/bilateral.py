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
This module contains functions associated to the bilateral filter used to filter the disparity map.
"""

from typing import Dict, Union

import cv2
import xarray as xr
from json_checker import Checker, And

from . import filter # pylint: disable=redefined-builtin


@filter.AbstractFilter.register_subclass('bilateral')
class BilateralFilter(filter.AbstractFilter):
    """
    BilateralFilter class allows to perform the filtering step
    """
    # Default configuration, do not change these values
    _SIGMA_COLOR = 2.
    _SIGMA_SPACE = 6.

    def __init__(self, **cfg: Union[str, float]):
        """
        :param cfg: optional configuration, {'sigmaColor' : value, 'sigmaSpace' : value}
        :type cfg: dict
        """
        self.cfg = self.check_conf(**cfg)
        self._sigma_color = self.cfg['sigma_color']
        self._sigma_space = self.cfg['sigma_space']

    def check_conf(self, **cfg: Union[str, float]) -> Dict[str, Union[str, float]]:
        """
        Add default values to the dictionary if there are missing elements and check if the dictionary is correct

        :param cfg: filter configuration
        :type cfg: dict
        :return cfg: filter configuration updated
        :rtype: dict
        """
        # Give the default value if the required element is not in the configuration
        if 'sigma_color' not in cfg:
            cfg['sigma_color'] = self._SIGMA_COLOR
        if 'sigma_space' not in cfg:
            cfg['sigma_space'] = self._SIGMA_SPACE

        schema = {
            'filter_method': And(str, lambda input: 'bilateral'),
            'sigma_color': And(float, lambda input: input > 0),
            'sigma_space': And(float, lambda input: input > 0)
        }

        checker = Checker(schema)
        checker.validate(cfg)
        return cfg

    def desc(self):
        """
        Describes the filtering method
        """
        print('Bilateral filter description')

    def filter_disparity(self, disp: xr.Dataset, img_left: xr.Dataset = None, img_right: xr.Dataset = None,
                         cv: xr.Dataset = None) -> None:
        """
        Apply bilateral filter using openCV.
        Filter size is computed from sigmaSpace

        :param disp: the disparity map dataset
        :type disp:
            xarray.Dataset with the variables :
                - disparity_map 2D xarray.DataArray (row, col)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
                - validity_mask 2D xarray.DataArray (row, col)
        :param img_left: left Dataset image
        :tye img_left: xarray.Dataset
        :param img_right: right Dataset image
        :type img_right: xarray.Dataset
        :param cv: cost volume dataset
        :type cv: xarray.Dataset
        :return: None
        """
        disp['disparity_map'].data = cv2.bilateralFilter(disp['disparity_map'].data, d=0, sigmaColor=self._sigma_color,
                                                         sigmaSpace=self._sigma_space)
        disp.attrs['filter'] = 'bilateral'
