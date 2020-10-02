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
This module contains classes and functions associated to the cost volume optimization step.
"""

import sys
import logging
from abc import ABCMeta, abstractmethod
from json_checker import Checker, And
from typing import Dict
import xarray as xr


class AbstractOptimization(object):
    __metaclass__ = ABCMeta

    optimization_methods_avail = {}

    def __new__(cls, **cfg: Dict[str, dict]) -> None:
        """
        Return the plugin associated with the optimization_method given in the configuration

        :param cfg: configuration {'optimization_method': value}
        :type cfg: dictionary
        """
        if cls is AbstractOptimization:
            if type(cfg['optimization_method']) is str:
                try:
                    return super(AbstractOptimization, cls).__new__(cls.optimization_methods_avail[cfg['optimization_method']])
                except KeyError:
                    logging.error("No optimization method named {} supported".format(cfg['optimization_method']))
                    sys.exit(1)
            else:
                if type(cfg['optimization_method']) is unicode:
                    # creating a plugin from registered short name given as unicode (py2 & 3 compatibility)
                    try:
                        return super(AbstractOptimization, cls).__new__(cls.optimization_methods_avail[cfg['optimization_method'].encode('utf-8')])
                    except KeyError:
                        logging.error("No optimization method named {} supported".format(cfg['optimization_method']))
                        sys.exit(1)
        else:
            return super(AbstractOptimization, cls).__new__(cls)

    @classmethod
    def register_subclass(cls, short_name: str):
        """
        Allows to register the subclass with its short name

        :param short_name: the subclass to be registered
        :type short_name: string
        """
        def decorator(subclass):
            cls.optimization_methods_avail[short_name] = subclass
            return subclass

        return decorator

    @abstractmethod
    def desc(self):
        """
        Describes the optimization method

        """
        print('Optimization method description')

    @abstractmethod
    def optimize_cv(self, cv: xr.Dataset, img_ref: xr.Dataset, img_sec: xr.Dataset) -> xr.Dataset:
        """
        Optimizes the cost volume

        :param cv: the cost volume dataset
        :type cv:
            xarray.Dataset, with the data variables:
                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
        :param img_ref: reference Dataset image
        :type img_ref: xarray.DataArray
        :param img_sec: secondary Dataset image
        :type img_sec: xarray.DataArray
        :return: the cost volume dataset
        :rtype:
            xarray.Dataset, with the data variables:
                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
        """
