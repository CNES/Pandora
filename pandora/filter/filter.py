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
This module contains classes and functions associated to the disparity map filtering.
"""

import logging
import sys
from abc import ABCMeta, abstractmethod
from json_checker import Checker, And
from typing import Dict
import xarray as xr


class AbstractFilter(object):
    __metaclass__ = ABCMeta

    filter_methods_avail = {}

    def __new__(cls, **cfg: dict):
        """
        Return the plugin associated with the filter_method given in the configuration

        :param cfg: the configuration {'filter_method': value}
        :type cfg: dictionary
        """
        if cls is AbstractFilter:
            if type(cfg['filter_method']) is str:
                try:
                    return super(AbstractFilter, cls).__new__(cls.filter_methods_avail[cfg['filter_method']])
                except KeyError:
                    logging.error("No filter method named {} supported".format(cfg['filter_method']))
                    sys.exit(1)
            else:
                if type(cfg['filter_method']) is unicode:
                    # creating a plugin from registered short name given as unicode (py2 & 3 compatibility)
                    try:
                        return super(AbstractFilter, cls).__new__(cls.filter_methods_avail[cfg['filter_method'].encode('utf-8')])
                    except KeyError:
                        logging.error("No filter method named {} supported".format(cfg['filter_method']))
                        sys.exit(1)
        else:
            return super(AbstractFilter, cls).__new__(cls)

    @classmethod
    def register_subclass(cls, short_name: str):
        """
        Allows to register the subclass with its short name

        :param short_name: the subclass to be registered
        :type short_name: string
        """
        def decorator(subclass):
            cls.filter_methods_avail[short_name] = subclass
            return subclass

        return decorator

    @abstractmethod
    def desc(self):
        """
        Describes the filtering method

        """
        print('Filtering method description')

    @abstractmethod
    def filter_disparity(self, disp: xr.Dataset, img_ref: xr.Dataset = None, img_sec: xr.Dataset = None,
                         cv: xr.Dataset = None) -> xr.Dataset:
        """
        Post processing the disparity map by applying a filter on valid pixels

        :param disp: the disparity map dataset
        :type disp:
            xarray.Dataset with the variables :
                - disparity_map 2D xarray.DataArray (row, col)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
                - validity_mask 2D xarray.DataArray (row, col)
        :param img_ref: reference Dataset image
        :tye img_ref: xarray.Dataset
        :param img_sec: secondary Dataset image
        :type img_sec: xarray.Dataset
        :param cv: cost volume dataset
        :type cv: xarray.Dataset
        :return: the Dataset with the filtered DataArray disparity_map
        :rtype:
            xarray.Dataset with the variables :
                - disparity_map 2D xarray.DataArray (row, col)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
                - validity_mask 2D xarray.DataArray (row, col)
        """


@AbstractFilter.register_subclass('none')
class NoneFilter(AbstractFilter):
    """
    Default plugin that does not perform filtering
    """

    def __init__(self, **cfg: str) -> None:
        """
        :param cfg: optional configuration, {}
        :type cfg: dict
        """
        self.cfg = self.check_conf(**cfg)

    @staticmethod
    def check_conf(**cfg: str) -> Dict[str, str]:
        """
        Add default values to the dictionary if there are missing elements and check if the dictionary is correct

        :param cfg: filter configuration
        :type cfg: dict
        :return cfg: filter configuration updated
        :rtype: dict
        """
        schema = {
            "filter_method": And(str, lambda x: 'none')
        }

        checker = Checker(schema)
        checker.validate(cfg)
        return cfg

    def desc(self):
        """
        Describes the filtering method
        """
        print('No filtering method')

    def filter_disparity(self, disp: xr.Dataset, img_ref: xr.Dataset = None, img_sec: xr.Dataset = None,
                         cv: xr.Dataset = None) -> xr.Dataset:
        """
        Returns the disparity map without filtering

        :param disp: the disparity map dataset
        :type disp:
            xarray.Dataset with the variables :
                - disparity_map 2D xarray.DataArray (row, col)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
                - validity_mask 2D xarray.DataArray (row, col)
        :param img_ref: reference Dataset image
        :tye img_ref: xarray.Dataset
        :param img_sec: secondary Dataset image
        :type img_sec: xarray.Dataset
        :param cv: cost volume dataset
        :type cv: xarray.Dataset
        :return: the Dataset with the filtered DataArray disparity_map
        :rtype:
            xarray.Dataset with the variables :
                - disparity_map 2D xarray.DataArray (row, col)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
                - validity_mask 2D xarray.DataArray (row, col)
        """
        return disp
