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
This module contains classes and functions associated to the cost volume aggregation step.
"""

import sys
import logging
import xarray as xr
from abc import ABCMeta, abstractmethod
from json_checker import Checker, And
from typing import Dict, Union


class AbstractAggregation(object):
    __metaclass__ = ABCMeta

    aggreg_methods_avail = {}

    def __new__(cls, **cfg: dict):
        """
        Return the plugin associated with the aggregation_method given in the configuration

        :param cfg: the configuration {'aggregation_method': value}
        :type cfg: dictionary
        """
        if cls is AbstractAggregation:
            if type(cfg['aggregation_method']) is str:
                try:
                    return super(AbstractAggregation, cls).__new__(cls.aggreg_methods_avail[cfg['aggregation_method']])
                except KeyError:
                    logging.error("No aggregation method named {} supported".format(cfg['aggregation_method']))
                    sys.exit(1)
            else:
                if type(cfg['aggregation_method']) is unicode:
                    # creating a plugin from registered short name given as unicode (py2 & 3 compatibility)
                    try:
                        return super(AbstractAggregation, cls).__new__(cls.aggreg_methods_avail[cfg['aggregation_method'].encode('utf-8')])
                    except KeyError:
                        logging.error("No aggregation method named {} supported".format(cfg['aggregation_method']))
                        sys.exit(1)
        else:
            return super(AbstractAggregation, cls).__new__(cls)

    @classmethod
    def register_subclass(cls, short_name: str):
        """
        Allows to register the subclass with its short name

        :param short_name: the subclass to be registered
        :type short_name: string
        """
        def decorator(subclass):
            cls.aggreg_methods_avail[short_name] = subclass
            return subclass

        return decorator

    @abstractmethod
    def desc(self):
        """
        Describes the aggregation method

        """
        print('Aggregation description')

    @abstractmethod
    def cost_volume_aggregation(self, img_left: xr.Dataset, img_right: xr.Dataset, cv: xr.Dataset,
                                **cfg: Union[str, int]) -> None:
        """
        Aggregate the cost volume for a pair of images

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
        :param cv: the cost volume dataset
        :type cv:
            xarray.Dataset, with the data variables:
                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
        :param cfg: images configuration containing the mask convention : valid_pixels, no_data
        :type cfg: dict
        :return: None
        """

