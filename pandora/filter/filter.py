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
This module contains classes and functions associated to the disparity map filtering.
"""

import logging
import sys
from abc import ABCMeta, abstractmethod
from typing import Dict
import xarray as xr


class AbstractFilter:
    """
    Abstract Filter class
    """

    __metaclass__ = ABCMeta

    filter_methods_avail: Dict = {}
    cfg = None

    def __new__(cls, **cfg: dict):
        """
        Return the plugin associated with the filter_method given in the configuration

        :param cfg: the configuration {'filter_method': value}
        :type cfg: dictionary
        """
        if cls is AbstractFilter:
            if isinstance(cfg["filter_method"], str):
                try:
                    return super(AbstractFilter, cls).__new__(cls.filter_methods_avail[cfg["filter_method"]])
                except KeyError:
                    logging.error("No filter method named % supported", cfg["filter_method"])
                    sys.exit(1)
            else:
                if isinstance(cfg["filter_method"], unicode):  # type: ignore # pylint: disable=undefined-variable
                    # creating a plugin from registered short name given as unicode (py2 & 3 compatibility)
                    try:
                        return super(AbstractFilter, cls).__new__(
                            cls.filter_methods_avail[cfg["filter_method"].encode("utf-8")]
                        )
                    except KeyError:
                        logging.error("No filter method named % supported", cfg["filter_method"])
                        sys.exit(1)
        else:
            return super(AbstractFilter, cls).__new__(cls)
        return None

    @classmethod
    def register_subclass(cls, short_name: str):
        """
        Allows to register the subclass with its short name

        :param short_name: the subclass to be registered
        :type short_name: string
        """

        def decorator(subclass):
            """
            Registers the subclass in the available methods

            :param subclass: the subclass to be registered
            :type subclass: object
            """
            cls.filter_methods_avail[short_name] = subclass
            return subclass

        return decorator

    @abstractmethod
    def desc(self):
        """
        Describes the filtering method

        """
        print("Filtering method description")

    @abstractmethod
    def filter_disparity(
        self,
        disp: xr.Dataset,
        img_left: xr.Dataset = None,
        img_right: xr.Dataset = None,
        cv: xr.Dataset = None,
    ) -> None:
        """
        Post processing the disparity map by applying a filter on valid pixels

        :param disp: the disparity map dataset with the variables :

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
