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
This module contains classes and functions associated to the cost volume optimization step.
"""

import logging
import sys
from abc import ABCMeta, abstractmethod
from typing import Dict

import xarray as xr


class AbstractOptimization:
    """
    Abstract Optimizationinput class
    """

    __metaclass__ = ABCMeta

    optimization_methods_avail: Dict = {}
    cfg = None

    def __new__(cls, **cfg: Dict[str, dict]):
        """
        Return the plugin associated with the optimization_method given in the configuration

        :param cfg: configuration {'optimization_method': value}
        :type cfg: dictionary
        """
        if cls is AbstractOptimization:
            if isinstance(cfg["optimization_method"], str):
                try:
                    return super(AbstractOptimization, cls).__new__(
                        cls.optimization_methods_avail[cfg["optimization_method"]]
                    )
                except KeyError:
                    logging.error(
                        "No optimization method named % supported",
                        cfg["optimization_method"],
                    )
                    sys.exit(1)
            else:
                if isinstance(cfg["optimization_method"], unicode):  # type: ignore # pylint: disable=undefined-variable
                    # creating a plugin from registered short name given as unicode (py2 & 3 compatibility)
                    try:
                        return super(AbstractOptimization, cls).__new__(
                            cls.optimization_methods_avail[cfg["optimization_method"].encode("utf-8")]
                        )
                    except KeyError:
                        logging.error(
                            "No optimization method named % supported",
                            cfg["optimization_method"],
                        )
                        sys.exit(1)
        else:
            return super(AbstractOptimization, cls).__new__(cls)
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
            cls.optimization_methods_avail[short_name] = subclass
            return subclass

        return decorator

    @abstractmethod
    def desc(self) -> None:
        """
        Describes the optimization method
        :return: None
        """
        print("Optimization method description")

    @abstractmethod
    def optimize_cv(self, cv: xr.Dataset, img_left: xr.Dataset, img_right: xr.Dataset) -> xr.Dataset:
        """
        Optimizes the cost volume

        :param cv: the cost volume dataset with the data variables:

                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
        :type cv: xarray.Dataset
        :param img_left: left Dataset image
        :type img_left: xarray.DataArray
        :param img_right: right Dataset image
        :type img_right: xarray.DataArray
        :return: the cost volume dataset with the data variables:

                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
        :rtype: xarray.Dataset
        """
