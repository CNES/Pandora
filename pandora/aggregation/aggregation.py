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
This module contains classes and functions associated to the cost volume aggregation step.
"""

import logging
import sys
from abc import ABCMeta, abstractmethod
from typing import Union, Dict

import xarray as xr


class AbstractAggregation:
    """
    Abstract Aggregation class
    """

    __metaclass__ = ABCMeta

    aggreg_methods_avail: Dict = {}
    cfg = None

    def __new__(cls, **cfg: dict):
        """
        Return the plugin associated with the aggregation_method given in the configuration

        :param cfg: the configuration {'aggregation_method': value}
        :type cfg: dictionary
        """

        if cls is AbstractAggregation:
            if isinstance(cfg["aggregation_method"], str):
                try:
                    return super(AbstractAggregation, cls).__new__(cls.aggreg_methods_avail[cfg["aggregation_method"]])
                except KeyError:
                    logging.error(
                        "No aggregation method named % supported",
                        cfg["aggregation_method"],
                    )
                    sys.exit(1)
            else:
                if isinstance(cfg["aggregation_method"], unicode):  # type: ignore # pylint: disable=undefined-variable
                    # creating a plugin from registered short name given as unicode (py2 & 3 compatibility)
                    try:
                        return super(AbstractAggregation, cls).__new__(
                            cls.aggreg_methods_avail[cfg["aggregation_method"].encode("utf-8")]
                        )
                    except KeyError:
                        logging.error(
                            "No aggregation method named % supported",
                            cfg["aggregation_method"],
                        )
                        sys.exit(1)
        else:
            return super(AbstractAggregation, cls).__new__(cls)
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
            cls.aggreg_methods_avail[short_name] = subclass
            return subclass

        return decorator

    @abstractmethod
    def desc(self):
        """
        Describes the aggregation method

        """
        print("Aggregation description")

    @abstractmethod
    def cost_volume_aggregation(
        self, img_left: xr.Dataset, img_right: xr.Dataset, cv: xr.Dataset, **cfg: Union[str, int]
    ) -> None:
        """
        Aggregate the cost volume for a pair of images

        :param img_left: left Dataset image containing :

                - im : 2D (row, col) xarray.DataArray
                - msk (optional): 2D (row, col) xarray.DataArray
        :type img_left: xarray.Dataset
        :param img_right: right Dataset image containing :

                - im : 2D (row, col) xarray.DataArray
                - msk (optional): 2D (row, col) xarray.DataArray
        :type img_right: xarray.Dataset
        :param cv: the cost volume dataset with the data variables:

                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
        :type cv: xarray.Dataset
        :param cfg: images configuration containing the mask convention : valid_pixels, no_data
        :type cfg: dict
        :return: None
        """
