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
This module contains functions associated to the multiscale step.
"""

import logging
from abc import ABCMeta, abstractmethod
from typing import Tuple, Union, Dict

import numpy as np
import xarray as xr

import pandora.constants as cst


class AbstractMultiscale:
    """
    Abstract Multiscale class
    """

    __metaclass__ = ABCMeta

    multiscale_methods_avail: Dict = {}
    cfg = None

    def __new__(cls, **cfg: Union[str, int]):
        """
        Return the plugin associated with the multiscale method given in the configuration

        :param cfg: configuration {'multiscale_method': value, 'margin': value}
        :type cfg: dictionary
        """
        if cls is AbstractMultiscale:
            if isinstance(cfg["multiscale_method"], str):
                try:
                    return super(AbstractMultiscale, cls).__new__(
                        cls.multiscale_methods_avail[cfg["multiscale_method"]]
                    )
                except KeyError:
                    logging.error(
                        "No multiscale matching method named % supported",
                        cfg["multiscale_method"],
                    )
                    raise KeyError
            else:
                if isinstance(cfg["multiscale_method"], unicode):  # type: ignore # pylint: disable=undefined-variable
                    # creating a plugin from registered short name given as unicode (py2 & 3 compatibility)
                    try:
                        return super(AbstractMultiscale, cls).__new__(
                            cls.multiscale_methods_avail[cfg["multiscale_method"].encode("utf-8")]
                        )
                    except KeyError:
                        logging.error(
                            "No multiscale matching method named % supported",
                            cfg["multiscale_method"],
                        )
                        raise KeyError
        else:
            return super(AbstractMultiscale, cls).__new__(cls)
        return None

    @classmethod
    def register_subclass(cls, short_name: str, *args):
        """
        Allows to register the subclass with its short name

        :param short_name: the subclass to be registered
        :type short_name: string
        :param args: allows to register one plugin that contains different methods
        """

        def decorator(subclass):
            """
            Registers the subclass in the available methods

            :param subclass: the subclass to be registered
            :type subclass: object
            """
            cls.multiscale_methods_avail[short_name] = subclass
            for arg in args:
                cls.multiscale_methods_avail[arg] = subclass
            return subclass

        return decorator

    @abstractmethod
    def desc(self):
        """
        Describes the multiscale method
        """
        print("Multiscale matching description")

    @abstractmethod
    def disparity_range(self, disp: xr.Dataset, disp_min: int, disp_max: int) -> Tuple[np.array, np.array]:
        """
        Disparity range computation by seeking the max and min values in the window.
        Unvalid disparities are given the full disparity range

        :param disp: the disparity dataset
        :type disp: xarray.Dataset with the data variables :
                - disparity_map 2D xarray.DataArray (row, col)
                - confidence_measure 3D xarray.DataArray(row, col, indicator)
        :param disp_min: absolute min disparity
        :type disp_min: int
        :param disp_max: absolute max disparity
        :type disp_max: int
        :return: Two np.darray :
                - disp_min_range : minimum disparity value for all pixels :
                - disp_max_range : maximum disparity value for all pixels.

        :rtype: tuple (np.ndarray, np.ndarray)
        """

    @staticmethod
    def mask_invalid_disparities(disp: xr.Dataset) -> np.ndarray:
        """
        Return a copied disparity map with all invalid disparities set to Nan

        :param disp: the disparity dataset
        :type disp: xarray.Dataset with the data variables :
                - disparity_map 2D xarray.DataArray (row, col)
                - confidence_measure 3D xarray.DataArray(row, col, indicator)

        :return: np.darray :
                - filtered_disp_map : disparity map with invalid values set to Nzn

        :rtype: tuple (np.ndarray, np.ndarray)
        """
        filtered_disp_map = disp["disparity_map"].data.copy()

        # Set all invalid disparities to nan
        for idxes, val in np.ndenumerate(disp["validity_mask"].data):
            if val & cst.PANDORA_MSK_PIXEL_INVALID != 0:
                filtered_disp_map[idxes] = np.nan

        return filtered_disp_map
