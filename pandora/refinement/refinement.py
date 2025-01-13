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
This module contains classes and functions associated to the subpixel refinement step.
"""

from abc import ABCMeta, abstractmethod
from typing import Dict, Tuple

import numpy as np
import xarray as xr

import pandora.constants as cst
from pandora.margins.descriptors import NullMargins

from .cpp import refinement_cpp


class AbstractRefinement:
    """
    Abstract Refinement class
    """

    __metaclass__ = ABCMeta

    subpixel_methods_avail: Dict = {}
    _refinement_method_name = None
    cfg = None
    margins = NullMargins()

    def __new__(cls, **cfg: dict):
        """
        Return the plugin associated with the refinement_method given in the configuration

        :param cfg: configuration {'refinement_method': value}
        :type cfg: dictionary
        """
        if cls is AbstractRefinement:
            if isinstance(cfg["refinement_method"], str):
                try:
                    return super(AbstractRefinement, cls).__new__(cls.subpixel_methods_avail[cfg["refinement_method"]])
                except:
                    raise KeyError("No refinement method named {} supported".format(cfg["refinement_method"]))
            else:
                if isinstance(cfg["refinement_method"], unicode):  # type: ignore # pylint: disable=undefined-variable
                    # creating a plugin from registered short name given as unicode (py2 & 3 compatibility)
                    try:
                        return super(AbstractRefinement, cls).__new__(
                            cls.subpixel_methods_avail[cfg["refinement_method"].encode("utf-8")]
                        )
                    except:
                        raise KeyError("No refinement method named {} supported".format(cfg["refinement_method"]))
        else:
            return super(AbstractRefinement, cls).__new__(cls)
        return None

    def subpixel_refinement(self, cv: xr.Dataset, disp: xr.Dataset) -> None:
        """
        Subpixel refinement of disparities and costs.

        :param cv: the cost volume dataset with the data variables:

                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
        :type cv: xarray.Dataset
        :param disp: Dataset with the variables :

            - disparity_map 2D xarray.DataArray (row, col)
            - confidence_measure 3D xarray.DataArray (row, col, indicator)
            - validity_mask 2D xarray.DataArray (row, col)
        :type disp: xarray.Dataset
        :return: None
        """
        d_min = cv.coords["disp"].data[0]
        d_max = cv.coords["disp"].data[-1]
        subpixel = cv.attrs["subpixel"]
        measure = cv.attrs["type_measure"]

        (
            itp_coeff,
            disp["disparity_map"].data,
            disp["validity_mask"].data,
        ) = refinement_cpp.loop_refinement(
            cv["cost_volume"].data,
            disp["disparity_map"].data,
            disp["validity_mask"].data,
            d_min,
            d_max,
            subpixel,
            measure,
            self.refinement_method,
            cst.PANDORA_MSK_PIXEL_INVALID,
            cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION,
        )

        disp.attrs["refinement"] = self._refinement_method_name
        disp["interpolated_coeff"] = xr.DataArray(
            itp_coeff,
            coords=[("row", disp.coords["row"].data), ("col", disp.coords["col"].data)],
            dims=["row", "col"],
        )

    def approximate_subpixel_refinement(self, cv_left: xr.Dataset, disp_right: xr.Dataset) -> xr.Dataset:
        """
        Subpixel refinement of the right disparities map, which was created with the approximate method : a diagonal
        search for the minimum on the left cost volume

        :param cv_left: the left cost volume dataset with the data variables:

                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
        :type cv_left: xarray.Dataset
        :param disp_right: right disparity map with the variables :

            - disparity_map 2D xarray.DataArray (row, col)
            - confidence_measure 3D xarray.DataArray (row, col, indicator)
            - validity_mask 2D xarray.DataArray (row, col)
        :type disp_right: xarray.Dataset
        :return: disp_right Dataset with the variables :

                - disparity_map 2D xarray.DataArray (row, col) that contains the refined disparities
                - confidence_measure 3D xarray.DataArray (row, col, indicator) (unchanged)
                - validity_mask 2D xarray.DataArray (row, col) with the value of bit 3 ( Information: \
                calculations stopped at the pixel step, sub-pixel interpolation did not succeed )
                - interpolated_coeff 2D xarray.DataArray (row, col) that contains the refined cost
        :rtype: xarray.Dataset
        """
        d_min = cv_left.coords["disp"].data[0]
        d_max = cv_left.coords["disp"].data[-1]
        subpixel = cv_left.attrs["subpixel"]
        measure = cv_left.attrs["type_measure"]

        (
            itp_coeff,
            disp_right["disparity_map"].data,
            disp_right["validity_mask"].data,
        ) = refinement_cpp.loop_approximate_refinement(
            cv_left["cost_volume"].data,
            disp_right["disparity_map"].data,
            disp_right["validity_mask"].data,
            d_min,
            d_max,
            subpixel,
            measure,
            self.refinement_method,
            cst.PANDORA_MSK_PIXEL_INVALID,
            cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION,
        )

        disp_right.attrs["refinement"] = self._refinement_method_name
        disp_right["interpolated_coeff"] = xr.DataArray(
            itp_coeff,
            coords=[
                ("row", disp_right.coords["row"].data),
                ("col", disp_right.coords["col"].data),
            ],
            dims=["row", "col"],
        )
        return disp_right

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
            cls.subpixel_methods_avail[short_name] = subclass
            return subclass

        return decorator

    @abstractmethod
    def desc(self) -> None:
        """
        Describes the subpixel method
        :return: None
        """
        print("Subpixel method description")

    @staticmethod
    @abstractmethod
    def refinement_method(cost: np.ndarray, disp: float, measure: str) -> Tuple[float, float, int]:
        """
        Return the subpixel disparity and cost

        :param cost: cost of the values disp - 1, disp, disp + 1
        :type cost: 1D numpy array : [cost[disp -1], cost[disp], cost[disp + 1]]
        :param disp: the disparity
        :type disp: float
        :param measure: the type of measure used to create the cost volume
        :type measure: string = min | max
        :return: the refined disparity (disp + (sub_disp/subpix)), the refined cost and the state of the pixel
         ( Information: calculations stopped at the pixel step, sub-pixel interpolation did not succeed )
        :rtype: float, float, int
        """
