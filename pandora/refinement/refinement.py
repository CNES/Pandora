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
This module contains classes and functions associated to the subpixel refinement step.
"""

import logging
import warnings
from abc import ABCMeta, abstractmethod
from typing import Tuple, Callable, Dict

import numpy as np
import xarray as xr
from numba import njit, prange

import pandora.constants as cst


class AbstractRefinement:
    """
    Abstract Refinement class
    """

    __metaclass__ = ABCMeta

    subpixel_methods_avail: Dict = {}
    _refinement_method_name = None
    cfg = None

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
                except KeyError:
                    logging.error("No subpixel method named % supported", cfg["refinement_method"])
                    raise KeyError
            else:
                if isinstance(cfg["refinement_method"], unicode):  # type: ignore # pylint: disable=undefined-variable
                    # creating a plugin from registered short name given as unicode (py2 & 3 compatibility)
                    try:
                        return super(AbstractRefinement, cls).__new__(
                            cls.subpixel_methods_avail[cfg["refinement_method"].encode("utf-8")]
                        )
                    except KeyError:
                        logging.error(
                            "No subpixel method named % supported",
                            cfg["refinement_method"],
                        )
                        raise KeyError
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

        # This silences numba's TBB threading layer warning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # Conversion to numpy array ( .data ), because Numba does not support Xarray
            (itp_coeff, disp["disparity_map"].data, disp["validity_mask"].data,) = self.loop_refinement(
                cv["cost_volume"].data,
                disp["disparity_map"].data,
                disp["validity_mask"].data,
                d_min,
                d_max,
                subpixel,
                measure,
                self.refinement_method,
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

        # This silences numba's TBB threading layer warning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # Conversion to numpy array ( .data ), because Numba does not support Xarray
            (
                itp_coeff,
                disp_right["disparity_map"].data,
                disp_right["validity_mask"].data,
            ) = self.loop_approximate_refinement(
                cv_left["cost_volume"].data,
                disp_right["disparity_map"].data,
                disp_right["validity_mask"].data,
                d_min,
                d_max,
                subpixel,
                measure,
                self.refinement_method,
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
    @njit(parallel=True)
    def loop_refinement(
        cv: np.ndarray,
        disp: np.ndarray,
        mask: np.ndarray,
        d_min: int,
        d_max: int,
        subpixel: int,
        measure: str,
        method: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str], Tuple[int, int, int]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
         Apply for each pixels the refinement method

        :param cv: cost volume to refine
        :type cv: 3D numpy array (row, col, disp)
        :param disp: disparity map
        :type disp: 2D numpy array (row, col)
        :param mask: validity mask
        :type mask: 2D numpy array (row, col)
        :param d_min: minimal disparity
        :type d_min: int
        :param d_max: maximal disparity
        :type d_max: int
        :param subpixel: subpixel precision used to create the cost volume
        :type subpixel: int ( 1 | 2 | 4 )
        :param measure: the measure used to create the cot volume
        :param measure: string
        :param method: the refinement method
        :param method: function
        :return: the refine coefficient, the refine disparity map, and the validity mask
        :rtype: tuple(2D numpy array (row, col), 2D numpy array (row, col), 2D numpy array (row, col))
        """
        n_row, n_col, _ = cv.shape
        itp_coeff = np.zeros((n_row, n_col), dtype=np.float64)

        for row in prange(n_row):
            for col in prange(n_col):
                # No interpolation on invalid points
                if (mask[row, col] & cst.PANDORA_MSK_PIXEL_INVALID) != 0:
                    itp_coeff[row, col] = np.nan
                else:
                    # conversion to numpy indexing
                    dsp = int((disp[row, col] - d_min) * subpixel)
                    itp_coeff[row, col] = cv[row, col, dsp]
                    if not np.isnan(cv[row, col, dsp]):
                        if (disp[row, col] != d_min) and (disp[row, col] != d_max):

                            sub_disp, sub_cost, valid = method(  # type: ignore
                                [
                                    cv[row, col, dsp - 1],
                                    cv[row, col, dsp],
                                    cv[row, col, dsp + 1],
                                ],
                                disp[row, col],
                                measure,
                            )

                            disp[row, col] = disp[row, col] + (sub_disp / subpixel)
                            itp_coeff[row, col] = sub_cost
                            mask[row, col] += valid
                        else:
                            # If Information: calculations stopped at the pixel step, sub-pixel interpolation did
                            # not succeed
                            mask[row, col] += cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION

        return itp_coeff, disp, mask

    @abstractmethod
    @njit(cache=True)
    def refinement_method(self, cost: np.ndarray, disp: float, measure: str) -> Tuple[float, float, int]:
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

    @staticmethod
    @njit(parallel=True)
    def loop_approximate_refinement(
        cv: np.ndarray,
        disp: np.ndarray,
        mask: np.ndarray,
        d_min: int,
        d_max: int,
        subpixel: int,
        measure: str,
        method: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str], Tuple[int, int, int]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
         Apply for each pixels the refinement method on the right disparity map which was created with the approximate
          method : a diagonal search for the minimum on the left cost volume

        :param cv: the left cost volume
        :type cv: 3D numpy array (row, col, disp)
        :param disp: right disparity map
        :type disp: 2D numpy array (row, col)
        :param mask: right validity mask
        :type mask: 2D numpy array (row, col)
        :param d_min: minimal disparity
        :type d_min: int
        :param d_max: maximal disparity
        :type d_max: int
        :param subpixel: subpixel precision used to create the cost volume
        :type subpixel: int ( 1 | 2 | 4 )
        :param measure: the type of measure used to create the cost volume
        :type measure: string = min | max
        :param method: the refinement method
        :type method: function
        :return: the refine coefficient, the refine disparity map, and the validity mask
        :rtype: tuple(2D numpy array (row, col), 2D numpy array (row, col), 2D numpy array (row, col))
        """
        n_row, n_col, _ = cv.shape
        itp_coeff = np.zeros((n_row, n_col), dtype=np.float64)

        for row in prange(n_row):
            for col in prange(n_col):
                # No interpolation on invalid points
                if (mask[row, col] & cst.PANDORA_MSK_PIXEL_INVALID) != 0:
                    itp_coeff[row, col] = np.nan
                else:
                    # Conversion to numpy indexing
                    dsp = int((-disp[row, col] - d_min) * subpixel)
                    # Position of the best cost in the left cost volume is cv[r, diagonal, d]
                    diagonal = int(col + disp[row, col])
                    itp_coeff[row, col] = cv[row, diagonal, dsp]
                    if not np.isnan(cv[row, diagonal, dsp]):
                        if (
                            (disp[row, col] != -d_min)
                            and (disp[row, col] != -d_max)
                            and (diagonal != 0)
                            and (diagonal != (n_col - 1))
                        ):
                            # (1 * subpixel) because in fast mode, we can not have sub-pixel disparity for the right
                            # image.
                            # We therefore interpolate between pixel disparities
                            sub_disp, cost, valid = method(  # type:ignore
                                [
                                    cv[row, diagonal - 1, dsp + (1 * subpixel)],
                                    cv[row, diagonal, dsp],
                                    cv[row, diagonal + 1, dsp - (1 * subpixel)],
                                ],
                                disp[row, col],
                                measure,
                            )

                            disp[row, col] = disp[row, col] + (sub_disp / subpixel)
                            itp_coeff[row, col] = cost
                            mask[row, col] += valid
                        else:
                            # If Information: calculations stopped at the pixel step, sub-pixel interpolation did
                            # not succeed
                            mask[row, col] += cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION

        return itp_coeff, disp, mask
