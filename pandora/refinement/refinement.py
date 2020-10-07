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
This module contains classes and functions associated to the subpixel refinement step.
"""

import logging
from numba import njit, prange
import numpy as np
from abc import ABCMeta, abstractmethod
from json_checker import Checker, And
from typing import Dict, Tuple, Callable
import xarray as xr

from pandora.constants import *


class AbstractRefinement(object):
    __metaclass__ = ABCMeta

    subpixel_methods_avail = {}

    def __new__(cls, **cfg: dict):
        """
        Return the plugin associated with the refinement_method given in the configuration

        :param cfg: configuration {'refinement_method': value}
        :type cfg: dictionary
        """
        if cls is AbstractRefinement:
            if type(cfg['refinement_method']) is str:
                try:
                    return super(AbstractRefinement, cls).__new__(cls.subpixel_methods_avail[cfg['refinement_method']])
                except KeyError:
                    logging.error("No subpixel method named {} supported".format(cfg['refinement_method']))
                    raise KeyError
            else:
                if type(cfg['refinement_method']) is unicode:
                    # creating a plugin from registered short name given as unicode (py2 & 3 compatibility)
                    try:
                        return super(AbstractRefinement, cls).__new__(
                            cls.subpixel_methods_avail[cfg['refinement_method'].encode('utf-8')])
                    except KeyError:
                        logging.error("No subpixel matching method named {} supported".format(cfg['refinement_method']))
                        raise KeyError
        else:
            return super(AbstractRefinement, cls).__new__(cls)

    @classmethod
    def register_subclass(cls, short_name: str):
        """
        Allows to register the subclass with its short name

        :param short_name: the subclass to be registered
        :type short_name: string
        """
        def decorator(subclass):
            cls.subpixel_methods_avail[short_name] = subclass
            return subclass

        return decorator

    @abstractmethod
    def desc(self):
        """
        Describes the subpixel method
        """
        print('Subpixel method description')

    @abstractmethod
    def subpixel_refinement(self, cv: xr.Dataset, disp: xr.Dataset, img_left: xr.Dataset = None,
                            img_right: xr.Dataset = None) -> None:
        """
        Subpixel refinement of disparities and costs.

        :param cv: the cost volume dataset
        :type cv:
            xarray.Dataset, with the data variables:
                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
        :param disp: Dataset
        :type disp: xarray.Dataset with the variables :
            - disparity_map 2D xarray.DataArray (row, col)
            - confidence_measure 3D xarray.DataArray (row, col, indicator)
            - validity_mask 2D xarray.DataArray (row, col)
        :param img_left: left Dataset image
        :type img_left:
            xarray.Dataset containing:
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :param img_right: right Dataset image
        :type img_right:
            xarray.Dataset containing:
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :return: None
        """

    @abstractmethod
    def approximate_subpixel_refinement(self, cv_left: xr.Dataset, disp_right: xr.Dataset, img_left: xr.Dataset = None,
                                        img_right: xr.Dataset = None) -> xr.Dataset:
        """
        Subpixel refinement of the right disparities map, which was created with the approximate method : a diagonal
        search for the minimum on the left cost volume

        :param cv_leftf: the left cost volume dataset
        :type cv_left:
            xarray.Dataset, with the data variables:
                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
        :param disp_right: right disparity map
        :type disp_right: xarray.Dataset with the variables :
            - disparity_map 2D xarray.DataArray (row, col)
            - confidence_measure 3D xarray.DataArray (row, col, indicator)
            - validity_mask 2D xarray.DataArray (row, col)
        :param img_left: left Dataset image
        :type img_left:
            xarray.Dataset containing:
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :param img_right: right Dataset image
        :type img_right:
            xarray.Dataset containing:
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :return:
            disp_right Dataset with the variables :
                - disparity_map 2D xarray.DataArray (row, col) that contains the refined disparities
                - confidence_measure 3D xarray.DataArray (row, col, indicator) (unchanged)
                - validity_mask 2D xarray.DataArray (row, col) with the value of bit 3 ( Information:
                    calculations stopped at the pixel step, sub-pixel interpolation did not succeed )
                - interpolated_coeff 2D xarray.DataArray (row, col) that contains the refined cost
        :rtype: Dataset
        """

    @staticmethod
    @njit(parallel=True)
    def loop_refinement(cv: np.ndarray, disp: np.ndarray, mask: np.ndarray, d_min: int, d_max: int, subpixel: int,
                        measure: str, method: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str],
                                                       Tuple[int, int, int]]) -> Tuple[np.ndarray, np.ndarray,
                                                                                       np.ndarray]:
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
        row, col, _ = cv.shape
        itp_coeff = np.zeros((row, col), dtype=np.float64)

        for r in prange(row):
            for c in prange(col):
                # No interpolation on invalid points
                if (mask[r, c] & PANDORA_MSK_PIXEL_INVALID) != 0:
                    itp_coeff[r, c] = np.nan
                else:
                    # conversion to numpy indexing
                    d = int((disp[r, c] - d_min) * subpixel)
                    itp_coeff[r, c] = cv[r, c, d]
                    if not (np.isnan(cv[r, c, d])):
                        if (disp[r, c] != d_min) and (disp[r, c] != d_max):

                            sub_disp, sub_cost, valid = method([cv[r, c, d - 1], cv[r, c, d], cv[r, c, d + 1]],
                                                               disp[r, c],
                                                               measure)

                            disp[r, c] = sub_disp
                            itp_coeff[r, c] = sub_cost
                            mask[r, c] += valid
                        else:
                            # If Information: calculations stopped at the pixel step, sub-pixel interpolation did
                            # not succeed
                            mask[r, c] += PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION

        return itp_coeff, disp, mask

    @staticmethod
    @njit(parallel=True)
    def loop_approximate_refinement(cv: np.ndarray, disp: np.ndarray, mask: np.ndarray, d_min: int, d_max: int,
                                    subpixel: int, measure: str, method: Callable[[np.ndarray, np.ndarray, np.ndarray,
                                    np.ndarray, str], Tuple[int, int, int]]) -> Tuple[np.ndarray, np.ndarray,
                                                                                       np.ndarray]:
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
        :param measure: string = min | max
        :param method: the refinement method
        :param method: function
        :return: the refine coefficient, the refine disparity map, and the validity mask
        :rtype: tuple(2D numpy array (row, col), 2D numpy array (row, col), 2D numpy array (row, col))
         """
        row, col, _ = cv.shape
        itp_coeff = np.zeros((row, col), dtype=np.float64)

        for r in prange(row):
            for c in prange(col):
                # No interpolation on invalid points
                if (mask[r, c] & PANDORA_MSK_PIXEL_INVALID) != 0:
                    itp_coeff[r, c] = np.nan
                else:
                    # Conversion to numpy indexing
                    d = int((-disp[r, c] - d_min) * subpixel)
                    # Position of the best cost in the left cost volume is cv[r, diagonal, d]
                    diagonal = int(c + disp[r, c])
                    itp_coeff[r, c] = cv[r, diagonal, d]
                    if not (np.isnan(cv[r, diagonal, d])):
                        if (disp[r, c] != -d_min) and (disp[r, c] != -d_max) and (diagonal != 0) and (
                                diagonal != (col - 1)):
                            # (1 * subpixel) because in fast mode, we can not have sub-pixel disparity for the right
                            # image.
                            # We therefore interpolate between pixel disparities
                            sub_disp, cost, valid = method([cv[r, diagonal - 1, d + (1 * subpixel)], cv[r, diagonal, d],
                                                            cv[r, diagonal + 1, d - (1 * subpixel)]], disp[r, c],
                                                           measure)

                            disp[r, c] = sub_disp
                            itp_coeff[r, c] = cost
                            mask[r, c] += valid
                        else:
                            # If Information: calculations stopped at the pixel step, sub-pixel interpolation did
                            # not succeed
                            mask[r, c] += PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION

        return itp_coeff, disp, mask
