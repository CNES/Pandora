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
    def subpixel_refinement(self, cv: xr.Dataset, disp: xr.Dataset, img_ref: xr.Dataset = None,
                            img_sec: xr.Dataset = None) -> Tuple[xr.Dataset, xr.Dataset]:
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
        :param img_ref: reference Dataset image
        :type img_ref:
            xarray.Dataset containing:
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :param img_sec: secondary Dataset image
        :type img_sec:
            xarray.Dataset containing:
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :return:
            cv Dataset with the variables (unchanged):
                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
            disp Dataset with the variables:
                - disparity_map 2D xarray.DataArray (row, col) that contains the refined disparities
                - confidence_measure 3D xarray.DataArray (row, col, indicator) (unchanged)
                - validity_mask 2D xarray.DataArray (row, col) with the state of the pixel ( Information:
                    calculations stopped at the pixel step, sub-pixel interpolation did not succeed )
                - interpolated_coeff 2D xarray.DataArray (row, col) that contains the refined cost
        :rtype: tuple(Dataset cv, Dataset disp)
        """

    @abstractmethod
    def approximate_subpixel_refinement(self, cv_ref: xr.Dataset, disp_sec: xr.Dataset, img_ref: xr.Dataset = None,
                                        img_sec: xr.Dataset = None) -> xr.Dataset:
        """
        Subpixel refinement of the secondary disparities map, which was created with the approximate method : a diagonal
        search for the minimum on the reference cost volume

        :param cv_ref: the reference cost volume dataset
        :type cv_ref:
            xarray.Dataset, with the data variables:
                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
        :param disp_sec: secondary disparity map
        :type disp_sec: xarray.Dataset with the variables :
            - disparity_map 2D xarray.DataArray (row, col)
            - confidence_measure 3D xarray.DataArray (row, col, indicator)
            - validity_mask 2D xarray.DataArray (row, col)
        :param img_ref: reference Dataset image
        :type img_ref:
            xarray.Dataset containing:
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :param img_sec: secondary Dataset image
        :type img_sec:
            xarray.Dataset containing:
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :return:
            disp_sec Dataset with the variables :
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
         Apply for each pixels the refinement method on the secondary disparity map which was created with the approximate
          method : a diagonal search for the minimum on the reference cost volume

        :param cv: the reference cost volume
        :type cv: 3D numpy array (row, col, disp)
        :param disp: secondary disparity map
        :type disp: 2D numpy array (row, col)
        :param mask: secondary validity mask
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
                    # Position of the best cost in the reference cost volume is cv[r, diagonal, d]
                    diagonal = int(c + disp[r, c])
                    itp_coeff[r, c] = cv[r, diagonal, d]
                    if not (np.isnan(cv[r, diagonal, d])):
                        if (disp[r, c] != -d_min) and (disp[r, c] != -d_max) and (diagonal != 0) and (
                                diagonal != (col - 1)):
                            # (1 * subpixel) because in fast mode, we can not have sub-pixel disparity for the secondary
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


@AbstractRefinement.register_subclass('none')
class NoneRefinement(AbstractRefinement):
    """
    Default plugin that does not perform refinement
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

        :param cfg: refinement configuration
        :type cfg: dict
        :return cfg: refinement configuration updated
        :rtype: dict
        """
        schema = {
            "refinement_method": And(str, lambda x: 'none')
        }

        checker = Checker(schema)
        checker.validate(cfg)
        return cfg

    def desc(self):
        """
        Describes the subpixel method
        """
        print('No refinement')

    def subpixel_refinement(self, cv: xr.Dataset, disp: xr.Dataset, img_ref: xr.Dataset = None,
                            img_sec: xr.Dataset = None) -> Tuple[xr.Dataset, xr.Dataset]:
        """
        Returns the cost volume and the disparity map without subpixel refinement

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
        :param img_ref: reference Dataset image
        :type img_ref:
            xarray.Dataset containing:
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :param img_sec: secondary Dataset image
        :type img_sec:
            xarray.Dataset containing:
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :return:
            cv Dataset with the variables (unchanged):
                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
            disp Dataset with the variables:
                - disparity_map 2D xarray.DataArray (row, col) that contains the refined disparities
                - confidence_measure 3D xarray.DataArray (row, col, indicator) (unchanged)
                - validity_mask 2D xarray.DataArray (row, col) with the state of the pixel ( Information:
                    calculations stopped at the pixel step, sub-pixel interpolation did not succeed )
                - interpolated_coeff 2D xarray.DataArray (row, col) that contains the refined cost
        :rtype: tuple(Dataset cv, Dataset disp)
        """
        return cv, disp

    def approximate_subpixel_refinement(self, cv_ref: xr.Dataset, disp_sec: xr.Dataset, img_ref: xr.Dataset = None,
                                        img_sec: xr.Dataset = None) -> xr.Dataset:
        """
        Returns the secondary disparity map without subpixel refinement

        :param cv_ref: the reference cost volume dataset
        :type cv_ref:
            xarray.Dataset, with the data variables:
                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
        :param disp_sec: secondary disparity map
        :type disp_sec: xarray.Dataset with the variables :
            - disparity_map 2D xarray.DataArray (row, col)
            - confidence_measure 3D xarray.DataArray (row, col, indicator)
            - validity_mask 2D xarray.DataArray (row, col)
        :param img_ref: reference Dataset image
        :type img_ref:
            xarray.Dataset containing:
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :param img_sec: secondary Dataset image
        :type img_sec:
            xarray.Dataset containing:
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :return:
            disp_sec Dataset with the variables :
                - disparity_map 2D xarray.DataArray (row, col) that contains the refined disparities
                - confidence_measure 3D xarray.DataArray (row, col, indicator) (unchanged)
                - validity_mask 2D xarray.DataArray (row, col) with the value of bit 3 ( Information:
                    calculations stopped at the pixel step, sub-pixel interpolation did not succeed )
                - interpolated_coeff 2D xarray.DataArray (row, col) that contains the refined cost
        :rtype: Dataset
        """
        return disp_sec
