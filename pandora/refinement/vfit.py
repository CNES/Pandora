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
This module contains functions associated to the vfit method used in the refinement step.
"""

from numba import njit
import numpy as np
import xarray as xr
from json_checker import Checker, And
from typing import Dict, Tuple

from . import refinement
from pandora.constants import *


@refinement.AbstractRefinement.register_subclass('vfit')
class Vfit(refinement.AbstractRefinement):
    """
    Vfit class allows to perform the subpixel cost refinement step
    """

    def __init__(self, **cfg: str):
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
            "refinement_method": And(str, lambda x: 'vfit')
        }

        checker = Checker(schema)
        checker.validate(cfg)
        return cfg

    def desc(self):
        """
        Describes the subpixel refinement method
        """
        print('Vfit refinement method')

    def subpixel_refinement(self, cv: xr.Dataset, disp: xr.Dataset, img_left: xr.Dataset = None,
                            img_right: xr.Dataset = None) -> Tuple[xr.Dataset, xr.Dataset]:
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
        d_min = cv.coords['disp'].data[0]
        d_max = cv.coords['disp'].data[-1]
        subpixel = cv.attrs['subpixel']
        measure = cv.attrs['type_measure']

        # Conversion to numpy array ( .data ), because Numba does not support Xarray
        itp_coeff, disp['disparity_map'].data, disp['validity_mask'].data = \
            self.loop_refinement(cv['cost_volume'].data, disp['disparity_map'].data, disp['validity_mask'].data,
                                 d_min, d_max, subpixel, measure, self.vfit)

        disp.attrs['refinement'] = 'vfit'
        disp['interpolated_coeff'] = xr.DataArray(itp_coeff,
                                                      coords=[disp.coords['row'], disp.coords['col']],
                                                      dims=['row', 'col'])
        return cv, disp

    def approximate_subpixel_refinement(self, cv_left: xr.Dataset, disp_right: xr.Dataset, img_left: xr.Dataset = None,
                                        img_right: xr.Dataset = None) -> xr.Dataset:
        """
        Subpixel refinement of the right disparities map, which was created with the approximate method : a diagonal
        search for the minimum on the left cost volume

        :param cv_left: the left cost volume dataset
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
        d_min = cv_left.coords['disp'].data[0]
        d_max = cv_left.coords['disp'].data[-1]
        subpixel = cv_left.attrs['subpixel']
        measure = cv_left.attrs['type_measure']

        # Conversion to numpy array ( .data ), because Numba does not support Xarray
        itp_coeff, disp_right['disparity_map'].data, disp_right['validity_mask'].data = self.loop_approximate_refinement(
            cv_left['cost_volume'].data, disp_right['disparity_map'].data, disp_right['validity_mask'].data, d_min, d_max,
            subpixel, measure, self.vfit)

        disp_right.attrs['refinement'] = 'vfit'
        disp_right['interpolated_coeff'] = xr.DataArray(itp_coeff,
                                                      coords=[disp_right.coords['row'], disp_right.coords['col']],
                                                      dims=['row', 'col'])
        return disp_right

    @staticmethod
    @njit()
    def vfit(cost: np.ndarray, disp: float, measure: str) -> Tuple[float, float, int]:
        """
        Return the subpixel disparity and cost, by matching a symmetric V shape (linear interpolation)

        :param cost: cost of the values disp - 1, disp, disp + 1
        :type cost: 1D numpy array : [cost[disp -1], cost[disp], cost[disp + 1]]
        :param disp: the disparity
        :type disp: float
        :param measure: the type of measure used to create the cost volume
        :param measure: string = min | max
        :return: the refined disparity (disp + sub_disp), the refined cost and the state of the pixel( Information:
        calculations stopped at the pixel step, sub-pixel interpolation did not succeed )
        :rtype: float, float, int
        """
        if (np.isnan(cost[0])) or (np.isnan(cost[2])):
            # Information: calculations stopped at the pixel step, sub-pixel interpolation did not succeed
            return disp, cost[1], PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION

        inverse = 1
        if measure == 'max':
            # Additive inverse : if a < b then -a > -b
            inverse = -1

        # The problem is to approximate sub_cost function with an affine function: y = a * x + origin
        # Calculate the slope
        a = cost[2] - cost[1]

        # Compare the difference disparity between (cost[0]-cost[1]) and (cost[2]-cost[1]): the highest cost is used
        if (inverse * cost[0]) > (inverse * cost[2]):
            a = cost[0] - cost[1]

        if abs(a) < 1.e-15:
            return disp, cost[1], 0

        # Problem is resolved with tangents equality, due to the symmetric V shape of 3 points (cv0, cv2 and (x,y))
        # sub_disp is dx
        sub_disp = ((cost[0] - cost[2]) / (2 * a))

        # sub_cost is y
        sub_cost = a * (sub_disp - 1) + cost[2]

        return sub_disp + disp, sub_cost, 0
