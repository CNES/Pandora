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
This module contains classes and functions associated to the interpolation of the disparity map for the validation step.
"""

import numpy as np
import xarray as xr
import logging
from numba import njit
import math
from typing import Tuple

from abc import ABCMeta, abstractmethod
from pandora.constants import *


class AbstractInterpolation(object):
    __metaclass__ = ABCMeta

    interpolation_methods_avail = {}

    def __new__(cls, **cfg: dict):
        """
        Return the plugin associated with the interpolated_disparity given in the configuration

        :param cfg: configuration {'interpolated_disparity': value}
        :type cfg: dictionary
        """
        if cls is AbstractInterpolation:
            if type(cfg['interpolated_disparity']) is str:
                try:
                    return super(AbstractInterpolation, cls).__new__(
                        cls.interpolation_methods_avail[cfg['interpolated_disparity']])
                except KeyError:
                    logging.error("No interpolation method named {} supported".format(cfg['interpolated_disparity']))
                    raise KeyError
            else:
                if type(cfg['interpolated_disparity']) is unicode:
                    # creating a plugin from registered short name given as unicode (py2 & 3 compatibility)
                    try:
                        return super(AbstractInterpolation, cls).__new__(
                            cls.interpolation_methods_avail[cfg['interpolated_disparity'].encode('utf-8')])
                    except KeyError:
                        logging.error("No interpolation method named {} supported".format(cfg['interpolated_disparity']))
                        raise KeyError
        else:
            return super(AbstractInterpolation, cls).__new__(cls)

    @classmethod
    def register_subclass(cls, short_name: str):
        """
        Allows to register the subclass with its short name

        :param short_name: the subclass to be registered
        :type short_name: string
        """

        def decorator(subclass):
            cls.interpolation_methods_avail[short_name] = subclass
            return subclass

        return decorator

    @abstractmethod
    def desc(self) -> None:
        """
        Describes the disparity interpolation method for the validation step
        :return: None
        """
        print('Disparity interpolation method description for the validation step')

    @abstractmethod
    def interpolated_disparity(self, left: xr.Dataset, img_left: xr.Dataset = None, img_right: xr.Dataset = None,
                               cv: xr.Dataset = None) -> None:
        """
        Interpolation of the left disparity map to resolve occlusion and mismatch conflicts.

        :param left: left Dataset
        :type left: xarray.Dataset with the variables :
            - disparity_map 2D xarray.DataArray (row, col)
            - confidence_measure 3D xarray.DataArray (row, col, indicator)
            - validity_mask 2D xarray.DataArray (row, col)
        :param img_left: left Datset image
        :type img_left:
            xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :param img_right: right Dataset image
        :type img_right:
            xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :param cv: cost_volume Dataset
        :type cv:
            xarray.Dataset with the variables:
                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
        :return: None
        """


@AbstractInterpolation.register_subclass('mc-cnn')
class McCnnInterpolation(AbstractInterpolation):
    """
        McCnnInterpolation class allows to perform the interpolation of the disparity map
        """

    def __init__(self, **cfg: dict) -> None:
        """
        :param cfg: optional configuration, {}
        :type cfg: dictionary
        :return: None
        """
        self.check_config(**cfg)

    def check_config(self, **cfg: dict) -> None:
        """
        Check and update the configuration

        :param cfg: optional configuration, {}
        :type cfg: dictionary
        :return: None
        """
        # No optional configuration

    def desc(self) -> None:
        """
        Describes the disparity interpolation method
        :return: None
        """
        print('MC-CNN interpolation method')

    def interpolated_disparity(self, left: xr.Dataset, img_left: xr.Dataset = None, img_right: xr.Dataset = None,
                               cv: xr.Dataset = None) -> None:
        """
        Interpolation of the left disparity map to resolve occlusion and mismatch conflicts.

        :param left: left Dataset
        :type left: xarray.Dataset with the variables :
            - disparity_map 2D xarray.DataArray (row, col)
            - confidence_measure 3D xarray.DataArray (row, col, indicator)
            - validity_mask 2D xarray.DataArray (row, col)
        :param img_left: left Datset image
        :type img_left:
            xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :param img_right: right Dataset image
        :type img_right:
            xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :param cv: cost_volume Dataset
        :type cv:
            xarray.Dataset with the variables:
                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
        :return: None
        """

        left['disparity_map'].data, left['validity_mask'].data = \
            self.interpolate_occlusion_mc_cnn(left['disparity_map'].data, left['validity_mask'].data)
        left['disparity_map'].data, left['validity_mask'].data = \
            self.interpolate_mismatch_mc_cnn(left['disparity_map'].data, left['validity_mask'].data)

        left.attrs['interpolated_disparity'] = 'mc-cnn'


    @staticmethod
    @njit()
    def interpolate_occlusion_mc_cnn(disp: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolation of the left disparity map to resolve occlusion conflicts. Interpolate occlusion by moving left until
        we find a position labeled correct.

        Žbontar, J., & LeCun, Y. (2016). Stereo matching by training a convolutional neural network to compare image
        patches. The journal of machine learning research, 17(1), 2287-2318.

        :param disp: disparity map
        :type disp: 2D np.array (row, col)
        :param valid: validity mask
        :type valid: 2D np.array (row, col)
        :return: the interpolate left disparity map, with the validity mask update :
            - If out & MSK_PIXEL_FILLED_OCCLUSION != 0 : Invalid pixel : filled occlusion
        :rtype : tuple(2D np.array (row, col), 2D np.array (row, col))
        """
        # Output disparity map and validity mask
        out_disp = np.copy(disp)
        out_val = np.copy(valid)

        ny, nx = disp.shape
        for y in range(ny):
            for x in range(nx):
                # Occlusion
                if (valid[y, x] & PANDORA_MSK_PIXEL_OCCLUSION) != 0:
                    # interpolate occlusion by moving left until we find a position labeled correct

                    #  valid pixels mask
                    msk = (valid[y, 0:x + 1] & PANDORA_MSK_PIXEL_INVALID) == 0
                    # Find the first valid pixel
                    msk = msk[::-1]
                    arg_valid = np.argmax(msk)

                    # If occlusions are still present :  interpolate occlusion by moving right until we find a position
                    # labeled correct
                    if arg_valid == 0:
                        # valid pixels mask
                        msk = (valid[y, x:] & PANDORA_MSK_PIXEL_INVALID) == 0
                        # Find the first valid pixel
                        arg_valid = np.argmax(msk)

                        # Update the validity mask Information : filled occlusion
                        out_val[y, x] -= PANDORA_MSK_PIXEL_OCCLUSION * msk[arg_valid]
                        out_val[y, x] += PANDORA_MSK_PIXEL_FILLED_OCCLUSION * msk[arg_valid]
                        out_disp[y, x] = disp[y, x + arg_valid]
                    else:
                        # Update the validity mask : Information : filled occlusion
                        out_val[y, x] -= PANDORA_MSK_PIXEL_OCCLUSION * msk[arg_valid]
                        out_val[y, x] += PANDORA_MSK_PIXEL_FILLED_OCCLUSION * msk[arg_valid]
                        out_disp[y, x] = disp[y, x - arg_valid]

        return out_disp, out_val

    @staticmethod
    @njit()
    def interpolate_mismatch_mc_cnn(disp: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolation of the left disparity map to resolve mismatch conflicts. Interpolate mismatch by finding the nearest
        correct pixels in 16 different directions and use the median of their disparities.

        Žbontar, J., & LeCun, Y. (2016). Stereo matching by training a convolutional neural network to compare image
        patches. The journal of machine learning research, 17(1), 2287-2318.

        :param disp: disparity map
        :type disp: 2D np.array (row, col)
        :param valid: validity mask
        :type valid: 2D np.array (row, col)
        :return: the interpolate left disparity map, with the validity mask update :
            - If out & MSK_PIXEL_FILLED_MISMATCH != 0 : Invalid pixel : filled mismatch
        :rtype : tuple(2D np.array (row, col), 2D np.array (row, col))
        """
        # Output disparity map and validity mask
        out_disp = np.copy(disp)
        out_val = np.copy(valid)

        ny, nx = disp.shape

        # 16 directions : [x, y]
        dir = np.array([[0., 1.], [-0.5, 1.], [-1., 1.], [-1., 0.5], [-1., 0.], [-1., -0.5], [-1., -1.], [-0.5, -1.],
                        [0., -1.], [0.5, -1.], [1., -1.], [1., -0.5], [1., 0.], [1., 0.5], [1., 1.], [0.5, 1.]])

        # Maximum path length
        max_path_length = max(nx, ny)

        for y in range(ny):
            for x in range(nx):
                # Mismatch
                if (valid[y, x] & PANDORA_MSK_PIXEL_MISMATCH) != 0:
                    interp_mismatched = np.zeros(16, dtype=np.float32)
                    # For each directions
                    for d in range(16):
                        # Find the first valid pixel in the current path
                        for i in range(1, max_path_length):
                            xx = x + int(dir[d][0] * i)
                            yy = y + int(dir[d][1] * i)
                            xx = math.floor(xx)
                            yy = math.floor(yy)

                            # Edge of the image reached: there is no valid pixel in the current path
                            if (yy < 0) | (yy >= ny) | (xx < 0) | (xx >= nx):
                                interp_mismatched[d] = np.nan
                                break

                            # First valid pixel
                            if (valid[yy, xx] & PANDORA_MSK_PIXEL_INVALID) == 0:
                                interp_mismatched[d] = disp[yy, xx]
                                break

                    # Median of the 16 pixels
                    out_disp[y, x] = np.nanmedian(interp_mismatched)
                    # Update the validity mask : Information : filled mismatch
                    out_val[y, x] -= PANDORA_MSK_PIXEL_MISMATCH
                    out_val[y, x] += PANDORA_MSK_PIXEL_FILLED_MISMATCH

        return out_disp, out_val


@AbstractInterpolation.register_subclass('sgm')
class SgmInterpolation(AbstractInterpolation):
    """
        SgmInterpolation class allows to perform the interpolation of the disparity map
        """

    def __init__(self, **cfg: dict) -> None:
        """
        :param cfg: optional configuration, {}
        :type cfg: dictionary
        :return: None
        """
        self.check_config(**cfg)

    def check_config(self, **cfg: dict) -> None:
        """
        Check and update the configuration

        :param cfg: optional configuration, {}
        :type cfg: dictionary
        :return: None
        """
        # No optional configuration

    def desc(self) -> None:
        """
        Describes the disparity interpolation method
        :return: None
        """
        print('SGM interpolation method')

    def interpolated_disparity(self, left: xr.Dataset, img_left: xr.Dataset = None, img_right: xr.Dataset = None,
                               cv: xr.Dataset = None) -> None:
        """
        Interpolation of the left disparity map to resolve occlusion and mismatch conflicts.

        :param left: left Dataset
        :type left: xarray.Dataset with the variables :
            - disparity_map 2D xarray.DataArray (row, col)
            - confidence_measure 3D xarray.DataArray (row, col, indicator)
            - validity_mask 2D xarray.DataArray (row, col)
        :param img_left: left Datset image
        :type img_left:
            xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :param img_right: right Dataset image
        :type img_right:
            xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :param cv: cost_volume Dataset
        :type cv:
            xarray.Dataset with the variables:
                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
        :return: None
        """

        left['disparity_map'].data, left['validity_mask'].data = self.interpolate_mismatch_sgm(left['disparity_map'].data,
                                                                                             left['validity_mask'].data)
        left['disparity_map'].data, left['validity_mask'].data = self.interpolate_occlusion_sgm(left['disparity_map'].data,
                                                                                              left['validity_mask'].data)
        left.attrs['interpolated_disparity'] = 'sgm'


    @staticmethod
    @njit()
    def interpolate_occlusion_sgm(disp: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolation of the left disparity map to resolve occlusion conflicts. Interpolate occlusion by moving by selecting
        the right lowest value along paths from 8 directions.

        HIRSCHMULLER, Heiko. Stereo processing by semiglobal matching and mutual information.
        IEEE Transactions on pattern analysis and machine intelligence, 2007, vol. 30, no 2, p. 328-341.

        :param disp: disparity map
        :type disp: 2D np.array (row, col)
        :param valid: validity mask
        :type valid: 2D np.array (row, col)
        :return: the interpolate left disparity map, with the validity mask update :
            - If out & MSK_PIXEL_FILLED_OCCLUSION != 0 : Invalid pixel : filled occlusion
        :rtype : tuple(2D np.array (row, col), 2D np.array (row, col))
        """
        # Output disparity map and validity mask
        out_disp = np.copy(disp)
        out_val = np.copy(valid)

        ny, nx = disp.shape

        # 8 directions : [x, y]
        dir = np.array([[0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1]])

        # Maximum path length
        max_path_length = max(nx, ny)

        ny, nx = disp.shape
        for y in range(ny):
            for x in range(nx):
                # Occlusion
                if (valid[y, x] & PANDORA_MSK_PIXEL_OCCLUSION) != 0:
                    valid_neighbors = np.zeros(8, dtype=np.float32)
                    # For each directions
                    for d in range(8):
                        # Find the first valid pixel in the current path
                        xx = x
                        yy = y
                        for i in range(max_path_length):
                            xx += dir[d][0]
                            yy += dir[d][1]

                            # Edge of the image reached: there is no valid pixel in the current path
                            if (yy < 0) | (yy >= ny) | (xx < 0) | (xx >= nx):
                                valid_neighbors[d] = np.nan
                                break

                            # First valid pixel
                            if (valid[yy, xx] & PANDORA_MSK_PIXEL_INVALID) == 0:
                                valid_neighbors[d] = disp[yy, xx]
                                break

                    # Returns the indices that would sort the absolute array
                    # The absolute value is used to search for the right value closest to 0
                    valid_neighbors_argsort = np.argsort(np.abs(valid_neighbors))

                    # right lowest value
                    out_disp[y, x] = valid_neighbors[valid_neighbors_argsort[1]]

                    # Update the validity mask : Information : filled occlusion
                    out_val[y, x] -= PANDORA_MSK_PIXEL_OCCLUSION
                    out_val[y, x] += PANDORA_MSK_PIXEL_FILLED_OCCLUSION

        return out_disp, out_val

    @staticmethod
    @njit()
    def interpolate_mismatch_sgm(disp: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolation of the left disparity map to resolve mismatch conflicts. Interpolate mismatch by finding the nearest
        correct pixels in 8 different directions and use the median of their disparities.
        Mismatched pixel areas that are direct neighbors of occluded pixels are treated as occlusions.

        HIRSCHMULLER, Heiko. Stereo processing by semiglobal matching and mutual information.
        IEEE Transactions on pattern analysis and machine intelligence, 2007, vol. 30, no 2, p. 328-341.

        :param disp: disparity map
        :type disp: 2D np.array (row, col)
        :param valid: validity mask
        :type valid: 2D np.array (row, col)
        :return: the interpolate left disparity map, with the validity mask update :
            - If out & MSK_PIXEL_FILLED_MISMATCH != 0 : Invalid pixel : filled mismatch
        :rtype : tuple(2D np.array (row, col), 2D np.array (row, col))
        """
        # Output disparity map and validity mask
        out_disp = np.copy(disp)
        out_val = np.copy(valid)

        ny, nx = disp.shape

        # 8 directions : [x, y]
        dir = np.array([[0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1]])

        # Maximum path length
        max_path_length = max(nx, ny)

        ny, nx = disp.shape
        for y in range(ny):
            for x in range(nx):
                # Mismatched
                if valid[y, x] & PANDORA_MSK_PIXEL_MISMATCH != 0:

                    # Mismatched pixel areas that are direct neighbors of occluded pixels are treated as occlusions
                    if np.sum(valid[max(0, y - 1):min(ny - 1, y + 1) + 1, max(0, x - 1):min(nx - 1, x + 1) + 1] &
                              PANDORA_MSK_PIXEL_OCCLUSION) != 0:
                        out_val[y, x] -= PANDORA_MSK_PIXEL_MISMATCH
                        out_val[y, x] += PANDORA_MSK_PIXEL_OCCLUSION

                    else:
                        # For each directions
                        valid_neighbors = np.zeros(8, dtype=np.float32)
                        for d in range(8):
                            # Find the first valid pixel in the current path
                            xx = x
                            yy = y
                            for i in range(max_path_length):
                                xx += dir[d][0]
                                yy += dir[d][1]

                                # Edge of the image reached: there is no valid pixel in the current path
                                if (yy < 0) | (yy >= ny) | (xx < 0) | (xx >= nx):
                                    valid_neighbors[d] = np.nan
                                    break

                                # First valid pixel
                                if (valid[yy, xx] & PANDORA_MSK_PIXEL_INVALID) == 0:
                                    valid_neighbors[d] = disp[yy, xx]
                                    break

                        # Median of the 8 pixels
                        out_disp[y, x] = np.nanmedian(valid_neighbors)
                        # Update the validity mask : Information : filled mismatch
                        out_val[y, x] -= PANDORA_MSK_PIXEL_MISMATCH
                        out_val[y, x] += PANDORA_MSK_PIXEL_FILLED_MISMATCH

        return out_disp, out_val
