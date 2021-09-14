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
This module contains classes and functions associated to the interpolation of the disparity map for the validation step.
"""

import logging
import math
from abc import ABCMeta, abstractmethod
from typing import Tuple, Dict

import numpy as np
import xarray as xr
from numba import njit

import pandora.constants as cst
from pandora.img_tools import find_valid_neighbors


class AbstractInterpolation:
    """
    Abstract Interpolation class
    """

    __metaclass__ = ABCMeta

    interpolation_methods_avail: Dict = {}

    def __new__(cls, **cfg: dict):
        """
        Return the plugin associated with the interpolated_disparity given in the configuration

        :param cfg: configuration {'interpolated_disparity': value}
        :type cfg: dictionary
        """
        if cls is AbstractInterpolation:
            if isinstance(cfg["interpolated_disparity"], str):
                try:
                    return super(AbstractInterpolation, cls).__new__(
                        cls.interpolation_methods_avail[cfg["interpolated_disparity"]]
                    )
                except KeyError:
                    logging.error(
                        "No interpolation method named % supported",
                        cfg["interpolated_disparity"],
                    )
                    raise KeyError
            else:
                if isinstance(
                    cfg["interpolated_disparity"], unicode  # type: ignore # pylint: disable=undefined-variable
                ):
                    # creating a plugin from registered short name given as unicode (py2 & 3 compatibility)
                    try:
                        return super(AbstractInterpolation, cls).__new__(
                            cls.interpolation_methods_avail[cfg["interpolated_disparity"].encode("utf-8")]
                        )
                    except KeyError:
                        logging.error(
                            "No interpolation method named % supported",
                            cfg["interpolated_disparity"],
                        )
                        raise KeyError
        else:
            return super(AbstractInterpolation, cls).__new__(cls)
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
            cls.interpolation_methods_avail[short_name] = subclass
            return subclass

        return decorator

    @abstractmethod
    def desc(self) -> None:
        """
        Describes the disparity interpolation method for the validation step
        :return: None
        """
        print("Disparity interpolation method description for the validation step")

    @abstractmethod
    def interpolated_disparity(
        self,
        left: xr.Dataset,
        img_left: xr.Dataset = None,
        img_right: xr.Dataset = None,
        cv: xr.Dataset = None,
    ) -> None:
        """
        Interpolation of the left disparity map to resolve occlusion and mismatch conflicts.

        :param left: left Dataset with the variables :

            - disparity_map 2D xarray.DataArray (row, col)
            - confidence_measure 3D xarray.DataArray (row, col, indicator)
            - validity_mask 2D xarray.DataArray (row, col)
        :type left: xarray.Dataset
        :param img_left: left Datset image containing :

                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :type img_left: xarray.Dataset
        :param img_right: right Dataset image containing :

                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :type img_right: xarray.Dataset
        :param cv: cost_volume Dataset with the variables:

                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
        :type cv: xarray.Dataset
        :return: None
        """


@AbstractInterpolation.register_subclass("mc-cnn")
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
        print("MC-CNN interpolation method")

    def interpolated_disparity(
        self,
        left: xr.Dataset,
        img_left: xr.Dataset = None,
        img_right: xr.Dataset = None,
        cv: xr.Dataset = None,
    ) -> None:
        """
        Interpolation of the left disparity map to resolve occlusion and mismatch conflicts.

        :param left: left Dataset with the variables :

            - disparity_map 2D xarray.DataArray (row, col)
            - confidence_measure 3D xarray.DataArray (row, col, indicator)
            - validity_mask 2D xarray.DataArray (row, col)
        :type left: xarray.Dataset
        :param img_left: left Datset image containing :

                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :type img_left: xarray.Dataset
        :param img_right: right Dataset image containing :

                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :type img_right: xarray.Dataset
        :param cv: cost_volume Dataset with the variables:

                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
        :type cv: xarray.Dataset
        :return: None
        """

        (
            left["disparity_map"].data,
            left["validity_mask"].data,
        ) = self.interpolate_occlusion_mc_cnn(left["disparity_map"].data, left["validity_mask"].data)
        (
            left["disparity_map"].data,
            left["validity_mask"].data,
        ) = self.interpolate_mismatch_mc_cnn(left["disparity_map"].data, left["validity_mask"].data)

        left.attrs["interpolated_disparity"] = "mc-cnn"

    @staticmethod
    @njit()
    def interpolate_occlusion_mc_cnn(disp: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolation of the left disparity map to resolve occlusion conflicts.
        Interpolate occlusion by moving left until
        we find a position labeled correct.

        Žbontar, J., & LeCun, Y. (2016). Stereo matching by training a convolutional neural network to compare image
        patches. The journal of machine learning research, 17(1), 2287-2318.

        :param disp: disparity map
        :type disp: 2D np.array (row, col)
        :param valid: validity mask
        :type valid: 2D np.array (row, col)
        :return: the interpolate left disparity map, with the validity mask update :

            - If out & MSK_PIXEL_FILLED_OCCLUSION != 0 : Invalid pixel : filled occlusion
        :rtype: tuple(2D np.array (row, col), 2D np.array (row, col))
        """
        # Output disparity map and validity mask
        out_disp = np.copy(disp)
        out_val = np.copy(valid)

        ncol, nrow = disp.shape
        for col in range(ncol):
            for row in range(nrow):
                # Occlusion
                if (valid[col, row] & cst.PANDORA_MSK_PIXEL_OCCLUSION) != 0:
                    # interpolate occlusion by moving left until we find a position labeled correct

                    #  valid pixels mask
                    msk = (valid[col, 0 : row + 1] & cst.PANDORA_MSK_PIXEL_INVALID) == 0
                    # Find the first valid pixel
                    msk = msk[::-1]
                    arg_valid = np.argmax(msk)

                    # If occlusions are still present :  interpolate occlusion by moving right until we find a position
                    # labeled correct
                    if arg_valid == 0:
                        # valid pixels mask
                        msk = (valid[col, row:] & cst.PANDORA_MSK_PIXEL_INVALID) == 0
                        # Find the first valid pixel
                        arg_valid = np.argmax(msk)

                        # Update the validity mask Information : filled occlusion
                        out_val[col, row] -= cst.PANDORA_MSK_PIXEL_OCCLUSION * msk[arg_valid]
                        out_val[col, row] += cst.PANDORA_MSK_PIXEL_FILLED_OCCLUSION * msk[arg_valid]
                        out_disp[col, row] = disp[col, row + arg_valid]
                    else:
                        # Update the validity mask : Information : filled occlusion
                        out_val[col, row] -= cst.PANDORA_MSK_PIXEL_OCCLUSION * msk[arg_valid]
                        out_val[col, row] += cst.PANDORA_MSK_PIXEL_FILLED_OCCLUSION * msk[arg_valid]
                        out_disp[col, row] = disp[col, row - arg_valid]

        return out_disp, out_val

    @staticmethod
    @njit()
    def interpolate_mismatch_mc_cnn(disp: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolation of the left disparity map to resolve mismatch conflicts.
        Interpolate mismatch by finding the nearest
        correct pixels in 16 different directions and use the median of their disparities.

        Žbontar, J., & LeCun, Y. (2016). Stereo matching by training a convolutional neural network to compare image
        patches. The journal of machine learning research, 17(1), 2287-2318.

        :param disp: disparity map
        :type disp: 2D np.array (row, col)
        :param valid: validity mask
        :type valid: 2D np.array (row, col)
        :return: the interpolate left disparity map, with the validity mask update :

            - If out & MSK_PIXEL_FILLED_MISMATCH != 0 : Invalid pixel : filled mismatch
        :rtype: tuple(2D np.array (row, col), 2D np.array (row, col))
        """
        # Output disparity map and validity mask
        out_disp = np.copy(disp)
        out_val = np.copy(valid)

        ncol, nrow = disp.shape

        # 16 directions : [row, col]
        dirs = np.array(
            [
                [0.0, 1.0],
                [-0.5, 1.0],
                [-1.0, 1.0],
                [-1.0, 0.5],
                [-1.0, 0.0],
                [-1.0, -0.5],
                [-1.0, -1.0],
                [-0.5, -1.0],
                [0.0, -1.0],
                [0.5, -1.0],
                [1.0, -1.0],
                [1.0, -0.5],
                [1.0, 0.0],
                [1.0, 0.5],
                [1.0, 1.0],
                [0.5, 1.0],
            ]
        )

        # Maximum path length
        max_path_length = max(nrow, ncol)

        for col in range(ncol):
            for row in range(nrow):
                # Mismatch
                if (valid[col, row] & cst.PANDORA_MSK_PIXEL_MISMATCH) != 0:
                    interp_mismatched = np.zeros(16, dtype=np.float32)
                    # For each directions
                    for direction in range(16):
                        # Find the first valid pixel in the current path
                        for i in range(1, max_path_length):
                            tmp_row = row + int(dirs[direction][0] * i)
                            tmp_col = col + int(dirs[direction][1] * i)
                            tmp_row = math.floor(tmp_row)
                            tmp_col = math.floor(tmp_col)

                            # Edge of the image reached: there is no valid pixel in the current path
                            if (tmp_col < 0) | (tmp_col >= ncol) | (tmp_row < 0) | (tmp_row >= nrow):
                                interp_mismatched[direction] = np.nan
                                break

                            # First valid pixel
                            if (valid[tmp_col, tmp_row] & cst.PANDORA_MSK_PIXEL_INVALID) == 0:
                                interp_mismatched[direction] = disp[tmp_col, tmp_row]
                                break

                    # Median of the 16 pixels
                    out_disp[col, row] = np.nanmedian(interp_mismatched)
                    # Update the validity mask : Information : filled mismatch
                    out_val[col, row] -= cst.PANDORA_MSK_PIXEL_MISMATCH
                    out_val[col, row] += cst.PANDORA_MSK_PIXEL_FILLED_MISMATCH

        return out_disp, out_val


@AbstractInterpolation.register_subclass("sgm")
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
        print("SGM interpolation method")

    def interpolated_disparity(
        self,
        left: xr.Dataset,
        img_left: xr.Dataset = None,
        img_right: xr.Dataset = None,
        cv: xr.Dataset = None,
    ) -> None:
        """
        Interpolation of the left disparity map to resolve occlusion and mismatch conflicts.

        :param left: left Dataset with the variables :

            - disparity_map 2D xarray.DataArray (row, col)
            - confidence_measure 3D xarray.DataArray (row, col, indicator)
            - validity_mask 2D xarray.DataArray (row, col)
        :type left: xarray.Dataset
        :param img_left: left Datset image containing :

                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :type img_left: xarray.Dataset
        :param img_right: right Dataset image containing :

                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :type img_right: xarray.Dataset
        :param cv: cost_volume Dataset with the variables:

                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
        :type cv: xarray.Dataset
        :return: None
        """

        (
            left["disparity_map"].data,
            left["validity_mask"].data,
        ) = self.interpolate_mismatch_sgm(left["disparity_map"].data, left["validity_mask"].data)
        (
            left["disparity_map"].data,
            left["validity_mask"].data,
        ) = self.interpolate_occlusion_sgm(left["disparity_map"].data, left["validity_mask"].data)
        left.attrs["interpolated_disparity"] = "sgm"

    @staticmethod
    @njit()
    def interpolate_occlusion_sgm(disp: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolation of the left disparity map to resolve occlusion conflicts.
        Interpolate occlusion by moving by selecting
        the right lowest value along paths from 8 directions.

        HIRSCHMULLER, Heiko. Stereo processing by semiglobal matching and mutual information.
        IEEE Transactions on pattern analysis and machine intelligence, 2007, vol. 30, no 2, p. 328-341.

        :param disp: disparity map
        :type disp: 2D np.array (row, col)
        :param valid: validity mask
        :type valid: 2D np.array (row, col)
        :return: the interpolate left disparity map, with the validity mask update :

            - If out & MSK_PIXEL_FILLED_OCCLUSION != 0 : Invalid pixel : filled occlusion
        :rtype: : tuple(2D np.array (row, col), 2D np.array (row, col))
        """
        # Output disparity map and validity mask
        out_disp = np.copy(disp)
        out_val = np.copy(valid)

        # 8 directions : [row, col]
        dirs = np.array([[0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1]])

        ncol, nrow = disp.shape
        for col in range(ncol):
            for row in range(nrow):
                # Occlusion
                if (valid[col, row] & cst.PANDORA_MSK_PIXEL_OCCLUSION) != 0:
                    valid_neighbors = find_valid_neighbors(dirs, disp, valid, row, col)

                    # Returns the indices that would sort the absolute array
                    # The absolute value is used to search for the right value closest to 0
                    valid_neighbors_argsort = np.argsort(np.abs(valid_neighbors))

                    # right lowest value
                    out_disp[col, row] = valid_neighbors[valid_neighbors_argsort[1]]

                    # Update the validity mask : Information : filled occlusion
                    out_val[col, row] -= cst.PANDORA_MSK_PIXEL_OCCLUSION
                    out_val[col, row] += cst.PANDORA_MSK_PIXEL_FILLED_OCCLUSION
        return out_disp, out_val

    @staticmethod
    @njit()
    def interpolate_mismatch_sgm(disp: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolation of the left disparity map to resolve mismatch conflicts. Interpolate mismatch by finding the
        nearest correct pixels in 8 different directions and use the median of their disparities.
        Mismatched pixel areas that are direct neighbors of occluded pixels are treated as occlusions.

        HIRSCHMULLER, Heiko. Stereo processing by semiglobal matching and mutual information.
        IEEE Transactions on pattern analysis and machine intelligence, 2007, vol. 30, no 2, p. 328-341.

        :param disp: disparity map
        :type disp: 2D np.array (row, col)
        :param valid: validity mask
        :type valid: 2D np.array (row, col)
        :return: the interpolate left disparity map, with the validity mask update :

            - If out & MSK_PIXEL_FILLED_MISMATCH != 0 : Invalid pixel : filled mismatch
        :rtype: tuple(2D np.array (row, col), 2D np.array (row, col))
        """
        # Output disparity map and validity mask
        out_disp = np.copy(disp)
        out_val = np.copy(valid)

        # 8 directions : [row, col]
        dirs = np.array([[0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1]])

        ncol, nrow = disp.shape
        for col in range(ncol):
            for row in range(nrow):
                # Mismatched
                if valid[col, row] & cst.PANDORA_MSK_PIXEL_MISMATCH != 0:

                    # Mismatched pixel areas that are direct neighbors of occluded pixels are treated as occlusions
                    if (
                        np.sum(
                            valid[
                                max(0, col - 1) : min(ncol - 1, col + 1) + 1,
                                max(0, row - 1) : min(nrow - 1, row + 1) + 1,
                            ]
                            & cst.PANDORA_MSK_PIXEL_OCCLUSION
                        )
                        != 0
                    ):
                        out_val[col, row] -= cst.PANDORA_MSK_PIXEL_MISMATCH
                        out_val[col, row] += cst.PANDORA_MSK_PIXEL_OCCLUSION

                    else:
                        valid_neighbors = find_valid_neighbors(dirs, disp, valid, row, col)

                        # Median of the 8 pixels
                        out_disp[col, row] = np.nanmedian(valid_neighbors)
                        # Update the validity mask : Information : filled mismatch
                        out_val[col, row] -= cst.PANDORA_MSK_PIXEL_MISMATCH
                        out_val[col, row] += cst.PANDORA_MSK_PIXEL_FILLED_MISMATCH

        return out_disp, out_val
