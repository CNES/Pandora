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
This module contains classes and functions associated to the interpolation of the disparity map for the validation step.
"""

import logging
from abc import ABCMeta, abstractmethod
from typing import Dict

import numpy as np
import xarray as xr

import pandora.constants as cst
from pandora.criteria import mask_border

from .cpp import validation_cpp


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

                - im: 2D (row, col) or 3D (band_im, row, col) xarray.DataArray float32
                - disparity (optional): 3D (disp, row, col) xarray.DataArray float32
                - msk (optional): 2D (row, col) xarray.DataArray int16
                - classif (optional): 3D (band_classif, row, col) xarray.DataArray int16
                - segm (optional): 2D (row, col) xarray.DataArray int16
        :type img_left: xarray.Dataset
        :param img_right: right Dataset image containing :

                - im: 2D (row, col) or 3D (band_im, row, col) xarray.DataArray float32
                - disparity (optional): 3D (disp, row, col) xarray.DataArray float32
                - msk (optional): 2D (row, col) xarray.DataArray int16
                - classif (optional): 3D (band_classif, row, col) xarray.DataArray int16
                - segm (optional): 2D (row, col) xarray.DataArray int16
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

                - im: 2D (row, col) or 3D (band_im, row, col) xarray.DataArray float32
                - disparity (optional): 3D (disp, row, col) xarray.DataArray float32
                - msk (optional): 2D (row, col) xarray.DataArray int16
                - classif (optional): 3D (band_classif, row, col) xarray.DataArray int16
                - segm (optional): 2D (row, col) xarray.DataArray int16
        :type img_left: xarray.Dataset
        :param img_right: right Dataset image containing :

                - im: 2D (row, col) or 3D (band_im, row, col) xarray.DataArray float32
                - disparity (optional): 3D (disp, row, col) xarray.DataArray float32
                - msk (optional): 2D (row, col) xarray.DataArray int16
                - classif (optional): 3D (band_classif, row, col) xarray.DataArray int16
                - segm (optional): 2D (row, col) xarray.DataArray int16
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

        # Update validity mask to make sure that PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER criteria is marked
        if left.attrs["offset_row_col"] > 0:
            left["validity_mask"] = mask_border(left)

    @staticmethod
    def interpolate_occlusion_mc_cnn(disp: np.ndarray, valid: np.ndarray):

        return validation_cpp.interpolate_occlusion_mc_cnn(
            disp,
            valid,
            cst.PANDORA_MSK_PIXEL_OCCLUSION,
            cst.PANDORA_MSK_PIXEL_FILLED_OCCLUSION,
            cst.PANDORA_MSK_PIXEL_INVALID,
        )

    @staticmethod
    def interpolate_mismatch_mc_cnn(disp: np.ndarray, valid: np.ndarray):

        return validation_cpp.interpolate_mismatch_mc_cnn(
            disp,
            valid,
            cst.PANDORA_MSK_PIXEL_MISMATCH,
            cst.PANDORA_MSK_PIXEL_FILLED_MISMATCH,
            cst.PANDORA_MSK_PIXEL_INVALID,
        )


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

                - im: 2D (row, col) or 3D (band_im, row, col) xarray.DataArray float32
                - disparity (optional): 3D (disp, row, col) xarray.DataArray float32
                - msk (optional): 2D (row, col) xarray.DataArray int16
                - classif (optional): 3D (band_classif, row, col) xarray.DataArray int16
                - segm (optional): 2D (row, col) xarray.DataArray int16
        :type img_left: xarray.Dataset
        :param img_right: right Dataset image containing :

                - im: 2D (row, col) or 3D (band_im, row, col) xarray.DataArray float32
                - disparity (optional): 3D (disp, row, col) xarray.DataArray float32
                - msk (optional): 2D (row, col) xarray.DataArray int16
                - classif (optional): 3D (band_classif, row, col) xarray.DataArray int16
                - segm (optional): 2D (row, col) xarray.DataArray int16
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
    def interpolate_occlusion_sgm(disp: np.ndarray, valid: np.ndarray):

        return validation_cpp.interpolate_occlusion_sgm(
            disp,
            valid,
            cst.PANDORA_MSK_PIXEL_OCCLUSION,
            cst.PANDORA_MSK_PIXEL_FILLED_OCCLUSION,
            cst.PANDORA_MSK_PIXEL_INVALID,
        )

    @staticmethod
    def interpolate_mismatch_sgm(disp: np.ndarray, valid: np.ndarray):

        return validation_cpp.interpolate_mismatch_sgm(
            disp,
            valid,
            cst.PANDORA_MSK_PIXEL_MISMATCH,
            cst.PANDORA_MSK_PIXEL_FILLED_MISMATCH,
            cst.PANDORA_MSK_PIXEL_OCCLUSION,
            cst.PANDORA_MSK_PIXEL_INVALID,
        )
