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
This module contains classes and functions associated to the validation step.
"""

import logging
from abc import ABCMeta, abstractmethod
from typing import Dict, Union

import numpy as np
import xarray as xr
from json_checker import Checker, And, Or, OptionalKey

import pandora.constants as cst
from pandora import common
from pandora.cost_volume_confidence.cost_volume_confidence import (
    AbstractCostVolumeConfidence,
)


class AbstractValidation:
    """
    Abstract Validation class
    """

    __metaclass__ = ABCMeta

    validation_methods_avail: Dict = {}
    cfg = None

    def __new__(cls, **cfg: dict):
        """
        Return the plugin associated with the validation_method given in the configuration

        :param cfg: configuration {'validation_method': value}
        :type cfg: dictionary
        """
        if cls is AbstractValidation:
            if isinstance(cfg["validation_method"], str):
                try:
                    return super(AbstractValidation, cls).__new__(
                        cls.validation_methods_avail[cfg["validation_method"]]
                    )
                except KeyError:
                    logging.error(
                        "No validation method named % supported",
                        cfg["validation_method"],
                    )
                    raise KeyError
            else:
                if isinstance(cfg["validation_method"], unicode):  # type: ignore # pylint: disable=undefined-variable
                    # creating a plugin from registered short name given as unicode (py2 & 3 compatibility)
                    try:
                        return super(AbstractValidation, cls).__new__(
                            cls.validation_methods_avail[cfg["validation_method"].encode("utf-8")]
                        )
                    except KeyError:
                        logging.error(
                            "No validation method named % supported",
                            cfg["validation_method"],
                        )
                        raise KeyError
        else:
            return super(AbstractValidation, cls).__new__(cls)
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
            cls.validation_methods_avail[short_name] = subclass
            return subclass

        return decorator

    @abstractmethod
    def desc(self) -> None:
        """
        Describes the validation method
        :return: None
        """
        print("Validation method description")

    @abstractmethod
    def disparity_checking(
        self,
        dataset_left: xr.Dataset,
        dataset_right: xr.Dataset,
        img_left: xr.Dataset = None,
        img_right: xr.Dataset = None,
        cv: xr.Dataset = None,
    ) -> xr.Dataset:
        """
        Determination of occlusions and false matches by performing a consistency check on valid pixels.
        Update the validity_mask :

            - If out & MSK_PIXEL_OCCLUSION != 0 : Invalid pixel : occluded pixel
            - If out & MSK_PIXEL_MISMATCH  != 0  : Invalid pixel : mismatched pixel

        | Update the measure map: add the disp RL / disp LR distances

        :param dataset_left: left Dataset with the variables :

            - disparity_map 2D xarray.DataArray (row, col)
            - confidence_measure 3D xarray.DataArray (row, col, indicator)
            - validity_mask 2D xarray.DataArray (row, col)
        :type dataset_left: xarray.Dataset
        :param dataset_right: right Dataset with the variables :

            - disparity_map 2D xarray.DataArray (row, col)
            - confidence_measure 3D xarray.DataArray (row, col, indicator)
            - validity_mask 2D xarray.DataArray (row, col)
        :type dataset_right: xarray.Dataset
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
        :return: the left dataset with the variables :

            - disparity_map 2D xarray.DataArray (row, col)
            - confidence_measure 3D xarray.DataArray (row, col, indicator)
            - validity_mask 2D xarray.DataArray (row, col) with the bit 8 and 9 of the validity_mask :
                - If out & MSK_PIXEL_OCCLUSION != 0 : Invalid pixel : occluded pixel
                - If out & MSK_PIXEL_MISMATCH  != 0  : Invalid pixel : mismatched pixel
        :rtype: xarray.Dataset
        """


@AbstractValidation.register_subclass("cross_checking")
class CrossChecking(AbstractValidation):
    """
    CrossChecking class allows to perform the validation step
    """

    # Default configuration, do not change this value
    _THRESHOLD = 1.0

    def __init__(self, **cfg) -> None:
        """
        :param cfg: optional configuration, {'cross_checking_threshold': value,
                                            'interpolated_disparity': value, 'filter_interpolated_disparities': value}
        :type cfg: dictionary
        :return: None
        """
        self.cfg = self.check_conf(**cfg)
        self._threshold = self.cfg["cross_checking_threshold"]

    def check_conf(self, **cfg: Union[str, int, float, bool]) -> Dict[str, Union[str, int, float, bool]]:
        """
        Add default values to the dictionary if there are missing elements and check if the dictionary is correct

        :param cfg: optimization configuration
        :type cfg: dict
        :return: optimization configuration updated
        :rtype: dict
        """
        # Give the default value if the required element is not in the configuration
        if "cross_checking_threshold" not in cfg:
            cfg["cross_checking_threshold"] = self._THRESHOLD

        schema = {
            "validation_method": And(str, lambda input: "cross_checking"),
            "cross_checking_threshold": Or(int, float),
            OptionalKey("interpolated_disparity"): And(str, lambda input: common.is_method(input, ["mc-cnn", "sgm"])),
        }

        checker = Checker(schema)
        checker.validate(cfg)
        return cfg

    def desc(self) -> None:
        """
        Describes the validation method
        :return: None
        """
        print("Cross-checking method")

    def disparity_checking(
        self,
        dataset_left: xr.Dataset,
        dataset_right: xr.Dataset,
        img_left: xr.Dataset = None,
        img_right: xr.Dataset = None,
        cv: xr.Dataset = None,
    ) -> xr.Dataset:
        """
        Determination of occlusions and false matches by performing a consistency check on valid pixels.

        Update the validity_mask :

            - If out & MSK_PIXEL_OCCLUSION != 0 : Invalid pixel : occluded pixel
            - If out & MSK_PIXEL_MISMATCH  != 0  : Invalid pixel : mismatched pixel

        | Update the measure map: add the disp RL / disp LR distances

        :param dataset_left: left Dataset with the variables :

            - disparity_map 2D xarray.DataArray (row, col)
            - validity_mask 2D xarray.DataArray (row, col)
        :type dataset_left: xarray.Dataset
        :param dataset_right: right Dataset with the variables :

            - disparity_map 2D xarray.DataArray (row, col)
            - validity_mask 2D xarray.DataArray (row, col)
        :type dataset_right: xarray.Dataset
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
        :return: the left dataset with the variables :

            - disparity_map 2D xarray.DataArray (row, col)
            - confidence_measure 3D xarray.DataArray (row, col, indicator)
            - validity_mask 2D xarray.DataArray (row, col) with the bit 8 and 9 of the validity_mask :

                - If out & MSK_PIXEL_OCCLUSION != 0 : Invalid pixel : occluded pixel
                - If out & MSK_PIXEL_MISMATCH  != 0  : Invalid pixel : mismatched pixel
        :rtype: xarray.Dataset
        """
        nb_row, nb_col = dataset_left["disparity_map"].shape
        disparity_range = np.arange(dataset_left.attrs["disp_min"], dataset_left.attrs["disp_max"] + 1)

        # Confidence measure which calculates the distance LR / RL
        conf_measure = np.full((nb_row, nb_col), np.nan, dtype=np.float32)

        for row in range(0, nb_row):
            # Exclude invalid pixel :
            valid_pixel = np.where((dataset_left["validity_mask"].data[row, :] & cst.PANDORA_MSK_PIXEL_INVALID) == 0)

            col_left = np.arange(nb_col, dtype=np.int)
            col_left = col_left[valid_pixel]

            col_right = col_left + dataset_left["disparity_map"].data[row, col_left]
            # Round elements of the array to the nearest integer
            col_right = np.rint(col_right).astype(int)

            # Left-Right consistency, for pixel i :
            # If | Disp_right(i + rint(Disp_left(i)) + Disp_left(i) | > self._threshold :
            # i is invalid, mismatched or occlusion detected
            # If | Disp_right(i + rint(Disp_left(i)) + Disp_left(i) | <= self._threshold : i is valid

            # Apply cross checking on pixels i + round(Disp_left(i) inside the right image
            inside_right = np.where((col_right >= 0) & (col_right < nb_col))

            # Conversion from nan to inf
            right_disp = dataset_right["disparity_map"].data[row, col_right[inside_right]]
            right_disp[np.isnan(right_disp)] = np.inf
            left_disp = dataset_left["disparity_map"].data[row, col_left[inside_right]]
            left_disp[np.isnan(left_disp)] = np.inf

            # Allocate to the measure map, the distance disp LR / disp RL indicator
            conf_measure[row, col_left[inside_right]] = np.abs(right_disp + left_disp)

            # left image pixels invalidated by the cross checking
            invalid = np.abs(right_disp + left_disp) > self._threshold

            # Detect mismatched and occlusion :
            # For a left image pixel i invalidated by the cross checking :
            # mismatch if : Disp_right(i + d) = -d, for any other d
            # occlusion otherwise

            # Index : i + d, for any other d. 2D np array (nb invalid pixels, nb disparity )
            index = (
                np.tile(disparity_range, (len(col_left[inside_right][invalid]), 1)).astype(np.float32)
                + np.tile(col_left[inside_right][invalid], (len(disparity_range), 1)).transpose()
            )

            inside_col_disp = np.where((index >= 0) & (index < nb_col))

            # disp_right : Disp_right(i + d)
            disp_right = np.full(index.shape, np.inf, dtype=np.float32)
            disp_right[inside_col_disp] = dataset_right["disparity_map"].data[row, index[inside_col_disp].astype(int)]

            # Check if rint(Disp_right(i + d)) == -d
            comp = np.rint(disp_right) == np.tile(
                -1 * disparity_range, (len(col_left[inside_right][invalid]), 1)
            ).astype(np.float32)
            comp = np.sum(comp, axis=1)
            comp[comp > 1] = 1

            dataset_left["validity_mask"].data[row, col_left[inside_right][invalid]] += cst.PANDORA_MSK_PIXEL_OCCLUSION
            dataset_left["validity_mask"].data[row, col_left[inside_right][invalid]] += (
                cst.PANDORA_MSK_PIXEL_MISMATCH * comp
            ).astype(np.uint16)
            dataset_left["validity_mask"].data[row, col_left[inside_right][invalid]] -= (
                cst.PANDORA_MSK_PIXEL_OCCLUSION * comp
            ).astype(np.uint16)

            # Pixels i + round(Disp_left(i) outside the right image are occlusions
            outside_right = np.where((col_right < 0) & (col_right >= nb_col))
            dataset_left["validity_mask"].data[row, col_left[outside_right]] += cst.PANDORA_MSK_PIXEL_OCCLUSION

        dataset_left.attrs["validation"] = "cross_checking"

        dataset_left, _ = AbstractCostVolumeConfidence.allocate_confidence_map(
            "validation_pandora_distanceOfDisp", conf_measure, dataset_left, cv
        )

        return dataset_left
