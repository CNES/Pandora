#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
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
This module contains classes and functions to estimate confidence.
"""

from abc import ABCMeta, abstractmethod
from typing import Tuple

import xarray as xr
import numpy as np


class AbstractCostVolumeConfidence:
    """
    Abstract Cost Volume Confidence class
    """

    __metaclass__ = ABCMeta

    confidence_methods_avail = {}
    cfg = None
    # Indicator
    _indicator = ""

    def __new__(cls, **cfg: dict):
        """
        Return the plugin associated with the confidence_method given in the configuration

        :param cfg: the configuration {'confidence_method': value}
        :type cfg: dictionary
        """
        if cls is AbstractCostVolumeConfidence:
            if isinstance(cfg["confidence_method"], str):
                try:
                    return super(AbstractCostVolumeConfidence, cls).__new__(
                        cls.confidence_methods_avail[cfg["confidence_method"]]
                    )
                except:
                    raise KeyError("No confidence method named {} supported".format(cfg["confidence_method"]))
        else:
            return super(AbstractCostVolumeConfidence, cls).__new__(cls)
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
            cls.confidence_methods_avail[short_name] = subclass
            return subclass

        return decorator

    @abstractmethod
    def desc(self):
        """
        Describes the confidence method

        """
        print("Confidence method description")

    @abstractmethod
    def confidence_prediction(
        self,
        disp: xr.Dataset,
        img_left: xr.Dataset,
        img_right: xr.Dataset,
        cv: xr.Dataset,
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        """
        Computes a confidence prediction.

        :param disp: the disparity map dataset or None
        :type disp: xarray.Dataset or None
        :param img_left: left Dataset image
        :tye img_left: xarray.Dataset
        :param img_right: right Dataset image
        :type img_right: xarray.Dataset
        :param cv: cost volume dataset
        :type cv: xarray.Dataset
        :return: None
        """

    @staticmethod
    def normalize_with_extremum(
        confidence: np.ndarray, dataset: xr.Dataset, nbr_etas: int, subpix: int = 1
    ) -> np.ndarray:
        """
        Normalize ambiguity with extremum

        :param confidence: confidence
        :type confidence: 2D np.ndarray (row, col) dtype = float32
        :param dataset: Dataset image
        :tye dataset: xarray.Dataset
        :param nbr_etas: size of etas
        :type nbr_etas: int
        :param subpix:  subpix used in matching cost
        :type subpix: int
        :return: the normalized confidence
        :rtype: 2D np.ndarray (row, col) dtype = float32
        """
        norm_confidence = np.copy(confidence)
        global_disp_max = dataset.attrs["global_disparity"][1]
        global_disp_min = dataset.attrs["global_disparity"][0]
        max_norm = (global_disp_max - global_disp_min) * nbr_etas * subpix

        return norm_confidence / max_norm

    @staticmethod
    def allocate_confidence_map(
        name_confidence_measure: str,
        confidence_map: np.ndarray,
        disp: xr.Dataset,
        cv: xr.Dataset,
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        """
        Create or update the confidence measure : confidence_measure (xarray.DataArray of the cost volume and the
        disparity map) by adding an indicator

        :param name_confidence_measure: the name of the new confidence indicator
        :type name_confidence_measure: string
        :param confidence_map: the confidence map
        :type confidence_map: 2D np.ndarray (row, col) dtype=np.float32
        :param disp: the disparity map dataset or None
        :type disp: xarray.Dataset or None
        :param cv: cost volume dataset
        :type cv: xarray.Dataset
        :return: the disparity map and the cost volume with updated confidence measure
        :rtype:
            Tuple(xarray.Dataset, xarray.Dataset) with the data variables:
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
        """

        # Add common prefix to confidence measure name
        if "disp_min" not in name_confidence_measure and "disp_max" not in name_confidence_measure:
            name_confidence_measure = "confidence_from_" + name_confidence_measure

        if cv is not None:
            # cost volume already contains a confidence map, it must be updated
            if "confidence_measure" in cv.data_vars:
                nb_row, nb_col, nb_indicator = cv["confidence_measure"].shape

                # Add a new indicator to the confidence measure DataArray
                conf_measure = np.full((nb_row, nb_col, nb_indicator + 1), np.nan, dtype=np.float32)
                # old confidence measures
                conf_measure[:, :, :-1] = cv["confidence_measure"].data
                # new confidence measure
                conf_measure[:, :, -1] = confidence_map

                indicator = np.copy(cv.coords["indicator"])
                indicator = np.append(indicator, name_confidence_measure)

                # Remove confidence_measure dataArray from the dataset to update it
                cv = cv.drop_dims("indicator")
                coords_confidence_measure = [
                    cv.coords["row"],
                    cv.coords["col"],
                    indicator,
                ]
                cv["confidence_measure"] = xr.DataArray(
                    data=conf_measure,
                    coords=coords_confidence_measure,
                    dims=["row", "col", "indicator"],
                )
            # Allocate the confidence measure in the cost volume Dataset
            else:
                coords_confidence_measure = [
                    cv.coords["row"],
                    cv.coords["col"],
                    [name_confidence_measure],
                ]
                cv["confidence_measure"] = xr.DataArray(
                    data=confidence_map[:, :, np.newaxis].astype(np.float32),
                    coords=coords_confidence_measure,
                    dims=["row", "col", "indicator"],
                )

        if disp is not None:
            # disparity already contains a confidence map, it must be updated
            if "confidence_measure" in disp.data_vars:
                nb_row, nb_col, nb_indicator = disp["confidence_measure"].shape

                # Add a new indicator to the confidence measure DataArray
                conf_measure = np.full((nb_row, nb_col, nb_indicator + 1), np.nan, dtype=np.float32)
                # old confidence measures
                conf_measure[:, :, :-1] = disp["confidence_measure"].data
                #  new confidence measure
                conf_measure[:, :, -1] = confidence_map

                indicator = np.copy(disp.coords["indicator"])
                indicator = np.append(indicator, name_confidence_measure)

                # Remove confidence_measure dataArray from the dataset to update it
                disp = disp.drop_dims("indicator")
                coords_confidence_measure = [
                    disp.coords["row"],
                    disp.coords["col"],
                    indicator,
                ]
                disp["confidence_measure"] = xr.DataArray(
                    data=conf_measure,
                    coords=coords_confidence_measure,
                    dims=["row", "col", "indicator"],
                )
            else:
                if cv is not None:
                    disp["confidence_measure"] = cv["confidence_measure"]
                else:
                    coords_confidence_measure = [
                        disp.coords["row"],
                        disp.coords["col"],
                        [name_confidence_measure],
                    ]
                    disp["confidence_measure"] = xr.DataArray(
                        data=confidence_map[:, :, np.newaxis].astype(np.float32),
                        coords=coords_confidence_measure,
                        dims=["row", "col", "indicator"],
                    )
        return disp, cv
