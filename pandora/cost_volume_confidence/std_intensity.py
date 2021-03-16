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
This module contains functions for estimating confidence from image.
"""

from typing import Dict, Tuple

import numpy as np
from json_checker import Checker, And
import xarray as xr

from pandora.img_tools import compute_std_raster
from . import cost_volume_confidence


@cost_volume_confidence.AbstractCostVolumeConfidence.register_subclass("std_intensity")
class StdIntensity(cost_volume_confidence.AbstractCostVolumeConfidence):
    """
    StdIntensity class allows to estimate a confidence measure from the left image by calculating the standard
     deviation of the intensity
    """

    def __init__(self, **cfg: str) -> None:
        """
        :param cfg: optional configuration, {'confidence_method': 'std_intensity'}
        :type cfg: dict
        :return: None
        """
        self.cfg = self.check_conf(**cfg)

    @staticmethod
    def check_conf(**cfg: str) -> Dict[str, str]:
        """
        Add default values to the dictionary if there are missing elements and check if the dictionary is correct

        :param cfg: std_intensity configuration
        :type cfg: dict
        :return cfg: std_intensity configuration updated
        :rtype: dict
        """
        schema = {"confidence_method": And(str, lambda input: "std_intensity")}

        checker = Checker(schema)
        checker.validate(cfg)
        return cfg

    def desc(self) -> None:
        """
        Describes the confidence method
        :return: None
        """
        print("Intensity confidence method")

    def confidence_prediction(
        self,
        disp: xr.Dataset,
        img_left: xr.Dataset = None,
        img_right: xr.Dataset = None,
        cv: xr.Dataset = None,
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        """
        Computes a confidence measure that evaluates the standard deviation of intensity of the left image

        :param disp: the disparity map dataset
        :type disp: xarray.Dataset
        :param img_left: left Dataset image
        :tye img_left: xarray.Dataset
        :param img_right: right Dataset image
        :type img_right: xarray.Dataset
        :param cv: cost volume dataset
        :type cv: xarray.Dataset
        :return: the disparity map and the cost volume with a new indicator 'ambiguity_confidence' in the DataArray
                 confidence_measure
        :rtype: Tuple(xarray.Dataset, xarray.Dataset) with the data variables:

                - confidence_measure 3D xarray.DataArray (row, col, indicator)
        """
        nb_row, nb_col = img_left["im"].shape

        window_size = cv.attrs["window_size"]
        confidence_measure = np.full((nb_row, nb_col), np.nan, dtype=np.float32)

        offset_row_col = int((window_size - 1) / 2)
        if offset_row_col != 0:
            confidence_measure[offset_row_col:-offset_row_col, offset_row_col:-offset_row_col] = compute_std_raster(
                img_left, window_size
            )
        else:
            confidence_measure = compute_std_raster(img_left, window_size)

        disp, cv = self.allocate_confidence_map("stereo_pandora_intensityStd", confidence_measure, disp, cv)
        return disp, cv
