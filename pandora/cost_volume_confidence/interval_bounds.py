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
This module contains functions for estimating interval bounds for the disparity
"""

from typing import Dict, Tuple, Union

import numpy as np
from json_checker import And, Checker
import xarray as xr

from . import cost_volume_confidence
from ..interval_tools import interval_regularization
from .cpp import cost_volume_confidence_cpp


@cost_volume_confidence.AbstractCostVolumeConfidence.register_subclass("interval_bounds")
class IntervalBounds(cost_volume_confidence.AbstractCostVolumeConfidence):
    """
    IntervalBounds class allows to estimate a confidence interval from the cost volume
    """

    # Default configuration, do not change this value
    _POSSIBILITY_THRESHOLD = 0.9
    _AMBIGUITY_THRESHOLD = 0.6
    _AMBIGUITY_KERNEL_SIZE = 5
    _VERTICAL_DEPTH = 0
    _QUANTILE_REGULARIZATION = 1.0

    # Method name
    _method = "interval_bounds"

    def __init__(self, **cfg: str) -> None:
        """
        :param cfg: optional configuration, {
            'confidence_method': 'interval_bounds',
            'possibility_threshold': float,
            'ambiguity_threshold': float,
            'ambiguity_kernel_size': int,
            'ambiguity_indicator': str}
        :type cfg: dict
        :return: None
        """
        self.cfg = self.check_conf(**cfg)
        self._possibility_threshold = float(self.cfg["possibility_threshold"])

        self._ambiguity_indicator = str(self.cfg["ambiguity_indicator"])
        self._ambiguity_threshold = float(self.cfg["ambiguity_threshold"])
        self._ambiguity_kernel_size = int(self.cfg["ambiguity_kernel_size"])
        self._regularization = bool(self.cfg["regularization"])
        self._vertical_depth = int(self.cfg["vertical_depth"])
        self._quantile_regularization = float(self.cfg["quantile_regularization"])
        self._indicator = self._method + str(self.cfg["indicator"])
        self._indicator_inf = self._method + "_inf" + str(self.cfg["indicator"])
        self._indicator_sup = self._method + "_sup" + str(self.cfg["indicator"])

    def check_conf(self, **cfg: Union[str, float, int, bool]) -> Dict[str, Union[str, float, int, bool]]:
        """
        Add default values to the dictionary if there are missing elements and check if the dictionary is correct

        :param cfg: interval_bounds configuration
        :type cfg: dict
        :return cfg: interval_bounds configuration updated
        :rtype: dict
        """
        if "possibility_threshold" not in cfg:
            cfg["possibility_threshold"] = self._POSSIBILITY_THRESHOLD

        if "regularization" not in cfg:
            cfg["regularization"] = False
        if "ambiguity_indicator" not in cfg:
            cfg["ambiguity_indicator"] = ""
        if "ambiguity_threshold" not in cfg:
            cfg["ambiguity_threshold"] = self._AMBIGUITY_THRESHOLD
        if "ambiguity_kernel_size" not in cfg:
            cfg["ambiguity_kernel_size"] = self._AMBIGUITY_KERNEL_SIZE
        if "vertical_depth" not in cfg:
            cfg["vertical_depth"] = self._VERTICAL_DEPTH
        if "quantile_regularization" not in cfg:
            cfg["quantile_regularization"] = self._QUANTILE_REGULARIZATION

        if "indicator" not in cfg:
            cfg["indicator"] = self._indicator

        schema = {
            "confidence_method": And(str, lambda input: "interval_bounds"),
            "possibility_threshold": And(float, lambda input: 0 <= input <= 1),
            "regularization": bool,
            "ambiguity_indicator": str,
            "ambiguity_threshold": And(float, lambda input: 0 <= input <= 1),
            "ambiguity_kernel_size": And(int, lambda input: (input % 2 == 1) & (input > 0)),
            "vertical_depth": And(int, lambda input: (input >= 0)),
            "quantile_regularization": And(float, lambda input: 0 <= input <= 1),
            "indicator": str,
        }

        checker = Checker(schema)
        checker.validate(cfg)
        return cfg

    def desc(self) -> None:
        """
        Describes the confidence method
        :return: None
        """
        print("Interval bounds confidence method with regularization")

    def confidence_prediction(
        self,
        disp: xr.Dataset,
        img_left: xr.Dataset = None,
        img_right: xr.Dataset = None,
        cv: xr.Dataset = None,
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        """
        Computes a confidence measure that evaluates the minimum and maximum disparity at
        each point with a confidence of possibility_threshold %

        :param disp: the disparity map dataset
        :type disp: xarray.Dataset
        :param img_left: left Dataset image
        :tye img_left: xarray.Dataset
        :param img_right: right Dataset image
        :type img_right: xarray.Dataset
        :param cv: cost volume dataset
        :type cv: xarray.Dataset
        :return: the disparity map and the cost volume with new indicators 'interval_bounds.inf' and
            'interval_bounds.sup' in the DataArray confidence_measure
        :rtype: Tuple(xarray.Dataset, xarray.Dataset) with the data variables:
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
        """

        # The possibility is different depending on the type of the measure
        if cv.attrs["type_measure"] == "min":
            type_factor = -1.0
        else:
            type_factor = 1.0

        grids = np.array(
            [img_left["disparity"].sel(band_disp="min"), img_left["disparity"].sel(band_disp="max")], dtype=np.int64
        )
        # Get disparity intervals parameters
        disparity_range = cv["disp"].data.astype(np.float32)

        # Computes interval bounds using numpy
        interval_bound_inf, interval_bound_sup = self.compute_interval_bounds(
            cv["cost_volume"].data,
            cv["disp"].data.astype(np.float32),
            self._possibility_threshold,
            type_factor,
            grids,
            disparity_range,
        )
        if self._regularization:
            indicator = (
                "confidence_from_ambiguity"
                if (self._ambiguity_indicator == "")
                else "confidence_from_ambiguity." + self._ambiguity_indicator
            )
            interval_bound_inf, interval_bound_sup, _ = interval_regularization(
                interval_bound_inf,
                interval_bound_sup,
                cv.confidence_measure.sel({"indicator": indicator}).data,
                self._ambiguity_threshold,
                self._ambiguity_kernel_size,
                self._vertical_depth,
                self._quantile_regularization,
            )
        # For empty cost volume, the interval gets its max length
        # interval_bound_inf[np.isnan(interval_bound_inf)] = cv["disp"].data.astype(np.float32)[0]
        # interval_bound_sup[np.isnan(interval_bound_sup)] = cv["disp"].data.astype(np.float32)[-1]

        disp, cv = self.allocate_confidence_map(self._indicator_inf, interval_bound_inf, disp, cv)
        disp, cv = self.allocate_confidence_map(self._indicator_sup, interval_bound_sup, disp, cv)

        return disp, cv

    @staticmethod
    def compute_interval_bounds(
        cv: np.ndarray,
        disp_interval: np.ndarray,
        possibility_threshold: float,
        type_factor: float,
        grids: np.ndarray,
        disparity_range: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes interval bounds on the disparity.

        :param cv: cost volume
        :type cv: 3D np.ndarray (row, col, disp)
        :param disp_interval: disparity data
        :type disp_interval: 1D np.ndarray (disp,)
        :param possibility_threshold: possibility threshold used for interval computation
        :type possibility_threshold: float
        :param type_factor: Either 1 or -1. Used to adapt the possibility computation to max or min measures
        :type type_factor: float
        :param grids: array containing min and max disparity grids
        :type grids: 2D np.ndarray (min, max)
        :param disparity_range: array containing disparity range
        :type disparity_range: np.ndarray

        :return: the infimum and supremum (not regularized) of the set containing the true disparity
        :rtype: Tuple(2D np.ndarray (row, col) dtype = float32, 2D np.ndarray (row, col) dtype = float32)
        """
        return cost_volume_confidence_cpp.compute_interval_bounds(
            cv, disp_interval, possibility_threshold, type_factor, grids, disparity_range
        )
