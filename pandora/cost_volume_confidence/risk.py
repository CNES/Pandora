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
This module contains functions for estimating the risk.
"""
from typing import Dict, Tuple, Union

import numpy as np
import xarray as xr
from json_checker import And, Checker

from .cpp import cost_volume_confidence_cpp

from . import cost_volume_confidence


@cost_volume_confidence.AbstractCostVolumeConfidence.register_subclass("risk")
class Risk(cost_volume_confidence.AbstractCostVolumeConfidence):
    """
    Allows to estimate a risk confidence from the cost volume
    """

    # Default configuration, do not change this value
    _ETA_MIN = 0.0
    _ETA_MAX = 0.7
    _ETA_STEP = 0.01
    # Percentile value to normalize ambiguity
    _PERCENTILE = 1.0
    # Method name
    _method_max = "risk_max"
    _method_min = "risk_min"

    _method_disp_inf = "disp_inf_from_risk"
    _method_disp_sup = "disp_sup_from_risk"

    def __init__(self, **cfg: str) -> None:
        """
        :param cfg: optional configuration, {'confidence_method': 'risk', 'eta_min': float, 'eta_max': float,
        'eta_step': float}
        :type cfg: dict
        :return: None
        """
        self.cfg = self.check_conf(**cfg)
        self._eta_min = self._ETA_MIN
        self._percentile = self._PERCENTILE
        self._eta_step = float(self.cfg["eta_step"])
        self._eta_max = float(self.cfg["eta_max"])
        self._indicator_max = self._method_max + str(self.cfg["indicator"])
        self._indicator_min = self._method_min + str(self.cfg["indicator"])
        self._indicator_disp_sup = self._method_disp_sup + str(self.cfg["indicator"])
        self._indicator_disp_inf = self._method_disp_inf + str(self.cfg["indicator"])
        self._etas = np.arange(self._eta_min, self._eta_max, self._eta_step)
        self._nbr_etas = self._etas.shape[0]

    def check_conf(self, **cfg: Union[str, float]) -> Dict[str, Union[str, float]]:
        """
        Add default values to the dictionary if there are missing elements and check if the dictionary is correct

        :param cfg: ambiguity configuration
        :type cfg: dict
        :return cfg: ambiguity configuration updated
        :rtype: dict
        """
        if "eta_max" not in cfg:
            cfg["eta_max"] = self._ETA_MAX
        if "eta_step" not in cfg:
            cfg["eta_step"] = self._ETA_STEP
        if "indicator" not in cfg:
            cfg["indicator"] = self._indicator

        schema = {
            "confidence_method": And(str, lambda input: "risk"),
            "eta_max": And(float, lambda input: 0 < input < 1),
            "eta_step": And(float, lambda input: 0 < input < 1),
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
        print("Risk method")

    def confidence_prediction(
        self,
        disp: xr.Dataset,
        img_left: xr.Dataset = None,
        img_right: xr.Dataset = None,
        cv: xr.Dataset = None,
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        """
        Computes a risk confidence measure that evaluates the matching cost function at each point

        :param disp: the disparity map dataset
        :type disp: xarray.Dataset
        :param img_left: left Dataset image
        :tye img_left: xarray.Dataset
        :param img_right: right Dataset image
        :type img_right: xarray.Dataset
        :param cv: cost volume dataset
        :type cv: xarray.Dataset
        :return: the disparity map and the cost volume with new indicators 'risk_max_confidence'
                and 'risk_min_confidence' in the
        :rtype: Tuple(xarray.Dataset, xarray.Dataset)
        """

        type_measure_max = cv.attrs["type_measure"] == "max"
        if type_measure_max:
            cv["cost_volume"].data *= -1

        grids = np.array(
            [img_left["disparity"].sel(band_disp="min"), img_left["disparity"].sel(band_disp="max")], dtype=np.int64
        )
        # Get disparity intervals parameters
        disparity_range = cv["disp"].data.astype(np.float32)

        _, sampled_ambiguity = cost_volume_confidence_cpp.compute_ambiguity_and_sampled_ambiguity(
            cv["cost_volume"].data, self._etas, self._nbr_etas, grids, disparity_range, True
        )

        risk_max, risk_min, disp_sup, disp_inf = self.compute_risk(
            cv["cost_volume"].data,
            sampled_ambiguity,
            self._etas,
            self._nbr_etas,
            grids,
            disparity_range,
        )

        disp, cv = self.allocate_confidence_map(self._indicator_max, risk_max, disp, cv)
        disp, cv = self.allocate_confidence_map(self._indicator_min, risk_min, disp, cv)
        disp, cv = self.allocate_confidence_map(self._indicator_disp_sup, disp_sup, disp, cv)
        disp, cv = self.allocate_confidence_map(self._indicator_disp_inf, disp_inf, disp, cv)

        if type_measure_max:
            cv["cost_volume"].data *= -1

        return disp, cv

    @staticmethod
    def compute_risk(
        cv: np.ndarray,
        sampled_ambiguity: np.ndarray,
        etas: np.ndarray,
        nbr_etas: int,
        grids: np.ndarray,
        disparity_range: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes minimum and maximum risk.
        Cost Volume must correspond to min similarity measure.

        :param cv: cost volume
        :type cv: 3D np.ndarray (row, col, disp)
        :param sampled_ambiguity: sampled cost volume ambiguity
        :type sampled_ambiguity: 3D np.ndarray (row, col, eta)
        :param etas: range between eta_min and eta_max with step eta_step
        :type etas: np.ndarray
        :param nbr_etas: number of etas
        :type nbr_etas: int
        :param grids: array containing min and max disparity grids
        :type grids: 2D np.ndarray (min, max)
        :param disparity_range: array containing disparity range
        :type disparity_range: np.ndarray
        :return: the minimum and maximum risk
        :rtype: Tuple(2D np.ndarray (row, col) dtype = float32, 2D np.ndarray (row, col) dtype = float32\
        2D np.ndarray (row, col) dtype = float32, 2D np.ndarray (row, col) dtype = float32)
        """
        return cost_volume_confidence_cpp.compute_risk_and_sampled_risk(
            cv, sampled_ambiguity, etas, nbr_etas, grids, disparity_range, False
        )

    @staticmethod
    def compute_risk_and_sampled_risk(
        cv: np.ndarray,
        sampled_ambiguity: np.ndarray,
        etas: np.ndarray,
        nbr_etas: int,
        grids: np.ndarray,
        disparity_range: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes minimum and maximum risk and sampled_risk.
        Cost Volume must correspond to min similarity measure.

        :param cv: cost volume
        :type cv: 3D np.ndarray (row, col, disp)
        :param sampled_ambiguity: sampled cost volume ambiguity
        :type sampled_ambiguity: 3D np.ndarray (row, col, eta)
        :param etas: range between eta_min and eta_max with step eta_step
        :type etas: np.ndarray
        :param nbr_etas: nuber of etas
        :type nbr_etas: int
        :param grids: array containing min and max disparity grids
        :type grids: 2D np.ndarray (min, max)
        :param disparity_range: array containing disparity range
        :type disparity_range: np.ndarray
        :return: the risk
        :rtype: Tuple(2D np.ndarray (row, col) dtype = float32, 2D np.ndarray (row, col) dtype = float32,
                     2D np.ndarray (row, col) dtype = float32, 2D np.ndarray (row, col) dtype = float32,
                     3D np.ndarray (row, col) dtype = float32, 3D np.ndarray (row, col) dtype = float32)
        """
        return cost_volume_confidence_cpp.compute_risk_and_sampled_risk(
            cv, sampled_ambiguity, etas, nbr_etas, grids, disparity_range, True
        )
