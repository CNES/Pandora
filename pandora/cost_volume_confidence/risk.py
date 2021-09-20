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
This module contains functions for estimating the risk.
"""

import warnings
from typing import Dict, Tuple, Union

import numpy as np
from json_checker import Checker, And
from numba import njit, prange
import xarray as xr


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

        schema = {
            "confidence_method": And(str, lambda input: "risk"),
            "eta_max": And(float, lambda input: 0 < input < 1),
            "eta_step": And(float, lambda input: 0 < input < 1),
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
        # This silences numba's TBB threading layer warning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # Obtain sampled_ambiguity, necessary for risk_min computation
            ambiguity_ = cost_volume_confidence.AbstractCostVolumeConfidence(  # type: ignore
                **{"confidence_method": "ambiguity"}  # type: ignore
            )
            _, sampled_ambiguity = ambiguity_.compute_ambiguity_and_sampled_ambiguity(  # type: ignore
                cv["cost_volume"].data, self._eta_min, self._eta_max, self._eta_step
            )
            # Computes risk using numba in parallel for memory and computation time optimization
            risk_max, risk_min = self.compute_risk(
                cv["cost_volume"].data, sampled_ambiguity, self._eta_min, self._eta_max, self._eta_step
            )

        disp, cv = self.allocate_confidence_map("risk_max_confidence", risk_max, disp, cv)
        disp, cv = self.allocate_confidence_map("risk_min_confidence", risk_min, disp, cv)

        return disp, cv

    @staticmethod
    @njit("Tuple((f4[:, :],f4[:, :]))(f4[:, :, :], f4[:, :, :], f4, f4, f4)", parallel=True, cache=True)
    def compute_risk(
        cv: np.ndarray, sampled_ambiguity: np.ndarray, _eta_min: float, _eta_max: float, _eta_step: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes minimum and maximum risk.

        :param cv: cost volume
        :type cv: 3D np.array (row, col, disp)
        :param sampled_ambiguity: sampled cost volume ambiguity
        :type sampled_ambiguity: 3D np.array (row, col, eta)
        :param _eta_min: minimal eta
        :type _eta_min: float
        :param _eta_max: maximal eta
        :type _eta_max: float
        :param _eta_step: eta step
        :type _eta_step: float
        :return: the minimum and maximum risk
        :rtype: Tuple(2D np.array (row, col) dtype = float32, 2D np.array (row, col) dtype = float32)
        """
        #  Miniumum and maximum of all costs, useful to normalize the cost volume
        min_cost = np.nanmin(cv)
        max_cost = np.nanmax(cv)

        n_row, n_col, nb_disps = cv.shape

        etas = np.arange(_eta_min, _eta_max, _eta_step)

        # Numba does not support the np.tile operation
        two_dim_etas = np.repeat(etas, nb_disps).reshape((-1, nb_disps)).T.flatten()

        # Initialize min and max risk integral
        risk_max = np.zeros((n_row, n_col), dtype=np.float32)
        risk_min = np.zeros((n_row, n_col), dtype=np.float32)

        for row in prange(n_row):  # pylint: disable=not-an-iterable
            for col in prange(n_col):  # pylint: disable=not-an-iterable
                # Normalized minimum cost for one point
                normalized_min_cost = (np.nanmin(cv[row, col, :]) - min_cost) / (max_cost - min_cost)

                # If all costs are at nan, set the risk at nan for this point
                if np.isnan(normalized_min_cost):
                    risk_max[row, col] = np.nan
                    risk_min[row, col] = np.nan
                else:
                    normalized_min_cost = np.repeat(normalized_min_cost, nb_disps * etas.shape[0])
                    # Normalized cost volume for one point
                    normalized_cv = (cv[row, col, :] - min_cost) / (max_cost - min_cost)
                    #  Mask nan to -inf to later discard values out of [min; min + eta]
                    normalized_cv[np.isnan(normalized_cv)] = -np.inf
                    normalized_cv = np.repeat(normalized_cv, etas.shape[0])
                    # Initialize all disparties
                    disp_cv = np.arange(nb_disps) * 1.0
                    disp_cv = np.repeat(disp_cv, etas.shape[0])
                    # Remove disparities for every similarity value outside of [min;min+eta[
                    disp_cv[normalized_cv > (normalized_min_cost + two_dim_etas)] = np.nan
                    # Reshape to distinguish each sample's disparity range
                    disp_cv = disp_cv.reshape((nb_disps, etas.shape[0]))
                    # Initialize minimum and maximum disparities
                    min_disp = np.zeros(etas.shape[0])
                    max_disp = np.zeros(etas.shape[0])
                    # Obtain min and max disparities for each sample
                    for i in range(etas.shape[0]):
                        min_disp[i] = np.nanmin(disp_cv[:, i])
                        max_disp[i] = np.nanmax(disp_cv[:, i])
                    # fill mean risks
                    risk_max[row, col] = np.nanmean(max_disp - min_disp)
                    # fill mean min risk. risk min is defined as mean( (1+risk(p,k)) - amb(p,k) )
                    risk_min[row, col] = np.nanmean((1 + (max_disp - min_disp)) - sampled_ambiguity[row, col, :])

        return risk_max, risk_min

    @staticmethod
    @njit(
        "Tuple((f4[:, :],f4[:, :],f4[:, :, :],f4[:, :, :]))(f4[:, :, :], f4[:, :, :], f4, f4, f4)",
        parallel=True,
        cache=True,
    )
    def compute_risk_and_sampled_risk(
        cv: np.ndarray, sampled_ambiguity: np.ndarray, _eta_min: float, _eta_max: float, _eta_step: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes minimum and maximum risk and sampled_risk.

        :param cv: cost volume
        :type cv: 3D np.array (row, col, disp)
        :param sampled_ambiguity: sampled cost volume ambiguity
        :type sampled_ambiguity: 3D np.array (row, col, eta)
        :param _eta_min: minimal eta
        :type _eta_min: float
        :param _eta_max: maximal eta
        :type _eta_max: float
        :param _eta_step: eta step
        :type _eta_step: float
        :return: the risk
        :rtype: Tuple(2D np.array (row, col) dtype = float32, 2D np.array (row, col) dtype = float32,
                     3D np.array (row, col) dtype = float32, 3D np.array (row, col) dtype = float32)
        """
        #  Miniumum and maximum of all costs, useful to normalize the cost volume
        min_cost = np.nanmin(cv)
        max_cost = np.nanmax(cv)

        n_row, n_col, nb_disps = cv.shape

        etas = np.arange(_eta_min, _eta_max, _eta_step)

        # Numba does not support the np.tile operation
        two_dim_etas = np.repeat(etas, nb_disps).reshape((-1, nb_disps)).T.flatten()

        # Initialize min and max risk integral
        risk_max = np.zeros((n_row, n_col), dtype=np.float32)
        risk_min = np.zeros((n_row, n_col), dtype=np.float32)
        # Initialize min and max sampled risks
        sampled_risk_min = np.zeros((n_row, n_col, etas.shape[0]), dtype=np.float32)
        sampled_risk_max = np.zeros((n_row, n_col, etas.shape[0]), dtype=np.float32)

        for row in prange(n_row):  # pylint: disable=not-an-iterable
            for col in prange(n_col):  # pylint: disable=not-an-iterable
                # Normalized minimum cost for one point
                normalized_min_cost = (np.nanmin(cv[row, col, :]) - min_cost) / (max_cost - min_cost)

                # If all costs are at nan, set the risk at nan for this point
                if np.isnan(normalized_min_cost):
                    sampled_risk_min[row, col, :] = np.nan
                    sampled_risk_max[row, col, :] = np.nan
                    risk_max[row, col] = np.nan
                    risk_min[row, col] = np.nan
                else:
                    normalized_min_cost = np.repeat(normalized_min_cost, nb_disps * etas.shape[0])
                    # Normalized cost volume for one point
                    normalized_cv = (cv[row, col, :] - min_cost) / (max_cost - min_cost)
                    #  Mask nan to -inf to later discard values out of [min; min + eta]
                    normalized_cv[np.isnan(normalized_cv)] = -np.inf
                    normalized_cv = np.repeat(normalized_cv, etas.shape[0])
                    # Initialize all disparties
                    disp_cv = np.arange(nb_disps) * 1.0
                    disp_cv = np.repeat(disp_cv, etas.shape[0])
                    # Remove disparities for every similarity value outside of [min;min+eta[
                    disp_cv[normalized_cv > (normalized_min_cost + two_dim_etas)] = np.nan
                    # Reshape to distinguish each sample's disparity range
                    disp_cv = disp_cv.reshape((nb_disps, etas.shape[0]))
                    # Initialize minimum and maximum disparities
                    min_disp = np.zeros(etas.shape[0])
                    max_disp = np.zeros(etas.shape[0])
                    # Obtain min and max disparities for each sample
                    for i in range(etas.shape[0]):
                        min_disp[i] = np.nanmin(disp_cv[:, i])
                        max_disp[i] = np.nanmax(disp_cv[:, i])
                    # fill sampled max risk
                    sampled_risk_max[row, col, :] += max_disp - min_disp
                    # fill sampled min risk. risk min is defined as ( (1+risk(p,k)) - amb(p,k) )
                    sampled_risk_min[row, col, :] += (1 + (max_disp - min_disp)) - sampled_ambiguity[row, col, :]
                    # fill mean risks
                    risk_max[row, col] = np.nanmean(max_disp - min_disp)
                    # fill mean min risk. risk min is defined as mean( (1+risk(p,k)) - amb(p,k) )
                    risk_min[row, col] = np.nanmean((1 + (max_disp - min_disp)) - sampled_ambiguity[row, col, :])

        return risk_max, risk_min, sampled_risk_max, sampled_risk_min
