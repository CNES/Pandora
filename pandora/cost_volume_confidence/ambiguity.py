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
This module contains functions for estimating confidence from ambiguity.
"""

import warnings
from typing import Dict, Tuple, Union

import numpy as np
from json_checker import Checker, And
from numba import njit, prange
import xarray as xr


from . import cost_volume_confidence


@cost_volume_confidence.AbstractCostVolumeConfidence.register_subclass("ambiguity")
class Ambiguity(cost_volume_confidence.AbstractCostVolumeConfidence):
    """
    Ambiguity class allows to estimate a confidence from the cost volume
    """

    # Default configuration, do not change this value
    _ETA_MIN = 0.0
    _ETA_MAX = 0.7
    _ETA_STEP = 0.01
    # Percentile value to normalize ambiguity
    _PERCENTILE = 1.0

    def __init__(self, **cfg: str) -> None:
        """
        :param cfg: optional configuration, {'confidence_method': 'ambiguity', 'eta_min': float, 'eta_max': float,
        'eta_step': float}
        :type cfg: dict
        :return: None
        """
        self.cfg = self.check_conf(**cfg)
        self._eta_min = self._ETA_MIN
        self._percentile = self._PERCENTILE
        self._eta_max = float(self.cfg["eta_max"])
        self._eta_step = float(self.cfg["eta_step"])

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
            "confidence_method": And(str, lambda input: "ambiguity"),
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
        print("Ambiguity confidence method")

    def confidence_prediction(
        self,
        disp: xr.Dataset,
        img_left: xr.Dataset = None,
        img_right: xr.Dataset = None,
        cv: xr.Dataset = None,
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        """
        Computes a confidence measure that evaluates the matching cost function at each point

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
        # This silences numba's TBB threading layer warning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # Computes ambiguity using numba in parallel for memory and computation time optimization
            ambiguity = self.compute_ambiguity(cv["cost_volume"].data, self._eta_min, self._eta_max, self._eta_step)

        # Ambiguity normalization with percentile
        ambiguity = self.normalize_with_percentile(ambiguity)

        # Conversion of ambiguity into a confidence measure
        ambiguity = 1 - ambiguity

        disp, cv = self.allocate_confidence_map("ambiguity_confidence", ambiguity, disp, cv)

        return disp, cv

    def normalize_with_percentile(self, ambiguity):
        """
        Normalize ambiguity with percentile

        :param ambiguity: ambiguity
        :type ambiguity: 2D np.array (row, col) dtype = float32
        :return: the normalized ambiguity
        :rtype: 2D np.array (row, col) dtype = float32
        """
        norm_amb = np.copy(ambiguity)
        perc_min = np.percentile(norm_amb, self._percentile)
        perc_max = np.percentile(norm_amb, 100 - self._percentile)
        np.clip(norm_amb, perc_min, perc_max, out=norm_amb)

        return (norm_amb - np.min(norm_amb)) / (np.max(norm_amb) - np.min(norm_amb))

    @staticmethod
    @njit("f4[:, :](f4[:, :, :], f4, f4, f4)", parallel=True, cache=True)
    def compute_ambiguity(cv: np.ndarray, _eta_min: float, _eta_max: float, _eta_step: float) -> np.ndarray:
        """
        Computes ambiguity.

        :param cv: cost volume
        :type cv: 3D np.array (row, col, disp)
        :param _eta_min: minimal eta
        :type _eta_min: float
        :param _eta_max: maximal eta
        :type _eta_max: float
        :param _eta_step: eta step
        :type _eta_step: float
        :return: the normalized ambiguity
        :rtype: 2D np.array (row, col) dtype = float32
        """
        #  Miniumum and maximum of all costs, useful to normalize the cost volume
        min_cost = np.nanmin(cv)
        max_cost = np.nanmax(cv)

        n_row, n_col, nb_disps = cv.shape

        etas = np.arange(_eta_min, _eta_max, _eta_step)

        # Numba does not support the np.tile operation
        two_dim_etas = np.repeat(etas, nb_disps).reshape((-1, nb_disps)).T.flatten()

        # integral of ambiguity
        ambiguity = np.zeros((n_row, n_col), dtype=np.float32)

        for row in prange(n_row):  # pylint: disable=not-an-iterable
            for col in prange(n_col):  # pylint: disable=not-an-iterable
                # Normalized minimum cost for one point
                normalized_min_cost = (np.nanmin(cv[row, col, :]) - min_cost) / (max_cost - min_cost)

                # If all costs are at nan, set the maximum value of the ambiguity for this point
                if np.isnan(normalized_min_cost):
                    ambiguity[row, col] = etas.shape[0] * nb_disps
                else:
                    normalized_min_cost = np.repeat(normalized_min_cost, nb_disps * etas.shape[0])

                    # Normalized cost volume for one point
                    normalized_cv = (cv[row, col, :] - min_cost) / (max_cost - min_cost)
                    #  Mask nan to -inf to increase the value of the ambiguity if a point contains nan costs
                    normalized_cv[np.isnan(normalized_cv)] = -np.inf
                    normalized_cv = np.repeat(normalized_cv, etas.shape[0])

                    ambiguity[row, col] += np.sum(normalized_cv <= (normalized_min_cost + two_dim_etas))

        return ambiguity

    @staticmethod
    @njit(
        "Tuple((f4[:, :],f4[:, :, :]))(f4[:, :, :], f4, f4, f4)",
        parallel=True,
        cache=True,
    )
    def compute_ambiguity_and_sampled_ambiguity(cv: np.ndarray, _eta_min: float, _eta_max: float, _eta_step: float):
        """
        Return the ambiguity and sampled ambiguity, useful for evaluating ambiguity in notebooks

        :param cv: cost volume
        :type cv: 3D np.array (row, col, disp)
        :param _eta_min: minimal eta
        :type _eta_min: float
        :param _eta_max: maximal eta
        :type _eta_max: float
        :param _eta_step: eta step
        :type _eta_step: float
        :return: the normalized ambiguity and sampled ambiguity
        :rtype: Tuple(2D np.array (row, col) dtype = float32, 3D np.array (row, col) dtype = float32)
        """
        #  Miniumum and maximum of all costs, useful to normalize the cost volume
        min_cost = np.nanmin(cv)
        max_cost = np.nanmax(cv)

        n_row, n_col, nb_disps = cv.shape

        etas = np.arange(_eta_min, _eta_max, _eta_step)

        # Numba does not support the np.tile operation
        two_dim_etas = np.repeat(etas, nb_disps).reshape((-1, nb_disps)).T.flatten()

        # integral of ambiguity
        ambiguity = np.zeros((n_row, n_col), dtype=np.float32)
        sampled_ambiguity = np.zeros((n_row, n_col, etas.shape[0]), dtype=np.float32)

        for row in prange(n_row):  # pylint: disable=not-an-iterable
            for col in prange(n_col):  # pylint: disable=not-an-iterable
                # Normalized minimum cost for one point
                normalized_min_cost = (np.nanmin(cv[row, col, :]) - min_cost) / (max_cost - min_cost)

                # If all costs are at nan, set the maximum value of the ambiguity for this point
                if np.isnan(normalized_min_cost):
                    ambiguity[row, col] = etas.shape[0] * nb_disps
                    sampled_ambiguity[row, col, :] = nb_disps
                else:
                    normalized_min_cost = np.repeat(normalized_min_cost, nb_disps * etas.shape[0])

                    # Normalized cost volume for one point
                    normalized_cv = (cv[row, col, :] - min_cost) / (max_cost - min_cost)

                    #  Mask nan to -inf to increase the value of the ambiguity if a point contains nan costs
                    normalized_cv[np.isnan(normalized_cv)] = -np.inf
                    normalized_cv = np.repeat(normalized_cv, etas.shape[0])

                    # fill integral ambiguity
                    ambiguity[row, col] += np.sum(normalized_cv <= (normalized_min_cost + two_dim_etas))

                    # fill sampled ambiguity
                    costs_comparison = normalized_cv <= (normalized_min_cost + two_dim_etas)
                    costs_comparison = costs_comparison.reshape((nb_disps, etas.shape[0]))
                    sampled_ambiguity[row, col, :] = np.sum(costs_comparison, axis=0)

        return ambiguity, sampled_ambiguity
