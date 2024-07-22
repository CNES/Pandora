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
This module contains functions for estimating confidence from ambiguity.
"""
import logging
import warnings
import os
from typing import Dict, Tuple, Union
from ast import literal_eval
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
    # Ambiguity normalization
    _NORMALIZATION = True
    # Method name
    _method = "ambiguity"

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
        self._normalization = self.cfg["normalization"]
        self._eta_max = float(self.cfg["eta_max"])
        self._eta_step = float(self.cfg["eta_step"])
        self._indicator = self._method + str(self.cfg["indicator"])
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
        if "normalization" not in cfg:
            cfg["normalization"] = self._NORMALIZATION

        schema = {
            "confidence_method": And(str, lambda input: "ambiguity"),
            "eta_max": And(float, lambda input: 0 < input < 1),
            "eta_step": And(float, lambda input: 0 < input < 1),
            "normalization": bool,
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

        grids = np.array(
            [img_left["disparity"].sel(band_disp="min"), img_left["disparity"].sel(band_disp="max")], dtype=np.int64
        )
        # Get disparity intervals parameters
        disparity_range = cv["disp"].data.astype(np.float32)
        # This silences numba's TBB threading layer warning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # Computes ambiguity using numba in parallel for memory and computation time optimization
            ambiguity = self.compute_ambiguity(
                cv["cost_volume"].data, self._etas, self._nbr_etas, grids, disparity_range
            )

        # If activated, ambiguity normalization with percentile
        if self._normalization:
            if "global_disparity" in img_left.attrs:
                ambiguity = self.normalize_with_extremum(ambiguity, img_left)
                logging.info(
                    "You are not using ambiguity normalization by percentile; \n"
                    "you are in a specific case with the instantiation of global_disparity."
                )
            # in case of cross correlation
            elif "global_disparity" in img_right.attrs:
                ambiguity = self.normalize_with_extremum(ambiguity, img_right)
            else:
                ambiguity = self.normalize_with_percentile(ambiguity)

        # Conversion of ambiguity into a confidence measure
        ambiguity = 1 - ambiguity

        disp, cv = self.allocate_confidence_map(self._indicator, ambiguity, disp, cv)

        return disp, cv

    def normalize_with_percentile(self, ambiguity: np.ndarray) -> np.ndarray:
        """
        Normalize ambiguity with percentile

        :param ambiguity: ambiguity
        :type ambiguity: 2D np.ndarray (row, col) dtype = float32
        :return: the normalized ambiguity
        :rtype: 2D np.ndarray (row, col) dtype = float32
        """

        norm_amb = np.copy(ambiguity)
        perc_min = np.percentile(norm_amb, self._percentile)
        perc_max = np.percentile(norm_amb, 100 - self._percentile)
        np.clip(norm_amb, perc_min, perc_max, out=norm_amb)

        return (norm_amb - np.min(norm_amb)) / (np.max(norm_amb) - np.min(norm_amb))

    def normalize_with_extremum(self, ambiguity: np.ndarray, dataset: xr.Dataset) -> np.ndarray:
        """
        Normalize ambiguity with extremum

        :param ambiguity: ambiguity
        :type ambiguity: 2D np.ndarray (row, col) dtype = float32
        :param dataset: Dataset image
        :tye dataset: xarray.Dataset
        :return: the normalized ambiguity
        :rtype: 2D np.ndarray (row, col) dtype = float32
        """
        norm_amb = np.copy(ambiguity)
        global_disp_max = dataset.attrs["global_disparity"][1]
        global_disp_min = dataset.attrs["global_disparity"][0]
        max_norm = (global_disp_max - global_disp_min) * self._nbr_etas

        return norm_amb / max_norm

    @staticmethod
    @njit(
        "f4[:, :](f4[:, :, :], f8[:], i8, i8[:, :, :],f4[:])",
        parallel=literal_eval(os.environ.get("PANDORA_NUMBA_PARALLEL", "False")),
        cache=True,
    )
    def compute_ambiguity(
        cv: np.ndarray,
        etas: np.ndarray,
        nbr_etas: int,
        grids: np.ndarray,
        disparity_range: np.ndarray,
    ) -> np.ndarray:
        """
        Computes ambiguity.

        :param cv: cost volume
        :type cv: 3D np.ndarray (row, col, disp)
        :param etas: range between eta_min and eta_max with step eta_step
        :type etas: np.ndarray
        :param nbr_etas: number of etas
        :type nbr_etas: int
        :param grids: array containing min and max disparity grids
        :type grids: 2D np.ndarray (min, max)
        :param disparity_range: array containing disparity range
        :type disparity_range: np.ndarray
        :return: the normalized ambiguity
        :rtype: 2D np.ndarray (row, col) dtype = float32
        """

        # Minimum and maximum of all costs, useful to normalize the cost volume
        min_cost = np.nanmin(cv)
        max_cost = np.nanmax(cv)

        n_row, n_col, nb_disps = cv.shape

        # Numba does not support the np.tile operation
        two_dim_etas = np.repeat(etas, nb_disps).reshape((-1, nb_disps)).T.flatten()

        # integral of ambiguity
        ambiguity = np.zeros((n_row, n_col), dtype=np.float32)

        diff_cost = max_cost - min_cost

        for row in prange(n_row):  # pylint: disable=not-an-iterable
            for col in prange(n_col):  # pylint: disable=not-an-iterable
                # Normalized minimum cost for one point
                normalized_min_cost = (np.nanmin(cv[row, col, :]) - min_cost) / diff_cost

                # If all costs are at nan, set the maximum value of the ambiguity for this point
                if np.isnan(normalized_min_cost):
                    ambiguity[row, col] = nbr_etas * nb_disps
                else:

                    idx_disp_min = np.searchsorted(disparity_range, grids[0][row, col])
                    idx_disp_max = np.searchsorted(disparity_range, grids[1][row, col]) + 1

                    normalized_min_cost = np.repeat(normalized_min_cost, nb_disps * nbr_etas)
                    # Normalized cost volume for one point
                    normalized_cv = (cv[row, col, :] - min_cost) / diff_cost

                    # Mask nan to -inf to increase the value of the ambiguity if a point contains nan costs
                    normalized_cv[idx_disp_min:idx_disp_max][
                        np.isnan(normalized_cv[idx_disp_min:idx_disp_max])
                    ] = -np.inf

                    normalized_cv[:idx_disp_min][np.isnan(normalized_cv[:idx_disp_min])] = np.inf
                    normalized_cv[idx_disp_max:][np.isnan(normalized_cv[idx_disp_max:])] = np.inf

                    normalized_cv = np.repeat(normalized_cv, nbr_etas)

                    ambiguity[row, col] += np.nansum(normalized_cv <= (normalized_min_cost + two_dim_etas))

        return ambiguity

    @staticmethod
    @njit(
        "Tuple((f4[:, :],f4[:, :, :]))(f4[:, :, :], f8[:], i8, i8[:, :, :], f4[:])",
        parallel=literal_eval(os.environ.get("PANDORA_NUMBA_PARALLEL", "False")),
        cache=True,
    )
    def compute_ambiguity_and_sampled_ambiguity(
        cv: np.ndarray,
        etas: np.ndarray,
        nbr_etas: int,
        grids: np.ndarray,
        disparity_range: np.ndarray,
    ):
        """
        Return the ambiguity and sampled ambiguity, useful for evaluating ambiguity in notebooks

        :param cv: cost volume
        :type cv: 3D np.ndarray (row, col, disp)
        :param etas: range between eta_min and eta_max with step eta_step
        :type etas: np.ndarray
        :param nbr_etas: nuber of etas
        :type nbr_etas: int
        :param grids: array containing min and max disparity grids
        :type grids: 2D np.ndarray (min, max)
        :param disparity_range: array containing disparity range
        :type disparity_range: np.ndarray
        :return: the normalized ambiguity and sampled ambiguity
        :rtype: Tuple(2D np.ndarray (row, col) dtype = float32, 3D np.ndarray (row, col) dtype = float32)
        """
        # Minimum and maximum of all costs, useful to normalize the cost volume
        min_cost = np.nanmin(cv)
        max_cost = np.nanmax(cv)

        n_row, n_col, nb_disps = cv.shape

        # Numba does not support the np.tile operation
        two_dim_etas = np.repeat(etas, nb_disps).reshape((-1, nb_disps)).T.flatten()

        # integral of ambiguity
        ambiguity = np.zeros((n_row, n_col), dtype=np.float32)
        sampled_ambiguity = np.zeros((n_row, n_col, nbr_etas), dtype=np.float32)

        diff_cost = max_cost - min_cost

        for row in prange(n_row):  # pylint: disable=not-an-iterable
            for col in prange(n_col):  # pylint: disable=not-an-iterable
                # Normalized minimum cost for one point
                normalized_min_cost = (np.nanmin(cv[row, col, :]) - min_cost) / diff_cost

                # If all costs are at nan, set the maximum value of the ambiguity for this point
                if np.isnan(normalized_min_cost):
                    ambiguity[row, col] = nbr_etas * nb_disps
                    sampled_ambiguity[row, col, :] = nb_disps
                else:
                    normalized_min_cost = np.repeat(normalized_min_cost, nb_disps * nbr_etas)

                    # Normalized cost volume for one point
                    normalized_cv = (cv[row, col, :] - min_cost) / (max_cost - min_cost)

                    idx_disp_min = np.searchsorted(disparity_range, grids[0][row, col])
                    idx_disp_max = np.searchsorted(disparity_range, grids[1][row, col]) + 1

                    # Mask nan to -inf to increase the value of the ambiguity if a point contains nan costs
                    normalized_cv[idx_disp_min:idx_disp_max][
                        np.isnan(normalized_cv[idx_disp_min:idx_disp_max])
                    ] = -np.inf
                    normalized_cv[:idx_disp_min][np.isnan(normalized_cv[:idx_disp_min])] = np.inf
                    normalized_cv[idx_disp_max:][np.isnan(normalized_cv[idx_disp_max:])] = np.inf

                    normalized_cv = np.repeat(normalized_cv, nbr_etas)

                    # fill integral ambiguity
                    ambiguity[row, col] += np.nansum(normalized_cv <= (normalized_min_cost + two_dim_etas))

                    # fill sampled ambiguity
                    costs_comparison = normalized_cv <= (normalized_min_cost + two_dim_etas)
                    costs_comparison = costs_comparison.reshape((nb_disps, nbr_etas))
                    sampled_ambiguity[row, col, :] = np.sum(costs_comparison, axis=0)

        return ambiguity, sampled_ambiguity
