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
This module contains functions associated to the median filter used to filter the disparity map.
"""
from typing import Dict, cast

import numpy as np
from json_checker import Checker, And
import xarray as xr

from pandora.margins import Margins

from . import filter  # pylint: disable= redefined-builtin
from ..constants import PANDORA_MSK_PIXEL_INTERVAL_REGULARIZED
from ..interval_tools import interval_regularization
from .median import MedianFilter


@filter.AbstractFilter.register_subclass("median_for_intervals")
class MedianForIntervalsFilter(filter.AbstractFilter):
    """
    MedianForIntervalsFilter class allows to perform the filtering step on intervals
    """

    # Default configuration, do not change this value
    _FILTER_SIZE = 3
    _AMBIGUITY_THRESHOLD = 0.6
    _AMBIGUITY_KERNEL_SIZE = 5
    _VERTICAL_DEPTH = 0
    _QUANTILE_REGULARIZATION = 1.0

    def __init__(self, *args, cfg: Dict, step: int = 1, **kwargs):  # pylint:disable=unused-argument
        """
        :param cfg: optional configuration, {'filter_size': value}
        :type cfg: dictionary
        """
        self.cfg = self.check_conf(cfg)
        self._filter_size = cast(int, self.cfg["filter_size"])
        self._interval_indicator = str(self.cfg["interval_indicator"])
        self._regularization = bool(self.cfg["regularization"])
        self._vertical_depth = int(self.cfg["vertical_depth"])
        self._quantile_regularization = float(self.cfg["quantile_regularization"])
        self._ambiguity_indicator = str(self.cfg["ambiguity_indicator"])
        self._ambiguity_threshold = float(self.cfg["ambiguity_threshold"])
        self._ambiguity_kernel_size = int(self.cfg["ambiguity_kernel_size"])
        self._step = step

    def check_conf(self, cfg: Dict) -> Dict:
        """
        Add default values to the dictionary if there are missing elements and check if the dictionary is correct

        :param cfg: filter configuration
        :type cfg: dict
        :return cfg: filter configuration updated
        :rtype: dict
        """
        # Give the default value if the required element is not in the configuration
        if "filter_size" not in cfg:
            cfg["filter_size"] = self._FILTER_SIZE
        if "interval_indicator" not in cfg:
            cfg["interval_indicator"] = ""
        if "regularization" not in cfg:
            cfg["regularization"] = False
        if "vertical_depth" not in cfg:
            cfg["vertical_depth"] = self._VERTICAL_DEPTH
        if "quantile_regularization" not in cfg:
            cfg["quantile_regularization"] = self._QUANTILE_REGULARIZATION
        if "ambiguity_indicator" not in cfg:
            cfg["ambiguity_indicator"] = ""
        if "ambiguity_threshold" not in cfg:
            cfg["ambiguity_threshold"] = self._AMBIGUITY_THRESHOLD
        if "ambiguity_kernel_size" not in cfg:
            cfg["ambiguity_kernel_size"] = self._AMBIGUITY_KERNEL_SIZE

        schema = {
            "filter_method": And(str, lambda input: "median_for_intervals"),
            "filter_size": And(int, lambda input: input >= 1 and input % 2 != 0),
            "interval_indicator": str,
            "regularization": bool,
            "ambiguity_indicator": str,
            "ambiguity_threshold": And(float, lambda input: 0 <= input <= 1),
            "ambiguity_kernel_size": And(int, lambda input: (input % 2 == 1) & (input > 0)),
            "vertical_depth": And(int, lambda input: (input >= 0)),
            "quantile_regularization": And(float, lambda input: 0 <= input <= 1),
        }

        checker = Checker(schema)
        checker.validate(cfg)
        return cfg

    def desc(self):
        """
        Describes the filtering method
        """
        print("Median filter for intervals description")

    @property
    def margins(self):
        value = self._filter_size * self._step
        return Margins(value, value, value, value)

    def filter_disparity(
        self,
        disp: xr.Dataset,
        img_left: xr.Dataset = None,
        img_right: xr.Dataset = None,
        cv: xr.Dataset = None,
    ) -> None:
        """
        Apply a median filter on interval bounds for valid pixels.
        Invalid pixels are not filtered. If a valid pixel contains an invalid pixel in its filter, the invalid pixel is
        ignored for the calculation of the median.

        :param disp: the disparity map dataset with the variables :

                - disparity_map 2D xarray.DataArray (row, col)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
                - validity_mask 2D xarray.DataArray (row, col)
        :type disp: xarray.Dataset
        :param img_left: left Dataset image
        :type img_left: xarray.Dataset
        :param img_right: right Dataset image
        :type img_right: xarray.Dataset
        :param cv: cost volume dataset
        :type cv: xarray.Dataset
        :return: None
        """
        indicator_interval_inf = (
            "confidence_from_interval_bounds_inf"
            if (self._interval_indicator == "")
            else "confidence_from_interval_bounds_inf." + self._interval_indicator
        )
        indicator_interval_sup = (
            "confidence_from_interval_bounds_sup"
            if (self._interval_indicator == "")
            else "confidence_from_interval_bounds_sup." + self._interval_indicator
        )

        cfg_median = {"filter_size": self._filter_size, "filter_method": "median"}
        med_filter = MedianFilter(cfg=cfg_median)

        for ind_interval in [indicator_interval_inf, indicator_interval_sup]:
            masked_data = disp["confidence_measure"].sel({"indicator": ind_interval}).copy(deep=True).data

            disp_median = med_filter.median_filter(masked_data)

            disp["confidence_measure"].loc[{"indicator": ind_interval}] = disp_median

        if self._regularization:
            indicator_amb = (
                "confidence_from_ambiguity"
                if (self._ambiguity_indicator == "")
                else "confidence_from_ambiguity." + self._ambiguity_indicator
            )
            interval_bound_inf, interval_bound_sup, mask_regularization = interval_regularization(
                disp["confidence_measure"].loc[{"indicator": indicator_interval_inf}].copy(deep=True).data,
                disp["confidence_measure"].loc[{"indicator": indicator_interval_sup}].copy(deep=True).data,
                disp["confidence_measure"].sel({"indicator": indicator_amb}).data,
                self._ambiguity_threshold,
                self._ambiguity_kernel_size,
                self._vertical_depth,
                self._quantile_regularization,
            )

            # This is based on 'allocate_confidence_map'
            conf_measure = disp["confidence_measure"].data

            # Regularization can be done multiple times, so the mask cannot simply be added
            disp["validity_mask"].data[mask_regularization] |= PANDORA_MSK_PIXEL_INTERVAL_REGULARIZED

            indicator_inf_index = np.argwhere(disp.coords["indicator"].data == indicator_interval_inf)[0, 0]
            indicator_sup_index = np.argwhere(disp.coords["indicator"].data == indicator_interval_sup)[0, 0]

            # Overwritting np.ndarray with the regularized values
            conf_measure[:, :, indicator_inf_index] = interval_bound_inf
            conf_measure[:, :, indicator_sup_index] = interval_bound_sup

            coords_confidence_measure = [
                disp.coords["row"],
                disp.coords["col"],
                disp.coords["indicator"],
            ]

            disp["confidence_measure"] = xr.DataArray(
                data=conf_measure,
                coords=coords_confidence_measure,
                dims=["row", "col", "indicator"],
            )
