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
This module contains functions associated to the median filter used to filter the disparity map.
"""
import warnings
from typing import Dict, Union, cast

import numpy as np
import xarray as xr
from json_checker import Checker, And

import pandora.constants as cst
from . import filter  # pylint: disable= redefined-builtin
from ..common import sliding_window


@filter.AbstractFilter.register_subclass("median")
class MedianFilter(filter.AbstractFilter):
    """
    MedianFilter class allows to perform the filtering step
    """

    # Default configuration, do not change this value
    _FILTER_SIZE = 3

    def __init__(self, **cfg: Union[str, int]):
        """
        :param cfg: optional configuration, {'filter_size': value}
        :type cfg: dictionary
        """
        self.cfg = self.check_conf(**cfg)
        self._filter_size = cast(int, self.cfg["filter_size"])

    def check_conf(self, **cfg: Union[str, int]) -> Dict[str, Union[str, int]]:
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

        schema = {
            "filter_method": And(str, lambda input: "median"),
            "filter_size": And(int, lambda input: input >= 1 and input % 2 != 0),
        }

        checker = Checker(schema)
        checker.validate(cfg)
        return cfg

    def desc(self):
        """
        Describes the filtering method
        """
        print("Median filter description")

    def filter_disparity(
        self,
        disp: xr.Dataset,
        img_left: xr.Dataset = None,
        img_right: xr.Dataset = None,
        cv: xr.Dataset = None,
    ) -> None:
        """
        Apply a median filter on valid pixels.
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
        # Invalid pixels are nan
        masked_data = disp["disparity_map"].copy(deep=True).data
        masked_data[np.where((disp["validity_mask"].data & cst.PANDORA_MSK_PIXEL_INVALID) != 0)] = np.nan

        valid = np.isfinite(masked_data)
        disp_median = self.median_filter(masked_data)

        disp["disparity_map"].data[valid] = disp_median[valid]
        disp.attrs["filter"] = "median"
        del (
            disp_median,
            masked_data,
        )

    def median_filter(self, data) -> np.ndarray:
        """
        Apply median filter on valid pixels (pixels that are not nan).
        Invalid pixels are not filtered. If a valid pixel contains an invalid pixel in its filter, the invalid pixel is
        ignored for the calculation of the median.

        :param data: input data to be filtered
        :type data: 2D np.array (row, col)
        :return: The filtered array
        :rtype: 2D np.array(row, col)
        """
        data_median = np.copy(data)
        invalid = np.isnan(data_median)
        ny_, nx_ = data.shape

        aggregation_window = sliding_window(data, (self._filter_size, self._filter_size))

        radius = int(self._filter_size / 2)

        # To reduce memory, the data array is split (along the row axis) into multiple sub-arrays with a step of 100
        chunk_size = 100
        disp_chunked_y = np.array_split(aggregation_window, np.arange(chunk_size, ny_, chunk_size), axis=0)
        y_begin = radius

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
            # numpy.nanmedian : Compute the median along the specified axis, while ignoring NaNs (i.e if valid pixel
            # contains an invalid pixel in its filter, the invalid pixel is ignored because invalid pixels are nan )
            for col, disp_y in enumerate(disp_chunked_y):  # pylint: disable= unused-variable
                # To reduce memory, the data array is split (along the col axis) into multiple sub-arrays,
                # with a step of 100
                disp_chunked_x = np.array_split(disp_y, np.arange(chunk_size, nx_, chunk_size), axis=1)
                x_begin = radius

                for rox, disp_x in enumerate(disp_chunked_x):  # pylint: disable= unused-variable
                    y_end = y_begin + disp_y.shape[0]
                    x_end = x_begin + disp_x.shape[1]
                    data_median[y_begin:y_end, x_begin:x_end] = np.nanmedian(disp_x, axis=(2, 3))
                    x_begin += disp_x.shape[1]

                y_begin += disp_y.shape[0]

        del aggregation_window

        data_median[invalid] = np.nan
        return data_median
