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
This module contains functions associated to the median filter used to filter the disparity map.
"""
import logging
import sys
import numpy as np
import warnings
from json_checker import Checker, And
from typing import Dict, Union, cast
import xarray as xr

from . import filter
from pandora.constants import *


@filter.AbstractFilter.register_subclass('median')
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
        self._filter_size = cast(int, self.cfg['filter_size'])

    def check_conf(self, **cfg: Union[str, int]) -> Dict[str, Union[str, int]]:
        """
        Add default values to the dictionary if there are missing elements and check if the dictionary is correct

        :param cfg: filter configuration
        :type cfg: dict
        :return cfg: filter configuration updated
        :rtype: dict
        """
        # Give the default value if the required element is not in the configuration
        if 'filter_size' not in cfg:
            cfg['filter_size'] = self._FILTER_SIZE

        schema = {
            "filter_method": And(str, lambda x: 'median'),
            "filter_size": And(int, lambda x: x >= 1 and x % 2 != 0)
        }

        checker = Checker(schema)
        checker.validate(cfg)
        return cfg

    def desc(self):
        """
        Describes the filtering method
        """
        print('Median filter description')

    def filter_disparity(self, disp: xr.Dataset, img_ref: xr.Dataset = None, img_sec: xr.Dataset = None,
                         cv: xr.Dataset = None) -> xr.Dataset:
        """
        Apply a median filter on valid pixels.
        Invalid pixels are not filtered. If a valid pixel contains an invalid pixel in its filter, the invalid pixel is
        ignored for the calculation of the median.

        :param disp: the disparity map dataset
        :type disp:
            xarray.Dataset with the variables :
                - disparity_map 2D xarray.DataArray (row, col)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
                - validity_mask 2D xarray.DataArray (row, col)
        :param img_ref: reference Dataset image
        :tye img_ref: xarray.Dataset
        :param img_sec: secondary Dataset image
        :type img_sec: xarray.Dataset
        :param cv: cost volume dataset
        :type cv: xarray.Dataset
        :return: the Dataset with the filtered DataArray disparity_map
        :rtype:
            xarray.Dataset with the variables :
                - disparity_map 2D xarray.DataArray (row, col)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
                - validity_mask 2D xarray.DataArray (row, col)
        """
        # Invalid pixels are nan
        masked_data = disp['disparity_map'].copy(deep=True).data
        masked_data[np.where((disp['validity_mask'].data & PANDORA_MSK_PIXEL_INVALID) != 0)] = np.nan
        disp_median = np.copy(masked_data)

        valid = np.isfinite(masked_data)
        ny_, nx_ = masked_data.shape

        # Created a view of each window, by manipulating the internal data structure of array
        # The view allow to looking at the array data in memory in a new way, without additional cost on the memory.
        str_row, str_col = masked_data.strides
        shape_windows = (
            ny_ - (self._filter_size - 1), nx_ - (self._filter_size - 1), self._filter_size, self._filter_size)
        strides_windows = (str_row, str_col, str_row, str_col)
        aggregation_window = np.lib.stride_tricks.as_strided(masked_data, shape_windows, strides_windows)

        radius = int(self._filter_size / 2)

        # To reduce memory, the disparity card is split (along the row axis) into multiple sub-arrays with a step of 100
        disp_chunked_y = np.array_split(aggregation_window, np.arange(100, ny_, 100), axis=0)
        y_begin = radius

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
            # numpy.nanmedian : Compute the median along the specified axis, while ignoring NaNs (i.e if valid pixel
            # contains an invalid pixel in its filter, the invalid pixel is ignored because invalid pixels are nan )
            for y in range(len(disp_chunked_y)):
                # To reduce memory, the disparity card is split (along the col axis) into multiple sub-arrays with a step of 100
                disp_chunked_x = np.array_split(disp_chunked_y[y], np.arange(100, nx_, 100), axis=1)
                x_begin = radius

                for x in range(len(disp_chunked_x)):
                    y_end = y_begin + disp_chunked_y[y].shape[0]
                    x_end = x_begin + disp_chunked_x[x].shape[1]
                    disp_median[y_begin:y_end, x_begin:x_end] = np.nanmedian(disp_chunked_x[x], axis=(2, 3))
                    x_begin += disp_chunked_x[x].shape[1]

                y_begin += disp_chunked_y[y].shape[0]

        disp['disparity_map'].data[valid] = disp_median[valid]
        disp.attrs['filter'] = 'median'
        del disp_median, masked_data, aggregation_window
        return disp
