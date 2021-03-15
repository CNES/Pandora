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
This module contains functions associated to the multi-scale pyramid method.
"""

import warnings
from typing import Dict, Union, Tuple

import numpy as np
import xarray as xr
from json_checker import Checker, And
from scipy.ndimage.interpolation import zoom

from . import multiscale
from ..common import sliding_window


@multiscale.AbstractMultiscale.register_subclass("fixed_zoom_pyramid")
class FixedZoomPyramid(multiscale.AbstractMultiscale):
    """
    FixedZoomPyramid class, allows to perform the multiscale processing
    """

    _PYRAMID_NUM_SCALES = 2
    _PYRAMID_SCALE_FACTOR = 2
    _PYRAMID_MARGE = 1

    def __init__(self, **cfg: dict):
        """
        :param cfg: optional configuration, {  "num_scales": int, "scale_factor": int, "marge" int}
        :type cfg: dict
        """
        self.cfg = self.check_conf(**cfg)  # type: ignore
        self._num_scales = self.cfg["num_scales"]
        self._scale_factor = self.cfg["scale_factor"]
        self._marge = self.cfg["marge"]

    def check_conf(self, **cfg: Union[str, float, int]) -> Dict[str, Union[str, float, int]]:
        """
        Add default values to the dictionary if there are missing elements and check if the dictionary is correct

        :param cfg: aggregation configuration
        :type cfg: dict
        :return cfg: aggregation configuration updated
        :rtype: dict
        """
        # Give the default value if the required element is not in the configuration
        if "num_scales" not in cfg:
            cfg["num_scales"] = self._PYRAMID_NUM_SCALES
        if "scale_factor" not in cfg:
            cfg["scale_factor"] = self._PYRAMID_SCALE_FACTOR
        if "marge" not in cfg:
            cfg["marge"] = self._PYRAMID_MARGE

        schema = {
            "multiscale_method": And(str, lambda x: "fixed_zoom_pyramid"),
            "num_scales": And(int, lambda x: x > 1),
            "scale_factor": And(int, lambda x: x > 1),
            "marge": And(int, lambda x: x >= 0),
        }

        checker = Checker(schema)
        checker.validate(cfg)
        return cfg

    def desc(self):
        """
        Describes the aggregation method
        """
        print("FixedZoomPyramid method")

    def disparity_range(self, disp: xr.Dataset, disp_min: int, disp_max: int) -> Tuple[np.array, np.array]:
        """
        Disparity range computation by seeking the max and min values in the window.
        Invalid disparities are given the full disparity range

        :param disp: the disparity dataset
        :type disp: xarray.Dataset with the data variables :
                - disparity_map 2D xarray.DataArray (row, col)
                - confidence_measure 3D xarray.DataArray(row, col, indicator)
        :param disp_min: absolute min disparity
        :type disp_min: int
        :param disp_max: absolute max disparity
        :type disp_max: int
        :return: Two np.darray :
                - disp_min_range : minimum disparity value for all pixels.
                - disp_max_range : maximum disparity value for all pixels.

        :rtype: tuple (np.ndarray, np.ndarray)
        """
        ncol, nrow = disp["disparity_map"].shape
        offset = int((disp.attrs["window_size"] - 1) / 2)

        # Initialize ranges on max and min disparity values
        disp_max_range = np.ones((ncol, nrow), dtype=np.float32) * disp_max
        disp_min_range = np.ones((ncol, nrow), dtype=np.float32) * disp_min

        # Set invalid disparities as nan and store its indices
        tmp_disp_map = self.mask_invalid_disparities(disp)
        invalid_ind = np.where(np.isnan(tmp_disp_map))

        # Disparity windows
        disparity_windows = sliding_window(tmp_disp_map, (disp.attrs["window_size"], disp.attrs["window_size"]))
        # Ignore warning in case the window is All-NaN
        warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")

        # To reduce memory, the data array is split (along the row axis) into multiple sub-arrays with a step of 100
        chunk_size = 100
        disp_chunked_y = np.array_split(disparity_windows, np.arange(chunk_size, ncol, chunk_size), axis=0)
        y_begin = offset

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
            # numpy.nanmin/nanmax : Compute the median along the specified axis, while ignoring NaNs
            for col in np.arange(len(disp_chunked_y)):
                # To reduce memory, the data array is split (along the col axis) into multiple sub-arrays,
                # with a step of 100
                disp_chunked_x = np.array_split(disp_chunked_y[col], np.arange(chunk_size, nrow, chunk_size), axis=1)
                x_begin = offset

                for row in np.arange(len(disp_chunked_x)):
                    y_end = y_begin + disp_chunked_y[col].shape[0]
                    x_end = x_begin + disp_chunked_x[row].shape[1]

                    disp_min_range[y_begin:y_end, x_begin:x_end] = (
                        np.nanmin(disp_chunked_x[row], axis=(2, 3)) - self._marge
                    )
                    disp_max_range[y_begin:y_end, x_begin:x_end] = (
                        np.nanmax(disp_chunked_x[row], axis=(2, 3)) + self._marge
                    )
                    x_begin += disp_chunked_x[row].shape[1]

                y_begin += disp_chunked_y[col].shape[0]

        del disparity_windows

        # Indices where disparity was invalid are set the max/min absolute value
        disp_min_range[invalid_ind] = disp_min
        disp_max_range[invalid_ind] = disp_max

        if self._scale_factor == 1:
            return disp_min_range, disp_max_range

        # Upsampling disparity range maps for next pyramid level
        disp_min_range = zoom(disp_min_range, self._scale_factor, order=0)
        disp_max_range = zoom(disp_max_range, self._scale_factor, order=0)

        return disp_min_range, disp_max_range
