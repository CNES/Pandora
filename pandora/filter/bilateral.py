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
This module contains functions associated to the bilateral filter used to filter the disparity map.
"""

import warnings
from typing import Dict, Union

import numpy as np
import xarray as xr
from json_checker import Checker, And

import pandora.constants as cst
from . import filter  # pylint: disable=redefined-builtin
from ..common import sliding_window


@filter.AbstractFilter.register_subclass("bilateral")
class BilateralFilter(filter.AbstractFilter):
    """
    BilateralFilter class allows to perform the filtering step
    """

    # Default configuration, do not change these values
    _SIGMA_COLOR = 2.0
    _SIGMA_SPACE = 6.0

    def __init__(self, **cfg: Union[str, float]):
        """
        :param cfg: optional configuration, {'sigmaColor' : value, 'sigmaSpace' : value}
        :type cfg: dict
        """
        self.cfg = self.check_conf(**cfg)
        self._sigma_color = float(self.cfg["sigma_color"])
        self._sigma_space = float(self.cfg["sigma_space"])

    def check_conf(self, **cfg: Union[str, float]) -> Dict[str, Union[str, float]]:
        """
        Add default values to the dictionary if there are missing elements and check if the dictionary is correct

        :param cfg: filter configuration
        :type cfg: dict
        :return cfg: filter configuration updated
        :rtype: dict
        """
        # Give the default value if the required element is not in the configuration
        if "sigma_color" not in cfg:
            cfg["sigma_color"] = self._SIGMA_COLOR
        if "sigma_space" not in cfg:
            cfg["sigma_space"] = self._SIGMA_SPACE

        schema = {
            "filter_method": And(str, lambda input: "bilateral"),
            "sigma_color": And(float, lambda input: input > 0),
            "sigma_space": And(float, lambda input: input > 0),
        }

        checker = Checker(schema)
        checker.validate(cfg)
        return cfg

    def desc(self):
        """
        Describes the filtering method
        """
        print("Bilateral filter description")

    def filter_disparity(
        self,
        disp: xr.Dataset,
        img_left: xr.Dataset = None,
        img_right: xr.Dataset = None,
        cv: xr.Dataset = None,
    ) -> None:
        """
        Apply bilateral filter.
        Filter size is computed from sigmaSpace

        :param disp: the disparity map dataset  with the variables :

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
        masked_data = disp["disparity_map"].copy(deep=True).data
        masked_data[np.where((disp["validity_mask"].data & cst.PANDORA_MSK_PIXEL_INVALID) != 0)] = np.nan

        valid = np.isfinite(masked_data)
        disp_bilateral = self.filter_bilateral(masked_data, self._sigma_space, self._sigma_color)

        disp["disparity_map"].data[valid] = disp_bilateral[valid]
        disp.attrs["filter"] = "bilateral"
        del (
            disp_bilateral,
            masked_data,
        )

    def filter_bilateral(self, data: np.ndarray, sigma_space: float, sigma_color: float) -> np.ndarray:
        """
        Apply bilateral filter on valid pixels (pixels that are not nan).
        Invalid pixels are not filtered. If a valid pixel contains an invalid pixel in its filter, the invalid pixel is
        ignored for the calculation of the median.

        :param data: input data to be filtered
        :type data: 2D np.array (row, col)
        :param sigma_space: spatial sigma value
        :type sigma_space: float
        :param sigma_color: color sigma value
        :type sigma_color: float
        :return: The gaussian spatial kernel
        :rtype: 2D np.array(row, col)
        :return: The filtered array
        :rtype: 2D np.array(row, col)
        """
        data_bilateral = np.copy(data)
        invalid = np.isnan(data_bilateral)
        ny_, nx_ = data.shape

        # Window width is obtained from the spatial sigma
        win_width = min(min(ny_, nx_), int(3 * sigma_space + 1))
        offset = int(win_width / 2)
        # Obtain all filter windows
        filter_windows = sliding_window(data, (win_width, win_width))
        # Obtain gaussian spatial kernel
        gauss_spatial_kernel = self.gauss_spatial_kernel(win_width, sigma_space)

        # To reduce memory, the data array is split (along the row axis) into multiple sub-arrays with a step of 50
        chunk_size = 50
        disp_chunked_y = np.array_split(filter_windows, np.arange(chunk_size, ny_, chunk_size), axis=0)
        y_begin = offset

        with warnings.catch_warnings():
            for disp_y in disp_chunked_y:
                # To reduce memory, the data array is split (along the col axis) into multiple sub-arrays,
                # with a step of 50
                disp_chunked_x = np.array_split(disp_y, np.arange(chunk_size, nx_, chunk_size), axis=1)
                x_begin = offset

                for disp_x in disp_chunked_x:
                    y_end = y_begin + disp_y.shape[0]
                    x_end = x_begin + disp_x.shape[1]
                    data_bilateral[y_begin:y_end, x_begin:x_end] = self.bilateral_kernel(
                        disp_x, gauss_spatial_kernel, sigma_color, offset
                    )
                    x_begin += disp_x.shape[1]

                y_begin += disp_y.shape[0]

        del filter_windows

        data_bilateral[invalid] = np.nan
        return data_bilateral

    def gauss_spatial_kernel(self, kernel_size: int, sigma: float) -> np.ndarray:
        """
        Compute gaussian spatial kernel

        :param kernel_size: Kernel size
        :type kernel_size: float
        :param sigma: sigma value
        :type sigma: float
        :return: The gaussian spatial kernel
        :rtype: 2D np.array(row, col)
        """
        arr = np.zeros((kernel_size, kernel_size))
        for [i, j], val in np.ndenumerate(arr):  # pylint:disable=unused-variable
            arr[i, j] = np.sqrt(abs(i - kernel_size // 2) ** 2 + abs(j - kernel_size // 2) ** 2)
        return self.normalized_gaussian(arr, sigma)

    @staticmethod
    def normalized_gaussian(array: np.ndarray, sigma: float) -> np.ndarray:
        """
        Apply normalized gaussian to the input array
        :param array: input array
        :type array: 2D np.array
        :param sigma: sigma
        :type sigma: float
        :return: The filtered array
        :rtype: 2D np.array
        """
        return np.exp(-((array / sigma) ** 2) * 0.5) / (sigma * np.sqrt(2 * np.pi))

    def bilateral_kernel(
        self,
        windows: np.ndarray,
        gauss_spatial_kernel: np.ndarray,
        sigma_color: float,
        offset: int,
    ) -> np.ndarray:
        """
        Bilateral filtering on each window.

        :param windows: batch of windows to be filtered
        :type windows: 4D np.array (stepsize, stepsize, windowsize, windowsize)
        :param gauss_spatial_kernel: gaussian spatial kernel
        :type gauss_spatial_kernel: 2D np.array (row, col)
        :param sigma_color: color sigma value
        :type sigma_color: float
        :param offset: distance to window's center
        :type offset: int
        :return: The filtered pixels
        :rtype: 2D np.array(stepsize, stepsize)
        """
        # Intensity difference kernel. For each window, its center is in (offset, offset) coordinates
        int_kernel = np.transpose(np.transpose(windows) - np.transpose(windows[:, :, offset, offset]))
        # Apply gaussian to Intensity difference kernel
        gauss_int_kernel = self.normalized_gaussian(int_kernel, sigma_color)
        # Multiply gaussian spatial kernel and gaussian intensity kernel to obtain weights
        weights = np.multiply(gauss_spatial_kernel, gauss_int_kernel)
        # Multiply kernels by its pixel
        pixel_weights = np.multiply(windows, weights)
        del gauss_int_kernel, int_kernel, windows
        # Return filtered pixel
        warnings.filterwarnings("ignore", r"invalid value encountered in true_divide")
        return np.nansum(pixel_weights, axis=(2, 3)) / np.nansum(weights, axis=(2, 3))
