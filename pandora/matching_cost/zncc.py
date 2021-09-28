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
This module contains functions associated to ZNCC method used in the cost volume measure step.
"""

from typing import Dict, Union

import numpy as np
import xarray as xr
from json_checker import Checker, And

from pandora.img_tools import shift_right_img, compute_mean_raster, compute_std_raster
from pandora.matching_cost import matching_cost


@matching_cost.AbstractMatchingCost.register_subclass("zncc")
class Zncc(matching_cost.AbstractMatchingCost):
    """
    Zero mean normalized cross correlation
    Zncc class allows to compute the cost volume
    """

    # Default configuration, do not change these values
    _WINDOW_SIZE = 5
    _SUBPIX = 1

    def __init__(self, **cfg: Union[str, int]) -> None:
        """
        :param cfg: optional configuration,  {'window_size': value, 'subpix': value}
        :type cfg: dictionary
        :return: None
        """
        self.cfg = self.check_conf(**cfg)
        self._window_size = self.cfg["window_size"]
        self._subpix = self.cfg["subpix"]

    def check_conf(self, **cfg: Union[str, int]) -> Dict[str, Union[str, int]]:
        """
        Add default values to the dictionary if there are missing elements and check if the dictionary is correct

        :param cfg: matching cost configuration
        :type cfg: dict
        :return cfg: matching cost configuration updated
        :rtype: dict
        """
        # Give the default value if the required element is not in the configuration
        if "window_size" not in cfg:
            cfg["window_size"] = self._WINDOW_SIZE
        if "subpix" not in cfg:
            cfg["subpix"] = self._SUBPIX

        schema = {
            "matching_cost_method": And(str, lambda input: "zncc"),
            "window_size": And(int, lambda input: input > 0 and (input % 2) != 0),
            "subpix": And(int, lambda input: input > 0 and ((input % 2) == 0) or input == 1),
        }

        checker = Checker(schema)
        checker.validate(cfg)
        return cfg

    def desc(self) -> None:
        """
        Describes the matching cost method
        :return: None
        """
        print("zncc similarity measure")

    def compute_cost_volume(
        self, img_left: xr.Dataset, img_right: xr.Dataset, disp_min: int, disp_max: int
    ) -> xr.Dataset:
        """
        Computes the cost volume for a pair of images

        :param img_left: left Dataset image
        :type img_left:
            xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :param img_right: right Dataset image
        :type img_right:
            xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :param disp_min: minimum disparity
        :type disp_min: int
        :param disp_max: maximum disparity
        :type disp_max: int
        :return: the cost volume dataset
        :rtype:
            xarray.Dataset, with the data variables:
                - cost_volume 3D xarray.DataArray (row, col, disp)
        """
        # Contains the shifted right images
        img_right_shift = shift_right_img(img_right, self._subpix)

        # Computes the standard deviation raster for the whole images
        # The standard deviation raster is truncated for points that are not calculable
        img_left_std = compute_std_raster(img_left, self._window_size)
        img_right_std = []
        for i, img in enumerate(img_right_shift):  # pylint: disable=unused-variable
            img_right_std.append(compute_std_raster(img, self._window_size))

        # Computes the mean raster for the whole images
        # The standard mean raster is truncated for points that are not calculable
        img_left_mean = compute_mean_raster(img_left, self._window_size)
        img_right_mean = []
        for i, img in enumerate(img_right_shift):
            img_right_mean.append(compute_mean_raster(img, self._window_size))

        # Maximal cost of the cost volume with zncc measure
        cmax = 1

        # Cost volume metadata
        offset_row_col = int((self._window_size - 1) / 2)
        metadata = {
            "measure": "zncc",
            "subpixel": self._subpix,
            "offset_row_col": offset_row_col,
            "window_size": self._window_size,
            "type_measure": "max",
            "cmax": cmax,
        }

        # Disparity range
        if self._subpix == 1:
            disparity_range = np.arange(disp_min, disp_max + 1)
        else:
            disparity_range = np.arange(disp_min, disp_max, step=1 / float(self._subpix))
            disparity_range = np.append(disparity_range, [disp_max])

        # Allocate the numpy cost volume cv = (disp, col, row), for efficient memory management
        cv = np.zeros(
            (len(disparity_range), img_left["im"].shape[1], img_right["im"].shape[0]),
            dtype=np.float32,
        )
        cv += np.nan

        # If offset, do not consider border position for cost computation
        if offset_row_col != 0:
            cv_crop = cv[:, offset_row_col:-offset_row_col, offset_row_col:-offset_row_col]
        else:
            cv_crop = cv

        # Computes the matching cost
        for disp in disparity_range:
            i_right = int((disp % 1) * self._subpix)
            point_p, point_q = self.point_interval(img_left, img_right_shift[i_right], disp)
            dsp = int((disp - disp_min) * self._subpix)

            # Point interval in the left standard deviation image
            # -  (win_radius * 2) because img_std is truncated for points that are not calculable
            p_std = (point_p[0], point_p[1] - (int(self._window_size / 2) * 2))
            # Point interval in the right standard deviation image
            q_std = (point_q[0], point_q[1] - (int(self._window_size / 2) * 2))

            # Compute the normalized summation of the product of intensities
            zncc_ = (
                img_left["im"].data[:, point_p[0] : point_p[1]]
                * img_right_shift[i_right]["im"].data[:, point_q[0] : point_q[1]]
            )
            zncc_ = xr.Dataset(
                {"im": (["row", "col"], zncc_)},
                coords={
                    "row": np.arange(zncc_.shape[0]),
                    "col": np.arange(zncc_.shape[1]),
                },
            )
            zncc_ = compute_mean_raster(zncc_, self._window_size)
            # Subtracting  the  local mean  value  of  intensities
            zncc_ -= img_left_mean[:, p_std[0] : p_std[1]] * img_right_mean[i_right][:, q_std[0] : q_std[1]]

            # Divide by the standard deviation of the intensities of the images :
            # If the standard deviation of the intensities of the images is greater than 0
            divide_standard = np.multiply(
                img_left_std[:, p_std[0] : p_std[1]],
                img_right_std[i_right][:, q_std[0] : q_std[1]],
            )
            valid = np.where(divide_standard > 0)
            zncc_[valid] /= divide_standard[valid]

            # Otherwise zncc is equal to 0
            zncc_[np.where(divide_standard <= 0)] = 0

            # Places the result in the cost_volume
            cv_crop[dsp, point_p[0] : p_std[1], :] = np.swapaxes(zncc_, 0, 1)

        # Create the xarray.DataSet that will contain the cost_volume of dimensions (row, col, disp)
        cv = self.allocate_costvolume(
            img_left,
            self._subpix,
            disp_min,
            disp_max,
            self._window_size,
            metadata,
            np.swapaxes(cv, 0, 2),
        )

        return cv
