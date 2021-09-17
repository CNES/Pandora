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
This module contains functions associated to SAD and SSD methods used in the cost volume measure step.
"""

from typing import Dict, Union, Tuple

import numpy as np
import xarray as xr
from json_checker import Checker, And

from pandora.img_tools import shift_right_img
from pandora import common
from pandora.matching_cost import matching_cost


@matching_cost.AbstractMatchingCost.register_subclass("sad", "ssd")
class SadSsd(matching_cost.AbstractMatchingCost):
    """
    SadSsd class allows to compute the cost volume
    """

    # Default configuration, do not change these values
    _WINDOW_SIZE = 5
    _SUBPIX = 1

    def __init__(self, **cfg: Union[str, int]) -> None:
        """
        :param cfg: optional configuration,  {'matching_cost_method': value, 'window_size': value, 'subpix': value}
        :type cfg: dict
        :return: None
        """
        self.cfg = self.check_conf(**cfg)
        self._method = str(self.cfg["matching_cost_method"])
        self._window_size = self.cfg["window_size"]
        self._subpix = self.cfg["subpix"]
        self._pixel_wise_methods = {"sad": self.ad_cost, "ssd": self.sd_cost}

    def check_conf(self, **cfg: Union[str, int]) -> Dict[str, Union[str, int]]:
        """
        Add default values to the dictionary if there are missing elements and check if the dictionary is correct

        :param cfg: matching cost configuration
        :type cfg: dict
        :return cfg: matching cost configuration updated
        :rtype: dict
        """
        # Give the default value if the required element is not in the conf
        if "window_size" not in cfg:
            cfg["window_size"] = self._WINDOW_SIZE
        if "subpix" not in cfg:
            cfg["subpix"] = self._SUBPIX

        schema = {
            "matching_cost_method": And(str, lambda input: common.is_method(input, ["ssd", "sad"])),
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
        print(str(self._method) + " similarity measure")

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

        # Computes the maximal cost of the cost volume
        min_left = np.amin(img_left["im"].data)
        max_left = np.amax(img_left["im"].data)

        min_right = np.amin(img_right["im"].data)
        max_right = np.amax(img_right["im"].data)
        cmax = None

        if self._method == "sad":
            # Maximal cost of the cost volume with sad measure
            cmax = int(max(abs(max_left - min_right), abs(max_right - min_left)) * (self._window_size ** 2))
        if self._method == "ssd":
            # Maximal cost of the cost volume with ssd measure
            cmax = int(max(abs(max_left - min_right) ** 2, abs(max_right - min_left) ** 2) * (self._window_size ** 2))
        offset_row_col = int((self._window_size - 1) / 2)
        metadata = {
            "measure": self._method,
            "subpixel": self._subpix,
            "offset_row_col": offset_row_col,
            "window_size": self._window_size,
            "type_measure": "min",
            "cmax": cmax,
        }

        # Disparity range # pylint: disable=undefined-variable
        if self._subpix == 1:
            disparity_range = np.arange(disp_min, disp_max + 1)
        else:
            disparity_range = np.arange(disp_min, disp_max, step=1 / float(self._subpix))
            disparity_range = np.append(disparity_range, [disp_max])

        # Allocate the numpy cost volume cv = (disp, col, row), for efficient memory management
        # If offset , over allocate the cost volume by adding 2 * offset on row and col dimension
        # The following computation will reduce the dimension during the pixel wise aggregation so the final
        # cv dimension will be equal to the correct one.
        if offset_row_col != 0:
            cv_enlarge = np.zeros(
                (
                    len(disparity_range),
                    img_left["im"].shape[1] + 2 * offset_row_col,
                    img_left["im"].shape[0] + 2 * offset_row_col,
                ),
                dtype=np.float32,
            )
            cv_enlarge += np.nan
            cv = cv_enlarge[:, offset_row_col:-offset_row_col, offset_row_col:-offset_row_col]
        else:
            cv = np.zeros(
                (
                    len(disparity_range),
                    img_left["im"].shape[1],
                    img_left["im"].shape[0],
                ),
                dtype=np.float32,
            )
            cv += np.nan

        # Giving the 2 images, the matching cost will be calculated as :
        #                 1, 1, 1                2, 5, 6
        #                 2, 1, 4                1, 1, 3
        #                 1, 3, 1                6, 1, 2
        #
        # for disp = 0, the costvolume will be
        #                        (1 - 2) (1 - 5) (1 - 6)
        #                        (2 - 1) (1 - 1) (4 - 3)
        #                        (1 - 6) (3 - 1) (1 - 2)
        # , we make the difference between the two full images
        #
        # for disp = -1, compute the difference between
        #                    1, 1                   2, 5
        #                    1, 4                   1, 1
        #                    3, 1                   6, 1
        # , the costvolume will be
        #                        nan  (1-2) (1-5)
        #                        nan  (1-1) (4-1)
        #                        nan  (3-6) (1-1)
        # , nan correspond to the first column of the left image that cannot be calculated,
        #  because there is no corresponding column for the disparity -1.
        #
        # In the following loop, the tuples point_p,
        # point_q describe to which part of the image the difference will be applied,
        # for disp = 0, we compute the difference on the whole images so point_p=(:,0:3) et point_q=(:,0:3),
        # for disp = -1, we use only a part of the images so point_p=(:,1:3) et point_q=(:,0:2)

        # Computes the matching cost
        # In the loop, cv is of shape (disp, col, row) and images / masks of shape (row, col)
        # np.swapaxes allow to interchange row and col in images and masks
        for disp in disparity_range:
            i_right = int((disp % 1) * self._subpix)
            point_p, point_q = self.point_interval(img_left, img_right_shift[i_right], disp)
            dsp = int((disp - disp_min) * self._subpix)

            cv[dsp, point_p[0] : point_p[1], :] = np.swapaxes(
                self._pixel_wise_methods[self._method](point_p, point_q, img_left, img_right_shift[i_right]),
                0,
                1,
            )

        # Pixel wise aggregation modifies border values so it is important to reconvert to nan values
        if offset_row_col != 0:
            cv = self.pixel_wise_aggregation(cv_enlarge.data)
            cv = np.swapaxes(cv, 0, 2)
            cv[:offset_row_col, :, :] = np.nan
            cv[
                -offset_row_col:,
                :,
            ] = np.nan
            cv[:, :offset_row_col, :] = np.nan
            cv[:, -offset_row_col:, :] = np.nan
        else:
            cv = self.pixel_wise_aggregation(cv.data)
            cv = np.swapaxes(cv, 0, 2)

        # Create the xarray.DataSet that will contain the cv of dimensions (row, col, disp)
        cv = self.allocate_costvolume(img_left, self._subpix, disp_min, disp_max, self._window_size, metadata, cv)

        return cv

    @staticmethod
    def ad_cost(
        point_p: Tuple[int, int],
        point_q: Tuple[int, int],
        img_left: xr.Dataset,
        img_right: xr.Dataset,
    ) -> np.ndarray:
        """
        Computes the absolute difference

        :param point_p: Point interval, in the left image, over which the squared difference will be applied
        :type point_p: tuple
        :param point_q: Point interval, in the right image, over which the squared difference will be applied
        :type point_q: tuple
        :param img_left: left Dataset image
        :type img_left:
            xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk (optional): 2D (row, col) xarray.DataArray
        :param img_right: right Dataset image
        :type img_right:
            xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk (optional): 2D (row, col) xarray.DataArray
        :return: the absolute difference pixel-wise between elements in the interval
        :rtype: numpy array
        """
        return abs(img_left["im"].data[:, point_p[0] : point_p[1]] - img_right["im"].data[:, point_q[0] : point_q[1]])

    @staticmethod
    def sd_cost(point_p: Tuple, point_q: Tuple, img_left: xr.Dataset, img_right: xr.Dataset) -> np.ndarray:
        """
        Computes the square difference

        :param point_p: Point interval, in the left image, over which the squared difference will be applied
        :type point_p: tuple
        :param point_q: Point interval, in the right image, over which the squared difference will be applied
        :type point_q: tuple
        :param img_left: left Dataset image
        :type img_left:
            xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk (optional): 2D (row, col) xarray.DataArray
        :param img_right: right Dataset image
        :type img_right:
            xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk (optional): 2D (row, col) xarray.DataArray
        :return: the squared difference pixel-wise between elements in the interval
        :rtype: numpy array
        """
        return (img_left["im"].data[:, point_p[0] : point_p[1]] - img_right["im"].data[:, point_q[0] : point_q[1]]) ** 2

    def pixel_wise_aggregation(self, cost_volume: np.ndarray) -> np.ndarray:
        """
        Summing pixel wise matching cost over square windows

         :param cost_volume: the cost volume
         :type cost_volume: numpy array 3D (disp, col, row)
         :return: the cost volume aggregated
         :rtype: numpy array 3D ( disp, col, row)
        """
        nb_disp, nx_, ny_ = cost_volume.shape

        # Create a sliding window of using as_strided function : this function create a new a view (by manipulating
        # data pointer) of the cost_volume array with a different shape. The new view pointing to the same memory block
        # as cost_volume so it does not consume any additional memory.
        str_disp, str_col, str_row = cost_volume.strides

        shape_windows = (
            self._window_size,
            self._window_size,
            nb_disp,
            nx_ - (self._window_size - 1),
            ny_ - (self._window_size - 1),
        )
        strides_windows = (str_row, str_col, str_disp, str_col, str_row)
        aggregation_window = np.lib.stride_tricks.as_strided(
            cost_volume, shape_windows, strides_windows, writeable=False
        )
        cost_volume = np.sum(aggregation_window, (0, 1))
        return cost_volume
