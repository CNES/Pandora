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
This module contains functions associated to SAD and SSD methods used in the cost volume measure step.
"""

from typing import Dict, Union, Tuple, List

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

    def __init__(self, **cfg: Union[str, int]) -> None:
        """
        :param cfg: optional configuration,  {'matching_cost_method': value, 'window_size': value, 'subpix': value}
        :type cfg: dict
        :return: None
        """

        super().instantiate_class(**cfg)
        self._pixel_wise_methods = {"sad": self.ad_cost, "ssd": self.sd_cost}

    def check_conf(self, **cfg: Dict[str, Union[str, int]]) -> Dict[str, Union[str, int]]:
        """
        Add default values to the dictionary if there are missing elements and check if the dictionary is correct

        :param cfg: matching cost configuration
        :type cfg: dict
        :return cfg: matching cost configuration updated
        :rtype: dict
        """
        cfg = super().check_conf(**cfg)

        schema = self.schema
        schema["matching_cost_method"] = And(str, lambda input: common.is_method(input, ["ssd", "sad"]))
        schema["window_size"] = And(int, lambda input: input > 0 and (input % 2) != 0)

        checker = Checker(schema)
        checker.validate(cfg)
        return cfg

    def compute_cost_volume(
        self,
        img_left: xr.Dataset,
        img_right: xr.Dataset,
        cost_volume: xr.Dataset,
    ) -> xr.Dataset:
        """
        Computes the cost volume for a pair of images

        :param img_left: left Dataset image containing :

                - im: 2D (row, col) or 3D (band_im, row, col) xarray.DataArray float32
                - disparity (optional): 3D (disp, row, col) xarray.DataArray float32
                - msk (optional): 2D (row, col) xarray.DataArray int16
                - classif (optional): 3D (band_classif, row, col) xarray.DataArray int16
                - segm (optional): 2D (row, col) xarray.DataArray int16
        :type img_left: xarray.Dataset
        :param img_right: right Dataset image containing :

                - im: 2D (row, col) or 3D (band_im, row, col) xarray.DataArray float32
                - disparity (optional): 3D (disp, row, col) xarray.DataArray float32
                - msk (optional): 2D (row, col) xarray.DataArray int16
                - classif (optional): 3D (band_classif, row, col) xarray.DataArray int16
                - segm (optional): 2D (row, col) xarray.DataArray int16
        :type img_right: xarray.Dataset
        :param cost_volume: an empty cost volume
        :type cost_volume: xr.Dataset
        :return: the cost volume dataset , with the data variables:

                - cost_volume 3D xarray.DataArray (row, col, disp)
        :rtype: xarray.Dataset
        """
        # check band parameter
        self.check_band_input_mc(img_left, img_right)

        # Contains the shifted right images
        img_right_shift = shift_right_img(img_right, self._subpix, self._band, self._spline_order)
        if self._band is not None:
            band_index_left = list(img_left.band_im.data).index(self._band)
            band_index_right = list(img_right.band_im.data).index(self._band)
            selected_band_right = img_right["im"].data[band_index_right, :, :]
            selected_band_left = img_left["im"].data[band_index_left, :, :]
        else:
            selected_band_right = img_right["im"].data
            selected_band_left = img_left["im"].data

        # Computes the maximal cost of the cost volume
        min_left = np.amin(selected_band_left)
        max_left = np.amax(selected_band_left)

        min_right = np.amin(selected_band_right)
        max_right = np.amax(selected_band_right)
        cmax = None

        if self._method == "sad":
            # Maximal cost of the cost volume with sad measure
            cmax = int(max(abs(max_left - min_right), abs(max_right - min_left)) * (self._window_size**2))
        if self._method == "ssd":
            # Maximal cost of the cost volume with ssd measure
            cmax = int(max(abs(max_left - min_right) ** 2, abs(max_right - min_left) ** 2) * (self._window_size**2))
        offset_row_col = cost_volume.attrs["offset_row_col"]
        cost_volume.attrs.update(
            {
                "type_measure": "min",
                "cmax": cmax,
            }
        )

        disparity_range = cost_volume.coords["disp"].data
        cv_enlarge = self.allocate_numpy_cost_volume(img_left, disparity_range, offset_row_col)
        cv = self.crop_cost_volume(cv_enlarge, offset_row_col)

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
        for disp_index, disp in enumerate(disparity_range):
            i_right = int((disp % 1) * self._subpix)
            point_p, point_q = self.point_interval(img_left, img_right_shift[i_right], disp)

            cv[disp_index, point_p[0] : point_p[1], :] = np.swapaxes(
                self._pixel_wise_methods[self._method](point_p, point_q, img_left, img_right_shift[i_right]),
                0,
                1,
            )

        cv = self.pixel_wise_aggregation(cv_enlarge.data if offset_row_col else cv.data)  # type: ignore

        # Computations were optimized with a cost_volume of dimensions (disp, row, col)
        # As we are expected to return a cost_volume of dimensions (row, col, disp),
        # we swap axes.
        cv = np.swapaxes(cv, 0, 2)
        index_col = cost_volume.attrs["col_to_compute"]
        index_col = index_col - img_left.coords["col"].data[0]  # If first col coordinate is not 0

        if offset_row_col:
            # Pixel wise aggregation modifies border values so it is important to reconvert to nan values
            cv[:offset_row_col, :, :] = np.nan
            cv[-offset_row_col:, :, :] = np.nan
            cv[:, :offset_row_col, :] = np.nan
            cv[:, -offset_row_col:, :] = np.nan

        cost_volume["cost_volume"].data = cv[:, index_col, :]
        return cost_volume

    def allocate_numpy_cost_volume(
        self, img_left: xr.Dataset, disparity_range: Union[np.ndarray, List], offset_row_col: int = 0
    ) -> np.ndarray:
        # Allocate the numpy cost volume cv = (disp, col, row), for efficient memory management
        # If offset , over allocate the cost volume by adding 2 * offset on row and col dimension
        # The following computation will reduce the dimension during the pixel wise aggregation so the final
        # cv dimension will be equal to the correct one.
        return np.full(
            (
                len(disparity_range),
                int((img_left.sizes["col"] + 2 * offset_row_col)),
                int((img_left.sizes["row"] + 2 * offset_row_col)),
            ),
            np.nan,
            dtype=np.float32,
        )

    def ad_cost(
        self,
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
                - im: 2D (row, col) or 3D (band_im, row, col) xarray.DataArray float32
                - disparity (optional): 3D (disp, row, col) xarray.DataArray float32
                - msk (optional): 2D (row, col) xarray.DataArray int16
                - classif (optional): 3D (band_classif, row, col) xarray.DataArray int16
                - segm (optional): 2D (row, col) xarray.DataArray int16
        :param img_right: right Dataset image
        :type img_right:
            xarray.Dataset containing :
                - im: 2D (row, col) or 3D (band_im, row, col) xarray.DataArray float32
                - disparity (optional): 3D (disp, row, col) xarray.DataArray float32
                - msk (optional): 2D (row, col) xarray.DataArray int16
                - classif (optional): 3D (band_classif, row, col) xarray.DataArray int16
                - segm (optional): 2D (row, col) xarray.DataArray int16
        :return: the absolute difference pixel-wise between elements in the interval
        :rtype: numpy array
        """
        if self._band is not None:
            band_index_left = list(img_left.band_im.data).index(self._band)
            band_index_right = list(img_right.band_im.data).index(self._band)
            # Right image can have 3 dim if its from dataset or 2 if its from shift_right_image function
            if len(img_right["im"].data.shape) > 2:
                cost = abs(
                    img_left["im"].data[
                        band_index_left,
                        :,
                        point_p[0] : point_p[1],
                    ]
                    - img_right["im"].data[band_index_right, :, point_q[0] : point_q[1]]
                )
            else:
                cost = abs(
                    img_left["im"].data[band_index_left, :, point_p[0] : point_p[1]]
                    - img_right["im"].data[:, point_q[0] : point_q[1]]
                )
        else:
            cost = abs(
                img_left["im"].data[:, point_p[0] : point_p[1]] - img_right["im"].data[:, point_q[0] : point_q[1]]
            )
        return cost

    def sd_cost(self, point_p: Tuple, point_q: Tuple, img_left: xr.Dataset, img_right: xr.Dataset) -> np.ndarray:
        """
        Computes the square difference

        :param point_p: Point interval, in the left image, over which the squared difference will be applied
        :type point_p: tuple
        :param point_q: Point interval, in the right image, over which the squared difference will be applied
        :type point_q: tuple
        :param img_left: left Dataset image
        :type img_left:
            xarray.Dataset containing :
                - im: 2D (row, col) or 3D (band_im, row, col) xarray.DataArray float32
                - disparity (optional): 3D (disp, row, col) xarray.DataArray float32
                - msk (optional): 2D (row, col) xarray.DataArray int16
                - classif (optional): 3D (band_classif, row, col) xarray.DataArray int16
                - segm (optional): 2D (row, col) xarray.DataArray int16
        :param img_right: right Dataset image
        :type img_right:
            xarray.Dataset containing :
                - im: 2D (row, col) or 3D (band_im, row, col) xarray.DataArray float32
                - disparity (optional): 3D (disp, row, col) xarray.DataArray float32
                - msk (optional): 2D (row, col) xarray.DataArray int16
                - classif (optional): 3D (band_classif, row, col) xarray.DataArray int16
                - segm (optional): 2D (row, col) xarray.DataArray int16
        :return: the squared difference pixel-wise between elements in the interval
        :rtype: numpy array
        """

        if self._band is not None:
            band_index_left = list(img_left.band_im.data).index(self._band)
            band_index_right = list(img_right.band_im.data).index(self._band)
            # Right image can have 3 dim if its from dataset or 2 if its from shift_right_image function
            if len(img_right["im"].data.shape) > 2:
                cost = (
                    img_left["im"].data[
                        band_index_left,
                        :,
                        point_p[0] : point_p[1],
                    ]
                    - img_right["im"].data[band_index_right, :, point_q[0] : point_q[1]]
                ) ** 2
            else:
                cost = (
                    img_left["im"].data[band_index_left, :, point_p[0] : point_p[1]]
                    - img_right["im"].data[:, point_q[0] : point_q[1]]
                ) ** 2
        else:
            cost = (
                img_left["im"].data[:, point_p[0] : point_p[1]] - img_right["im"].data[:, point_q[0] : point_q[1]]
            ) ** 2

        return cost

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
