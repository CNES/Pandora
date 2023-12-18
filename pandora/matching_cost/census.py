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
This module contains functions associated to census method used in the cost volume measure step.
"""

from typing import Dict, Union, Tuple, List

import numpy as np
import xarray as xr
from json_checker import Checker, And

from pandora.img_tools import shift_right_img, census_transform
from pandora.matching_cost import matching_cost


@matching_cost.AbstractMatchingCost.register_subclass("census")
class Census(matching_cost.AbstractMatchingCost):
    """
    Census class allows to compute the cost volume
    """

    def __init__(self, **cfg: Dict[str, Union[str, int]]) -> None:
        """
        :param cfg: optional configuration,  {'window_size': value, 'subpix': value}
        :type cfg: dict
        :return: None
        """

        super().instantiate_class(**cfg)  # type: ignore

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
        schema["matching_cost_method"] = And(str, lambda input: "census")
        schema["window_size"] = And(int, lambda input: input in (3, 5))

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
        img_right_shift = shift_right_img(img_right, self._subpix, self._band)

        # Maximal cost of the cost volume with census measure
        cmax = int(self._window_size**2)
        cost_volume.attrs.update(
            {
                "type_measure": "min",
                "cmax": cmax,
            }
        )

        # Apply census transformation
        left = census_transform(img_left, self._window_size, self._band)
        for i, img in enumerate(img_right_shift):
            img_right_shift[i] = census_transform(img, self._window_size, self._band)

        disparity_range = cost_volume.coords["disp"].data
        cv = self.allocate_numpy_cost_volume(img_left, disparity_range)
        cv_crop = self.crop_cost_volume(cv, cost_volume.attrs["offset_row_col"])

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
        # In the following loop, the tuples point_p,point_q describe
        # to which part of the image the difference will be applied,
        # for disp = 0, we compute the difference on the whole images so point_p=(:,0:3) et point_q=(:,0:3),
        # for disp = -1, we use only a part of the images so point_p=(:,1:3) et point_q=(:,0:2)

        # Computes the matching cost
        # In the loop, cv is of shape (disp, col, row) and images / masks of shape (row, col)
        # np.swapaxes allow to interchange row and col in images and masks
        for disp_index, disp in enumerate(disparity_range):
            i_right = int((disp % 1) * self._subpix)
            point_p, point_q = self.point_interval(left, img_right_shift[i_right], disp)

            cv_crop[disp_index, point_p[0] : point_p[1], :] = np.swapaxes(
                self.census_cost(point_p, point_q, left, img_right_shift[i_right]), 0, 1
            )

        # Create the xarray.DataSet that will contain the cv of dimensions (row, col, disp)
        # Computations were optimized with a cost_volume of dimensions (disp, row, col)
        # As we are expected to return a cost_volume of dimensions (row, col, disp),
        # we swap axes.
        cv = np.swapaxes(cv, 0, 2)
        index_col = cost_volume.attrs["col_to_compute"]
        index_col = index_col - img_left.coords["col"].data[0]  # If first col coordinate is not 0
        cost_volume["cost_volume"].data = cv[:, index_col, :]

        return cost_volume

    def census_cost(
        self,
        point_p: Tuple[int, int],
        point_q: Tuple[int, int],
        img_left: xr.Dataset,
        img_right: xr.Dataset,
    ) -> List[int]:
        """
        Computes xor pixel-wise between pre-processed images by census transform

        :param point_p: Point interval, in the left image, over which the squared difference will be applied
        :type point_p: tuple
        :param point_q: Point interval, in the right image, over which the squared difference will be applied
        :type point_q: tuple
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
        :return: the xor pixel-wise between elements in the interval
        :rtype: numpy array
        """
        xor_ = img_left["im"].data[:, point_p[0] : point_p[1]].astype("uint32") ^ img_right["im"].data[
            :, point_q[0] : point_q[1]
        ].astype("uint32")
        return list(map(self.popcount32b, xor_))

    @staticmethod
    def popcount32b(row: int) -> int:
        """
        Computes the Hamming weight for the input row,
        Hamming weight is the number of symbols that are different from the zero

        :param row: 32-bit integer
        :type row: int
        :return: the number of symbols that are different from the zero
        :rtype: int
        """
        row -= (row >> 1) & 0x55555555
        row = (row & 0x33333333) + ((row >> 2) & 0x33333333)
        row = (row + (row >> 4)) & 0x0F0F0F0F
        row += row >> 8
        row += row >> 16
        return row & 0x7F
