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

    # Default configuration, do not change these values
    _WINDOW_SIZE = 5
    _SUBPIX = 1

    def __init__(self, **cfg: Dict[str, Union[str, int]]) -> None:
        """
        :param cfg: optional configuration,  {'window_size': value, 'subpix': value}
        :type cfg: dict
        :return: None
        """
        self.cfg = self.check_conf(**cfg)
        self._window_size = self.cfg["window_size"]
        self._subpix = self.cfg["subpix"]

    def check_conf(self, **cfg: Dict[str, Union[str, int]]) -> Dict[str, Union[str, int]]:
        """
        Add default values to the dictionary if there are missing elements and check if the dictionary is correct

        :param cfg: matching cost configuration
        :type cfg: dict
        :return cfg: matching cost configuration updated
        :rtype: dict
        """
        # Give the default value if the required element is not in the conf
        if "window_size" not in cfg:
            cfg["window_size"] = self._WINDOW_SIZE  # type: ignore
        if "subpix" not in cfg:
            cfg["subpix"] = self._SUBPIX  # type: ignore

        schema = {
            "matching_cost_method": And(str, lambda input: "census"),
            "window_size": And(int, lambda input: input in (3, 5)),
            "subpix": And(int, lambda input: input > 0 and ((input % 2) == 0) or input == 1),
        }

        checker = Checker(schema)
        checker.validate(cfg)
        return cfg  # type: ignore

    def desc(self) -> None:
        """
        Describes the matching cost method
        :return: None
        """
        print("census similarity measure")

    def compute_cost_volume(
        self, img_left: xr.Dataset, img_right: xr.Dataset, disp_min: int, disp_max: int
    ) -> xr.Dataset:
        """
        Computes the cost volume for a pair of images

        :param img_left: left Dataset image containing :

                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :type img_left: xarray.Dataset
        :param img_right: right Dataset image containing :

                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :type img_right: xarray.Dataset
        :param disp_min: minimum disparity
        :type disp_min: int
        :param disp_max: maximum disparity
        :type disp_max: int
        :return: the cost volume dataset , with the data variables:

                - cost_volume 3D xarray.DataArray (row, col, disp)
        :rtype: xarray.Dataset
        """
        # Contains the shifted right images
        img_right_shift = shift_right_img(img_right, self._subpix)

        # Maximal cost of the cost volume with census measure
        cmax = int(self._window_size ** 2)
        offset_row_col = int((self._window_size - 1) / 2)
        metadata = {
            "measure": "census",
            "subpixel": self._subpix,
            "offset_row_col": offset_row_col,
            "window_size": self._window_size,
            "type_measure": "min",
            "cmax": cmax,
        }

        # Apply census transformation
        left = census_transform(img_left, self._window_size)
        for i, img in enumerate(img_right_shift):
            img_right_shift[i] = census_transform(img, self._window_size)

        # Disparity range # pylint: disable=undefined-variable
        if self._subpix == 1:
            disparity_range = np.arange(disp_min, disp_max + 1)
        else:
            disparity_range = np.arange(disp_min, disp_max, step=1 / float(self._subpix))
            disparity_range = np.append(disparity_range, [disp_max])

        # Allocate the numpy cost volume cv = (disp, col, row), for efficient memory management
        cv = np.zeros(
            (len(disparity_range), img_left["im"].shape[1], img_left["im"].shape[0]),
            dtype=np.float32,
        )
        cv += np.nan

        # If offset, do not consider border position for cost computation
        if offset_row_col != 0:
            cv_crop = cv[:, offset_row_col:-offset_row_col, offset_row_col:-offset_row_col]
        else:
            cv_crop = cv

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
        for disp in disparity_range:
            i_right = int((disp % 1) * self._subpix)
            point_p, point_q = self.point_interval(left, img_right_shift[i_right], disp)
            dsp = int((disp - disp_min) * self._subpix)

            cv_crop[dsp, point_p[0] : point_p[1], :] = np.swapaxes(
                self.census_cost(point_p, point_q, left, img_right_shift[i_right]), 0, 1
            )

        # Create the xarray.DataSet that will contain the cv of dimensions (row, col, disp)
        cv = self.allocate_costvolume(
            img_left,
            self._subpix,
            disp_min,
            disp_max,
            self._window_size,
            metadata,
            np.swapaxes(cv, 0, 2),
        )

        # Remove temporary values
        del left, img_right_shift

        return cv

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

                - im : 2D (row, col) xarray.DataArray
                - msk (optional): 2D (row, col) xarray.DataArray
        :type img_left: xarray.Dataset
        :param img_right: right Dataset image containing :

                - im : 2D (row, col) xarray.DataArray
                - msk (optional): 2D (row, col) xarray.DataArray
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
