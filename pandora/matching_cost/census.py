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
This module contains functions associated to census method used in the cost volume measure step.
"""

from typing import Dict, Union, Tuple, List

import numpy as np
import xarray as xr
from json_checker import Checker, And

from .cpp import matching_cost_cpp

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
        imgs_right = shift_right_img(img_right, self._subpix, self._band, self._spline_order)

        # Maximal cost of the cost volume with census measure
        cmax = int(self._window_size**2)
        cost_volume.attrs.update(
            {
                "type_measure": "min",
                "cmax": cmax,
            }
        )

        if self._band is None:
            img_left_np = img_left["im"].data
        else:
            band_index_right = list(img_right.band_im.data).index(band)
            img_left_np = img_right["im"].data[band_index_right, :, :]

        cv = matching_cost_cpp.compute_matching_costs(
            img_left_np.astype(np.float32),
            [img["im"].data.astype(np.float32) for img in imgs_right],
            cost_volume["cost_volume"].data,
            cost_volume["disp"].data,
            self._window_size,
            self._window_size
        )

        index_col = cost_volume.attrs["col_to_compute"]
        index_col = index_col - img_left.coords["col"].data[0]  # If first col coordinate is not 0
        cost_volume["cost_volume"].data = cv[:, index_col, :]

        return cost_volume
