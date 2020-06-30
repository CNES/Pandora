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
This module contains functions associated to census method used in the cost volume measure step.
"""

import numpy as np
from json_checker import Checker, And
from typing import Dict, Union, Tuple
import xarray as xr

from pandora.stereo import stereo
from pandora.img_tools import shift_sec_img, census_transform


@stereo.AbstractStereo.register_subclass('census')
class Census(stereo.AbstractStereo):
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
        """
        self.cfg = self.check_conf(**cfg)
        self._window_size = self.cfg['window_size']
        self._subpix = self.cfg['subpix']

    def check_conf(self, **cfg: Dict[str, Union[str, int]]) -> Dict[str, Union[str, int]]:
        """
        Add default values to the dictionary if there are missing elements and check if the dictionary is correct

        :param cfg: stereo configuration
        :type cfg: dict
        :return cfg: stereo configuration updated
        :rtype: dict
        """
        # Give the default value if the required element is not in the configuration
        if 'window_size' not in cfg:
            cfg['window_size'] = self._WINDOW_SIZE
        if 'subpix' not in cfg:
            cfg['subpix'] = self._SUBPIX

        schema = {
            "stereo_method": And(str, lambda x: 'census'),
            "window_size": And(int, lambda x: x == 3 or x == 5),
            "subpix": And(int, lambda x: x == 1 or x == 2 or x == 4)
        }

        checker = Checker(schema)
        checker.validate(cfg)
        return cfg

    def desc(self):
        """
        Describes the stereo method
        """
        print('census similarity measure')

    def compute_cost_volume(self, img_ref: xr.Dataset, img_sec: xr.Dataset, disp_min: int, disp_max: int,
                            **cfg: Dict[str, Union[str, int]]) -> xr.Dataset:
        """
        Computes the cost volume for a pair of images with the census measure

        :param img_ref: reference Dataset image
        :type img_ref:
            xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :param img_sec: secondary Dataset image
        :type img_sec:
            xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :param disp_min: minimum disparity
        :type disp_min: int
        :param disp_max: maximum disparity
        :type disp_max: int
        :param cfg: images configuration containing the mask convention : valid_pixels, no_data
        :type cfg: dict
        :return: the cost volume dataset
        :rtype:
            xarray.Dataset, with the data variables:
                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
        """
        # Contains the shifted secondary images
        img_sec_shift = shift_sec_img(img_sec, self._subpix)

        # Maximal cost of the cost volume with census measure
        cmax = int(self._window_size ** 2)

        metadata = {"measure": 'census', "subpixel": self._subpix,
                    "offset_row_col": int((self._window_size - 1) / 2), "window_size": self._window_size,
                    "type_measure": "min", "cmax": cmax}

        # Apply census transformation
        ref = census_transform(img_ref, self._window_size)
        for i in range(0, len(img_sec_shift)):
            img_sec_shift[i] = census_transform(img_sec_shift[i], self._window_size)

        # Disparity range
        if self._subpix == 1:
            disparity_range = np.arange(disp_min, disp_max + 1)
        else:
            disparity_range = np.arange(disp_min, disp_max, step=1 / float(self._subpix))
            disparity_range = np.append(disparity_range, [disp_max])

        # Allocate the numpy cost volume cv = (disp, col, row), for efficient memory management
        cv = np.zeros((len(disparity_range), ref['im'].shape[1], ref['im'].shape[0]), dtype=np.float32)
        cv += np.nan

        # First pixel in the image that is fully computable (aggregation windows are complete)
        offset = int((self._window_size - 1) / 2)
        mask_ref, mask_sec = self.masks_dilatation(img_ref, img_sec, offset, self._window_size, self._subpix, cfg)

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
        # , nan correspond to the first column of the reference image that cannot be calculated,
        #  because there is no corresponding column for the disparity -1.
        #
        # In the following loop, the tuples p,q describe to which part of the image the difference will be applied,
        # for disp = 0, we compute the difference on the whole images so p=(:,0:3) et q=(:,0:3),
        # for disp = -1, we use only a part of the images so p=(:,1:3) et q=(:,0:2)

        # Computes the matching cost
        # In the loop, cv is of shape (disp, col, row) and images / masks of shape (row, col)
        # np.swapaxes allow to interchange row and col in images and masks
        for disp in disparity_range:
            i_sec = int((disp % 1) * self._subpix)
            p, q = self.point_interval(ref, img_sec_shift[i_sec], disp)
            # mask_sec is of size 2
            i_mask_sec = min(1, i_sec)
            d = int((disp - disp_min) * self._subpix)

            cv[d, p[0]:p[1], :] = np.swapaxes(self.census_cost(p, q, ref, img_sec_shift[i_sec]), 0, 1) + \
                np.swapaxes(mask_sec[i_mask_sec].data[:, q[0]:q[1]], 0, 1) + \
                np.swapaxes(mask_ref.data[:, p[0]:p[1]], 0, 1)

        # Create the xarray.DataSet that will contain the cost_volume of dimensions (row, col, disp)
        cv = self.allocate_costvolume(img_ref, self._subpix, disp_min, disp_max, self._window_size, metadata,
                                      np.swapaxes(cv, 0, 2))

        # Remove temporary values
        del mask_ref, mask_sec, ref, img_sec_shift
        return cv

    def census_cost(self, p: Tuple[int, int], q: Tuple[int, int], img_ref: xr.Dataset, img_sec: xr.Dataset) ->\
            np.ndarray:
        """
        Computes xor pixel-wise between pre-processed images by census transform

        :param p: Point interval, in the reference image, over which the squared difference will be applied
        :type p: tuple
        :param q: Point interval, in the secondary image, over which the squared difference will be applied
        :type q: tuple
        :param img_ref: reference Dataset image
        :type img_ref:
            xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk (optional): 2D (row, col) xarray.DataArray
        :param img_sec: secondary Dataset image
        :type img_sec:
            xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk (optional): 2D (row, col) xarray.DataArray
        :return: the xor pixel-wise between elements in the interval
        :rtype: numpy array
        """
        xor_ = img_ref['im'].data[:, p[0]:p[1]].astype('uint32') ^ img_sec['im'].data[:, q[0]:q[1]].astype('uint32')
        return list(map(self.popcount32b, xor_))

    def popcount32b(self, x: int) -> int:
        """
        Computes the Hamming weight for the input x,
        Hamming weight is the number of symbols that are different from the zero

        :param x: 32-bit integer
        :type x: int
        :return: the number of symbols that are different from the zero
        :rtype: int
        """
        x -= (x >> 1) & 0x55555555
        x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
        x = (x + (x >> 4)) & 0x0f0f0f0f
        x += x >> 8
        x += x >> 16
        return x & 0x7f
