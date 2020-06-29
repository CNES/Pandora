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
This module contains functions associated to ZNCC method used in the cost volume measure step.
"""

import numpy as np
import xarray as xr
from json_checker import Checker, And
from typing import Dict, Union

from pandora.img_tools import shift_sec_img, compute_mean_raster, compute_std_raster
from . import stereo


@stereo.AbstractStereo.register_subclass('zncc')
class Zncc(stereo.AbstractStereo):
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
        """
        self.cfg = self.check_conf(**cfg)
        self._window_size = self.cfg['window_size']
        self._subpix = self.cfg['subpix']

    def check_conf(self, **cfg: Union[str, int]) -> Dict[str, Union[str, int]]:
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
            "stereo_method": And(str, lambda x: 'zncc'),
            "window_size": And(int, lambda x: x > 0 and (x % 2) != 0),
            "subpix": And(int, lambda x: x == 1 or x == 2 or x == 4)
        }

        checker = Checker(schema)
        checker.validate(cfg)
        return cfg

    def desc(self):
        """
        Describes the stereo method
        """
        print('zncc similarity measure')

    def compute_cost_volume(self, img_ref: xr.Dataset, img_sec: xr.Dataset, disp_min: int, disp_max: int,
                            **cfg: Union[str, int]) -> xr.Dataset:
        """
        Computes the cost volume for a pair of images

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

        # Computes the standard deviation raster for the whole images
        # The standard deviation raster is truncated for points that are not calculable
        img_ref_std = compute_std_raster(img_ref, self._window_size)
        img_sec_std = []
        for i in range(0, len(img_sec_shift)):
            img_sec_std.append(compute_std_raster(img_sec_shift[i], self._window_size))

        # Computes the mean raster for the whole images
        # The standard mean raster is truncated for points that are not calculable
        img_ref_mean = compute_mean_raster(img_ref, self._window_size)
        img_sec_mean = []
        for i in range(0, len(img_sec_shift)):
            img_sec_mean.append(compute_mean_raster(img_sec_shift[i], self._window_size))

        # Maximal cost of the cost volume with zncc measure
        cmax = 1

        # Cost volume metadata
        metadata = {"measure": 'zncc', "subpixel": self._subpix,
                    "offset_row_col": int((self._window_size - 1) / 2), "window_size": self._window_size,
                    "type_measure": "max", "cmax": cmax}

        # Disparity range
        if self._subpix == 1:
            disparity_range = np.arange(disp_min, disp_max + 1)
        else:
            disparity_range = np.arange(disp_min, disp_max, step=1 / float(self._subpix))
            disparity_range = np.append(disparity_range, [disp_max])

        # Allocate the numpy cost volume cv = (disp, col, row), for efficient memory management
        cv = np.zeros((len(disparity_range), img_ref['im'].shape[1] - (self._window_size - 1),
                       img_sec['im'].shape[0] - (self._window_size - 1)), dtype=np.float32)
        cv += np.nan

        mask_ref, mask_sec = self.masks_dilatation(img_ref, img_sec_shift[0], int((self._window_size - 1) / 2),
                                                   self._window_size, self._subpix, cfg)

        # Computes the matching cost
        for disp in disparity_range:
            i_sec = int((disp % 1) * self._subpix)
            # mask_sec is of size 2
            i_mask_sec = min(1, i_sec)
            d = int((disp - disp_min) * self._subpix)

            p, q = self.point_interval(img_ref, img_sec_shift[i_sec], disp)

            # Point interval in the reference standard deviation image
            # -  (win_radius * 2) because img_std is truncated for points that are not calculable
            p_std = (p[0], p[1] - (int(self._window_size / 2) * 2))
            # Point interval in the secondary standard deviation image
            q_std = (q[0], q[1] - (int(self._window_size / 2) * 2))

            # Compute the normalized summation of the product of intensities
            zncc_ = img_ref['im'].data[:, p[0]:p[1]] * img_sec_shift[i_sec]['im'].data[:, q[0]:q[1]]
            zncc_ = xr.Dataset({'im': (['row', 'col'], zncc_)},
                               coords={'row': np.arange(zncc_.shape[0]), 'col': np.arange(zncc_.shape[1])})
            zncc_ = compute_mean_raster(zncc_, self._window_size)
            # Subtracting  the  local mean  value  of  intensities
            zncc_ -= (img_ref_mean[:, p_std[0]:p_std[1]] * img_sec_mean[i_sec][:, q_std[0]:q_std[1]])

            # Divide by the standard deviation of the intensities of the images :
            # If the standard deviation of the intensities of the images is greater than 0
            divide_standard = np.multiply(img_ref_std[:, p_std[0]:p_std[1]], img_sec_std[i_sec][:, q_std[0]:q_std[1]])
            valid = np.where(divide_standard > 0)
            zncc_[valid] /= divide_standard[valid]
            # Else, zncc = 0
            zncc_[np.where(divide_standard <= 0)] = 0

            # Places the result in the cost_volume
            # We use p_std indices because cost_volume and img_std have the same shape,
            # they are both truncated for points that are not calculable
            cv[d, p[0]:p_std[1], :] = np.swapaxes(zncc_, 0, 1) + \
                                      np.swapaxes(mask_sec[i_mask_sec].data[:, q_std[0]:q_std[1]], 0, 1) + \
                                      np.swapaxes(mask_ref.data[:, p_std[0]:p_std[1]], 0, 1)

        # Create the xarray.DataSet that will contain the cost_volume of dimensions (row, col, disp)
        cv = self.allocate_costvolume(img_ref, self._subpix, disp_min, disp_max, self._window_size, metadata,
                                      np.swapaxes(cv, 0, 2))

        return cv
