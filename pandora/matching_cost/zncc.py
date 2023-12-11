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

from typing import Dict, Union, Tuple, List

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

    def __init__(self, **cfg: Union[str, int]) -> None:
        """
        :param cfg: optional configuration,  {'window_size': value, 'subpix': value}
        :type cfg: dictionary
        :return: None
        """
        super().instantiate_class(**cfg)

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
        schema["matching_cost_method"] = And(str, lambda input: "zncc")
        schema["window_size"] = And(int, lambda input: input > 0 and (input % 2) != 0)

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
        self, img_left: xr.Dataset, img_right: xr.Dataset, grid_disp_min: np.ndarray, grid_disp_max: np.ndarray
    ) -> xr.Dataset:
        """
        Computes the cost volume for a pair of images

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
        :param grid_disp_min: minimum disparity
        :type grid_disp_min: np.ndarray
        :param grid_disp_max: maximum disparity
        :type grid_disp_max: np.ndarray
        :return: the cost volume dataset
        :rtype:
            xarray.Dataset, with the data variables:
                - cost_volume 3D xarray.DataArray (row, col, disp)
        """
        # check band parameter
        self.check_band_input_mc(img_left, img_right)

        # Contains the shifted right images
        img_right_shift = shift_right_img(img_right, self._subpix, self._band)

        # Computes the standard deviation raster for the whole images
        # The standard deviation raster is truncated for points that are not calculable
        img_left_std = compute_std_raster(img_left, self._window_size, self._band)
        img_right_std = [compute_std_raster(img, self._window_size, self._band) for img in img_right_shift]

        # Computes the mean raster for the whole images
        # The standard mean raster is truncated for points that are not calculable
        img_left_mean = compute_mean_raster(img_left, self._window_size, self._band)
        img_right_mean = [compute_mean_raster(img, self._window_size, self._band) for img in img_right_shift]

        # Obtain absolute min and max disparities
        disp_min, disp_max = self.get_min_max_from_grid(grid_disp_min, grid_disp_max)
        disparity_range = self.get_disparity_range(disp_min, disp_max, self._subpix)
        cv = self.allocate_numpy_cost_volume(img_left, disparity_range)

        offset_row_col = int((self._window_size - 1) / 2)
        cv_crop = self.crop_cost_volume(cv, offset_row_col)

        left_data = self.extract_image_data(img_left)

        # Computes the matching cost
        for disp_index, disp in enumerate(disparity_range):
            i_right = int((disp % 1) * self._subpix)
            shifted_right_image = img_right_shift[i_right]
            point_p, point_q = self.point_interval(img_left, shifted_right_image, disp)
            right_data = self.extract_image_data(shifted_right_image)

            zncc_ = left_data[:, slice(*point_p)] * right_data[:, slice(*point_q)]

            zncc_ = xr.Dataset(
                {"im": (["row", "col"], zncc_)},
                coords={
                    "row": np.arange(zncc_.shape[0]),
                    "col": np.arange(zncc_.shape[1]),
                },
            )
            zncc_ = compute_mean_raster(zncc_, self._window_size, self._band)

            p_std = self.std_point_interval(point_p)
            q_std = self.std_point_interval(point_q)

            # Subtracting  the  local mean  value  of  intensities
            zncc_ -= img_left_mean[:, p_std[0] : p_std[1]] * img_right_mean[i_right][:, q_std[0] : q_std[1]]

            # Divide by the standard deviation of the intensities of the images
            apply_divide_standard(zncc_, img_left_std, img_right_std, p_std, q_std, i_right)

            # Places the result in the cost_volume
            cv_crop[disp_index, point_p[0] : p_std[1], :] = np.swapaxes(zncc_, 0, 1)

        # Cost volume metadata
        metadata = {
            "measure": "zncc",
            "subpixel": self._subpix,
            "offset_row_col": offset_row_col,
            "window_size": self._window_size,
            "type_measure": "max",
            "cmax": 1,  # Maximal cost of the cost volume with zncc measure
            "band_correl": self._band,
        }
        # Create the xarray.DataSet that will contain the cost_volume of dimensions (row, col, disp)
        # Computations were optimized with a cost_volume of dimensions (disp, row, col)
        # As we are expected to return a cost_volume of dimensions (row, col, disp),
        # we swap axes.
        cv = self.allocate_costvolume(
            img_left, self._subpix, disp_min, disp_max, self._window_size, metadata, np.swapaxes(cv, 0, 2)
        )

        return cv

    def std_point_interval(self, point_interval: Tuple[int, int]) -> Tuple[int, int]:
        """
        Compute standard deviation point interval.

        (win_radius * 2) because img_std is truncated for points that are not calculable
        :param point_interval: point interval
        :type point_interval: Tuple[int, int]
        :return: standard deviation point interval
        :rtype: Tuple[int, int]
        """
        return point_interval[0], point_interval[1] - (int(self._window_size / 2) * 2)

    def extract_image_data(self, image_dataset: xr.Dataset) -> np.ndarray:
        """
        Extract image data from dataset taking band into account if relevant.
        :param image_dataset:
        :type image_dataset:
        :return:
        :rtype:
        """
        try:
            return image_dataset.sel(band_im=self._band)["im"].data
        except KeyError:
            # Raised when there is no band_im coordinates or when self._band is None
            return image_dataset["im"].data


def apply_divide_standard(
    zncc: np.ndarray,
    img_left: np.ndarray,
    img_right: List[np.ndarray],
    p_std: Tuple[int, int],
    q_std: Tuple[int, int],
    i_right: int,
):
    """
    Divide by the standard deviation of the intensities of the images

    :param zncc:
    :type zncc: np.ndarray
    :param img_left: standard deviation raster of left image
    :type img_left: np.ndarray
    :param img_right: standard deviation raster list of right image (for each subpix)
    :type img_right: List[np.ndarray]
    :param p_std: point interval in the left standard deviation image
    :type p_std: Tuple[int, int]
    :param q_std: Point interval in the right standard deviation image
    :type q_std: Tuple[int, int]
    :param i_right: ith image
    :type i_right: int
    """
    # If the standard deviation of the intensities of the images is greater than 0
    divide_standard = np.multiply(
        img_left[:, p_std[0] : p_std[1]],
        img_right[i_right][:, q_std[0] : q_std[1]],
    )
    valid = np.where(divide_standard > 0)
    zncc[valid] /= divide_standard[valid]

    # Otherwise zncc is equal to 0
    zncc[np.where(divide_standard <= 0)] = 0
