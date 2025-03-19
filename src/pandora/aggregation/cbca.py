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
This module contains functions associated to the Cross Based Cost Aggregation (cbca) method.
"""

from typing import Dict, Union, Tuple, List

import numpy as np
import xarray as xr
from json_checker import Checker, And

from pandora.filter import AbstractFilter
from pandora.img_tools import shift_right_img
from .cpp import aggregation_cpp
from . import aggregation


@aggregation.AbstractAggregation.register_subclass("cbca")
class CrossBasedCostAggregation(aggregation.AbstractAggregation):
    """
    CrossBasedCostAggregation class, allows to perform the aggregation step
    """

    # Default configuration, do not change these values
    _CBCA_INTENSITY = 30.0
    _CBCA_DISTANCE = 5

    def __init__(self, **cfg: dict):
        """
        :param cfg: optional configuration, {'cbca_intensity': value, 'cbca_distance': value}
        :type cfg: dict
        """
        self.cfg = self.check_conf(**cfg)  # type: ignore
        self._cbca_intensity = self.cfg["cbca_intensity"]
        self._cbca_distance = self.cfg["cbca_distance"]

    def check_conf(self, **cfg: Union[str, float, int]) -> Dict[str, Union[str, float, int]]:
        """
        Add default values to the dictionary if there are missing elements and check if the dictionary is correct

        :param cfg: aggregation configuration
        :type cfg: dict
        :return cfg: aggregation configuration updated
        :rtype: dict
        """
        # Give the default value if the required element is not in the configuration
        if "cbca_intensity" not in cfg:
            cfg["cbca_intensity"] = self._CBCA_INTENSITY
        if "cbca_distance" not in cfg:
            cfg["cbca_distance"] = self._CBCA_DISTANCE

        schema = {
            "aggregation_method": And(str, lambda input: "cbca"),
            "cbca_intensity": And(float, lambda input: input > 0),
            "cbca_distance": And(int, lambda input: input > 0),
        }

        checker = Checker(schema)
        checker.validate(cfg)
        return cfg

    def desc(self):
        """
        Describes the aggregation method
        """
        print("CrossBasedCostAggregation method")

    def cost_volume_aggregation(
        self, img_left: xr.Dataset, img_right: xr.Dataset, cv: xr.Dataset, **cfg: Union[str, int]
    ) -> None:
        """
        Aggregated the cost volume with Cross-Based Cost Aggregation, using the pipeline define in
        Zhang, K., Lu, J., & Lafruit, G. (2009).
        Cross-based local stereo matching using orthogonal integral images.
        IEEE transactions on circuits and systems for video technology, 19(7), 1073-1079.

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
        :param cv: cost volume dataset with the data variables:

                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
        :type cv: xarray.Dataset
        :param cfg: images configuration containing the mask convention : valid_pixels, no_data
        :type cfg: dict
        :return: None
        """
        cross_left, cross_right = self.computes_cross_supports(img_left, img_right, cv)

        offset = int(cv.attrs["offset_row_col"])
        # Cost volume has input image size, if offset > 0 do not consider the marge
        if offset > 0:
            cv_data = cv["cost_volume"].data[offset:-offset, offset:-offset]
        else:
            cv_data = cv["cost_volume"].data
        n_col_, n_row_, nb_disp = cv_data.shape

        # Allocate the numpy aggregated cost volume cv = (disp, col, row), for efficient memory management
        agg = np.zeros((nb_disp, n_row_, n_col_), dtype=np.float32)

        # Add invalid costs (i.e = np.nan ) to the output aggregated cost volume (because the step 1 of cbca do not
        # propagate invalid pixels, we need to retrieve them at the end of aggregation )
        # Much faster than :
        # id_nan = np.isnan(cv['cost_volume'].data)
        # compute the aggregation ..
        # cv['cost_volume'].data[id_nan] = np.nan
        agg += np.swapaxes(cv_data, 0, 2)
        agg *= 0

        disparity_range = cv.coords["disp"].data
        range_col = np.arange(0, n_row_)

        for dsp in range(nb_disp):
            i_right = int((disparity_range[dsp] % 1) * cv.attrs["subpixel"])

            range_col_right = range_col + disparity_range[dsp]
            valid_index = np.where((range_col_right >= 0) & (range_col_right < cross_right[i_right].shape[1]))

            step4, sum4 = aggregation_cpp.cbca(
                cv_data[:, :, dsp],
                cross_left,
                cross_right[i_right],
                range_col[valid_index],
                range_col_right[valid_index].astype(int),
            )

            # Added the pixel anchor pixel to the number of support pixels used during the aggregation
            sum4 += 1
            # Add the aggregate cost to the output
            agg[dsp, :, :] += np.swapaxes(step4, 0, 1)
            # Normalize the aggregated cost
            agg[dsp, :, :] /= np.swapaxes(sum4, 0, 1)

        cv_data = np.swapaxes(agg, 0, 2)
        if offset > 0:
            cv["cost_volume"].data[offset:-offset, offset:-offset] = cv_data
        else:
            cv["cost_volume"].data = cv_data
        cv.attrs["aggregation"] = "cbca"

        # Maximal cost of the cost volume after agregation
        cmax = cv.attrs["cmax"] * ((self._cbca_distance * 2) - 1) ** 2  # type: ignore
        cv.attrs["cmax"] = cmax

    def computes_cross_supports(
        self, img_left: xr.Dataset, img_right: xr.Dataset, cv: xr.Dataset
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Prepare images and compute the cross support region of the left and right images.
        A 3x3 median filter is applied to the images before calculating the cross support region.

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
        :param cv: cost volume dataset with the data variables:

                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
        :type cv: xarray.Dataset
        :return: the left and right cross support region
        :rtype: Tuples(left cross support region, List(right cross support region))
        """
        subpix = cv.attrs["subpixel"]
        offset = int(cv.attrs["offset_row_col"])

        # shift the right image
        img_right_shift = shift_right_img(img_right, subpix)

        # Median filter on valid pixels
        filter_ = AbstractFilter(cfg={"filter_method": "median", "filter_size": 3})  # type: ignore

        # Invalid and no data pixels are masked with np.nan to avoid propagating the values with the median filter
        left_masked = np.copy(img_left["im"].data)
        if "msk" in img_left.data_vars:
            left_masked[np.where(img_left["msk"].data != img_left.attrs["valid_pixels"])] = np.nan

        left_masked = filter_.median_filter(left_masked)  # type: ignore
        # Convert nan to inf to be able to use the comparison operators < and > in cross_support function
        np.nan_to_num(left_masked, copy=False, nan=np.inf)
        # Compute left cross support using C++ to reduce running time
        if offset != 0:
            # Cross support to the size of the cost volume
            cross_left = aggregation_cpp.cross_support(
                left_masked[offset:-offset, offset:-offset],
                self._cbca_distance,
                self._cbca_intensity,
            )
        else:
            cross_left = aggregation_cpp.cross_support(left_masked, self._cbca_distance, self._cbca_intensity)

        # Compute the right cross support. Apply a 3×3 median filter to the input image
        cross_right = []
        for shift, img in enumerate(img_right_shift):
            # Invalid and nodata pixels are masked with np.nan to avoid propagating the values with the median filter
            right_masked = np.copy(img["im"].data)

            # Pixel precision
            if ("msk" in img_right.data_vars) and (shift == 0):
                right_masked[np.where(img_right["msk"].data != img_right.attrs["valid_pixels"])] = np.nan

            # Subpixel precision : computes the shifted right mask
            if ("msk" in img_right.data_vars) and (shift != 0):
                shift_mask = np.zeros(img_right["msk"].data.shape)
                shift_mask[np.where(img_right["msk"].data != img_right.attrs["valid_pixels"])] = np.nan

                # Since the interpolation of the right image is of order 1, the shifted right mask corresponds
                # to an aggregation of two columns of the right mask

                # Create a sliding window of shape 2 using as_strided function : this function create a new a view (by
                # manipulating data pointer)of the shift_mask array with a different shape. The new view pointing to the
                # same memory block as shift_mask so it does not consume any additional memory.
                (  # pylint: disable=unpacking-non-sequence
                    str_row,
                    str_col,
                ) = shift_mask.strides
                shape_windows = (shift_mask.shape[0], shift_mask.shape[1] - 1, 2)
                strides_windows = (str_row, str_col, str_col)
                aggregation_window = np.lib.stride_tricks.as_strided(
                    shift_mask, shape_windows, strides_windows, writeable=False
                )
                shift_mask = np.sum(aggregation_window, 2)
                right_masked += shift_mask

            #  Apply a 3×3 median filter to the input image
            right_masked = filter_.median_filter(right_masked)  # type: ignore
            # Convert nan to inf to be able to use the comparison operators < and > in cross_support function
            np.nan_to_num(right_masked, copy=False, nan=np.inf)
            # Compute right cross support using C++ to reduce running time
            if offset != 0:
                # Cross support to the size of the cost volume
                curr_c_r = aggregation_cpp.cross_support(
                    right_masked[offset:-offset, offset:-offset],
                    self._cbca_distance,
                    self._cbca_intensity,
                )
            else:
                curr_c_r = aggregation_cpp.cross_support(right_masked, self._cbca_distance, self._cbca_intensity)

            cross_right.append(curr_c_r)

        return cross_left, cross_right


cross_support = aggregation_cpp.cross_support
