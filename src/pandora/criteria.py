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
This module contains functions associated to the validity mask created in the cost volume step.
"""
from typing import Union, Tuple

import numpy as np
from scipy.ndimage import binary_dilation
import xarray as xr
import pandora.constants as cst


def binary_dilation_msk(img: xr.Dataset, window_size: int) -> np.ndarray:
    """
    Apply scipy binary_dilation on our image dataset.
    Get the no_data pixels.

    :param img: Dataset image containing :

            - im: 2D (row, col) or 3D (band_im, row, col) xarray.DataArray float32
            - disparity (optional): 3D (disp, row, col) xarray.DataArray float32
            - msk (optional): 2D (row, col) xarray.DataArray int16
            - classif (optional): 3D (band_classif, row, col) xarray.DataArray int16
            - segm (optional): 2D (row, col) xarray.DataArray int16
    :type img: xarray.Dataset
    :param window_size: window size of the cost volume
    :type window_size: int
    :return: np.ndarray with location of pixels that are marked as no_data according to the image mask
    :rtype: np.ndarray
    """

    dil = binary_dilation(
        img["msk"].data == img.attrs["no_data_mask"],
        structure=np.ones((window_size, window_size)),
        iterations=1,
    )

    return dil


def validity_mask(
    img_left: xr.Dataset,
    img_right: xr.Dataset,
    cv: xr.Dataset,
) -> xr.Dataset:
    """
    Create the validity mask of the cost volume

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
            - confidence_measure (optional) 3D xarray.DataArray (row, col, indicator)
    :type cv: xarray.Dataset
    :return: Dataset with the cost volume and the validity_mask with the data variables :

            - cost_volume 3D xarray.DataArray (row, col, disp)
            - confidence_measure 3D xarray.DataArray (row, col, indicator)
            - validity_mask 2D xarray.DataArray (row, col)
    :rtype: xarray.Dataset
    """

    # Allocate the validity mask
    cv["validity_mask"] = xr.DataArray(
        np.full((cv.sizes["row"], cv.sizes["col"]), 0),
        dims=["row", "col"],
    )

    # From the grid_estimation function, which creates the cost volume xarray dataset
    d_min, d_max = cv.coords["disp"].data[[0, -1]]
    col = cv.coords["col"].data

    offset = cv.attrs["offset_row_col"]

    # Negative disparity range
    if d_max < 0:
        bit_1 = np.where((col + d_max) < (col[0] + offset))
        # Information: the disparity interval is incomplete (border reached in the right image)
        cv["validity_mask"].data[
            :,
            np.where(((col + d_max) >= (col[0] + offset)) & ((col + d_min) < (col[0] + offset))),
        ] += cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
    else:
        # Positive disparity range
        if d_min > 0:
            bit_1 = np.where((col + d_min) > (col[-1] - offset))
            # Information: the disparity interval is incomplete (border reached in the right image)
            cv["validity_mask"].data[
                :,
                np.where(((col + d_min) <= (col[-1] - offset)) & ((col + d_max) > (col[-1] - offset))),
            ] += cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE

        # Disparity range contains 0
        else:
            bit_1 = ([],)  # type: ignore
            # Information: the disparity interval is incomplete (border reached in the right image)
            cv["validity_mask"].data[
                :,
                np.where(((col + d_min) < (col[0] + offset)) | (col + d_max > (col[-1]) - offset)),
            ] += cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE

    # Invalid pixel : the disparity interval is missing in the right image ( disparity range
    # outside the image )
    cv["validity_mask"].data[:, bit_1] += cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING

    if "msk" in img_left.data_vars:
        allocate_left_mask(cv, img_left)

    if "msk" in img_right.data_vars:
        allocate_right_mask(cv, img_right, bit_1)

    return cv


def allocate_left_mask(cv: xr.Dataset, img_left: xr.Dataset) -> None:
    """
    Allocate the left image mask

    :param cv: cost volume dataset with the data variables:

            - cost_volume 3D xarray.DataArray (row, col, disp)
            - confidence_measure (optional) 3D xarray.DataArray (row, col, indicator)
    :type cv: xarray.Dataset
    :param img_left: left Dataset image containing :

            - im: 2D (row, col) or 3D (band_im, row, col) xarray.DataArray float32
            - disparity (optional): 3D (disp, row, col) xarray.DataArray float32
            - msk (optional): 2D (row, col) xarray.DataArray int16
            - classif (optional): 3D (band_classif, row, col) xarray.DataArray int16
            - segm (optional): 2D (row, col) xarray.DataArray int16
    :type img_left: xarray.Dataset
    :return: None
    """

    _, r_mask = xr.align(cv["validity_mask"], img_left["msk"])  # pylint: disable=unbalanced-tuple-unpacking

    # Dilatation : pixels that contains no_data in their aggregation window become no_data
    dil = binary_dilation_msk(img_left, cv.attrs["window_size"])

    # Invalid pixel : no_data in the left image
    cv["validity_mask"] += dil.astype(np.uint16) * cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER

    # Invalid pixel : invalidated by the validity mask of the left image given as input
    cv["validity_mask"] += xr.where(
        (r_mask != img_left.attrs["no_data_mask"]) & (r_mask != img_left.attrs["valid_pixels"]),
        cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT,
        0,
    ).astype(np.uint16)


def allocate_right_mask(cv: xr.Dataset, img_right: xr.Dataset, bit_1: Union[np.ndarray, Tuple]) -> None:
    """
    Allocate the right image mask

    :param cv: cost volume dataset with the data variables:

            - cost_volume 3D xarray.DataArray (row, col, disp)
            - confidence_measure (optional) 3D xarray.DataArray (row, col, indicator)
    :type cv: xarray.Dataset
    :param img_right: right Dataset image containing :

            - im: 2D (row, col) or 3D (band_im, row, col) xarray.DataArray float32
            - disparity (optional): 3D (disp, row, col) xarray.DataArray float32
            - msk (optional): 2D (row, col) xarray.DataArray int16
            - classif (optional): 3D (band_classif, row, col) xarray.DataArray int16
            - segm (optional): 2D (row, col) xarray.DataArray int16
    :type img_right: xarray.Dataset
    :param bit_1: where the disparity interval is missing in the right image ( disparity range outside the image )
    :type: ndarray or Tuple
    :return: None
    """

    offset = cv.attrs["offset_row_col"]

    _, r_mask = xr.align(cv["validity_mask"], img_right["msk"])  # pylint: disable=unbalanced-tuple-unpacking
    d_min, d_max = cv.coords["disp"].data[[0, -1]].astype(int)

    # Dilatation : pixels that contains no_data in their aggregation window become no_data
    dil = binary_dilation_msk(img_right, cv.attrs["window_size"])

    r_mask = xr.where(
        (r_mask != img_right.attrs["no_data_mask"]) & (r_mask != img_right.attrs["valid_pixels"]),
        1,
        0,
    ).data

    # Useful to calculate the case where the disparity interval is incomplete, and all remaining right
    # positions are invalidated by the right mask
    b_2_7 = np.full((cv.sizes["row"], cv.sizes["col"]), 0)
    # Useful to calculate the case where no_data in the right image invalidated the disparity interval
    no_data_right = np.full((cv.sizes["row"], cv.sizes["col"]), 0)

    col_range = np.arange(cv.sizes["col"])
    for dsp in range(d_min, d_max + 1):
        # Diagonal in the cost volume
        col_d = col_range + dsp
        valid_index = np.where((col_d >= col_range[0] + offset) & (col_d <= col_range[-1] - offset))

        # No_data and masked pixels do not raise the same flag, we need to treat them differently
        b_2_7[:, col_range[valid_index]] += r_mask[:, col_d[valid_index]].astype(np.uint16)
        b_2_7[:, col_range[np.setdiff1d(col_range, valid_index)]] += 1

        no_data_right[:, col_range[valid_index]] += dil[:, col_d[valid_index]]
        no_data_right[:, col_range[np.setdiff1d(col_range, valid_index)]] += 1

        # Exclusion of pixels that have flag 1 already enabled
        b_2_7[:, bit_1[0]] = 0
        no_data_right[:, bit_1[0]] = 0

        # Invalid pixel: right positions invalidated by the mask of the right image given as input
        cv["validity_mask"].data[
            np.where(b_2_7 == len(range(d_min, d_max + 1)))
        ] += cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_RIGHT

        # If Invalid pixel : the disparity interval is missing in the right image (disparity interval
        # is invalidated by no_data in the right image )
        cv["validity_mask"].data[
            np.where(no_data_right == len(range(d_min, d_max + 1)))
        ] += cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING


def mask_invalid_variable_disparity_range(cv: xr.Dataset) -> None:
    """
    Mask the pixels that have a missing disparity range, searching in the cost volume
    the pixels where cost_volume(row,col, for all d) = np.nan

    :param cv: cost volume dataset with the data variables:

            - cost_volume 3D xarray.DataArray (row, col, disp)
            - confidence_measure (optional) 3D xarray.DataArray (row, col, indicator)
    :type cv: xarray.Dataset
    :return: None
    """

    indices_nan = np.isnan(cv["cost_volume"].data)
    missing_disparity_range = np.min(indices_nan, axis=2)
    missing_range_y, missing_range_x = np.where(missing_disparity_range)

    # Mask the positions which have an missing disparity range, not already taken into account
    condition_to_mask = (
        cv["validity_mask"].data[missing_range_y, missing_range_x]
        & cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING
        == 0
    )
    masking_value = (
        cv["validity_mask"].data[missing_range_y, missing_range_x]
        + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING
    )
    no_masking_value = cv["validity_mask"].data[missing_range_y, missing_range_x]

    cv["validity_mask"].data[missing_range_y, missing_range_x] = np.where(
        condition_to_mask, masking_value, no_masking_value
    )


def mask_border(dataset: xr.Dataset) -> xr.DataArray:
    """
    Mask border pixel  which haven't been calculated because of the window's size

    :param dataset: dataset that can be :

    - the cost volume, the confidence measure and the validity_mask with the data variables :
            - cost_volume 3D xarray.DataArray (row, col, disp)
            - confidence_measure (optional) 3D xarray.DataArray (row, col, indicator)
            - validity_mask 2D xarray.DataArray (row, col)
    - the disparity_map, the confidence measure and the validity mask with the data variables :
            - disparity_map 2D xarray.DataArray (row, col)
            - confidence_measure (optional) 3D xarray.DataArray (row, col, indicator)
            - validity_mask 2D xarray.DataArray (row, col)

    :type dataset: xarray.Dataset
    :return: DataArray with the updated validity_mask
    :rtype: xarray.Dataset
    """

    offset = dataset.attrs["offset_row_col"]

    # Border pixels have invalid disparity, erase the potential previous values
    dataset["validity_mask"].data[:offset, :] = cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER
    dataset["validity_mask"].data[-offset:, :] = cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER
    dataset["validity_mask"].data[offset:-offset, :offset] = cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER
    dataset["validity_mask"].data[offset:-offset, -offset:] = cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER

    return dataset["validity_mask"]
