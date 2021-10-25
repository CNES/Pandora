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
This module contains functions allowing to save the results and the configuration of Pandora pipeline.
"""

import errno
import json
import os
from typing import Dict, Tuple, List, Union
import logging

import numpy as np
import rasterio.dtypes
import xarray as xr

from pandora.output_tree_design import get_out_dir, get_out_file_path
from pandora.img_tools import rasterio_open


def write_data_array(
    data_array: xr.DataArray,
    filename: str,
    dtype: rasterio.dtypes = rasterio.dtypes.float32,
    band_names: List[str] = None,
    crs: Union[rasterio.crs.CRS, None] = None,
    transform: Union[rasterio.Affine, None] = None,
) -> None:
    """
    Write a xarray.DataArray in a tiff file

    :param data_array: data
    :type data_array: 2D xarray.DataArray (row, col) or 3D xarray.DataArray (row, col, indicator)
    :param filename:  output filename
    :type filename: string
    :param dtype: band types
    :type dtype: rasterio.dtypes
    :param band_names: band names
    :type dtype: List[str] or None
    :param crs: coordinate reference support
    :type dtype: rasterio.crs.CRS
    :param transform: geospatial transform matrix
    :type dtype: rasterio.Affine
    :return: None
    """
    if len(data_array.shape) == 2:
        row, col = data_array.shape
        with rasterio_open(
            filename,
            mode="w+",
            driver="GTiff",
            width=col,
            height=row,
            count=1,
            dtype=dtype,
            crs=crs,
            transform=transform,
        ) as source_ds:
            source_ds.write(data_array.data, 1)
    else:
        row, col, depth = data_array.shape
        with rasterio_open(
            filename,
            mode="w+",
            driver="GTiff",
            width=col,
            height=row,
            count=depth,
            dtype=dtype,
            crs=crs,
            transform=transform,
        ) as source_ds:
            for dsp in range(1, depth + 1):
                source_ds.write(data_array.data[:, :, dsp - 1], dsp)
            if band_names is not None:
                source_ds.descriptions = band_names


def mkdir_p(path: str) -> None:
    """
    Create a directory without complaining if it already exists.
    :return: None
    """
    try:
        os.makedirs(path)
    except OSError as exc:  # requires Python > 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def save_results(left: xr.Dataset, right: xr.Dataset, output: str) -> None:
    """
    Save results in the output directory

    :param left: left dataset, which contains the variables :

        - disparity_map : the disparity map in the geometry of the left image 2D DataArray (row, col)
        - confidence_measure : the confidence measure in the geometry of the left image 3D DataArray \
        (row, col, indicator)
        - validity_mask : the validity mask in the geometry of the left image 2D DataArray (row, col)
    :type left: xr.Dataset
    :param right: right dataset. If there is no validation step, the right Dataset will be empty.If a validation step \
    is configured, the dataset will contain the variables :

        - disparity_map: the disparity map in the geometry of the right image 2D DataArray (row, col)
        - confidence_measure: the confidence in the geometry of the right image 3D DataArray (row, col, indicator)
        - validity_mask: the validity mask in the geometry of the left image 2D DataArray (row, col)
    :type right: xr.Dataset
    :param output: output directory
    :type output: string
    :return: None
    """
    # Create the output dir
    mkdir_p(output)
    # Save the left results
    write_data_array(
        left["disparity_map"],
        os.path.join(output, get_out_file_path("left_disparity.tif")),
        crs=left.attrs["crs"],
        transform=left.attrs["transform"],
    )
    if "confidence_measure" in left:
        write_data_array(
            left["confidence_measure"],
            os.path.join(output, get_out_file_path("left_confidence_measure.tif")),
            crs=left.attrs["crs"],
            transform=left.attrs["transform"],
            band_names=left["confidence_measure"]["indicator"].data,
        )
    write_data_array(
        left["validity_mask"],
        os.path.join(output, get_out_file_path("left_validity_mask.tif")),
        dtype=rasterio.dtypes.uint16,
        crs=left.attrs["crs"],
        transform=left.attrs["transform"],
    )

    # If a validation step is configured, save the right results
    if len(right.sizes) != 0:
        write_data_array(
            right["disparity_map"],
            os.path.join(output, get_out_file_path("right_disparity.tif")),
            crs=right.attrs["crs"],
            transform=right.attrs["transform"],
        )
        if "confidence_measure" in right:
            write_data_array(
                right["confidence_measure"],
                os.path.join(output, get_out_file_path("right_confidence_measure.tif")),
                crs=right.attrs["crs"],
                transform=right.attrs["transform"],
            )
        write_data_array(
            right["validity_mask"],
            os.path.join(output, get_out_file_path("right_validity_mask.tif")),
            dtype=rasterio.dtypes.uint16,
            crs=right.attrs["crs"],
            transform=right.attrs["transform"],
        )


def sliding_window(base_array: np.array, shape: Tuple[int, int]) -> np.array:
    """
    Create a sliding window of using as_strided function : this function create a new a view (by manipulating
    data pointer) of the data array with a different shape. The new view pointing to the same memory block as
    data so it does not consume any additional memory.

    :param base_array: the 2D array through which slide the window
    :type base_array: np.array
    :param shape: shape of the sliding window
    :type shape: Tuple[int,int]

    :rtype: np.array
    """
    shp = (base_array.shape[0] - shape[0] + 1,) + (base_array.shape[1] - shape[1] + 1,) + shape
    strides = base_array.strides + base_array.strides
    return np.lib.stride_tricks.as_strided(base_array, shape=shp, strides=strides)


def save_config(output: str, user_cfg: Dict) -> None:
    """
    Save the user configuration in json file

    :param output: Path to output directory
    :type output: string
    :param user_cfg: user configuration
    :type user_cfg: dict
    :return: None
    """

    # Create the output dir
    mkdir_p(os.path.join(output, get_out_dir("config.json")))

    # Save user configuration in json file
    with open(  # pylint:disable=unspecified-encoding
        os.path.join(output, get_out_file_path("config.json")), "w"
    ) as file_:
        json.dump(user_cfg, file_, indent=2)


def is_method(string_method: str, methods: List[str]) -> bool:
    """
    Test if string_method is a method in methods

    :param string_method: String to test
    :type string_method: string
    :param methods: list of available methods
    :type methods: list of strings
    :returns: True if string_method a method and False otherwise
    :rtype: bool
    """

    if string_method in methods:
        return True

    logging.error("% is not in available methods : ", string_method + ", ".join(methods))
    return False
