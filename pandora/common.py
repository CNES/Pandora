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
This module contains functions allowing to save the results and the configuration of Pandora pipeline.
"""

import json
import errno
import os
import rasterio
import xarray as xr
from typing import Dict, Union
import numpy as np

from .output_tree_design import get_out_dir, get_out_file_path
from pandora.constants import *


def write_data_array(data_array: xr.DataArray, filename: str,
                     dtype: rasterio.dtypes = rasterio.dtypes.float32) -> None:
    """
    Write a xarray.DataArray in a tiff file

    :param data_array: data
    :type data_array: 2D xarray.DataArray (row, col) or 3D xarray.DataArray (row, col, indicator)
    :param filename:  output filename
    :type filename: string
    :param dtype: band types
    :type dtype: GDALDataType
    """
    if len(data_array.shape) == 2:
        row, col = data_array.shape
        with rasterio.open(filename, mode='w+', driver='GTiff', width=col, height=row, count=1,
                           dtype=dtype) as source_ds:
            source_ds.write(data_array.data, 1)

    else:
        row, col, depth = data_array.shape
        with rasterio.open(filename, mode='w+', driver='GTiff', width=col, height=row, count=depth, dtype=dtype) as source_ds:
            for d in range(1, depth + 1):
                source_ds.write(data_array.data[:, :, d-1], d)


def mkdir_p(path: str) -> None:
    """
    Create a directory without complaining if it already exists.
    """
    try:
        os.makedirs(path)
    except OSError as exc:   # requires Python > 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def save_results(ref: xr.Dataset, sec: xr.Dataset, output: str) -> None:
    """
    Save results in the output directory

    :param ref: reference dataset, which contains the variables :
                - disparity_map : the disparity map in the geometry of the reference image 2D DataArray (row, col)
                - confidence_measure : the confidence measure in the geometry of the reference image 3D DataArray (row, col, indicator)
                - validity_mask : the validity mask in the geometry of the reference image 2D DataArray (row, col)
    :type ref: xr.Dataset
    :param sec: secondary dataset. If there is no validation step, the secondary Dataset will be empty.
                If a validation step is configured, the dataset will contain the variables :
                - disparity_map : the disparity map in the geometry of the secondary image 2D DataArray (row, col)
                - confidence_measure : the confidence in the geometry of the secondary image 3D DataArray (row, col, indicator)
                - validity_mask : the validity mask in the geometry of the reference image 2D DataArray (row, col)
    :type sec: xr.Dataset
    :param output: output directory
    :type output: string
    """
    # Create the output dir
    mkdir_p(output)

    # Save the reference results
    write_data_array(ref['disparity_map'], os.path.join(output, get_out_file_path('ref_disparity.tif')))
    write_data_array(ref['confidence_measure'], os.path.join(output, get_out_file_path('ref_confidence_measure.tif')))
    write_data_array(ref['validity_mask'], os.path.join(output, get_out_file_path('ref_validity_mask.tif')),
                     dtype=rasterio.dtypes.uint16)

    # If a validation step is configured, save the secondary results
    if len(sec.sizes) != 0:
        write_data_array(sec['disparity_map'], os.path.join(output, get_out_file_path('sec_disparity.tif')))
        write_data_array(sec['confidence_measure'], os.path.join(output, get_out_file_path('sec_confidence_measure.tif')))
        write_data_array(sec['validity_mask'], os.path.join(output, get_out_file_path('sec_validity_mask.tif')),
                         dtype=rasterio.dtypes.uint16)


def save_config(output: str, user_cfg: Dict) -> None:
    """
    Save the user configuration in json file

    :param output: Path to output directory
    :type output: string
    :param user_cfg: user configuration
    :type user_cfg: dict
    """
    
    # Create the output dir
    mkdir_p(os.path.join(output, get_out_dir('config.json')))

    # Save user configuration in json file
    with open(os.path.join(output, get_out_file_path('config.json')), 'w') as f:
        json.dump(user_cfg, f, indent=2)


def resize(dataset: xr.Dataset, border_disparity: Union[int, float]) -> xr.Dataset:
    """
    Pixels whose aggregation window exceeds the reference image are truncated in the output products.
    This function returns the output products with the size of the input images : add rows and columns that
    have been
    truncated. These added pixels will have bit 0 = 1 ( Invalid pixel : border of the reference image )
    in the validity_mask  and will have the disparity = invalid_value in the disparity map.

    :param dataset: Dataset which contains the output products
    :type dataset: xarray.Dataset with the variables :
        - disparity_map 2D xarray.DataArray (row, col)
        - confidence_measure 3D xarray.DataArray(row, col, indicator)
        - validity_mask 2D xarray.DataArray (row, col)
    :param border_disparity: disparity to assign to border pixels
    :type border_disparity: float or int
    :return: the dataset with the size of the input images
    :rtype : xarray.Dataset with the variables :
        - disparity_map 2D xarray.DataArray (row, col)
        - confidence_measure 3D xarray.DataArray(row, col, indicator)
        - validity_mask 2D xarray.DataArray (row, col)
    """
    offset = dataset.attrs['offset_row_col']

    if offset == 0:
        return dataset

    c_row = dataset.coords['row']
    c_col = dataset.coords['col']

    row = np.arange(c_row[0] - offset, c_row[-1] + 1 + offset)
    col = np.arange(c_col[0] - offset, c_col[-1] + 1 + offset)

    resize_disparity = xr.Dataset()

    for array in dataset:
        if array == 'disparity_map':
            data = xr.DataArray(np.full((len(row), len(col)), border_disparity, dtype=np.float32),
                                coords=[row, col],
                                dims=['row', 'col'])
            resize_disparity[array] = dataset[array].combine_first(data)

        if array == 'confidence_measure':
            depth = len(dataset.coords['indicator'])
            data = xr.DataArray(data=np.full((len(row), len(col), depth), np.nan, dtype=np.float32),
                                coords={'row': row, 'col': col}, dims=['row', 'col', 'indicator'])
            resize_disparity[array] = dataset[array].combine_first(data)

        if array == 'validity_mask':
            data = xr.DataArray(np.zeros((len(row), len(col)), dtype=np.uint16), coords=[row, col],
                                dims=['row', 'col'])

            # Invalid pixel : border of the reference image
            data += PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER

            resize_disparity[array] = dataset[array].combine_first(data).astype(np.uint16)

        if array == 'interpolated_coeff':
            data = xr.DataArray(np.full((len(row), len(col)), np.nan, dtype=np.float32), coords=[row, col],
                                dims=['row', 'col'])
            resize_disparity[array] = dataset[array].combine_first(data)

    resize_disparity.attrs = dataset.attrs
    resize_disparity.attrs['offset_row_col'] = 0

    return resize_disparity
