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
This module contains functions associated to the disparity map computation step.
"""

import xarray as xr
import numpy as np
from scipy.ndimage.morphology import binary_dilation

from pandora.img_tools import compute_std_raster
from pandora.constants import *


def to_disp(cv: xr.Dataset, invalid_value: float = 0, img_ref: xr.Dataset = None, img_sec: xr.Dataset = None) -> xr.Dataset:
    """
    Disparity computation by applying the Winner Takes All strategy

    :param cv: the cost volume datset
    :type cv:
        xarray.Dataset, with the data variables:
            - cost_volume 3D xarray.DataArray (row, col, disp)
            - confidence_measure 3D xarray.DataArray (row, col, indicator)
    :param invalid_value: disparity to assign to invalid pixels
    :type invalid_value: float
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
    :return: Dataset with the disparity map and the confidence measure
    :rtype:
        xarray.Dataset with the data variables :
            - disparity_map 2D xarray.DataArray (row, col)
            - confidence_measure 3D xarray.DataArray (row, col, indicator)
    """
    indices_nan = np.isnan(cv['cost_volume'].data)

    # Winner Takes All strategy
    if cv.attrs['type_measure'] == 'max':
        # Disparities are computed by selecting the maximal cost at each pixel
        cv['cost_volume'].data[indices_nan] = -np.inf
        disp = argmax_split(cv)
    else:
        # Disparities are computed by selecting the minimal cost at each pixel
        cv['cost_volume'].data[indices_nan] = np.inf
        disp = argmin_split(cv)

    cv['cost_volume'].data[indices_nan] = np.nan
    row = cv.coords['row']
    col = cv.coords['col']

    # ----- Disparity map -----
    disp_map = xr.Dataset({'disparity_map': (['row', 'col'], disp)}, coords={'row': row, 'col': col})

    invalid_mc = np.min(indices_nan, axis=2)
    # Pixels where the disparity interval is missing in the secondary image, have a disparity value invalid_value
    invalid_pixel = np.where(invalid_mc == True)
    disp_map['disparity_map'].data[invalid_pixel] = invalid_value

    # Save the disparity map in the cost volume
    cv['disp_indices'] = disp_map['disparity_map'].copy(deep=True)

    disp_map.attrs = cv.attrs
    d_range = cv.coords['disp'].data
    disp_map.attrs['disp_min'] = d_range[0]
    disp_map.attrs['disp_max'] = d_range[-1]

    # ----- Confidence measure -----
    # Allocate the confidence measure in the disparity_map dataset
    disp_map['confidence_measure'] = cv['confidence_measure']

    # Remove temporary values
    del indices_nan
    del invalid_mc

    return disp_map


def validity_mask(disp: xr.Dataset, img_ref: xr.Dataset, img_sec: xr.Dataset, cv: xr.Dataset = None, **cfg: int) -> xr.Dataset:
    """
    Create the validity mask of the disparity map

    :param disp: dataset with the disparity map and the confidence measure
    :type disp:
        xarray.Dataset with the data variables :
            - disparity_map 2D xarray.DataArray (row, col)
            - confidence_measure 3D xarray.DataArray(row, col, indicator)
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
    :param cv: cost volume dataset
    :type cv:
        xarray.Dataset, with the data variables:
            - cost_volume 3D xarray.DataArray (row, col, disp)
            - confidence_measure 3D xarray.DataArray (row, col, indicator)
    :param cfg: images configuration containing the mask convention : valid_pixels, no_data
    :type cfg: dict
    :return: the dataset disparity with the data variable validity_mask
    :rtype :
        xarray.Dataset with the data variables :
            - disparity_map 2D xarray.DataArray (row, col)
            - confidence_measure 3D xarray.DataArray (row, col, indicator)
            - validity_mak 2D xarray.DataArray (row, col)
    """
    # Allocate the validity mask
    disp['validity_mask'] = xr.DataArray(np.zeros(disp['disparity_map'].shape, dtype=np.uint16), dims=['row', 'col'])

    d_min = int(disp.attrs['disp_min'])
    d_max = int(disp.attrs['disp_max'])
    col = disp.coords['col'].data
    row = disp.coords['row'].data

    # Negative disparity range
    if d_max < 0:
        bit_1 = np.where((col + d_max) < col[0])
        # Information: the disparity interval is incomplete (border reached in the secondary image)
        disp['validity_mask'].data[:, np.where(((col + d_max) >= col[0]) & ((col + d_min) < col[0]))] +=\
            PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE
    else:
        # Positive disparity range
        if d_min > 0:
            bit_1 = np.where((col + d_min) > col[-1])
            # Information: the disparity interval is incomplete (border reached in the secondary image)
            disp['validity_mask'].data[:, np.where(((col + d_min) <= col[-1]) & ((col + d_max) > col[-1]))] +=\
                PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE

        # Disparity range contains 0
        else:
            bit_1 = ([], )
            # Information: the disparity interval is incomplete (border reached in the secondary image)
            disp['validity_mask'].data[:, np.where(((col + d_min) < col[0]) | (col + d_max > col[-1]))] +=\
                PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE

    # Invalid pixel : the disparity interval is missing in the secondary image ( disparity range
    # outside the image )
    disp['validity_mask'].data[:, bit_1] += PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING

    if 'msk' in img_ref.data_vars:
        _, r_mask = xr.align(disp['validity_mask'], img_ref['msk'])

        # Dilatation : pixels that contains no_data in their aggregation window become no_data
        dil = binary_dilation(img_ref['msk'].data == cfg['no_data'],
                              structure=np.ones((disp.attrs['window_size'], disp.attrs['window_size'])), iterations=1)
        offset = disp.attrs['offset_row_col']
        if offset != 0:
            dil = dil[offset:-offset, offset:-offset]

        # Invalid pixel : no_data in the reference image
        disp['validity_mask'] += dil.astype(np.uint16) * PANDORA_MSK_PIXEL_REF_NODATA_OR_BORDER

        # Invalid pixel : invalidated by the validity mask of the reference image given as input
        disp['validity_mask'] += xr.where((r_mask != cfg['no_data']) & (r_mask != cfg['valid_pixels']),
                                          PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_REF, 0).astype(np.uint16)

    if 'msk' in img_sec.data_vars:
        _, r_mask = xr.align(disp['validity_mask'], img_sec['msk'])

        # Dilatation : pixels that contains no_data in their aggregation window become no_data
        dil = binary_dilation(img_sec['msk'].data == cfg['no_data'],
                              structure=np.ones((disp.attrs['window_size'], disp.attrs['window_size'])), iterations=1)
        offset = disp.attrs['offset_row_col']
        if offset != 0:
            dil = dil[offset:-offset, offset:-offset]

        r_mask = xr.where((r_mask != cfg['no_data']) & (r_mask != cfg['valid_pixels']), 1, 0).data

        # Useful to calculate the case where the disparity interval is incomplete, and all remaining secondary
        # positions are invalidated by the secondary mask
        b_2_7 = np.zeros((len(row), len(col)), dtype=np.uint16)
        # Useful to calculate the case where no_data in the secondary image invalidated the disparity interval
        no_data_sec = np.zeros((len(row), len(col)), dtype=np.uint16)

        col_range = np.arange(len(col))
        for d in range(d_min, d_max+1):
            # Diagonal in the cost volume
            col_d = col_range + d
            valid_index = np.where((col_d >= col_range[0]) & (col_d <= col_range[-1]))

            # No_data and masked pixels do not raise the same flag, we need to treat them differently
            b_2_7[:, col_range[valid_index]] += r_mask[:, col_d[valid_index]].astype(np.uint16)
            b_2_7[:, col_range[np.setdiff1d(col_range, valid_index)]] += 1

            no_data_sec[:, col_range[valid_index]] += dil[:, col_d[valid_index]]
            no_data_sec[:, col_range[np.setdiff1d(col_range, valid_index)]] += 1

        # Exclusion of pixels that have flag 1 already enabled
        b_2_7[:, bit_1[0]] = 0
        no_data_sec[:, bit_1[0]] = 0

        # Invalid pixel: secondary positions invalidated by the mask of the secondary image given as input
        disp['validity_mask'].data[np.where(b_2_7 == len(range(d_min, d_max+1)))] +=\
            PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_SEC

        # If Invalid pixel : the disparity interval is missing in the secondary image (disparity interval
        # is invalidated by no_data in the secondary image )
        disp['validity_mask'].data[np.where(no_data_sec == len(range(d_min, d_max + 1)))] +=\
            PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING

    return disp


def argmin_split(cost_volume: xr.Dataset) -> np.ndarray:
    """
    Find the indices of the minimum values for a 3D DataArray, along axis 2.
    Memory consumption is reduced by splitting the 3D Array.

    :param cost_volume: the cost volume dataset
    :type cost_volume: xarray.Dataset
    :return: the disparities for which the cost volume values are the smallest
    :rtype: np.ndarray
    """
    ny, nx, nd = cost_volume['cost_volume'].shape
    disp = np.zeros((ny, nx), dtype=np.float32)

    # Numpy argmin is making a copy of the cost volume.
    # To reduce memory, numpy argmin is applied on a small part of the cost volume.
    # The cost volume is split (along the row axis) into multiple sub-arrays with a step of 100
    cv_chunked_y = np.array_split(cost_volume['cost_volume'].data, np.arange(100, ny, 100), axis=0)

    y_begin = 0

    for y in range(len(cv_chunked_y)):
        # To reduce memory, the cost volume is split (along the col axis) into multiple sub-arrays with a step of 100
        cv_chunked_x = np.array_split(cv_chunked_y[y], np.arange(100, nx, 100), axis=1)
        x_begin = 0
        for x in range(len(cv_chunked_x)):

            disp[y_begin:y_begin + cv_chunked_y[y].shape[0], x_begin: x_begin + cv_chunked_x[x].shape[1]] = \
                cost_volume.coords['disp'].data[np.argmin(cv_chunked_x[x], axis=2)]
            x_begin += cv_chunked_x[x].shape[1]

        y_begin += cv_chunked_y[y].shape[0]

    return disp


def argmax_split(cost_volume: xr.Dataset) -> np.ndarray:
    """
    Find the indices of the maximum values for a 3D DataArray, along axis 2.
    Memory consumption is reduced by splitting the 3D Array.

    :param cost_volume: the cost volume dataset
    :type cost_volume: xarray.Dataset
    :return: the disparities for which the cost volume values are the highest
    :rtype: np.ndarray
    """
    ny, nx, nd = cost_volume['cost_volume'].shape
    disp = np.zeros((ny, nx), dtype=np.float32)

    # Numpy argmax is making a copy of the cost volume.
    # To reduce memory, numpy argmax is applied on a small part of the cost volume.
    # The cost volume is split (along the row axis) into multiple sub-arrays with a step of 100
    cv_chunked_y = np.array_split(cost_volume['cost_volume'].data, np.arange(100, ny, 100), axis=0)

    y_begin = 0

    for y in range(len(cv_chunked_y)):
        # To reduce memory, the cost volume is split (along the col axis) into multiple sub-arrays with a step of 100
        cv_chunked_x = np.array_split(cv_chunked_y[y], np.arange(100, nx, 100), axis=1)
        x_begin = 0
        for x in range(len(cv_chunked_x)):

            disp[y_begin:y_begin + cv_chunked_y[y].shape[0], x_begin: x_begin + cv_chunked_x[x].shape[1]] = \
                cost_volume.coords['disp'].data[np.argmax(cv_chunked_x[x], axis=2)]
            x_begin += cv_chunked_x[x].shape[1]

        y_begin += cv_chunked_y[y].shape[0]

    return disp


def coefficient_map(cv: xr.DataArray) -> xr.DataArray:
    """
    Return the coefficient map

    :param cv: cost volume
    :type cv: xarray.Dataset, with the data variables cost_volume 3D xarray.DataArray (row, col, disp)
    :return: the coefficient map
    :rtype : 2D DataArray (row, col)
    """
    row = cv.coords['row']
    col = cv.coords['col']

    # Create the coefficient map
    coeff_map = xr.DataArray(cv['cost_volume'].sel(disp=cv['disp_indices']).astype(np.float32),
                             coords=[('row', row), ('col', col)])
    coeff_map.name = 'Coefficient Map'
    coeff_map.attrs = cv.attrs

    return coeff_map


def approximate_right_disparity(cv: xr.Dataset, img_sec: xr.Dataset, invalid_value: float = 0,
                                img_ref: xr.Dataset = None) -> xr.Dataset:
    """
    Create the right disparity map, by a diagonal search for the minimum in the reference cost volume

    ERNST, Ines et HIRSCHMÜLLER, Heiko.
    Mutual information based semi-global stereo matching on the GPU.
    In : International Symposium on Visual Computing. Springer, Berlin, Heidelberg, 2008. p. 228-239.

    :param cv: the cost volume dataset
    :type cv:
        xarray.Dataset, with the data variables:
            - cost_volume 3D xarray.DataArray (row, col, disp)
            - confidence_measure 3D xarray.DataArray (row, col, indicator)
    :param img_sec: secondary Dataset image
    :type img_sec:
        xarray.Dataset containing :
            - im : 2D (row, col) xarray.DataArray
            - msk : 2D (row, col) xarray.DataArray
    :param invalid_value: disparity to assign to invalid pixels
    :type invalid_value: float
    :param img_ref: reference Dataset image
    :type img_ref:
        xarray.Dataset containing :
            - im : 2D (row, col) xarray.DataArray
            - msk : 2D (row, col) xarray.DataArray
    :return: Dataset with the secondary disparity map, the confidence measure and the validity mask
    :rtype:
        xarray.Dataset with the data variables :
            - disparity_map 2D xarray.DataArray (row, col)
            - confidence_measure 3D xarray.DataArray (row, col, indicator)
            - validity_mask 2D xarray.DataArray (row, col)
    """
    disp_range = cv.coords['disp'].data.astype(float)
    col_range = cv.coords['col'].data
    row_range = cv.coords['row'].data
    # Extract integer disparity
    disp_range = np.extract(np.mod(disp_range, 1) == 0, disp_range)

    # Allocate the disparity map
    data = np.zeros((len(row_range), len(col_range))).astype(np.float32)
    disp_map = xr.Dataset({'disparity_map': (['row', 'col'], data)},
                          coords={'row': cv.coords['row'], 'col': cv.coords['col']})

    # Allocate the confidence measure
    confidence_measure = compute_std_raster(img_sec, cv.attrs['window_size']).reshape(len(row_range), len(col_range), 1)
    disp_map = disp_map.assign_coords(indicator=['disparity_pandora_intensityStd'])
    disp_map['confidence_measure'] = xr.DataArray(data=confidence_measure.astype(np.float32),
                                                  dims=['row', 'col', 'indicator'])

    # Allocate the validity mask
    disp_map['validity_mask'] = xr.DataArray(np.zeros(disp_map['disparity_map'].shape, dtype=np.uint16),
                                             dims=['row', 'col'])

    disp_map.attrs = cv.attrs
    d_range = cv.coords['disp'].data
    disp_map.attrs['disp_min'] = d_range[0]
    disp_map.attrs['disp_max'] = d_range[-1]
    disp_map.attrs['right_left_mode'] = "approximate"

    for c in col_range:
        x_d = c - disp_range
        valid = np.where((x_d >= col_range[0]) & (x_d <= col_range[-1]))

        # The disparity interval is missing in the reference image
        if x_d[valid].size == 0:
            disp_map['disparity_map'].loc[dict(col=c)] = invalid_value

            # Invalid pixel : the disparity interval is missing in the secondary image
            disp_map['validity_mask'].loc[dict(col=c)] += PANDORA_MSK_PIXEL_SEC_NODATA_OR_DISPARITY_RANGE_MISSING
        else:
            # Diagonal search for the minimum or maximum
            if cv.attrs['measure'] == 'zncc':
                min_ = cv['cost_volume'].sel(col=xr.DataArray(np.flip(x_d[valid]), dims='disp_'),
                                             disp=xr.DataArray(np.flip(disp_range[valid]),
                                                               dims='disp_')).argmax(dim='disp_')
            else:
                min_ = cv['cost_volume'].sel(col=xr.DataArray(np.flip(x_d[valid]), dims='disp_'),
                                             disp=xr.DataArray(np.flip(disp_range[valid]),
                                                               dims='disp_')).argmin(dim='disp_')

            # Disparity interval is incomplete
            if x_d[valid].size != disp_range.size:
                #  Information: the disparity interval is incomplete (border reached in the secondary image)
                disp_map['validity_mask'].loc[dict(col=c)] += PANDORA_MSK_PIXEL_SEC_INCOMPLETE_DISPARITY_RANGE

            disp_map['disparity_map'].loc[dict(col=c)] = -1 * np.flip(disp_range[valid])[min_.data]

    return disp_map


def resize(dataset: xr.Dataset, invalid_value: float = 0) -> xr.Dataset:
    """
    Pixels whose aggregation window exceeds the reference image are truncated in the output products.
    This function returns the output products with the size of the input images : add rows and columns that have been
    truncated. These added pixels will have bit 0 = 1 ( Invalid pixel : border of the reference image )
    in the validity_mask  and will have the disparity = invalid_value in the disparity map.

    :param dataset: Dataset which contains the output products
    :type dataset: xarray.Dataset with the variables :
        - disparity_map 2D xarray.DataArray (row, col)
        - confidence_measure 3D xarray.DataArray(row, col, indicator)
        - validity_mask 2D xarray.DataArray (row, col)
    :param invalid_value: disparity to assign to invalid pixels ( pixels whose aggregation window exceeds the image)
    :type invalid_value: float
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
            data = xr.DataArray(np.full((len(row), len(col)), invalid_value, dtype=np.float32), coords=[row, col],
                                dims=['row', 'col'])
            resize_disparity[array] = dataset[array].combine_first(data)

        if array == 'confidence_measure':
            depth = len(dataset.coords['indicator'])
            data = xr.DataArray(data=np.full((len(row), len(col), depth), np.nan, dtype=np.float32),
                                coords={'row': row, 'col': col}, dims=['row', 'col', 'indicator'])
            resize_disparity[array] = dataset[array].combine_first(data)

        if array == 'validity_mask':
            data = xr.DataArray(np.zeros((len(row), len(col)), dtype=np.uint16), coords=[row, col], dims=['row', 'col'])

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
