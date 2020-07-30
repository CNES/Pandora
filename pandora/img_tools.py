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
This module contains functions associated to raster images.
"""

import xarray as xr
import numpy as np
import logging
from scipy.ndimage.interpolation import zoom
import rasterio
from typing import Dict, List, Union


def read_img(img: str, no_data: float, cfg: Dict, mask: xr.Dataset = None) -> xr.Dataset:
    """
    Read image and mask, and return the corresponding xarray.DataSet

    :param img: Path to the image
    :type img: string
    :type no_data: no_data value in the image
    :type no_data: float
    :param cfg: image configuration, containing the mask conventions generated by this function
    :param cfg: dict
    :param mask: Path to the mask (optional)
    :type mask: string
    :return: xarray.DataSet
    :rtype:
        xarray.DataSet containing the variables :
            - im : 2D (row, col) xarray.DataArray float32
            - msk : 2D (row, col) xarray.DataArray int16, with the convention defined in the configuration file
    """
    img_ds = rasterio.open(img)
    data = img_ds.read(1)
    dataset = xr.Dataset({'im': (['row', 'col'], data.astype(np.float32))},
                         coords={'row': np.arange(data.shape[0]),
                                 'col': np.arange(data.shape[1])})

    # If there is no mask, and no data in the images, do not create the mask to minimize calculation time
    no_data_pixels = np.where(data == no_data)
    if mask is None and no_data_pixels[0].size == 0:
        return dataset

    # Allocate the mask
    dataset['msk'] = xr.DataArray(np.full((data.shape[0], data.shape[1]), int(cfg['valid_pixels'])).astype(np.int16),
                                  dims=['row', 'col'])

    # Mask invalid pixels if needed
    if mask is not None:
        input_mask = rasterio.open(mask).read(1)
        # Masks invalid pixels
        # All pixels that are not no_data and valid_pixels are considered as invalid pixels
        dataset['msk'].data[np.where(input_mask > 0)] = int(cfg['valid_pixels']) + int(cfg['no_data']) + 1

    # Masks no_data pixels
    # If a pixel is invalid due to the input mask, and it is also no_data, then the value of this pixel in the
    # generated mask will be = no_data
    dataset['msk'].data[no_data_pixels] = int(cfg['no_data'])
    return dataset


def shift_sec_img(img_sec: xr.Dataset, subpix: int) -> List[xr.Dataset]:
    """
    Return an array that contains the shifted secondary images

    :param img_sec: secondary Dataset image
    :type img_sec:
    xarray.Dataset containing :
        - im : 2D (row, col) xarray.DataArray

    :param subpix: subpixel precision = (1 or 2 or 4)
    :type subpix: int
    :return: an array that contains the shifted secondary images
    :rtype : array of xarray.Dataset
    """
    img_sec_shift = [img_sec]
    ny_, nx_ = img_sec['im'].shape

    # zoom factor = (number of columns with zoom / number of original columns)
    if subpix == 2:
        # Shift the secondary image for subpixel precision 0.5
        data = zoom(img_sec['im'].data, (1, (nx_ * 4 - 3) / float(nx_)), order=1)[:, 2::4]
        col = np.arange(img_sec.coords['col'][0] + 0.5, img_sec.coords['col'][-1], step=1)
        img_sec_shift.append(xr.Dataset({'im': (['row', 'col'], data)},
                                        coords={'row': np.arange(ny_), 'col': col}))

    if subpix == 4:
        # Shift the secondary image for subpixel precision 0.25
        data = zoom(img_sec['im'].data, (1, (nx_ * 4 - 3) / float(nx_)), order=1)[:, 1::4]
        col = np.arange(img_sec.coords['col'][0] + 0.25, img_sec.coords['col'][-1], step=1)
        img_sec_shift.append(xr.Dataset({'im': (['row', 'col'], data)},
                                        coords={'row': np.arange(ny_), 'col': col}))

        # Shift the secondary image for subpixel precision 0.5
        data = zoom(img_sec['im'].data, (1, (nx_ * 4 - 3) / float(nx_)), order=1)[:, 2::4]
        col = np.arange(img_sec.coords['col'][0] + 0.5, img_sec.coords['col'][-1], step=1)
        img_sec_shift.append(xr.Dataset({'im': (['row', 'col'], data)},
                                        coords={'row': np.arange(ny_), 'col': col}))

        # Shift the secondary image for subpixel precision 0.75
        data = zoom(img_sec['im'].data, (1, (nx_ * 4 - 3) / float(nx_)), order=1)[:, 3::4]
        col = np.arange(img_sec.coords['col'][0] + 0.75, img_sec.coords['col'][-1], step=1)
        img_sec_shift.append(xr.Dataset({'im': (['row', 'col'], data)},
                                        coords={'row': np.arange(ny_), 'col': col}))
    return img_sec_shift


def check_inside_image(img: xr.Dataset, x: int, y: int) -> bool:
    """
    Check if the coordinates x,y are inside the image

    :param img: Dataset image
    :type img:
        xarray.Dataset containing :
            - im : 2D (row, col) xarray.DataArray
    :param x: row coordinates
    :type x: int
    :param y: column coordinates
    :type y: int
    :return: a boolean
    :rtype: boolean
    """
    nx_, ny_ = img['im'].shape
    return 0 <= x < nx_ and 0 <= y < ny_


def census_transform(image: xr.Dataset, window_size: int) -> xr.Dataset:
    """
    Generates the census transformed image from an image

    :param image: Dataset image
    :type image: xarray.Dataset containing the image im : 2D (row, col) xarray.Dataset
    :param window_size: Census window size
    :type window_size: int
    :return: Dataset census transformed uint32
    :rtype: xarray.Dataset containing the transformed image im: 2D (row, col) xarray.DataArray uint32
    """
    ny_, nx_ = image['im'].shape
    border = int((window_size - 1) / 2)

    # Create a view of each window, by manipulating the internal data structure of array
    str_row, str_col = image['im'].data.strides
    shape_windows = (ny_ - (window_size - 1), nx_ - (window_size - 1), window_size, window_size)
    strides_windows = (str_row, str_col, str_row, str_col)
    windows = np.lib.stride_tricks.as_strided(image['im'].data, shape_windows, strides_windows)

    # Pixels inside the image which can be centers of windows
    central_pixels = image['im'].data[border:-border, border:-border]

    # Allocate the census mask
    census = np.zeros((ny_ - (window_size - 1), nx_ - (window_size - 1)), dtype='uint32')

    shift = (window_size * window_size) - 1
    for row in range(window_size):
        for col in range(window_size):
            # Computes the difference and shift the result
            census[:, :] += ((windows[:, :, row, col] > central_pixels[:, :]) << shift).astype(np.uint32)
            shift -= 1

    census = xr.Dataset({'im': (['row', 'col'], census)},
                        coords={'row': np.arange(border, ny_ - border),
                                'col': np.arange(border, nx_ - border)})

    return census


def compute_mean_raster(img: xr.Dataset, win_size: int) -> np.ndarray:
    """
    Compute the mean within a sliding window for the whole image

    :param img: Dataset image
    :type img:
        xarray.Dataset containing :
            - im : 2D (row, col) xarray.DataArray
    :param win_size: window size
    :type win_size: int
    :return: mean raster
    :rtype: numpy array
    """
    ny_, nx_ = img['im'].shape

    # Example with win_size = 3 and the input :
    #           10 | 5  | 3
    #            2 | 10 | 5
    #            5 | 3  | 1

    # Compute the cumulative sum of the elements along the column axis
    r_mean = np.r_[np.zeros((1, nx_)), img['im']]
    # r_mean :   0 | 0  | 0
    #           10 | 5  | 3
    #            2 | 10 | 5
    #            5 | 3  | 1
    r_mean = np.cumsum(r_mean, axis=0)
    # r_mean :   0 | 0  | 0
    #           10 | 5  | 3
    #           12 | 15 | 8
    #           17 | 18 | 9
    r_mean = r_mean[win_size:, :] - r_mean[:-win_size, :]
    # r_mean :  17 | 18 | 9

    # Compute the cumulative sum of the elements along the row axis
    r_mean = np.c_[np.zeros(ny_ - (win_size - 1)), r_mean]
    # r_mean :   0 | 17 | 18 | 9
    r_mean = np.cumsum(r_mean, axis=1)
    # r_mean :   0 | 17 | 35 | 44
    r_mean = r_mean[:, win_size:] - r_mean[:, :-win_size]
    # r_mean : 44
    return r_mean / float(win_size * win_size)


def compute_mean_patch(img: xr.Dataset, x: int, y: int, win_size: int) -> float:
    """
    Compute the mean within a window centered at position x,y

    :param img: Dataset image
    :type img:
        xarray.Dataset containing :
            - im : 2D (row, col) xarray.DataArray
    :param x: row coordinates
    :type x: int
    :param y: column coordinates
    :type y: int
    :param win_size: window size
    :type win_size: int
    :return: mean
    :rtype : float
    """
    begin_window = (x-int(win_size/2), y-int(win_size/2))
    end_window = (x+int(win_size/2), y+int(win_size/2))

    # Check if the window is inside the image, and compute the mean
    if check_inside_image(img, begin_window[0], begin_window[1]) and \
            check_inside_image(img, end_window[0], end_window[1]):
        return np.mean(img['im'][begin_window[1]:end_window[1]+1, begin_window[0]:end_window[0]+1], dtype=np.float32)

    logging.error("The window is outside the image")
    raise IndexError


def compute_std_raster(img: xr.Dataset, win_size: int) -> np.ndarray:
    """
    Compute the standard deviation within a sliding window for the whole image
    with the formula : std = sqrt( E[x^2] - E[x]^2 )

    :param img: Dataset image
    :type img:
        xarray.Dataset containing :
            - im : 2D (row, col) xarray.DataArray
    :param win_size: window size
    :type win_size: int
    :return: std raster
    :rtype : numpy array
    """
    # Computes E[x]
    mean_ = compute_mean_raster(img, win_size)

    # Computes E[x^2]
    raster_power_two = xr.Dataset({'im': (['row', 'col'], img['im'].data ** 2)},
                                  coords={'row': np.arange(img['im'].shape[0]), 'col': np.arange(img['im'].shape[1])})
    mean_power_two = compute_mean_raster(raster_power_two, win_size)

    # Compute sqrt( E[x^2] - E[x]^2 )
    var = mean_power_two - mean_ ** 2
    # Avoid very small values
    var[np.where(var < (10 ** (-15) * abs(mean_power_two)))] = 0
    return np.sqrt(var)


def read_disp(disparity: Union[None, int, str]) -> Union[None, int, np.ndarray]:
    """
    Read the disparity :
        - if cfg_disp is the path of a disparity grid, read and return the grid (type numpy array)
        - else return the value of cfg_disp

    :param disparity: disparity, or path to the disparity grid
    :type disparity: None, int or str
    :return: the disparity
    :rtype: int or np.ndarray
    """
    if type(disparity) == str:
        disp_ = rasterio.open(disparity)
        data_disp = disp_.read(1)
    else:
        data_disp = disparity

    return data_disp
