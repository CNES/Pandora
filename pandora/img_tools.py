#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2020 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of PANDORA
#
#     https://github.com/CNES/Pandora
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
This module contains functions associated to raster images.
"""

import logging
import warnings
from typing import List, Union, Tuple
import sys

import numpy as np
import rasterio
import xarray as xr
from scipy.ndimage.interpolation import zoom
from skimage.transform.pyramids import pyramid_gaussian
from numba import njit

import pandora.constants as cst


def rasterio_open(*args: str, **kwargs: Union[int, str, None]) -> rasterio.io.DatasetReader:
    """
    rasterio.open wrapper to silence UserWarning like NotGeoreferencedWarning.

    (see https://rasterio.readthedocs.io/en/latest/api/rasterio.errors.html)

    :param args: args to be given to rasterio.open method
    :type args: str
    :param kwargs: kwargs to be given to rasterio.open method
    :type kwargs: Union[int, str, None]
    :return: rasterio DatasetReader
    :rtype: rasterio.io.DatasetReader
    """
    # this silence chosen category of warnings only for the following instructions
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        return rasterio.open(*args, **kwargs)


def read_img(img: str, no_data: float, mask: str = None, classif: str = None, segm: str = None) -> xr.Dataset:
    """
    Read image and mask, and return the corresponding xarray.DataSet

    :param img: Path to the image
    :type img: string
    :type no_data: no_data value in the image
    :type no_data: float
    :param mask: Path to the mask (optional): 0 value for valid pixels, !=0 value for invalid pixels
    :type mask: string
    :param classif: Path to the classif (optional)
    :type classif: string
    :param segm: Path to the mask (optional)
    :type segm: string
    :return: xarray.DataSet containing the variables :

            - im : 2D (row, col) xarray.DataArray float32
            - msk : 2D (row, col) xarray.DataArray int16, with the convention defined in the configuration file
    :rtype: xarray.DataSet
    """
    img_ds = rasterio_open(img)
    data = img_ds.read(1)

    if np.isnan(no_data):
        no_data_pixels = np.where(np.isnan(data))
    elif np.isinf(no_data):
        no_data_pixels = np.where(np.isinf(data))
    else:
        no_data_pixels = np.where(data == no_data)

    # We accept nan values as no data on input image but to not disturb cost volume processing as stereo computation
    # step,nan as no_data must be converted. We choose -9999 (can be another value). No_data position aren't erased
    # because stored in 'msk'
    if no_data_pixels[0].size != 0 and (np.isnan(no_data) or np.isinf(no_data)):
        data[no_data_pixels] = -9999
        no_data = -9999

    dataset = xr.Dataset(
        {"im": (["row", "col"], data.astype(np.float32))},
        coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
    )

    transform = img_ds.profile["transform"]
    crs = img_ds.profile["crs"]
    # If the image has no geotransform, its transform is the identity matrix, which may not be compatible with QGIS
    # To make it compatible, the attributes are set to None
    if crs is None:
        transform = None
        crs = None

    # Add image conf to the image dataset
    dataset.attrs = {
        "no_data_img": no_data,
        "crs": crs,
        "transform": transform,
        "valid_pixels": 0,  # arbitrary default value
        "no_data_mask": 1,
    }  # arbitrary default value

    if classif is not None:
        input_classif = rasterio_open(classif).read(1)
        dataset["classif"] = xr.DataArray(
            np.full((data.shape[0], data.shape[1]), 0).astype(np.int16),
            dims=["row", "col"],
        )
        dataset["classif"].data = input_classif

    if segm is not None:
        input_segm = rasterio_open(segm).read(1)
        dataset["segm"] = xr.DataArray(
            np.full((data.shape[0], data.shape[1]), 0).astype(np.int16),
            dims=["row", "col"],
        )
        dataset["segm"].data = input_segm

    # If there is no mask, and no data in the images, do not create the mask to minimize calculation time
    if mask is None and no_data_pixels[0].size == 0:
        return dataset

    # Allocate the internal mask (!= input_mask)
    # Mask convention:
    # value : meaning
    # dataset.attrs['valid_pixels'] : a valid pixel
    # dataset.attrs['no_data_mask'] : a no_data_pixel
    # other value : an invalid_pixel
    dataset["msk"] = xr.DataArray(
        np.full((data.shape[0], data.shape[1]), dataset.attrs["valid_pixels"]).astype(np.int16),
        dims=["row", "col"],
    )

    # Mask invalid pixels if needed
    # convention: input_mask contains information to identify valid / invalid pixels.
    # Value == 0 on input_mask represents a valid pixel
    # Value != 0 on input_mask represents an invalid pixel
    if mask is not None:
        input_mask = rasterio_open(mask).read(1)
        # Masks invalid pixels
        # All pixels that are not valid_pixels, on the input mask, are considered as invalid pixels
        dataset["msk"].data[np.where(input_mask > 0)] = (
            dataset.attrs["valid_pixels"] + dataset.attrs["no_data_mask"] + 1
        )

    # Masks no_data pixels
    # If a pixel is invalid due to the input mask, and it is also no_data, then the value of this pixel in the
    # generated mask will be = no_data
    dataset["msk"].data[no_data_pixels] = int(dataset.attrs["no_data_mask"])

    return dataset


def check_dataset(dataset: xr.Dataset) -> None:
    """
    Check if input dataset is correct, and return the corresponding xarray.DataSet

    :param dataset: dataset
    :type dataset: xr.Dataset
    """

    # Check image
    if "im" not in dataset:
        logging.error("User must provide an image im")
        sys.exit(1)

    # Check mask
    if "msk" not in dataset:
        logging.warning("User should provide a mask msk")
    else:
        if dataset["im"].data.shape != dataset["msk"].data.shape:
            logging.error("image and mask must have the same shape")
            sys.exit(1)

    # Check no_data_img
    if "no_data_img" not in dataset.attrs:
        logging.error("User must provide the image nodata value ")
        sys.exit(1)

    # Check valid_pixels
    if "valid_pixels" not in dataset.attrs:
        logging.error("User must provide the valid pixels value")
        sys.exit(1)

    # Check valid_pixels
    if "no_data_mask" not in dataset.attrs:
        logging.error("User must provide the no_data_mask pixels value")
        sys.exit(1)

    # Check georef
    if "crs" not in dataset.attrs:
        logging.error("User must provide image crs")
        sys.exit(1)
    if "transform" not in dataset.attrs:
        logging.error("User must provide image transform")
        sys.exit(1)


def prepare_pyramid(
    img_left: xr.Dataset, img_right: xr.Dataset, num_scales: int, scale_factor: int
) -> Tuple[List[xr.Dataset], List[xr.Dataset]]:
    """
    Return a List with the datasets at the different scales

    :param img_left: left Dataset image containing :

            - im : 2D (row, col) xarray.DataArray
    :type img_left: xarray.Dataset
    :param img_right: right Dataset containing :

            - im : 2D (row, col) xarray.DataArray
    :type img_right: xarray.Dataset
    :param num_scales: number of scales
    :type num_scales: int
    :param scale_factor: factor by which downsample the images
    :type scale_factor: int
    :return: a List that contains the different scaled datasets
    :rtype: List of xarray.Dataset
    """
    # Fill no data values in image with an interpolation. If no mask was given, create all valid masks
    img_left_fill, msk_left_fill = fill_nodata_image(img_left)
    img_right_fill, msk_right_fill = fill_nodata_image(img_right)

    # Create image pyramids
    images_left = list(
        pyramid_gaussian(
            img_left_fill,
            max_layer=num_scales - 1,
            downscale=scale_factor,
            sigma=1.2,
            order=1,
            mode="reflect",
            cval=0,
        )
    )
    images_right = list(
        pyramid_gaussian(
            img_right_fill,
            max_layer=num_scales - 1,
            downscale=scale_factor,
            sigma=1.2,
            order=1,
            mode="reflect",
            cval=0,
        )
    )
    # Create mask pyramids
    masks_left = masks_pyramid(msk_left_fill, scale_factor, num_scales)
    masks_right = masks_pyramid(msk_right_fill, scale_factor, num_scales)

    # Create dataset pyramids
    pyramid_left = convert_pyramid_to_dataset(img_left, images_left, masks_left)
    pyramid_right = convert_pyramid_to_dataset(img_right, images_right, masks_right)

    # The pyramid is intended to be from coarse to original size, so we inverse its order.
    return pyramid_left[::-1], pyramid_right[::-1]


def fill_nodata_image(dataset: xr.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate no data values in image. If no mask was given, create all valid masks

    :param dataset: Dataset image
    :type dataset: xarray.Dataset containing :

        - im : 2D (row, col) xarray.DataArray
    :return: a Tuple that contains the filled image and mask
    :rtype: Tuple of np.ndarray
    """
    if "msk" in dataset:
        img, msk = interpolate_nodata_sgm(dataset["im"].data, dataset["msk"].data)
    else:
        msk = np.full(
            (dataset["im"].data.shape[0], dataset["im"].data.shape[1]),
            int(dataset.attrs["valid_pixels"]),
        )
        img = dataset["im"].data
    return img, msk


@njit()
def interpolate_nodata_sgm(img: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolation of the input image to resolve invalid (nodata) pixels.
    Interpolate invalid pixels by finding the nearest correct pixels in 8 different directions
    and use the median of their disparities.

    HIRSCHMULLER, Heiko. Stereo processing by semiglobal matching and mutual information.
    IEEE Transactions on pattern analysis and machine intelligence, 2007, vol. 30, no 2, p. 328-341.

    :param img: input image
    :type img: 2D np.array (row, col)
    :param valid: validity mask
    :type valid: 2D np.array (row, col)
    :return: the interpolate input image, with the validity mask update :

        - If out & PANDORA_MSK_PIXEL_FILLED_NODATA != 0 : Invalid pixel : filled nodata pixel
    :rtype: tuple(2D np.array (row, col), 2D np.array (row, col))
    """
    # Output disparity map and validity mask
    out_img = np.copy(img)
    out_val = np.copy(valid)

    ncol, nrow = img.shape

    # 8 directions : [row, y]
    dirs = np.array([[0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1]])

    ncol, nrow = img.shape
    for col in range(ncol):
        for row in range(nrow):
            # Mismatched
            if valid[col, row] & cst.PANDORA_MSK_PIXEL_INVALID != 0:
                valid_neighbors = find_valid_neighbors(dirs, img, valid, row, col)

                # Median of the 8 pixels
                out_img[col, row] = np.nanmedian(valid_neighbors)
                # Update the validity mask : Information : filled nodata pixel
                out_val[col, row] = cst.PANDORA_MSK_PIXEL_FILLED_NODATA

    return out_img, out_val


def masks_pyramid(msk: np.ndarray, scale_factor: int, num_scales: int) -> List[np.ndarray]:
    """
    Return a List with the downsampled masks for each scale

    :param msk: full resolution mask
    :type msk: np.ndarray
    :param scale_factor: scale factor
    :type scale_factor: int
    :param num_scales: number of scales
    :type num_scales: int
    :return: a List that contains the different scaled masks
    :rtype: List of np.ndarray
    """
    msk_pyramid = []
    # Add the full resolution mask
    msk_pyramid.append(msk)
    tmp_msk = msk
    for scale in range(num_scales - 1):  # pylint: disable=unused-variable
        # Decimate in the two axis
        tmp_msk = tmp_msk[::scale_factor, ::scale_factor]
        msk_pyramid.append(tmp_msk)
    return msk_pyramid


def convert_pyramid_to_dataset(
    img_orig: xr.Dataset, images: List[np.ndarray], masks: List[np.ndarray]
) -> List[xr.Dataset]:
    """
    Return a List with the datasets at the different scales

    :param img_left: left Dataset image containing :

        - im : 2D (row, col) xarray.DataArray
    :type img_left: xarray.Dataset
    :param img_right: right Dataset image containing :

        - im : 2D (row, col) xarray.DataArray
    :type img_right: xarray.Dataset
    :param num_scales: number of scales
    :type num_scales: int
    :param scale_factor: factor by which downsample the images
    :type scale_factor: int
    :return: a List that contains the different scaled datasets
    :rtype: List of xarray.Dataset
    """

    pyramid = []
    for index, image in enumerate(images):
        # The full resolution image (first in list) has to be the original image and mask
        if index == 0:
            pyramid.append(img_orig)
            continue

        # Creating new dataset
        dataset = xr.Dataset(
            {"im": (["row", "col"], image.astype(np.float32))},
            coords={"row": np.arange(image.shape[0]), "col": np.arange(image.shape[1])},
        )

        # Allocate the mask
        dataset["msk"] = xr.DataArray(
            np.full((image.shape[0], image.shape[1]), masks[index].astype(np.int16)),
            dims=["row", "col"],
        )

        # Add image conf to the image dataset
        # - attributes are linked to each others
        dataset.attrs = img_orig.attrs
        pyramid.append(dataset)

    return pyramid


def shift_right_img(img_right: xr.Dataset, subpix: int) -> List[xr.Dataset]:
    """
    Return an array that contains the shifted right images

    :param img_right: right Dataset image containing :

        - im : 2D (row, col) xarray.DataArray
    :type img_right: xarray.Dataset
    :param subpix: subpixel precision = (1 or pair number)
    :type subpix: int
    :return: an array that contains the shifted right images
    :rtype: array of xarray.Dataset
    """
    img_right_shift = [img_right]
    ny_, nx_ = img_right["im"].shape

    if subpix > 1:
        for ind in np.arange(1, subpix):
            shift = 1 / subpix
            # For each index, shift the right image for subpixel precision 1/subpix*index
            data = zoom(img_right["im"].data, (1, (nx_ * subpix - (subpix - 1)) / float(nx_)), order=1)[:, ind::subpix]
            col = np.arange(img_right.coords["col"][0] + shift * ind, img_right.coords["col"][-1], step=1)
            img_right_shift.append(
                xr.Dataset(
                    {"im": (["row", "col"], data)},
                    coords={"row": np.arange(ny_), "col": col},
                )
            )
    return img_right_shift


def check_inside_image(img: xr.Dataset, row: int, col: int) -> bool:
    """
    Check if the coordinates row,col are inside the image

    :param img: Dataset image containing :

            - im : 2D (row, col) xarray.DataArray
    :type img: xarray.Dataset
    :param row: row coordinates
    :type row: int
    :param col: column coordinates
    :type col: int
    :return: a boolean
    :rtype: boolean
    """
    nx_, ny_ = img["im"].shape
    return 0 <= row < nx_ and 0 <= col < ny_


def census_transform(image: xr.Dataset, window_size: int) -> xr.Dataset:
    """
    Generates the census transformed image from an image

    :param image: Dataset image containing the image im : 2D (row, col) xarray.Dataset
    :type image: xarray.Dataset
    :param window_size: Census window size
    :type window_size: int
    :return: Dataset census transformed uint32 containing the transformed image im: 2D (row, col) xarray.DataArray \
    uint32
    :rtype: xarray.Dataset
    """
    ny_, nx_ = image["im"].shape
    border = int((window_size - 1) / 2)

    # Create a sliding window of using as_strided function : this function create a new a view (by manipulating data
    #  pointer) of the image array with a different shape. The new view pointing to the same memory block as
    # image so it does not consume any additional memory.
    str_row, str_col = image["im"].data.strides
    shape_windows = (
        ny_ - (window_size - 1),
        nx_ - (window_size - 1),
        window_size,
        window_size,
    )
    strides_windows = (str_row, str_col, str_row, str_col)
    windows = np.lib.stride_tricks.as_strided(image["im"].data, shape_windows, strides_windows, writeable=False)

    # Pixels inside the image which can be centers of windows
    central_pixels = image["im"].data[border:-border, border:-border]

    # Allocate the census mask
    census = np.zeros((ny_ - (window_size - 1), nx_ - (window_size - 1)), dtype="uint32")

    shift = (window_size * window_size) - 1
    for row in range(window_size):
        for col in range(window_size):
            # Computes the difference and shift the result
            census[:, :] += ((windows[:, :, row, col] > central_pixels[:, :]) << shift).astype(np.uint32)
            shift -= 1

    census = xr.Dataset(
        {"im": (["row", "col"], census)},
        coords={
            "row": np.arange(border, ny_ - border),
            "col": np.arange(border, nx_ - border),
        },
    )

    return census


def compute_mean_raster(img: xr.Dataset, win_size: int) -> np.ndarray:
    """
    Compute the mean within a sliding window for the whole image

    :param img: Dataset image containing :

            - im : 2D (row, col) xarray.DataArray
    :type img: xarray.Dataset
    :param win_size: window size
    :type win_size: int
    :return: mean raster
    :rtype: numpy array
    """
    ny_, nx_ = img["im"].shape

    # Example with win_size = 3 and the input :
    #           10 | 5  | 3
    #            2 | 10 | 5
    #            5 | 3  | 1

    # Compute the cumulative sum of the elements along the column axis
    r_mean = np.r_[np.zeros((1, nx_)), img["im"]]
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


@njit()
def find_valid_neighbors(dirs: np.ndarray, disp: np.ndarray, valid: np.ndarray, row: int, col: int):
    """
    Find valid neighbors along directions

    :param dirs: directions
    :type dirs: 2D np.array (row, col)
    :param disp: disparity map
    :type disp: 2D np.array (row, col)
    :param valid: validity mask
    :type valid: 2D np.array (row, col)
    :param row: row current value
    :type row: int
    :param col: col current value
    :type col: int
    :return: valid neighbors
    :rtype: 2D np.array
    """
    ncol, nrow = disp.shape
    # Maximum path length
    max_path_length = max(nrow, ncol)
    # For each directions
    valid_neighbors = np.zeros(8, dtype=np.float32)
    for direction in range(8):
        # Find the first valid pixel in the current path
        tmp_row = row
        tmp_col = col
        for i in range(max_path_length):  # pylint: disable= unused-variable
            tmp_row += dirs[direction][0]
            tmp_col += dirs[direction][1]

            # Edge of the image reached: there is no valid pixel in the current path
            if (tmp_col < 0) | (tmp_col >= ncol) | (tmp_row < 0) | (tmp_row >= nrow):
                valid_neighbors[direction] = np.nan
                break
            # First valid pixel
            if (valid[tmp_col, tmp_row] & cst.PANDORA_MSK_PIXEL_INVALID) == 0:
                valid_neighbors[direction] = disp[tmp_col, tmp_row]
                break
    return valid_neighbors


def compute_mean_patch(img: xr.Dataset, row: int, col: int, win_size: int) -> np.ndarray:
    """
    Compute the mean within a window centered at position row,col

    :param img: Dataset image containing :

            - im : 2D (row, col) xarray.DataArray
    :type img: xarray.Dataset
    :param row: row coordinates
    :type row: int
    :param col: column coordinates
    :type col: int
    :param win_size: window size
    :type win_size: int
    :return: mean
    :rtype: float
    """
    begin_window = (row - int(win_size / 2), col - int(win_size / 2))
    end_window = (row + int(win_size / 2), col + int(win_size / 2))

    # Check if the window is inside the image, and compute the mean
    if check_inside_image(img, begin_window[0], begin_window[1]) and check_inside_image(
        img, end_window[0], end_window[1]
    ):
        return np.mean(
            img["im"][begin_window[1] : end_window[1] + 1, begin_window[0] : end_window[0] + 1],
            dtype=np.float32,
        )

    logging.error("The window is outside the image")
    raise IndexError


def compute_std_raster(img: xr.Dataset, win_size: int) -> np.ndarray:
    """
    Compute the standard deviation within a sliding window for the whole image
    with the formula : std = sqrt( E[row^2] - E[row]^2 )

    :param img: Dataset image containing :

            - im : 2D (row, col) xarray.DataArray
    :type img: xarray.Dataset
    :param win_size: window size
    :type win_size: int
    :return: std raster
    :rtype: numpy array
    """
    # Computes E[row]
    mean_ = compute_mean_raster(img, win_size)

    # Computes E[row^2]
    raster_power_two = xr.Dataset(
        {"im": (["row", "col"], img["im"].data ** 2)},
        coords={
            "row": np.arange(img["im"].shape[0]),
            "col": np.arange(img["im"].shape[1]),
        },
    )
    mean_power_two = compute_mean_raster(raster_power_two, win_size)

    # Compute sqrt( E[row^2] - E[row]^2 )
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
    if isinstance(disparity, str):
        disp_ = rasterio_open(disparity)
        data_disp = disp_.read(1)
    else:
        data_disp = disparity

    return data_disp
