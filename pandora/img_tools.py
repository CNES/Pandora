#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
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

from __future__ import annotations

import warnings
from typing import Dict, List, Tuple, Union, cast

import numpy as np
import rasterio
import xarray as xr
from rasterio.windows import Window
from scipy.ndimage import zoom
from skimage.transform.pyramids import pyramid_gaussian

import pandora.constants as cst
from .cpp import img_tools_cpp


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


def get_window(roi: Dict, width: int, height: int) -> Window:
    """
    Get window from image size and roi

    :param roi: dictionnary with a roi

        "col": {"first": <value - int>, "last": <value - int>},
        "row": {"first": <value - int>, "last": <value - int>},
        "margins": [<value - int>, <value - int>, <value - int>, <value - int>]

    with margins : left, up, right, down
    :type roi: dict
    :param width: image width
    :type width: int
    :param height: image height
    :type height: int
    :return: Window : Windowed reading with rasterio
    :rtype: Window
    """
    # Window(col_off, row_off, width, height)
    col_off = max(roi["col"]["first"] - roi["margins"][0], 0)  # if overlapping on left side
    row_off = max(roi["row"]["first"] - roi["margins"][1], 0)  # if overlapping on up side
    roi_width = roi["col"]["last"] - col_off + roi["margins"][2] + 1
    roi_height = roi["row"]["last"] - row_off + roi["margins"][3] + 1

    # check roi outside
    if col_off > width or row_off > height or (col_off + roi_width) < 0 or (row_off + roi_height) < 0:
        raise ValueError("Roi specified is outside the image")

    # overlap roi and image
    # right side
    if (col_off + roi_width) > width:
        roi_width = width - col_off
    # down side
    if (row_off + roi_height) > height:
        roi_height = height - row_off

    return Window(col_off, row_off, roi_width, roi_height)


def add_disparity(
    dataset: xr.Dataset, disparity: Union[Tuple[int, int], list[int], str, None], window: Window
) -> xr.Dataset:
    """
    Add disparity to dataset

    If `disparity` is None, will return dataset unmodified.

    :param dataset: xarray dataset without classification
    :type dataset: xr.Dataset
    :param disparity: disparity, or path to the disparity grid
    :type disparity: Tuple[int, int] or list[int] or str or None

    :param window : Windowed reading with rasterio
    :type window: Window
    :return: dataset : updated dataset
    :rtype: xr.Dataset
    """
    if disparity is None:
        dataset.attrs["disparity_source"] = None
        return dataset

    # disparity is a grid
    if isinstance(disparity, str):
        disparity_data = rasterio_open(disparity).read(out_dtype=np.float32, window=window)
    # tuple or list
    else:
        disparity_data = np.array(
            [
                np.full((dataset.sizes["row"], dataset.sizes["col"]), disparity[0]),
                np.full((dataset.sizes["row"], dataset.sizes["col"]), disparity[1]),
            ]
        )

    return add_disparity_grid(dataset, xr.DataArray(disparity_data, dims=["band_disp", "row", "col"]), disparity)


def add_disparity_grid(
    dataset: xr.Dataset,
    disparity_grid: Union[xr.DataArray, None] = None,
    disparity_source: Union[Tuple[int, int], list[int], str, None] = "xr.Dataset",
) -> xr.Dataset:
    """
    Add a disparity grid to dataset.

    If `disparity_grid` is None, will return dataset unmodified.

    :param dataset: xarray dataset without classification
    :type dataset: xr.Dataset
    :param disparity_grid: disparity grid dataset with dimensions ["band_disp", "row", "col"] or None.
    :type disparity_grid: xr.DataArray or None
    :param disparity_source: source of the disparity:
                             either a path to a file or `xr.Dataset` if the dataset was directly provided.
    :type disparity_source: Union[Tuple[int, int], list[int], str, None]
    :return: dataset : updated dataset
    :rtype: xr.Dataset
    """
    if disparity_grid is not None:
        dataset.coords["band_disp"] = ["min", "max"]
        dataset["disparity"] = disparity_grid
        dataset.attrs["disparity_source"] = disparity_source
    return dataset


def add_classif(dataset: xr.Dataset, classif: Union[str, None], window: Window) -> xr.Dataset:
    """
    Add classification information and image to dataset

    :param dataset: xarray dataset without classification
    :type dataset: xr.Dataset
    :param classif: classification image path
    :type classif: str or None

    :param window : Windowed reading with rasterio
    :type window: Window
    :return: dataset : updated dataset
    :rtype: xr.Dataset
    """
    if classif is not None:
        classif_ds = rasterio_open(classif)
        # Add extra dataset coordinate for classification bands with band names from image metadat
        dataset.coords["band_classif"] = list(classif_ds.descriptions)
        dataset["classif"] = xr.DataArray(
            classif_ds.read(out_dtype=np.int16, window=window),
            dims=["band_classif", "row", "col"],
        )
    return dataset


def add_segm(dataset: xr.Dataset, segm: Union[str, None], window: Window) -> xr.Dataset:
    """
    Add Segmentation information and image to dataset

    :param dataset: xarray dataset without segmentation
    :type dataset: xr.Dataset
    :param segm: segmentation image path
    :type segm: str or None

    :param window : Windowed reading with rasterio
    :type window: Window
    :return: dataset : updated dataset
    :rtype: xr.Dataset
    """
    if segm is not None:
        dataset["segm"] = xr.DataArray(
            rasterio_open(segm).read(1, out_dtype=np.int16, window=window),
            dims=["row", "col"],
        )
    return dataset


def add_no_data(dataset: xr.Dataset, no_data: Union[int, float], no_data_pixels: np.ndarray) -> xr.Dataset:
    """
    Add no data information to dataset

    :param dataset: xarray dataset without no_data information
    :type dataset: xr.Dataset
    :param no_data: value
    :type no_data: int or float
    :param no_data_pixels: matrix with no_data value
    :type no_data_pixels: np.ndarray
    :return: dataset : updated dataset
    :rtype: xr.Dataset
    """
    # We accept nan values as no data on input image but to not disturb cost volume processing as stereo computation
    # step,nan as no_data must be converted. We choose -9999 (can be another value). No_data position aren't erased
    # because stored in 'msk'
    if no_data_pixels[0].size != 0 and (np.isnan(no_data) or np.isinf(no_data)):
        dataset["im"].data[no_data_pixels] = -9999
        no_data = -9999
    dataset.attrs.update({"no_data_img": no_data})
    return dataset


def add_mask(
    dataset: xr.Dataset, mask: Union[str, None], no_data_pixels: np.ndarray, width: int, height: int, window: Window
) -> xr.Dataset:
    """
    Add mask information and image to dataset

    :param dataset: xarray dataset without mask
    :type dataset: xr.Dataset
    :param mask: mask image path
    :type mask: str or None
    :param no_data_pixels: matrix with no_data value
    :type no_data_pixels: np.ndarray
    :param width: nb columns
    :type width: int
    :param height: nb rows
    :type height: int
    :param window: information about window
    :type window: rasterio.window
    :return: dataset : updated dataset
    :rtype: xr.Dataset
    """
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
        np.full((height, width), dataset.attrs["valid_pixels"]).astype(np.int16),
        dims=["row", "col"],
    )

    # Mask invalid pixels if needed
    # convention: input_mask contains information to identify valid / invalid pixels.
    # Value == 0 on input_mask represents a valid pixel
    # Value != 0 on input_mask represents an invalid pixel
    if mask is not None:
        input_mask = rasterio_open(mask).read(1, window=window)
        # Masks invalid pixels
        # All pixels that are not valid_pixels, on the input mask, are considered as invalid pixels
        dataset["msk"].data[np.where(input_mask > 0)] = (
            dataset.attrs["valid_pixels"] + dataset.attrs["no_data_mask"] + 1
        )

    # Masks no_data pixels
    # If a pixel is invalid due to the input mask, and it is also no_data, then the value of this pixel in the
    # generated mask will be = no_data
    # In 3D, the coordinates of dataset["im"] are [band_im, row, col] and in 2D are [row, col]. To be sure to always
    # having no_data_pixels[row], take the "-2" dimension, and no_data_pixel[col], take the last dimension.
    dataset["msk"].data[(no_data_pixels[-2], no_data_pixels[-1])] = int(dataset.attrs["no_data_mask"])
    return dataset


def add_global_disparity(dataset: xr.Dataset, global_disp_min: int, global_disp_max: int) -> xr.Dataset:
    """
    Add global disparity information to dataset

    :param dataset: xarray dataset without no_data information
    :type dataset: xr.Dataset
    :param global_disp_min: global minimum disparity
    :type global_disp_min: int
    :param global_disp_max: global maximum disparity
    :type global_disp_max: int
    :return: dataset : updated dataset
    :rtype: xr.Dataset
    """

    img_grid_min = dataset.attrs["disp_min"]
    img_grid_max = dataset.attrs["disp_max"]

    if global_disp_min > img_grid_min or global_disp_max < img_grid_max:
        raise ValueError("For ambiguity step, the global disparity must be outside the range of the grid disparity")

    # Add global disparity to dataset in case of tiling ambiguity
    dataset.attrs.update({"global_disparity": [global_disp_min, global_disp_max]})

    return dataset


def create_dataset_from_inputs(input_config: dict, roi: dict = None) -> xr.Dataset:
    """
    Read image and mask, and return the corresponding xarray.DataSet

    :param input_config: configuration used to create dataset.
    :type input_config: dict
    :param roi: dictionary with a roi

            "col": {"first": <value - int>, "last": <value - int>},
            "row": {"first": <value - int>, "last": <value - int>},
            "margins": [<value - int>, <value - int>, <value - int>, <value - int>]

            with margins : left, up, right, down
    :type roi: dict
    :return: xarray.DataSet containing the variables :

            - im: 2D (row, col) or 3D (band_im, row, col) xarray.DataArray float32
            - disparity (optional): 3D (disp, row, col) xarray.DataArray float32
            - msk (optional): 2D (row, col) xarray.DataArray int16
            - classif (optional): 3D (band_classif, row, col) xarray.DataArray int16
            - segm (optional): 2D (row, col) xarray.DataArray int16

    :rtype: xarray.DataSet
    """
    # Set default values
    input_parameters = {"mask": None, "classif": None, "segm": None}
    input_parameters.update(input_config)

    img_ds = rasterio_open(input_parameters["img"])
    nx_, ny_ = img_ds.width, img_ds.height

    # ROI
    window = get_window(roi, nx_, ny_) if roi else None
    # col_off and row_off = coordinates of top-left point of ROI
    col_off, row_off = (window.col_off, window.row_off) if roi else (0, 0)

    # If only one band is present, consider data as 2 dimensions
    if img_ds.count == 1:
        data = img_ds.read(1, out_dtype=np.float32, window=window)
        nx_, ny_ = data.shape[1], data.shape[0]
        image = {"im": (["row", "col"], data)}
        coords = {"row": np.arange(row_off, ny_ + row_off), "col": np.arange(col_off, nx_ + col_off)}
    # if image is 3 dimensions we create a dataset with [band row col] dims for dataArray
    else:
        data = img_ds.read(out_dtype=np.float32, window=window)
        nx_, ny_ = data.shape[2], data.shape[1]
        image = {"im": (["band_im", "row", "col"], data)}
        # Band names are in the image metadata
        coords = {
            "band_im": list(img_ds.descriptions),  # type: ignore
            "row": np.arange(row_off, ny_ + row_off),
            "col": np.arange(col_off, nx_ + col_off),
        }

    crs = img_ds.profile["crs"]
    # If the image has no geotransform, its transform is the identity matrix, which may not be compatible with QGIS
    # To make it compatible, the attributes are set to None
    transform = img_ds.profile["transform"] if crs is not None else None

    # Add image conf to the attributes of the dataset
    attributes = {
        "crs": crs,
        "transform": transform,
        "valid_pixels": 0,  # arbitrary default value
        "no_data_mask": 1,  # arbitrary default value
    }
    dataset = xr.Dataset(
        image,
        coords=coords,
        attrs=attributes,
    )

    # disparities
    if "disp" in input_parameters:
        dataset.pipe(add_disparity, disparity=input_config["disp"], window=window)

    # No data
    no_data = input_parameters["nodata"]
    if np.isnan(no_data):
        no_data_pixels = np.where(np.isnan(dataset["im"].data))
    elif np.isinf(no_data):
        no_data_pixels = np.where(np.isinf(dataset["im"].data))
    else:
        no_data_pixels = np.where(dataset["im"].data == no_data)

    return (
        dataset.pipe(add_classif, input_parameters["classif"], window)
        .pipe(add_segm, input_parameters["segm"], window)
        .pipe(add_no_data, no_data, no_data_pixels)
        .pipe(add_mask, input_parameters["mask"], no_data_pixels, nx_, ny_, window)
    )


def get_metadata(
    img: str, disparity: list[int] | str | None = None, classif: str = None, segm: str = None
) -> xr.Dataset:
    """
    Read metadata from image, and return the corresponding xarray.DataSet

    :param img: img_path
    :type img: str
    :param disparity: disparity couple of ints or path to disparity grid file (optional)
    :type disparity: list[int] | str | None
    :param classif: path to the classif (optional)
    :type classif: str
    :param segm: path to the segm (optional)
    :type segm: str
    :return: partial xarray.DataSet (attributes and coordinates)
    :rtype: xarray.DataSet
    """
    img_ds = rasterio_open(img)

    # create the dataset
    dataset = xr.Dataset(
        data_vars={},
        coords={
            "band_im": list(img_ds.descriptions),
            "row": np.arange(img_ds.height),
            "col": np.arange(img_ds.width),
        },
    )

    return (
        dataset.pipe(add_disparity, disparity=disparity, window=None)
        .pipe(add_classif, classif, window=None)
        .pipe(add_segm, segm, window=None)
    )


def get_pyramids(data, num_scales, scale_factor, channel_axis=None):
    """
    Returns the pyramid of images, of height num_scales and width factor scale_factor

    :param data: image
    :type data: 2D or 3D np.ndarray
    :param num_scales: number of scaled images
    :type num_scales: int
    :param scale_factor: ratio in width/height between two consecutives images of the pyramid
    :type scale_factor: float
    :type channel_axis: if specified, axis to use as the channel in the 3D data
    :type channel_axis: int
    :return: the list of images in the pyramid, large to small
    :rtype: list(np.ndarray)
    """
    return list(
        pyramid_gaussian(
            data,
            max_layer=num_scales - 1,
            downscale=scale_factor,
            sigma=1.2,
            order=1,
            mode="reflect",
            cval=0,
            channel_axis=channel_axis,
        )
    )


def prepare_pyramid(
    img_left: xr.Dataset, img_right: xr.Dataset, num_scales: int, scale_factor: int
) -> Tuple[List[xr.Dataset], List[xr.Dataset]]:
    """
    Return a List with the datasets at the different scales

    :param img_left: left Dataset image containing the image im : 2D (row, col) xarray.Dataset
    :type img_left: xarray.Dataset
    :param img_right: right Dataset image containing the image im : 2D (row, col) xarray.Dataset
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

    # If images are multiband, band coordinate is not to be reduced
    channel_axis = None
    if len(img_left_fill.shape) == 3:
        channel_axis = 0
    # Create image pyramids
    images_left = get_pyramids(img_left_fill, num_scales, scale_factor, channel_axis)
    images_right = get_pyramids(img_right_fill, num_scales, scale_factor, channel_axis)
    disparities_left_min = get_pyramids(
        # convert to float for the zoom
        img_left["disparity"].data[0].astype(np.float32),
        num_scales,
        scale_factor,
    )
    disparities_left_max = get_pyramids(
        # convert to float for the zoom
        img_left["disparity"].data[1].astype(np.float32),
        num_scales,
        scale_factor,
    )
    compute_right_disps = "disparity" in list(img_right.keys())
    disparities_right_min = None
    disparities_right_max = None
    if compute_right_disps:
        disparities_right_min = get_pyramids(
            img_right["disparity"].data[0].astype(np.float32), num_scales, scale_factor
        )
        disparities_right_max = get_pyramids(
            img_right["disparity"].data[1].astype(np.float32), num_scales, scale_factor
        )

    # Create mask pyramids
    masks_left = masks_pyramid(msk_left_fill, scale_factor, num_scales)
    masks_right = masks_pyramid(msk_right_fill, scale_factor, num_scales)

    # Create dataset pyramids
    pyramid_left = convert_pyramid_to_dataset(
        img_left, images_left, masks_left, [disparities_left_min, disparities_left_max]
    )
    pyramid_right = convert_pyramid_to_dataset(
        img_right,
        images_right,
        masks_right,
        [disparities_right_min, disparities_right_max] if compute_right_disps else None,
    )

    # The pyramid is intended to be from coarse to original size, so we inverse its order.
    return pyramid_left[::-1], pyramid_right[::-1]


def fill_nodata_image(dataset: xr.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate no data values in image. If no mask was given, create all valid masks

    :param dataset: Dataset image containing the image im : 2D (row, col) xarray.Dataset
    :type dataset: xarray.Dataset
    :return: a Tuple that contains the filled image and mask
    :rtype: Tuple of np.ndarray
    """
    if "msk" in dataset:
        if len(dataset["im"].data.shape) == 2:
            img, msk = interpolate_nodata_sgm(
                dataset["im"].data,
                dataset["msk"].data,
                cst.PANDORA_MSK_PIXEL_INVALID,
                cst.PANDORA_MSK_PIXEL_FILLED_NODATA,
            )
        else:
            img = dataset["im"].data
            msk = dataset["msk"].data
            nband = dataset["im"].data.shape[0]

            for band in range(nband):
                img[band, :, :], msk[:, :] = interpolate_nodata_sgm(
                    dataset["im"].data[band, :, :],
                    dataset["msk"].data[:, :],
                    cst.PANDORA_MSK_PIXEL_INVALID,
                    cst.PANDORA_MSK_PIXEL_FILLED_NODATA,
                )
    else:
        msk = np.full(
            (dataset.sizes["row"], dataset.sizes["col"]),
            int(dataset.attrs["valid_pixels"]),
        )
        img = dataset["im"].data
    return img, msk


interpolate_nodata_sgm = img_tools_cpp.interpolate_nodata_sgm


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
    msk_pyramid = [msk]
    # Add the full resolution mask
    tmp_msk = msk
    for _ in range(num_scales - 1):
        # Decimate in the two axis
        tmp_msk = tmp_msk[::scale_factor, ::scale_factor]
        msk_pyramid.append(tmp_msk)
    return msk_pyramid


def convert_pyramid_to_dataset(
    img_orig: xr.Dataset, images: List[np.ndarray], masks: List[np.ndarray], disps=None
) -> List[xr.Dataset]:
    """
    Return a List with the datasets at the different scales

    :param img_orig: Dataset image containing the image im : 2D (row, col) xarray.Dataset
    :type img_orig: xarray.Dataset
    :param images: list of images for the pyramid
    :type images: list[np.ndarray]
    :param masks: list of masks for the pyramid
    :type masks: list[np.ndarray]
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
        if len(img_orig["im"].shape) == 2:
            dataset = xr.Dataset(
                {"im": (["row", "col"], image.astype(np.float32))},
                coords={"row": np.arange(image.shape[0]), "col": np.arange(image.shape[1])},
            )
            # Allocate the mask
            dataset["msk"] = xr.DataArray(
                np.full((image.shape[0], image.shape[1]), masks[index].astype(np.int16)),
                dims=["row", "col"],
            )
            # add the disparity if provided
            if disps is not None:

                dataset.coords["band_disp"] = ["min", "max"]
                dataset["disparity"] = xr.DataArray(
                    np.array([disps[0][index].astype(np.int64), disps[1][index].astype(np.int64)]),
                    dims=["band_disp", "row", "col"],
                )

        else:
            dataset = xr.Dataset(
                {"im": (["band_im", "row", "col"], image.astype(np.float32))},
                coords={
                    "band_im": img_orig.coords["band_im"].data,
                    "row": np.arange(image.shape[1]),
                    "col": np.arange(image.shape[2]),
                },
            )
            # Allocate the mask
            dataset["msk"] = xr.DataArray(
                np.full((image.shape[1], image.shape[2]), masks[index].astype(np.int16)),
                dims=["row", "col"],
            )
            # add the disparity if provided
            if disps is not None:

                dataset.coords["band_disp"] = ["min", "max"]
                dataset["disparity"] = xr.DataArray(
                    np.array([disps[0][index].astype(np.int64), disps[1][index].astype(np.int64)]),
                    dims=["band_disp", "row", "col"],
                )
        # Add image conf to the image dataset
        # - attributes are linked to each others
        dataset.attrs = img_orig.attrs
        pyramid.append(dataset)

    return pyramid


def shift_right_img(img_right: xr.Dataset, subpix: int, band: str = None, order: int = 1) -> List[xr.Dataset]:
    """
    Return an array that contains the shifted right images

    :param img_right: Dataset image containing the image im : 2D (row, col) xarray.Dataset
    :type img_right: xarray.Dataset
    :param subpix: subpixel precision = (1 or pair number)
    :type subpix: int
    :param band: User's value for selected band
    :type band: str
    :param order: order parameter on zoom method
    :type order: int
    :return: an array that contains the shifted right images
    :rtype: array of xarray.Dataset
    """
    img_right_shift = [img_right]
    ny_, nx_ = img_right.sizes["row"], img_right.sizes["col"]

    if band is None:
        selected_band = img_right["im"].data
    else:
        # if image has more than one band we only shift the one specified in matching_cost
        band_index_right = list(img_right.band_im.data).index(band)
        selected_band = img_right["im"].data[band_index_right, :, :]

    if subpix > 1:
        for ind in np.arange(1, subpix):
            shift = 1 / subpix
            # For each index, shift the right image for subpixel precision 1/subpix*index
            data = zoom(selected_band, (1, (nx_ * subpix - (subpix - 1)) / float(nx_)), order=order)[:, ind::subpix]
            col = np.arange(
                img_right.coords["col"].values[0] + shift * ind, img_right.coords["col"].values[-1], step=1
            )  # type: np.ndarray
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

    :param img: Dataset image containing the image im : 2D (row, col) xarray.Dataset
    :type img: xarray.Dataset
    :param row: row coordinates
    :type row: int
    :param col: column coordinates
    :type col: int
    :return: True if the coordinates row,col are inside the image
    :rtype: boolean
    """
    ny_, nx_ = img.sizes["row"], img.sizes["col"]
    return 0 <= row < nx_ and 0 <= col < ny_


def census_transform(image: xr.Dataset, window_size: int, band: str = None) -> xr.Dataset:
    """
    Generates the census transformed image from an image

    :param image: Dataset image containing the image im : 2D (row, col) xarray.Dataset
    :type image: xarray.Dataset
    :param window_size: Census window size
    :type window_size: int
    :param band: User's value for selected band
    :type band: str
    :return: Dataset census transformed uint32 containing the transformed image im: 2D (row, col) xarray.DataArray \
    uint32
    :rtype: xarray.Dataset
    """

    ny_, nx_ = image.sizes["row"], image.sizes["col"]

    if len(image["im"].shape) > 2:
        band_index = list(image.band_im.data).index(band)
        selected_band = image["im"].data[band_index, :, :]
    else:
        selected_band = image["im"].data

    border = int((window_size - 1) / 2)

    # Create a sliding window of using as_strided function : this function create a new a view (by manipulating data
    #  pointer) of the image array with a different shape. The new view pointing to the same memory block as
    # image, so it does not consume any additional memory.
    str_row, str_col = selected_band.strides
    shape_windows = (
        ny_ - (window_size - 1),
        nx_ - (window_size - 1),
        window_size,
        window_size,
    )
    strides_windows = (str_row, str_col, str_row, str_col)
    windows = np.lib.stride_tricks.as_strided(selected_band, shape_windows, strides_windows, writeable=False)

    # Pixels inside the image which can be centers of windows
    central_pixels = selected_band[border:-border, border:-border]

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


def compute_mean_raster(img: xr.Dataset, win_size: int, band: str = None) -> np.ndarray:
    """
    Compute the mean within a sliding window for the whole image

    :param img: Dataset image containing the image im : 2D (row, col) xarray.Dataset
    :type img: xarray.Dataset
    :param win_size: window size
    :type win_size: int
    :param band: User's value for selected band
    :type band: str
    :return: mean raster
    :rtype: numpy array
    """
    # Right image can have 3 dim if its from dataset or 2 if its from shift_right_image function
    ny_, nx_ = img.sizes["row"], img.sizes["col"]
    if len(img["im"].shape) > 2:
        band_index = list(img.band_im.data).index(band)
        # Example with win_size = 3 and the input :
        #           10 | 5  | 3
        #            2 | 10 | 5
        #            5 | 3  | 1

        # Compute the cumulative sum of the elements along the column axis
        r_mean = np.r_[np.zeros((1, nx_)), img["im"].data[band_index, :, :]]
    else:
        r_mean = np.r_[np.zeros((1, nx_)), img["im"].data]
    # r_mean :   0 | 0  | 0
    #           10 | 5  | 3
    #            2 | 10 | 5
    #            5 | 3  | 1
    r_mean = np.nancumsum(r_mean, axis=0)
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


find_valid_neighbors = img_tools_cpp.find_valid_neighbors


def compute_mean_patch(img: xr.Dataset, row: int, col: int, win_size: int) -> np.float64:
    """
    Compute the mean within a window centered at position row,col

    :param img: Dataset image containing the image im : 2D (row, col) xarray.Dataset
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

    raise IndexError("The window is outside the image")


def compute_std_raster(img: xr.Dataset, win_size: int, band: str = None) -> np.ndarray:
    """
    Compute the standard deviation within a sliding window for the whole image
    with the formula : std = sqrt( E[row^2] - E[row]^2 )

    :param img: Dataset image containing the image im : 2D (row, col) xarray.Dataset
    :type img: xarray.Dataset
    :param win_size: window size
    :type win_size: int
    :param band: User's value for selected band
    :type band: str
    :return: std raster
    :rtype: numpy array
    """
    # Computes E[row]
    mean_ = compute_mean_raster(img, win_size, band)
    # Right image can have 3 dim if its from dataset or 2 if its from shift_right_image function
    if len(img["im"].shape) > 2:
        band_index = list(img.band_im.data).index(band)
        selected_band = img["im"].data[band_index, :, :]
    else:
        selected_band = img["im"].data

    # Computes E[row^2]
    raster_power_two = xr.Dataset(
        {"im": (["row", "col"], selected_band**2)},
        coords={
            "row": np.arange(selected_band.shape[0]),
            "col": np.arange(selected_band.shape[1]),
        },
    )
    mean_power_two = compute_mean_raster(raster_power_two, win_size)

    # Compute sqrt( E[row^2] - E[row]^2 )
    var = mean_power_two - mean_**2
    # Avoid very small values
    var[np.where(var < (10 ** (-15) * abs(mean_power_two)))] = 0
    return np.sqrt(var)


def read_disp(disparity: Tuple[int, int] | list[int] | str) -> Tuple[int, int] | Tuple[np.ndarray, np.ndarray]:
    """
    Read the disparity :
        - if cfg_disp is the path of a disparity grid, read and return the grids (type tuple of numpy arrays)
        - else return the value of cfg_disp

    :param disparity: disparity, or path to the disparity grid
    :type disparity: Tuple[int, int] or list[int] or str
    :return: the disparity
    :rtype: Tuple[int, int] | Tuple[np.ndarray, np.ndarray]
    """
    if disparity is None:
        raise ValueError("disparity should not be None")

    if not isinstance(disparity, str):
        # cast because of mypy when we give list as input while it expects a tuple as output
        # not sure if it is the best solution
        return cast(Tuple[int, int], tuple(disparity))

    raster_disparity = rasterio_open(disparity)
    return raster_disparity.read(1), raster_disparity.read(2)


def fuse_classification_bands(img: xr.Dataset, class_names: List[str]) -> np.ndarray:
    """
    Get the multiband classification map present in the input image dataset
    and select the given classes to make a single-band classification map

    :param img: image dataset
    :type img: xr.Dataset
    :param class_names: chosen classification classes
    :type class_names: List[str]
    :return: the map representing the selected classifications
    :rtype: np.ndarray
    """
    # Non classified pixels have value 0
    monoband_classif = np.zeros((len(img.coords["row"]), len(img.coords["col"])))
    for count, class_name in enumerate(class_names):
        # Each class must have a different pixel value
        band_index = list(img.band_classif.data).index(class_name)
        pixel_val = count + 1
        monoband_classif += pixel_val * img["classif"].data[band_index, :, :]

    return monoband_classif
