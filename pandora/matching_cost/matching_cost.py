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
This module contains functions associated to the cost volume measure step.
"""
# pylint:disable=too-many-branches
import sys
import logging
from abc import ABCMeta, abstractmethod
from math import ceil, floor
from typing import Tuple, List, Union, Dict

import numpy as np
import xarray as xr
from scipy.ndimage import binary_dilation

from pandora.img_tools import shift_right_img


class AbstractMatchingCost:
    """
    Abstract Matching Cost class
    """

    __metaclass__ = ABCMeta

    matching_cost_methods_avail: Dict = {}
    _subpix = None
    _window_size = None
    cfg = None
    _band = None

    # Default configuration, do not change these values
    _WINDOW_SIZE = 5
    _SUBPIX = 1
    _BAND = None

    def __new__(cls, **cfg: Union[str, int]):
        """
        Return the plugin associated with the matching_cost_method given in the configuration

        :param cfg: configuration {'matching_cost_method': value}
        :type cfg: dictionary
        """

        if cls is AbstractMatchingCost:
            if isinstance(cfg["matching_cost_method"], str):
                try:
                    return super(AbstractMatchingCost, cls).__new__(
                        cls.matching_cost_methods_avail[cfg["matching_cost_method"]]
                    )
                except:
                    raise KeyError("No matching cost method named {} supported".format(cfg["matching_cost_method"]))
            else:
                if isinstance(cfg["matching_cost_method"], unicode):  # type:ignore # pylint:disable=undefined-variable
                    # creating a plugin from registered short name given as unicode (py2 & 3 compatibility)
                    try:
                        return super(AbstractMatchingCost, cls).__new__(
                            cls.matching_cost_methods_avail[cfg["matching_cost_method"].encode("utf-8")]
                        )
                    except:
                        raise KeyError("No matching cost method named {} supported".format(cfg["matching_cost_method"]))
        else:
            return super(AbstractMatchingCost, cls).__new__(cls)
        return None

    @classmethod
    def register_subclass(cls, short_name: str, *args):
        """
        Allows to register the subclass with its short name

        :param short_name: the subclass to be registered
        :type short_name: string
        :param args: allows to register one plugin that contains different methods
        """

        def decorator(subclass):
            """
            Registers the subclass in the available methods

            :param subclass: the subclass to be registered
            :type subclass: object
            """
            cls.matching_cost_methods_avail[short_name] = subclass
            for arg in args:
                cls.matching_cost_methods_avail[arg] = subclass
            return subclass

        return decorator

    @abstractmethod
    def desc(self) -> None:
        """
        Describes the matching cost method
        :return: None
        """
        print("Matching cost description")

    def instantiate_class(self, **cfg: Union[str, int]) -> None:
        """
        :param cfg: optional configuration,  {'window_size': int, 'subpix': int,
                                                'band': str}
        :type cfg: dictionary
        :return: None
        """
        self.cfg = self.check_conf(**cfg)  # type: ignore
        self._window_size = self.cfg["window_size"]
        self._subpix = self.cfg["subpix"]
        self._band = self.cfg["band"]

    def check_conf(self, **cfg: Dict[str, Union[str, int]]) -> Dict[str, Union[str, int]]:
        """
        Add default values to the dictionary if there are missing elements and check if the dictionary is correct

        :param cfg: matching cost configuration
        :type cfg: dict
        :return cfg: matching cost configuration updated
        :rtype: dict
        """

        # Give the default value if the required element is not in the conf
        if "window_size" not in cfg:
            cfg["window_size"] = self._WINDOW_SIZE  # type: ignore
        if "subpix" not in cfg:
            cfg["subpix"] = self._SUBPIX  # type: ignore
        if "band" not in cfg:
            cfg["band"] = self._BAND

        return cfg  # type: ignore

    def check_band_input_mc(self, img_left: xr.Dataset, img_right: xr.Dataset) -> None:
        """
        Check coherence band parameter between inputs and matching cost step

        :param img_left: left Dataset image containing :

                - im : 2D (row, col) or 3D (band, row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :type img_left: xarray.Dataset
        :param img_right: right Dataset  containing :

                - im : 2D (row, col) or 3D (band, row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :type img_right: xarray.Dataset
        :return: None
        """
        if self._band is not None:
            try:
                list(img_right.band.data)
            except AttributeError:
                logging.error("Right dataset is monoband: %s band cannot be selected", self._band)
                sys.exit(1)
            try:
                list(img_left.band.data)
            except AttributeError:
                logging.error("Left dataset is monoband: %s band cannot be selected", self._band)
                sys.exit(1)
            if (self._band not in list(img_right.band.data)) or (self._band not in list(img_left.band.data)):
                logging.error("Wrong band instantiate : %s not in img_left or img_right", self._band)
                sys.exit(1)
        else:
            try:
                list(img_right.band.data)
            except AttributeError:
                return
            try:
                list(img_left.band.data)
            except AttributeError:
                return
            if (img_right.band.data is not None) or (img_left.band.data is not None):
                logging.error("Band must be instantiated in matching cost step")
                sys.exit(1)

    @abstractmethod
    def compute_cost_volume(
        self, img_left: xr.Dataset, img_right: xr.Dataset, disp_min: int, disp_max: int
    ) -> xr.Dataset:
        """
        Computes the cost volume for a pair of images

        :param img_left: left Dataset image containing :

                - im : 2D (row, col) or 3D (band, row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :type img_left: xarray.Dataset
        :param img_right: right Dataset  containing :

                - im : 2D (row, col) or 3D (band, row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :type img_right: xarray.Dataset
        :param disp_min: minimum disparity
        :type disp_min: int
        :param disp_max: maximum disparity
        :type disp_max: int
        :return: the cost volume dataset with the data variables:

                - cost_volume 3D xarray.DataArray (row, col, disp)
        :rtype: xarray.Dataset
        """

    @staticmethod
    def allocate_costvolume(
        img_left: xr.Dataset,
        subpix: int,
        disp_min: int,
        disp_max: int,
        window_size: int,
        metadata: dict,
        np_data: np.ndarray = None,
    ) -> xr.Dataset:
        """
        Allocate the cost volume

        :param img_left: left Dataset image containing :

                - im : 2D (row, col) or 3D (band, row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :type img_left: xarray.Dataset
        :param subpix: subpixel precision = (1 or 2 or 4)
        :type subpix: int
        :param disp_min: minimum disparity
        :type disp_min: int
        :param disp_max: maximum disparity
        :type disp_max: int
        :param window_size: size of the aggregation window
        :type window_size: int, odd number
        :param metadata: dictionary storing arbitrary metadata
        :type metadata: dictionary
        :param np_data: the array’s data
        :type np_data: 3D numpy array, dtype=np.float32
        :return: the dataset cost volume with the cost_volume :

                - cost_volume 3D xarray.DataArray (row, col, disp)
        :rtype: xarray.Dataset
        """
        c_row = img_left["im"].coords["row"]
        c_col = img_left["im"].coords["col"]

        # First pixel in the image that is fully computable (aggregation windows are complete)
        row = np.arange(c_row[0], c_row[-1] + 1)  # type: np.ndarray
        col = np.arange(c_col[0], c_col[-1] + 1)  # type: np.ndarray

        # Compute the disparity range
        if subpix == 1:
            disparity_range = np.arange(disp_min, disp_max + 1)
        else:
            disparity_range = np.arange(disp_min, disp_max, step=1 / float(subpix), dtype=np.float64)
            disparity_range = np.append(disparity_range, [disp_max])

        # Create the cost volume
        if np_data is None:
            np_data = np.zeros((len(row), len(col), len(disparity_range)), dtype=np.float32)

        cost_volume = xr.Dataset(
            {"cost_volume": (["row", "col", "disp"], np_data)},
            coords={"row": row, "col": col, "disp": disparity_range},
        )
        cost_volume.attrs = metadata

        cost_volume.attrs["crs"] = img_left.attrs["crs"]
        cost_volume.attrs["transform"] = img_left.attrs["transform"]

        cost_volume.attrs["window_size"] = window_size

        return cost_volume

    @staticmethod
    def point_interval(
        img_left: xr.Dataset, img_right: xr.Dataset, disp: float
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Computes the range of points over which the similarity measure will be applied

        :param img_left: left Dataset image containing :

                - im : 2D (row, col) or 3D (band, row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :type img_left: xarray.Dataset
        :param img_right: right Dataset image containing :

                - im : 2D (row, col) or 3D (band, row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :type img_right: xarray.Dataset
        :param disp: current disparity
        :type disp: float
        :return: the range of the left and right image over which the similarity measure will be applied
        :rtype: tuple
        """
        nx_left = img_left.dims["col"]
        nx_right = img_right.dims["col"]

        # range in the left image
        point_p = (max(0 - disp, 0), min(nx_left - disp, nx_left))
        # range in the right image
        point_q = (max(0 + disp, 0), min(nx_right + disp, nx_right))

        # Because the disparity can be floating
        if disp < 0:
            point_p = (int(ceil(point_p[0])), int(ceil(point_p[1])))
            point_q = (int(ceil(point_q[0])), int(ceil(point_q[1])))
        else:
            point_p = (int(floor(point_p[0])), int(floor(point_p[1])))
            point_q = (int(floor(point_q[0])), int(floor(point_q[1])))

        return point_p, point_q

    @staticmethod
    def masks_dilatation(
        img_left: xr.Dataset, img_right: xr.Dataset, window_size: int, subp: int
    ) -> Tuple[xr.DataArray, List[xr.DataArray]]:
        """
        Return the left and right mask with the convention :
            - Invalid pixels are nan
            - No_data pixels are nan
            - Valid pixels are 0

        Apply dilation on no_data : if a pixel contains a no_data in its aggregation window, then the central pixel
        becomes no_data

        :param img_left: left Dataset image containing :

                - im : 2D (row, col) or 3D (band, row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :type img_left: xarray.Dataset
        :param img_right: right Dataset image containing :

                - im : 2D (row, col) or 3D (band, row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :type img_right: xarray.Dataset
        :param window_size: window size of the measure
        :type window_size: int
        :param subp: subpixel precision = (1 or 2 or 4)
        :type subp: int
        :return: the left mask and the right masks:

                - left mask :  xarray.DataArray msk 2D(row, col)
                - right mask :  xarray.DataArray msk 2D(row, col)
                - right mask shifted :  xarray.DataArray msk 2D(row, shifted col by 0.5)
        :rtype: tuple (left mask, list[right mask, right mask shifted by 0.5])
        """
        # Create the left mask with the convention : 0 = valid, nan = invalid and no_data
        if "msk" in img_left.data_vars:
            dilatate_left_mask = np.zeros(img_left["msk"].shape)
            # Invalid pixels are nan
            dilatate_left_mask[
                np.where(
                    (img_left["msk"].data != img_left.attrs["valid_pixels"])
                    & (img_left["msk"].data != img_left.attrs["no_data_mask"])
                )
            ] = np.nan
            # Dilatation : pixels that contains no_data in their aggregation window become no_data = nan
            dil = binary_dilation(
                img_left["msk"].data == img_left.attrs["no_data_mask"],
                structure=np.ones((window_size, window_size)),
                iterations=1,
            )
            dilatate_left_mask[dil] = np.nan
        else:
            # All pixels are valid
            dilatate_left_mask = np.zeros((img_left.dims["row"], img_left.dims["col"]))

        # Create the right mask with the convention : 0 = valid, nan = invalid and no_data
        if "msk" in img_right.data_vars:
            dilatate_right_mask = np.zeros(img_right["msk"].shape)
            # Invalid pixels are nan
            dilatate_right_mask[
                np.where(
                    (img_right["msk"].data != img_right.attrs["valid_pixels"])
                    & (img_right["msk"].data != img_right.attrs["no_data_mask"])
                )
            ] = np.nan
            # Dilatation : pixels that contains no_data in their aggregation window become no_data = nan
            dil = binary_dilation(
                img_right["msk"].data == img_right.attrs["no_data_mask"],
                structure=np.ones((window_size, window_size)),
                iterations=1,
            )
            dilatate_right_mask[dil] = np.nan
        else:
            # All pixels are valid
            dilatate_right_mask = np.zeros((img_left.dims["row"], img_left.dims["col"]))

        ny_, nx_ = img_left.dims["row"], img_left.dims["col"]

        row = np.arange(0, ny_)
        col = np.arange(0, nx_)

        # Shift the right mask, for sub-pixel precision. If an no_data or invalid pixel was used to create the
        # sub-pixel point, then the sub-pixel point is invalidated / no_data.
        dilatate_right_mask_shift = xr.DataArray()

        if subp != 1:
            # Since the interpolation of the right image is of order 1, the shifted right mask corresponds
            # to an aggregation of two columns of the dilated right mask

            str_row, str_col = dilatate_right_mask.strides
            shape_windows = (
                dilatate_right_mask.shape[0],
                dilatate_right_mask.shape[1] - 1,
                2,
            )

            strides_windows = (str_row, str_col, str_col)
            aggregation_window = np.lib.stride_tricks.as_strided(dilatate_right_mask, shape_windows, strides_windows)
            dilatate_right_mask_shift = np.sum(aggregation_window, 2)

            # Whatever the sub-pixel precision, only one sub-pixel mask is created,
            # since 0.5 shifted mask == 0.25 shifted mask
            col_shift = np.arange(0 + 0.5, nx_ - 1, step=1)
            dilatate_right_mask_shift = xr.DataArray(
                dilatate_right_mask_shift, coords=[row, col_shift], dims=["row", "col"]
            )

        dilatate_left_mask_xr = xr.DataArray(dilatate_left_mask, coords=[row, col], dims=["row", "col"])
        dilatate_right_mask_xr = xr.DataArray(dilatate_right_mask, coords=[row, col], dims=["row", "col"])

        return dilatate_left_mask_xr, [dilatate_right_mask_xr, dilatate_right_mask_shift]

    @staticmethod
    def dmin_dmax(disp_min: Union[int, np.ndarray], disp_max: Union[int, np.ndarray]) -> Tuple[int, int]:
        """
        Find the smallest disparity present in disp_min, and the highest disparity present in disp_max

        :param disp_min: minimum disparity
        :type disp_min: int or np.ndarray
        :param disp_max: maximum disparity
        :type disp_max: int or np.ndarray
        :return: dmin_min: the smallest disparity in disp_min, dmax_max: the highest disparity in disp_max
        :rtype: Tuple(int, int)
        """
        # Disp_min is a fixed disparity
        if isinstance(disp_min, int):
            dmin_min = disp_min

        # Disp_min is a variable disparity
        else:
            dmin_min = int(np.nanmin(disp_min))

        # Disp_max is a fixed disparity
        if isinstance(disp_max, int):
            dmax_max = disp_max

        # Disp_max is a variable disparity
        else:
            dmax_max = int(np.nanmax(disp_max))

        return dmin_min, dmax_max

    def cv_masked(
        self,
        img_left: xr.Dataset,
        img_right: xr.Dataset,
        cost_volume: xr.Dataset,
        disp_min: Union[int, np.ndarray],
        disp_max: Union[int, np.ndarray],
    ) -> None:
        """
        Masks the cost volume :
            - costs which are not inside their disparity range, are masked with a nan value
            - costs of invalid_pixels (invalidated by the input image mask), are masked with a nan value
            - costs of no_data pixels, are masked with a nan value. If a valid pixel contains a no_data in its
                aggregation window, then the cost of the central pixel is masked with a nan value

        :param img_left: left Dataset image containing :

                - im : 2D (row, col) or 3D (band, row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :type img_left: xarray.Dataset
        :param img_right: right Dataset image containing :

                - im : 2D (row, col) or 3D (band, row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :type img_right: xarray.Dataset
        :param cost_volume: the cost_volume DataSet with the data variables:

                - cost_volume 3D xarray.DataArray (row, col, disp)
        :type cost_volume: xarray.Dataset
        :param disp_min: minimum disparity
        :type disp_min: int or np.ndarray
        :param disp_max: maximum disparity
        :type disp_max: int or np.ndarray
        :param cfg: images configuration containing the mask convention : valid_pixels, no_data
        :type cfg: dict
        :return: None
        """
        ny_, nx_, nd_ = cost_volume["cost_volume"].shape

        dmin, _ = self.dmin_dmax(disp_min, disp_max)

        # ----- Masking invalid pixels -----

        # Contains the shifted right images
        img_right_shift = shift_right_img(img_right, self._subpix, self._band)  # type: ignore

        # Computes the validity mask of the cost volume : invalid pixels or no_data are masked with the value nan.
        # Valid pixels are = 0
        mask_left, mask_right = self.masks_dilatation(
            img_left, img_right, self._window_size, self._subpix  # type: ignore
        )

        for disp in cost_volume.coords["disp"].data:
            i_right = int((disp % 1) * self._subpix)
            point_p, point_q = self.point_interval(img_left, img_right_shift[i_right], disp)

            # Point interval in the left image
            p_mask = (point_p[0], point_p[1])
            # Point interval in the right image
            q_mask = (point_q[0], point_q[1])

            i_mask_right = min(1, i_right)
            dsp = int((disp - dmin) * self._subpix)

            # Invalid costs in the cost volume
            cost_volume["cost_volume"].data[:, p_mask[0] : p_mask[1], dsp] += (
                mask_right[i_mask_right].data[:, q_mask[0] : q_mask[1]] + mask_left.data[:, p_mask[0] : p_mask[1]]
            )

        # ----- Masking disparity range -----

        # Fixed range of disparities
        if isinstance(disp_min, np.ndarray) and isinstance(disp_max, np.ndarray):
            # Disparity range may be one size bigger in y axis
            if disp_min.shape[0] > ny_:
                disp_min = disp_min[0:ny_, :]
                disp_max = disp_max[0:ny_, :]
            if disp_min.shape[1] > nx_:
                disp_min = disp_min[:, 0:nx_]
                disp_max = disp_max[:, 0:nx_]

            # Mask the costs computed with a disparity lower than disp_min and higher than disp_max
            for dsp in range(nd_):
                masking = np.where(
                    np.logical_or(
                        cost_volume.coords["disp"].data[dsp] < disp_min,
                        cost_volume.coords["disp"].data[dsp] > disp_max,
                    )
                )
                cost_volume["cost_volume"].data[masking[0], masking[1], dsp] = np.nan

    @staticmethod
    def allocate_numpy_cost_volume(
        img_left: xr.Dataset, disparity_range: np.ndarray, offset_row_col: int
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        """
        Allocate the numpy cost volume cv = (disp, col, row), for efficient memory management

        :param img_left: left Dataset image containing :

                - im : 2D (row, col) or 3D (band, row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :type img_left: xarray.Dataset
        :param disparity_range: disparity range
        :type disparity_range: np.ndarray
        :param offset_row_col: offset in row and col
        :type offset_row_col: int
        :return: the cost volume dataset , with the data variables:

                - cost_volume 3D xarray.DataArray (row, col, disp)
                - a cropped cost volume dataset
        :rtype: Tuple[xarray.Dataset, xarray.Dataset]
        """

        cv = np.zeros(
            (len(disparity_range), img_left.dims["col"], img_left.dims["row"]),
            dtype=np.float32,
        )
        cv += np.nan

        # If offset, do not consider border position for cost computation
        if offset_row_col != 0:
            cv_crop = cv[:, offset_row_col:-offset_row_col, offset_row_col:-offset_row_col]
        else:
            cv_crop = cv

        return cv, cv_crop  # type: ignore
