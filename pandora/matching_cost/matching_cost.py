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
from abc import ABCMeta, abstractmethod
from math import ceil, floor
from typing import Tuple, List, Union, Dict
from json_checker import And, Or
import numpy as np
import xarray as xr
from scipy.ndimage import binary_dilation

from pandora.margins.descriptors import HalfWindowMargins
from pandora.img_tools import shift_right_img
from pandora.criteria import mask_invalid_variable_disparity_range, mask_border


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
    _step_col = None
    _method = None

    # Default configuration, do not change these values
    _WINDOW_SIZE = 5
    _SUBPIX = 1
    _BAND = None
    _STEP_COL = 1

    # Matching cost schema confi
    schema = {
        "subpix": And(int, lambda input: input > 0 and ((input % 2) == 0) or input == 1),
        "band": Or(str, lambda input: input is None),
        "step": And(int, lambda y: y >= 1),
    }

    margins = HalfWindowMargins()

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

    def desc(self) -> None:
        """
        Describes the matching cost method
        :return: None
        """
        print(f"{self._method} similarity measure")

    def instantiate_class(self, **cfg: Union[str, int]) -> None:
        """
        :param cfg: optional configuration,  {'window_size': int, 'subpix': int,
                                                'band': str}
        :type cfg: dictionary
        :return: None
        """
        self.cfg = self.check_conf(**cfg)  # type: ignore
        self._window_size = int(self.cfg["window_size"])
        self._subpix = int(self.cfg["subpix"])
        self._band = self.cfg["band"]
        self._step_col = int(self.cfg["step"])
        self._method = str(self.cfg["matching_cost_method"])

    def check_conf(self, **cfg: Dict[str, Union[str, int]]) -> Dict:
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

        if "pandora2d" not in sys.modules:
            if "step" in cfg and cfg["step"] != 1:
                raise ValueError("Step parameter cannot be different from 1")
        if "step" not in cfg:
            cfg["step"] = self._STEP_COL  # type: ignore

        return cfg

    def check_band_input_mc(self, img_left: xr.Dataset, img_right: xr.Dataset) -> None:
        """
        Check coherence band parameter between inputs and matching cost step

        :param img_left: left Dataset image containing :

                - im: 2D (row, col) or 3D (band_im, row, col) xarray.DataArray float32
                - disparity (optional): 3D (disp, row, col) xarray.DataArray float32
                - msk (optional): 2D (row, col) xarray.DataArray int16
                - classif (optional): 3D (band_classif, row, col) xarray.DataArray int16
                - segm (optional): 2D (row, col) xarray.DataArray int16
        :type img_left: xarray.Dataset
        :param img_right: right Dataset  containing :

                - im: 2D (row, col) or 3D (band_im, row, col) xarray.DataArray float32
                - disparity (optional): 3D (disp, row, col) xarray.DataArray float32
                - msk (optional): 2D (row, col) xarray.DataArray int16
                - classif (optional): 3D (band_classif, row, col) xarray.DataArray int16
                - segm (optional): 2D (row, col) xarray.DataArray int16
        :type img_right: xarray.Dataset
        :return: None
        """
        if self._band is not None:
            try:
                list(img_right.band_im.data)
            except AttributeError:
                raise AttributeError(f"Right dataset is monoband: {self._band} band cannot be selected")
            try:
                list(img_left.band_im.data)
            except AttributeError:
                raise AttributeError(f"Left dataset is monoband: {self._band} band cannot be selected")
            if (self._band not in list(img_right.band_im.data)) or (self._band not in list(img_left.band_im.data)):
                raise AttributeError(f"Wrong band instantiate : {self._band} not in img_left or img_right")
        else:
            try:
                list(img_right.band_im.data)
            except AttributeError:
                return
            try:
                list(img_left.band_im.data)
            except AttributeError:
                return
            if (img_right.band_im.data is not None) or (img_left.band_im.data is not None):
                raise AttributeError("Band must be instantiated in matching cost step")

    @abstractmethod
    def compute_cost_volume(
        self,
        img_left: xr.Dataset,
        img_right: xr.Dataset,
        cost_volume: xr.Dataset,
    ) -> xr.Dataset:
        """
        Computes the cost volume for a pair of images

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
        :param cost_volume: an empty cost volume
        :type cost_volume: xr.Dataset
        :return: the cost volume dataset , with the data variables:

                - cost_volume 3D xarray.DataArray (row, col, disp)
        :rtype: xarray.Dataset
        """

    def grid_estimation(
        self, img: xr.Dataset, cfg: Union[Dict[str, dict], None], disparity_grids: Tuple[np.ndarray, np.ndarray]
    ) -> xr.Dataset:
        """
        Estimation of the grid xarray dataset that will store the cost volume.

        :param img: left Dataset image
        :type img: xarray.Dataset
        :param cfg: user configuration
        :type cfg: dict
        :param disparity_grids: Tuple of disparity grids min and max
        :type disparity_grids: Tuple[np.ndarray, np.ndarray]
        :return: a grid with coordinates and a attributes list with:

                - indexs of columns to compute
                - size
                - sampling interval
        :rtype: xarray.Dataset
        """
        # Get col dimension
        c_col = img["im"].coords["col"]
        col = np.arange(c_col[0], c_col[-1] + 1, self._step_col)  # type: np.ndarray

        # get disparity_range
        disparity_min, disparity_max = self.get_min_max_from_grid(*disparity_grids)
        disparity_range = self.get_disparity_range(disparity_min, disparity_max, self._subpix)

        # Index to be kept in the cost volume once it has been calculated
        index_compute_col = col

        # Get the index of the columns that should be computed
        if cfg and "ROI" in cfg:
            left_margin = cfg["ROI"]["margins"][0]

            # Check if the first column of roi is inside the final CV (with the step)
            if left_margin % self._step_col != 0:
                if left_margin < self._step_col:
                    # For example, given left_margin = 2 and step = 3
                    #
                    # roi_img = M M M M M M M
                    #           M M 1 2 3 M M
                    #           M M 4 5 6 M M
                    #           M M 7 8 9 M M
                    #           M M M M M M M
                    #
                    # Our starting point would be at index left_margin = 2 --> starting point = 1st point of ROI
                    # We are directly on the first point to compute

                    index_compute_col = np.arange(c_col[0] + left_margin, c_col[-1] + 1, self._step_col)
                else:
                    # For example, given left_margin = 3 and step = 2
                    #
                    # roi_img = M M M M M M M M M
                    #           M M M 1 2 3 M M M
                    #           M M M 4 5 6 M M M
                    #           M M M 7 8 9 M M M
                    #           M M M M M M M M M
                    #
                    # Our starting point would be at index 1 --> starting point = 2nd M
                    # With a step of 2, the first point of ROI is calculated without cropping margins too much
                    #
                    # For example, given left_margin = 4 and step = 3
                    #
                    # roi_img = M M M M M M M M M M M
                    #           M M M M 1 2 3 M M M M
                    #           M M M M 4 5 6 M M M M
                    #           M M M M 7 8 9 M M M M
                    #           M M M M M M M M M M M
                    #
                    # Our starting point would be at index 1 --> starting point = 2nd M
                    # With a step of 3, the first point of ROI is calculated without cropping margins too much

                    # give the number of the first column to compute
                    start = self._step_col - (self.find_nearest_multiple_of_step(left_margin) - left_margin)
                    index_compute_col = np.arange(c_col[0] + start, c_col[-1] + 1, self._step_col)

        # Instantiate grid
        grid = xr.Dataset(
            {},
            coords={"row": img["im"].coords["row"], "col": index_compute_col, "disp": disparity_range},
        )

        # Add img attributes
        grid.attrs = img.attrs
        # Add step in grid attributes
        grid.attrs["sampling_interval"] = self._step_col
        # Add index of columns to compute in grid attributes
        grid.attrs["col_to_compute"] = index_compute_col
        return grid

    def allocate_cost_volume(
        self, image: xr.Dataset, disparity_grids: Tuple[np.ndarray, np.ndarray], cfg: Dict = None
    ) -> xr.Dataset:
        """
        Create a cost_volume dataset.

        :param image: Image to compute cost volume from
        :type image: xr.Dataset
        :param cfg: user configuration
        :type cfg: Dict
        :param disparity_grids: Tuple of disparity grids min and max
        :type disparity_grids: Tuple[np.ndarray, np.ndarray]
        :return: a empty grid
        :rtype: xr.Dataset
        """
        cv = self.grid_estimation(image, cfg, disparity_grids)
        cv["cost_volume"] = (
            ["row", "col", "disp"],
            np.full((cv.sizes["row"], cv.sizes["col"], cv.sizes["disp"]), np.nan, dtype=np.float32),
        )
        cv.attrs.update(
            {
                "window_size": self._window_size,
                "subpixel": self._subpix,
                "band_correl": self._band,
                "offset_row_col": int((self._window_size - 1) / 2),
                "measure": self._method,
            }
        )
        return cv

    @staticmethod
    def get_disparity_range(disparity_min: int, disparity_max: int, subpix: int) -> np.ndarray:
        """
        Build disparity range and return it.

        :param disparity_min: minimum disparity
        :type disparity_min: int
        :param disparity_max: maximum disparity
        :type disparity_max: int
        :param subpix: subpixel precision = (1 or 2 or 4)
        :return: disparity range
        :rtype: np.ndarray
        """
        if subpix == 1:
            disparity_range = np.arange(disparity_min, disparity_max + 1)
        else:
            disparity_range = np.arange(disparity_min, disparity_max, 1 / float(subpix), dtype=np.float64)
            disparity_range = np.append(disparity_range, [disparity_max])
        return disparity_range

    def point_interval(
        self, img_left: xr.Dataset, img_right: xr.Dataset, disp: float
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Computes the range of points over which the similarity measure will be applied

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
        :param disp: current disparity
        :type disp: float
        :return: the range of the left and right image over which the similarity measure will be applied
        :rtype: tuple
        """
        nx_left = int(img_left.sizes["col"])
        nx_right = int(img_right.sizes["col"])

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
            dilatate_left_mask = np.zeros((img_left.sizes["row"], img_left.sizes["col"]))

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
            dilatate_right_mask = np.zeros((img_left.sizes["row"], img_left.sizes["col"]))

        ny_, nx_ = img_left.sizes["row"], img_left.sizes["col"]

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
    def get_min_max_from_grid(disp_min: np.ndarray, disp_max: np.ndarray) -> Tuple[int, int]:
        """
        Find the smallest disparity present in disp_min, and the highest disparity present in disp_max

        :param disp_min: minimum disparity
        :type disp_min: np.ndarray
        :param disp_max: maximum disparity
        :type disp_max: np.ndarray
        :return: dmin_min: the smallest disparity in disp_min, dmax_max: the highest disparity in disp_max
        :rtype: Tuple(int, int)
        """
        return int(np.nanmin(disp_min)), int(np.nanmax(disp_max))

    def find_nearest_multiple_of_step(self, value: int) -> int:
        """
        In case value is not a multiple of step, find nearest greater value for which it is the case.

        :param value: Initial value.
        :type: value: int
        :return: nearest multiple of step.
        :rtype: int
        """
        while value % self._step_col != 0:
            value += 1
        return value

    def cv_masked(
        self,
        img_left: xr.Dataset,
        img_right: xr.Dataset,
        cost_volume: xr.Dataset,
        disp_min: np.ndarray,
        disp_max: np.ndarray,
    ) -> None:
        """
        Masks the cost volume :
            - costs which are not inside their disparity range, are masked with a nan value
            - costs of invalid_pixels (invalidated by the input image mask), are masked with a nan value
            - costs of no_data pixels, are masked with a nan value. If a valid pixel contains a no_data in its
                aggregation window, then the cost of the central pixel is masked with a nan value

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
        :param cost_volume: the cost_volume DataSet with the data variables:

                - cost_volume 3D xarray.DataArray (row, col, disp)
        :type cost_volume: xarray.Dataset
        :param disp_min: minimum disparity
        :type disp_min: np.ndarray
        :param disp_max: maximum disparity
        :type disp_max: np.ndarray
        :return: None
        """

        ny_, nx_, nd_ = cost_volume["cost_volume"].shape

        dmin, _ = self.get_min_max_from_grid(disp_min, disp_max)

        # ----- Masking invalid pixels -----

        # Contains the shifted right images
        img_right_shift = shift_right_img(img_right, self._subpix, self._band)

        # Computes the validity mask of the cost volume : invalid pixels or no_data are masked with the value nan.
        # Valid pixels are = 0
        mask_left, mask_right = self.masks_dilatation(img_left, img_right, self._window_size, self._subpix)

        for disp in cost_volume.coords["disp"].data:
            i_right = int((disp % 1) * self._subpix)
            point_p, point_q = self.point_interval(img_left, img_right_shift[i_right], disp)

            # For the 2 images of shape (3,4) with a cost volume at disp
            #               1  4 -5  nan
            #               1  0  1  nan
            #               5  2 -1  nan
            # the intervals given by the point_interval function:
            #   - point_p [0, 1, 2]
            #   - point_q [1, 2 ,3]
            # and the _step_col parameter is 2
            # new index for point_p and point_q must be:
            #   - point_p [0, 2]
            #   - point_q [2] but here with np.arange(q_0, point_q[1], self._step_col) with have [1, 3]
            # To solve this problem, we look for the new point in the cost volume in relation to the step
            # chosen by the user by calling find_nearest_multiple_of_step.
            # This function return p_0 = 0 and q_0 = 2
            p_0 = self.find_nearest_multiple_of_step(point_p[0])
            q_0 = self.find_nearest_multiple_of_step(point_q[0])

            # Point interval in the left image
            p_mask = np.arange(p_0, point_p[1], self._step_col)
            # Point interval in the right image
            q_mask = np.arange(q_0, point_q[1], self._step_col)
            # Point interval in the cost volume
            # Here the indices for the left image must be divided by the step
            # to correspond to the cost volume column.
            p_cv = (p_mask / self._step_col).astype(int)

            i_mask_right = min(1, i_right)
            dsp = int((disp - dmin) * self._subpix)

            # Invalid costs in the cost volume
            if p_mask.size > 0:
                cost_volume["cost_volume"].data[:, p_cv, dsp] += mask_left.data[:, p_mask]
                if q_mask.size > 0:
                    cost_volume["cost_volume"].data[:, p_cv, dsp] += mask_right[i_mask_right].data[:, q_mask]
        # ----- Masking disparity range -----

        # Fixed range of disparities
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

        # The disp_min and disp_max used to search missing disparity interval are not the local disp_min and disp_max
        # in case of a variable range of disparities. So there may be pixels that have missing disparity range (all
        # cost are np.nan), but are not detected in the validity_mask function. To find the pixels that have a missing
        # disparity range, we search in the cost volume pixels where cost_volume(row,col, for all d) = np.nan

        mask_invalid_variable_disparity_range(cost_volume)

        # Mask border pixels
        offset = cost_volume.attrs["offset_row_col"]
        if offset > 0:
            mask_border(cost_volume)

    def allocate_numpy_cost_volume(self, img_left: xr.Dataset, disparity_range: Union[np.ndarray, List]) -> np.ndarray:
        """
        Allocate the numpy cost volume cv = (disp, col, row), for efficient memory management

        :param img_left: left Dataset image containing :

                - im: 2D (row, col) or 3D (band_im, row, col) xarray.DataArray float32
                - disparity (optional): 3D (disp, row, col) xarray.DataArray float32
                - msk (optional): 2D (row, col) xarray.DataArray int16
                - classif (optional): 3D (band_classif, row, col) xarray.DataArray int16
                - segm (optional): 2D (row, col) xarray.DataArray int16
        :type img_left: xarray.Dataset
        :param disparity_range: disparity range
        :type disparity_range: np.ndarray
        :param offset_row_col: offset in row and col
        :type offset_row_col: int
        :return: the cost volume dataset , with the data variables:

                - cost_volume 3D xarray.DataArray (row, col, disp)
        :rtype: xarray.Dataset
        """

        return np.full(
            (len(disparity_range), int(img_left.sizes["col"]), int(img_left.sizes["row"])),
            np.nan,
            dtype=np.float32,
        )

    @staticmethod
    def crop_cost_volume(cost_volume: np.ndarray, offset: int = 0) -> np.ndarray:
        """
        Return a cropped view of cost_volume.

        If offset, do not consider border position for cost computation.
        :param cost_volume: cost volume to crop
        :type cost_volume: np.ndarray
        :param offset: offset used to crop cost volume
        :type offset: int
        :return: cropped view of cost_volume.
        :rtype: np.ndarray
        """
        return cost_volume[:, offset:-offset, offset:-offset] if offset else cost_volume
