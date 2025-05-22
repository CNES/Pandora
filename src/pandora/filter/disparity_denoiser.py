#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
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
This module contains functions associated to the disparity denoiser filter used to filter the disparity map.
"""
from typing import Dict, Union

import xarray as xr
from json_checker import Checker, And, Or
import numpy as np
from scipy.ndimage import gaussian_filter

from pandora.profiler import profile
import pandora.constants as cst

from . import filter  # pylint: disable=redefined-builtin


def gaussian(value, sig=1.0):
    """
    :param value: Input scalar or NumPy array of values to apply the Gaussian function to.
    :type value: float or np.ndarray
    :param sig: Standard deviation (sigma) of the Gaussian function.
    :type sig: float
    :return: The result of applying the Gaussian function to `value`.
    :rtype: float or np.ndarray
    """
    return np.exp(-np.power(value / sig, 2.0) / 2.0)


@filter.AbstractFilter.register_subclass("disparity_denoiser")
class DisparityDenoiser(filter.AbstractFilter):
    """
    DisparityDenoiser class allows to perform the filtering step
    """

    # Default configuration, do not change these values
    _FILTER_SIZE = 11
    _SIGMA_EUCLIDIAN = 4.0
    _SIGMA_COLOR = 100.0
    _SIGMA_PLANAR = 12.0
    _SIGMA_GRAD = 1.5
    _BAND = None

    @profile("disparity_denoiser.__init__")
    def __init__(self, *args, cfg: Dict, **kwargs):  # pylint:disable=unused-argument
        """
        :param cfg: optional configuration, {'filterSize': value,  'sigmaEuclidian' : value,
        'sigmaColor' : value, 'sigmaPlanar' : value, 'sigmaGrad': value}
        :type cfg: dict
        """
        self.cfg = self.check_conf(cfg)
        self._filter_size = int(self.cfg["filter_size"])
        self._sigma_euclidian = float(self.cfg["sigma_euclidian"])
        self._sigma_color = float(self.cfg["sigma_color"])
        self._sigma_planar = float(self.cfg["sigma_planar"])
        self._sigma_grad = float(self.cfg["sigma_grad"])
        self._band = self.cfg["band"]

        assert self._filter_size % 2 != 0
        self.win_center = self._filter_size // 2  # index of the center of a window
        self.win_size = self._filter_size  # size of the window

        # coordinates within a window
        win_coords = np.meshgrid(
            np.arange(-(self._filter_size // 2), self._filter_size // 2 + 1),
            np.arange(-(self._filter_size // 2), self._filter_size // 2 + 1),
            indexing="ij",
        )
        self.win_coords = np.stack(win_coords, 0)

    def check_conf(self, cfg: Dict) -> Dict[str, Union[str, float]]:
        """
        Add default values to the dictionary if there are missing elements and check if the dictionary is correct

        :param cfg: filter configuration
        :type cfg: dict
        :return cfg: filter configuration updated
        :rtype: dict
        """
        # Give the default value if the required element is not in the configuration
        if "filter_size" not in cfg:
            cfg["filter_size"] = self._FILTER_SIZE
        if "sigma_euclidian" not in cfg:
            cfg["sigma_euclidian"] = self._SIGMA_EUCLIDIAN
        if "sigma_color" not in cfg:
            cfg["sigma_color"] = self._SIGMA_COLOR
        if "sigma_planar" not in cfg:
            cfg["sigma_planar"] = self._SIGMA_PLANAR
        if "sigma_grad" not in cfg:
            cfg["sigma_grad"] = self._SIGMA_GRAD
        if "band" not in cfg:
            cfg["band"] = self._BAND

        schema = {
            "filter_method": And(str, lambda input: "disparity_denoiser"),
            "filter_size": And(int, lambda input: input > 0),
            "sigma_euclidian": And(float, lambda input: input > 0),
            "sigma_color": And(float, lambda input: input > 0),
            "sigma_planar": And(float, lambda input: input > 0),
            "sigma_grad": And(float, lambda input: input >= 0),
            "band": Or(str, lambda input: input is None),
        }

        checker = Checker(schema)
        checker.validate(cfg)
        return cfg

    def desc(self):
        """
        Describes the filtering method
        """
        print("Disparity denoiser filter description")

    def get_grad(self, disp: np.ndarray) -> np.ndarray:
        """
        Get the disparity gradient

        :param disp: disparity map
        :type disp: np.array
        :return disp_grad: the gradient of the disparity map
        :rtype disp_grad: np.array
        """
        disp_blur = gaussian_filter(disp, sigma=self._sigma_grad)
        disp_grad = np.stack(np.gradient(disp_blur), axis=0)
        return disp_grad

    def sliding_window(self, im: np.ndarray) -> np.ndarray:
        """
        Get a sliding window view of the input image

        :param im: input image [C, H, W]
        :type im: np.array
        :return im_view: a windowed view of the image [H, W, C, ws, ws]
        :rtype im_view: np.array
        """
        pad = self.win_size // 2
        im_pad = np.pad(im, ((0,), (pad,), (pad,)), "reflect")

        im_view = np.lib.stride_tricks.sliding_window_view(
            im_pad,
            (im.shape[0], self.win_size, self.win_size),
        )
        return im_view.squeeze(0)

    def get_disparity_dist(self, disp_view: np.ndarray) -> np.ndarray:
        """
        Get the difference in disparity between the center
        of the window and the neighbours

        :param disp_view: windowed view of the disparity map
        :type disp_view: np.array
        :return dist: the signed disparity distance
        :rtype dist: np.array
        """
        c = self.win_center
        dist = disp_view - disp_view[..., :, c : c + 1, c : c + 1]
        return dist

    def get_color_dist(self, clr_view: np.ndarray) -> np.ndarray:
        """
        Get the color distance between the center of the window
        and the neighbours

        :param clr_view: windowed view of the color map
        :type clr_view: np.array
        :return dist: the signed color distance
        :rtype dist: np.array
        """
        c = self.win_center
        dist = clr_view - clr_view[..., :, c : c + 1, c : c + 1]
        return dist

    def get_planar_dist(
        self, disp_view: np.ndarray, disp_grad_view: np.ndarray, centered_plane: bool = False
    ) -> np.ndarray:
        """
        Get the distance from the tangent plane

        :param disp_view: windowed view of the disparity map
        :type disp_view: np.array
        :param disp_grad_view: windowed view of the disparity map gradient
        :type disp_grad_view: np.array
        :param centered_plane: center the plane with least squares.
            If False, the plane will be tangent to the center of the window
        :type centered_plane: bool
        :return dist: the gap in disparity from the local plane
        :rtype dist: np.array
        """
        c = self.win_center
        dist = disp_view - np.sum(
            self.win_coords * disp_grad_view[..., :, c : c + 1, c : c + 1],
            axis=-3,
            keepdims=True,
        )

        if centered_plane:
            offset = np.mean(dist, axis=(-2, -1), keepdims=True)  # average distance to the plane
        else:
            offset = disp_view[..., :, c : c + 1, c : c + 1]  # center of the window
        return dist - offset

    def bilateral_filter(self, disp: np.ndarray, planar_dist: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Apply a bilateral filter on the disparity map using the
        weighted contibutions of the neighbours

        :param disp: noisy disparity map
        :type disp: np.array
        :param planar_dist: signed distance in disparity with the local plane
        :type planar_dist: np.array
        :param weights: weight of each neighbour
        :type weights: np.array
        :return disp_filt: filtered disparity map
        :rtype disp_filt: np.array
        """
        disp_filt = disp.copy()
        if weights is not None:
            weights = weights / np.sum(weights, axis=(-2, -1), keepdims=True)
            disp_filt[0] += np.sum(planar_dist * weights, axis=(-2, -1)).squeeze()

        return disp_filt

    @profile("disparity_denoiser.filter_disparity")
    def filter_disparity(
        self,
        disp: xr.Dataset,
        img_left: xr.Dataset = None,
        img_right: xr.Dataset = None,
        cv: xr.Dataset = None,
    ) -> None:
        """
        Apply disparity denoiser filter.

        :param disp: the disparity map dataset  with the variables :

                - disparity_map 2D xarray.DataArray (row, col)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
                - validity_mask 2D xarray.DataArray (row, col)
        :type disp: xarray.Dataset
        :param img_left: left Dataset image
        :type img_left: xarray.Dataset
        :param img_right: right Dataset image
        :type img_right: xarray.Dataset
        :param cv: cost volume dataset
        :type cv: xarray.Dataset
        :return: None
        """
        disp_map = disp["disparity_map"].data
        disp_map = disp_map[None, ...]

        if self._band is None:
            if len(img_left["im"].shape) < 3:
                color_band = img_left["im"].data[None, ...]
            else:
                color_band = img_left["im"].data[1, :, :][None, ...]
        else:
            band_index = list(img_left.band_im.data).index(self._band)
            color_band = img_left["im"].data[band_index, :, :][None, ...]

        # Derive gradient
        disp_grad = self.get_grad(disp_map.squeeze())

        # Sliding window view
        disp_view = self.sliding_window(disp_map)  # [H, W, C, ws, ws]
        clr_view = self.sliding_window(color_band)
        disp_grad_view = self.sliding_window(disp_grad)

        # Compute distances
        euclidian_dist = np.tile(
            np.linalg.norm(self.win_coords, axis=0),
            (disp_view.shape[0], disp_view.shape[1], 1, 1, 1),
        )
        clr_dist = self.get_color_dist(clr_view)
        planar_dist = self.get_planar_dist(disp_view, disp_grad_view)
        planar_dist_centered = self.get_planar_dist(disp_view, disp_grad_view, centered_plane=True)

        # define neighbour weights
        weights = (
            1
            * gaussian(euclidian_dist, sig=self._sigma_euclidian)
            * gaussian(clr_dist, sig=self._sigma_color)
            * gaussian(planar_dist_centered, sig=self._sigma_planar)
        )

        masked_data = disp["disparity_map"].copy(deep=True).data
        masked_data[np.where((disp["validity_mask"].data & cst.PANDORA_MSK_PIXEL_INVALID) != 0)] = np.nan

        valid = np.isfinite(masked_data)
        # Apply bilateral filter
        disp_filt = self.bilateral_filter(disp_map, planar_dist, weights)
        disp["disparity_map"].data[valid] = disp_filt[0][valid]
        disp.attrs["filter"] = "disparity_denoiser"
