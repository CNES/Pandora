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
This module contains classes and functions associated to the semantic segmentation step.
"""

from abc import ABCMeta, abstractmethod
from typing import Dict

import xarray as xr


class AbstractSemanticSegmentation:
    """
    Abstract SemanticSegmentation class
    """

    __metaclass__ = ABCMeta

    segmentation_methods_avail: Dict = {}
    cfg = None

    def __new__(cls, _img: xr.Dataset, **cfg: Dict[str, dict]):
        """
        Return the plugin associated with the segmentation_method given in the configuration

        :param img: xarray.Dataset of left image
        :type img: xarray.Dataset
        :param cfg: configuration {'segmentation_method': value}
        :type cfg: dictionary
        """
        if cls is AbstractSemanticSegmentation:
            if isinstance(cfg["segmentation_method"], str):
                try:
                    return super(AbstractSemanticSegmentation, cls).__new__(
                        cls.segmentation_methods_avail[cfg["segmentation_method"]]
                    )
                except:
                    raise KeyError(
                        "No semantic segmentation method named {} supported".format(cfg["segmentation_method"])
                    )
            else:
                if isinstance(cfg["segmentation_method"], unicode):  # type: ignore # pylint: disable=undefined-variable
                    # creating a plugin from registered short name given as unicode (py2 & 3 compatibility)
                    try:
                        return super(AbstractSemanticSegmentation, cls).__new__(
                            cls.segmentation_methods_avail[cfg["segmentation_method"].encode("utf-8")]
                        )
                    except:
                        raise KeyError(
                            "No semantic segmentation method named {} supported".format(cfg["segmentation_method"])
                        )
        else:
            return super(AbstractSemanticSegmentation, cls).__new__(cls)
        return None

    @classmethod
    def register_subclass(cls, short_name: str):
        """
        Allows to register the subclass with its short name

        :param short_name: the subclass to be registered
        :type short_name: string
        """

        def decorator(subclass):
            """
            Registers the subclass in the available methods

            :param subclass: the subclass to be registered
            :type subclass: object
            """
            cls.segmentation_methods_avail[short_name] = subclass
            return subclass

        return decorator

    @abstractmethod
    def desc(self) -> None:
        """
        Describes the semantic segmentation method
        :return: None
        """
        print("Semantic segmentation method description")

    @abstractmethod
    def compute_semantic_segmentation(self, cv: xr.Dataset, img_left: xr.Dataset, img_right: xr.Dataset) -> xr.Dataset:
        """
        Compute semantic segmentation

        :param cv: the cost volume, with the data variables:

                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure (optional): 3D xarray.DataArray (row, col, indicator)
        :type cv: xarray.Dataset
        :param img_left: left Dataset image containing :

                - im: 3D (band_im, row, col) xarray.DataArray float32
                - disparity (optional): 3D (disp, row, col) xarray.DataArray float32
                - msk (optional): 2D (row, col) xarray.DataArray int16
                - classif (optional): 3D (band_classif, row, col) xarray.DataArray int16
                - segm (optional): 2D (row, col) xarray.DataArray int16
        :type img_left: xarray
        :param img_right: right Dataset image containing :

                - im: 3D (band_im, row, col) xarray.DataArray float32
                - disparity (optional): 3D (disp, row, col) xarray.DataArray float32
                - msk (optional): 2D (row, col) xarray.DataArray int16
                - classif (optional): 3D (band_classif, row, col) xarray.DataArray int16
                - segm (optional): 2D (row, col) xarray.DataArray int16
        :type img_right: xarray
        :return: The semantic segmentation in the left image dataset with the data variables:

                - im: 3D (band_im, row, col) xarray.DataArray float32
                - disparity (optional): 3D (disp, row, col) xarray.DataArray float32
                - msk (optional): 2D (row, col) xarray.DataArray int16
                - classif (optional): 3D (band_classif, row, col) xarray.DataArray int16
                - segm (optional): 2D (row, col) xarray.DataArray int16
                - initial : 2D (row, col) xarray.DataArray semantic segmentation
        :rtype: xarray.Dataset
        """
