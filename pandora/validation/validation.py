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
This module contains classes and functions associated to the validation step.
"""

import logging
import numpy as np
import xarray as xr
from json_checker import Checker, And, Or, OptionalKey
from pandora import JSON_checker as jcheck
from abc import ABCMeta, abstractmethod
from typing import Dict, Union
from pandora.constants import *


class AbstractValidation(object):
    __metaclass__ = ABCMeta

    validation_methods_avail = {}

    def __new__(cls, **cfg: dict):
        """
        Return the plugin associated with the validation_method given in the configuration

        :param cfg: configuration {'validation_method': value}
        :type cfg: dictionary
        """
        if cls is AbstractValidation:
            if type(cfg['validation_method']) is str:
                try:
                    return super(AbstractValidation, cls).__new__(cls.validation_methods_avail[cfg['validation_method']])
                except KeyError:
                    logging.error("No validation method named {} supported".format(cfg['validation_method']))
                    raise KeyError
            else:
                if type(cfg['validation_method']) is unicode:
                    # creating a plugin from registered short name given as unicode (py2 & 3 compatibility)
                    try:
                        return super(AbstractValidation, cls).__new__(
                            cls.validation_methods_avail[cfg['validation_method'].encode('utf-8')])
                    except KeyError:
                        logging.error("No validation matching method named {} supported".format(cfg['validation_method']))
                        raise KeyError
        else:
            return super(AbstractValidation, cls).__new__(cls)

    @classmethod
    def register_subclass(cls, short_name: str):
        """
        Allows to register the subclass with its short name

        :param short_name: the subclass to be registered
        :type short_name: string
        """

        def decorator(subclass):
            cls.validation_methods_avail[short_name] = subclass
            return subclass

        return decorator

    @abstractmethod
    def desc(self):
        """
        Describes the validation method
        """
        print('Validation method description')

    @abstractmethod
    def disparity_checking(self, dataset_ref: xr.Dataset, dataset_sec: xr.Dataset, img_ref: xr.Dataset = None,
                           img_sec: xr.Dataset = None, cv: xr.Dataset = None) -> xr.Dataset:
        """
        Determination of occlusions and false matches by performing a consistency check on valid pixels. \
        Update the validity_mask :
            - If out & MSK_PIXEL_OCCLUSION != 0 : Invalid pixel : occluded pixel
            - If out & MSK_PIXEL_MISMATCH  != 0  : Invalid pixel : mismatched pixel
        Update the measure map: add the disp RL / disp LR distances

        :param dataset_ref: Reference Dataset
        :type dataset_ref: xarray.Dataset with the variables :
            - disparity_map 2D xarray.DataArray (row, col)
            - confidence_measure 3D xarray.DataArray (row, col, indicator)
            - validity_mask 2D xarray.DataArray (row, col)
        :param dataset_sec: Secondary Dataset
        :type dataset_sec: xarray.Dataset with the variables :
            - disparity_map 2D xarray.DataArray (row, col)
            - confidence_measure 3D xarray.DataArray (row, col, indicator)
            - validity_mask 2D xarray.DataArray (row, col)
        :param img_ref: reference Datset image
        :type img_ref:
            xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :param img_sec: secondary Dataset image
        :type img_sec:
            xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :param cv: cost_volume Dataset
        :type cv:
            xarray.Dataset with the variables:
                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
        :return: the reference dataset, with the bit 8 and 9 of the validity_mask :
            - If out & MSK_PIXEL_OCCLUSION != 0 : Invalid pixel : occluded pixel
            - If out & MSK_PIXEL_MISMATCH  != 0  : Invalid pixel : mismatched pixel
        :rtype : xarray.Dataset with the variables :
            - disparity_map 2D xarray.DataArray (row, col)
            - confidence_measure 3D xarray.DataArray (row, col, indicator)
            - validity_mask 2D xarray.DataArray (row, col)
        """


@AbstractValidation.register_subclass('cross_checking')
class CrossChecking(AbstractValidation):
    """
    CrossChecking class allows to perform the validation step
    """
    # Default configuration, do not change this value
    _THRESHOLD = 1.

    def __init__(self, **cfg):
        """
        :param cfg: optional configuration, {'cross_checking_threshold': value,
                                            'interpolated_disparity': value, 'filter_interpolated_disparities': value}
        :type cfg: dictionary
        """
        self.cfg = self.check_conf(**cfg)
        self._threshold = self.cfg['cross_checking_threshold']

    def check_conf(self, **cfg: Union[str, int, float, bool]) -> Dict[str, Union[str, int, float, bool]]:
        """
        Add default values to the dictionary if there are missing elements and check if the dictionary is correct

        :param cfg: optimization configuration
        :type cfg: dict
        :return cfg: optimization configuration updated
        :rtype: dict
        """
        # Give the default value if the required element is not in the configuration
        if 'cross_checking_threshold' not in cfg:
            cfg['cross_checking_threshold'] = self._THRESHOLD
        if 'right_left_mode' not in cfg:
            cfg['right_left_mode'] = 'accurate'

        schema = {
            "validation_method": And(str, lambda x: 'cross_checking'),
            "cross_checking_threshold": Or(int, float),
            "right_left_mode": And(str, lambda x: jcheck.is_method(x, ['accurate', 'approximate'])),
            OptionalKey("interpolated_disparity"): And(str, lambda x: jcheck.is_method(x, ['mc-cnn', 'sgm']))
        }

        checker = Checker(schema)
        checker.validate(cfg)
        return cfg

    def desc(self):
        """
        Describes the validation method
        """
        print('Cross-checking method')

    def disparity_checking(self, dataset_ref: xr.Dataset, dataset_sec: xr.Dataset, img_ref: xr.Dataset = None,
                           img_sec: xr.Dataset = None, cv: xr.Dataset = None) -> xr.Dataset:
        """
        Determination of occlusions and false matches by performing a consistency check on valid pixels. \
        Update the validity_mask :
            - If out & MSK_PIXEL_OCCLUSION != 0 : Invalid pixel : occluded pixel
            - If out & MSK_PIXEL_MISMATCH  != 0  : Invalid pixel : mismatched pixel
        Update the measure map: add the disp RL / disp LR distances

        :param dataset_ref: Reference Dataset
        :type dataset_ref: xarray.Dataset with the variables :
            - disparity_map 2D xarray.DataArray (row, col)
            - confidence_measure 3D xarray.DataArray (row, col, indicator)
            - validity_mask 2D xarray.DataArray (row, col)
        :param dataset_sec: Secondary Dataset
        :type dataset_sec: xarray.Dataset with the variables :
            - disparity_map 2D xarray.DataArray (row, col)
            - confidence_measure 3D xarray.DataArray (row, col, indicator)
            - validity_mask 2D xarray.DataArray (row, col)
        :param img_ref: reference Datset image
        :type img_ref:
            xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :param img_sec: secondary Dataset image
        :type img_sec:
            xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :param cv: cost_volume Dataset
        :type cv:
            xarray.Dataset with the variables:
                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
        :return: the reference dataset, with the bit 8 and 9 of the validity_mask :
            - If out & MSK_PIXEL_OCCLUSION != 0 : Invalid pixel : occluded pixel
            - If out & MSK_PIXEL_MISMATCH  != 0  : Invalid pixel : mismatched pixel
        :rtype : xarray.Dataset with the variables :
            - disparity_map 2D xarray.DataArray (row, col)
            - confidence_measure 3D xarray.DataArray (row, col, indicator)
            - validity_mask 2D xarray.DataArray (row, col)
        """
        nb_row, nb_col, nb_indicator = dataset_ref['confidence_measure'].shape
        disparity_range = np.arange(dataset_ref.attrs['disp_min'], dataset_ref.attrs['disp_max'] + 1)

        # Add a new indicator to the confidence measure DataArray
        conf_measure = np.zeros((nb_row, nb_col, nb_indicator + 1), dtype=np.float32)
        conf_measure[:, :, :-1] = dataset_ref['confidence_measure'].data

        indicator = np.copy(dataset_ref.coords['indicator'])
        indicator = np.append(indicator, 'validation_pandora_distanceOfDisp')

        # Remove confidence_measure dataArray from the dataset to update it
        dataset_ref = dataset_ref.drop_dims('indicator')
        dataset_ref = dataset_ref.assign_coords(indicator=indicator)
        dataset_ref['confidence_measure'] = xr.DataArray(data=conf_measure, dims=['row', 'col', 'indicator'])

        for row in range(0, nb_row):
            # Exclude invalid pixel :
            valid_pixel = np.where((dataset_ref['validity_mask'].data[row, :] & PANDORA_MSK_PIXEL_INVALID) == 0)

            col_ref = np.arange(nb_col, dtype=np.int)
            col_ref = col_ref[valid_pixel]

            col_sec = col_ref + dataset_ref['disparity_map'].data[row, col_ref]
            # Round elements of the array to the nearest integer
            col_sec = np.rint(col_sec).astype(int)

            # Left-Right consistency, for pixel i :
            # If | Disp_sec(i + rint(Disp_ref(i)) + Disp_ref(i) | > self._threshold : i is invalid, mismatched or occlusion detected
            # If | Disp_sec(i + rint(Disp_ref(i)) + Disp_ref(i) | <= self._threshold : i is valid

            # Apply cross checking on pixels i + round(Disp_ref(i) inside the secondary image
            inside_sec = np.where((col_sec >= 0) & (col_sec < nb_col))

            # Conversion from nan to inf
            sec_disp = dataset_sec['disparity_map'].data[row, col_sec[inside_sec]]
            sec_disp[np.isnan(sec_disp)] = np.inf
            ref_disp = dataset_ref['disparity_map'].data[row, col_ref[inside_sec]]
            ref_disp[np.isnan(ref_disp)] = np.inf

            # Allocate to the measure map, the distance disp LR / disp RL indicator
            dataset_ref['confidence_measure'].data[row, inside_sec[0], -1] = np.abs(sec_disp + ref_disp)

            # Reference image pixels invalidated by the cross checking
            invalid = np.abs(sec_disp + ref_disp) > self._threshold

            # Detect mismatched and occlusion :
            # For a reference image pixel i invalidated by the cross checking :
            # mismatch if : Disp_sec(i + d) = -d, for any other d
            # occlusion otherwise

            # Index : i + d, for any other d. 2D np array (nb invalid pixels, nb disparity )
            index = np.tile(disparity_range, (len(col_ref[inside_sec][invalid]), 1)).astype(np.float32) + \
                    np.tile(col_ref[inside_sec][invalid], (len(disparity_range), 1)).transpose()

            inside_col_disp = np.where((index >= 0) & (index < nb_col))

            # disp_sec : Disp_sec(i + d)
            disp_sec = np.full(index.shape, np.inf, dtype=np.float32)
            disp_sec[inside_col_disp] = dataset_sec['disparity_map'].data[row, index[inside_col_disp].astype(int)]

            # Check if rint(Disp_sec(i + d)) == -d
            comp = (np.rint(disp_sec) == np.tile(-1 * disparity_range, (len(col_ref[inside_sec][invalid]), 1)).astype(
                np.float32))
            comp = np.sum(comp, axis=1)
            comp[comp > 1] = 1

            dataset_ref['validity_mask'].data[row, col_ref[inside_sec][invalid]] += PANDORA_MSK_PIXEL_OCCLUSION
            dataset_ref['validity_mask'].data[row, col_ref[inside_sec][invalid]] += \
                (PANDORA_MSK_PIXEL_MISMATCH * comp).astype(np.uint16)
            dataset_ref['validity_mask'].data[row, col_ref[inside_sec][invalid]] -= \
                (PANDORA_MSK_PIXEL_OCCLUSION * comp).astype(np.uint16)

            # Pixels i + round(Disp_ref(i) outside the secondary image are occlusions
            outside_sec = np.where((col_sec < 0) & (col_sec >= nb_col))
            dataset_ref['validity_mask'].data[row, col_ref[outside_sec]] += PANDORA_MSK_PIXEL_OCCLUSION

        dataset_ref.attrs['validation'] = 'cross_checking'

        return dataset_ref
