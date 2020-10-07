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
import logging
from json_checker import Checker, And, Or, OptionalKey
from abc import ABCMeta, abstractmethod
from typing import Dict, Union
from pandora.constants import *


class AbstractDisparity(object):
    __metaclass__ = ABCMeta

    disparity_methods_avail = {}

    def __new__(cls, **cfg: dict):
        """
        Return the plugin associated with the validation_method given in the configuration

        :param cfg: configuration {'validation_method': value}
        :type cfg: dictionary
        """
        if cls is AbstractDisparity:
            if type(cfg['disparity_method']) is str:
                try:
                    return super(AbstractDisparity, cls).__new__(cls.disparity_methods_avail[cfg['disparity_method']])
                except KeyError:
                    logging.error("No disparity method named {} supported".format(cfg['disparity_method']))
                    raise KeyError
            else:
                if type(cfg['disparity_method']) is unicode:
                    # creating a plugin from registered short name given as unicode (py2 & 3 compatibility)
                    try:
                        return super(AbstractDisparity, cls).__new__(
                            cls.disparity_methods_avail[cfg['disparity_method'].encode('utf-8')])
                    except KeyError:
                        logging.error("No disparity computing method named {} supported".format(cfg['disparity_method']))
                        raise KeyError
        else:
            return super(AbstractDisparity, cls).__new__(cls)

    @classmethod
    def register_subclass(cls, short_name: str):
        """
        Allows to register the subclass with its short name

        :param short_name: the subclass to be registered
        :type short_name: string
        """

        def decorator(subclass):
            cls.disparity_methods_avail[short_name] = subclass
            return subclass

        return decorator

    @abstractmethod
    def desc(self):
        """
        Describes the disparity method
        """
        print('Disparity method description')

    @abstractmethod
    def to_disp(self, cv: xr.Dataset, img_left: xr.Dataset = None, img_right: xr.Dataset = None) -> xr.Dataset:
        """
        Disparity computation by applying the Winner Takes All strategy

        :param cv: the cost volume datset
        :type cv:
            xarray.Dataset, with the data variables:
                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
        :param img_left: left Dataset image
        :type img_left:
            xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
            :param img_right: right Dataset image
        :type img_right:
            xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :return: Dataset with the disparity map and the confidence measure
        :rtype:
            xarray.Dataset with the data variables :
                - disparity_map 2D xarray.DataArray (row, col)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
        """

    def coefficient_map(self, cv: xr.DataArray) -> xr.DataArray:
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

    def approximate_right_disparity(self, cv: xr.Dataset, img_right: xr.Dataset, invalid_value: float = 0,
                                    img_left: xr.Dataset = None) -> xr.Dataset:
        """
        Create the right disparity map, by a diagonal search for the minimum in the left cost volume

        ERNST, Ines et HIRSCHMÜLLER, Heiko.
        Mutual information based semi-global stereo matching on the GPU.
        In : International Symposium on Visual Computing. Springer, Berlin, Heidelberg, 2008. p. 228-239.

        :param cv: the cost volume dataset
        :type cv:
            xarray.Dataset, with the data variables:
                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
        :param img_right: right Dataset image
        :type img_right:
            xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :param invalid_value: disparity to assign to invalid pixels
        :type invalid_value: float
        :param img_left: left Dataset image
        :type img_left:
            xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :return: Dataset with the right disparity map, the confidence measure and the validity mask
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
        confidence_measure = compute_std_raster(img_right, cv.attrs['window_size']).reshape(len(row_range),
                                                                                          len(col_range), 1)
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

            # The disparity interval is missing in the left image
            if x_d[valid].size == 0:
                disp_map['disparity_map'].loc[dict(col=c)] = invalid_value

                # Invalid pixel : the disparity interval is missing in the right image
                disp_map['validity_mask'].loc[
                    dict(col=c)] += PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING
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
                    #  Information: the disparity interval is incomplete (border reached in the right image)
                    disp_map['validity_mask'].loc[dict(col=c)] += PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE

                disp_map['disparity_map'].loc[dict(col=c)] = -1 * np.flip(disp_range[valid])[min_.data]

        return disp_map

    def validity_mask(self, disp: xr.Dataset, img_left: xr.Dataset, img_right: xr.Dataset, cv: xr.Dataset) -> None:
        """
        Create the validity mask of the disparity map

        :param disp: dataset with the disparity map and the confidence measure
        :type disp:
            xarray.Dataset with the data variables :
                - disparity_map 2D xarray.DataArray (row, col)
                - confidence_measure 3D xarray.DataArray(row, col, indicator)
        :param img_left: left Dataset image
        :type img_left:
            xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :param img_right: right Dataset image
        :type img_right:
            xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
        :param cv: cost volume dataset
        :type cv:
            xarray.Dataset, with the data variables:
                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
        :return: None
        """
        # Allocate the validity mask
        disp['validity_mask'] = xr.DataArray(np.zeros(disp['disparity_map'].shape, dtype=np.uint16),
                                             dims=['row', 'col'])

        d_min = int(disp.attrs['disp_min'])
        d_max = int(disp.attrs['disp_max'])
        col = disp.coords['col'].data
        row = disp.coords['row'].data

        # Negative disparity range
        if d_max < 0:
            bit_1 = np.where((col + d_max) < col[0])
            # Information: the disparity interval is incomplete (border reached in the right image)
            disp['validity_mask'].data[:, np.where(((col + d_max) >= col[0]) & ((col + d_min) < col[0]))] += \
                PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
        else:
            # Positive disparity range
            if d_min > 0:
                bit_1 = np.where((col + d_min) > col[-1])
                # Information: the disparity interval is incomplete (border reached in the right image)
                disp['validity_mask'].data[:, np.where(((col + d_min) <= col[-1]) & ((col + d_max) > col[-1]))] += \
                    PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE

            # Disparity range contains 0
            else:
                bit_1 = ([],)
                # Information: the disparity interval is incomplete (border reached in the right image)
                disp['validity_mask'].data[:, np.where(((col + d_min) < col[0]) | (col + d_max > col[-1]))] += \
                    PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE

        # Invalid pixel : the disparity interval is missing in the right image ( disparity range
        # outside the image )
        disp['validity_mask'].data[:, bit_1] += PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING

        # The disp_min and disp_max used to search missing disparity interval are not the local disp_min and disp_max in
        # case of a variable range of disparities. So there may be pixels that have missing disparity range (all cost are
        # np.nan), but are not detected in the code block above. To find the pixels that have a missing disparity range, we
        # search in the cost volume pixels where cost_volume(x,y, for all d) = np.nan
        indices_nan = np.isnan(cv['cost_volume'].data)
        missing_disparity_range = np.min(indices_nan, axis=2)
        missing_range_y, missing_range_x = np.where(missing_disparity_range == True)

        # Mask the positions which have an missing disparity range, not already taken into account
        condition_to_mask = (disp['validity_mask'].data[missing_range_y, missing_range_x] &
                             PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING) == 0
        masking_value = disp['validity_mask'].data[missing_range_y, missing_range_x] + \
                        PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING
        no_masking_value = disp['validity_mask'].data[missing_range_y, missing_range_x]

        disp['validity_mask'].data[missing_range_y, missing_range_x] = np.where(condition_to_mask,
                                                                                masking_value,
                                                                                no_masking_value)

        if 'msk' in img_left.data_vars:
            _, r_mask = xr.align(disp['validity_mask'], img_left['msk'])

            # Dilatation : pixels that contains no_data in their aggregation window become no_data
            dil = binary_dilation(img_left['msk'].data == img_left.attrs['no_data_mask'],
                                  structure=np.ones((disp.attrs['window_size'], disp.attrs['window_size'])),
                                  iterations=1)
            offset = disp.attrs['offset_row_col']
            if offset != 0:
                dil = dil[offset:-offset, offset:-offset]

            # Invalid pixel : no_data in the left image
            disp['validity_mask'] += dil.astype(np.uint16) * PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER

            # Invalid pixel : invalidated by the validity mask of the left image given as input
            disp['validity_mask'] += xr.where(
                (r_mask != img_left.attrs['no_data_mask']) & (r_mask != img_left.attrs['valid_pixels']),
                PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT, 0).astype(np.uint16)

        if 'msk' in img_right.data_vars:
            _, r_mask = xr.align(disp['validity_mask'], img_right['msk'])

            # Dilatation : pixels that contains no_data in their aggregation window become no_data
            dil = binary_dilation(img_right['msk'].data == img_right.attrs['no_data_mask'],
                                  structure=np.ones((disp.attrs['window_size'], disp.attrs['window_size'])),
                                  iterations=1)
            offset = disp.attrs['offset_row_col']
            if offset != 0:
                dil = dil[offset:-offset, offset:-offset]

            r_mask = xr.where((r_mask != img_right.attrs['no_data_mask']) & (r_mask != img_right.attrs['valid_pixels']),
                              1, 0).data

            # Useful to calculate the case where the disparity interval is incomplete, and all remaining right
            # positions are invalidated by the right mask
            b_2_7 = np.zeros((len(row), len(col)), dtype=np.uint16)
            # Useful to calculate the case where no_data in the right image invalidated the disparity interval
            no_data_right = np.zeros((len(row), len(col)), dtype=np.uint16)

            col_range = np.arange(len(col))
            for d in range(d_min, d_max + 1):
                # Diagonal in the cost volume
                col_d = col_range + d
                valid_index = np.where((col_d >= col_range[0]) & (col_d <= col_range[-1]))

                # No_data and masked pixels do not raise the same flag, we need to treat them differently
                b_2_7[:, col_range[valid_index]] += r_mask[:, col_d[valid_index]].astype(np.uint16)
                b_2_7[:, col_range[np.setdiff1d(col_range, valid_index)]] += 1

                no_data_right[:, col_range[valid_index]] += dil[:, col_d[valid_index]]
                no_data_right[:, col_range[np.setdiff1d(col_range, valid_index)]] += 1

            # Exclusion of pixels that have flag 1 already enabled
            b_2_7[:, bit_1[0]] = 0
            no_data_right[:, bit_1[0]] = 0

            # Invalid pixel: right positions invalidated by the mask of the right image given as input
            disp['validity_mask'].data[np.where(b_2_7 == len(range(d_min, d_max + 1)))] += \
                PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_RIGHT

            # If Invalid pixel : the disparity interval is missing in the right image (disparity interval
            # is invalidated by no_data in the right image )
            disp['validity_mask'].data[np.where(no_data_right == len(range(d_min, d_max + 1)))] += \
                PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING


@AbstractDisparity.register_subclass('wta')
class WinnerTakesAll(AbstractDisparity):
    """
    WinnerTakesAll class allows to perform the disparity computation step
    """
    # Default configuration, do not change this value
    _INVALID_DISPARITY = -9999

    def __init__(self, **cfg):
        """float
        :param cfg: optional configuration
        :type cfg: dictionary
        """
        self.cfg = self.check_conf(**cfg)
        self._invalid_disparity = self.cfg['invalid_disparity']

    def check_conf(self, **cfg: Union[str, int, float, bool]) -> Dict[str, Union[str, int, float, bool]]:
        """
        Add default values to the dictionary if there are missing elements and check if the dictionary is correct

        :param cfg: disparity configuration
        :type cfg: dict
        :return cfg: disparity configuration updated
        :rtype: dict
        """
        # Give the default value if the required element is not in the configuration
        if 'invalid_disparity' not in cfg:
            cfg['invalid_disparity'] = self._INVALID_DISPARITY

        schema = {
            "disparity_method": And(str, lambda x: 'wta'),
            "invalid_disparity": Or(int, float, lambda x: np.isnan(x))
        }

        checker = Checker(schema)
        checker.validate(cfg)
        return cfg

    def desc(self):
        """
        Describes the disparity method
        """
        print('Winner takes all method')


    def to_disp(self, cv: xr.Dataset, img_left: xr.Dataset = None, img_right: xr.Dataset = None) -> xr.Dataset:
        """
        Disparity computation by applying the Winner Takes All strategy

        :param cv: the cost volume datset
        :type cv:
            xarray.Dataset, with the data variables:
                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
        :param img_left: left Dataset image
        :type img_left:
            xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk : 2D (row, col) xarray.DataArray
            :param img_right: right Dataset image
        :type img_right:
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
            disp = self.argmax_split(cv)
        else:
            # Disparities are computed by selecting the minimal cost at each pixel
            cv['cost_volume'].data[indices_nan] = np.inf
            disp = self.argmin_split(cv)

        cv['cost_volume'].data[indices_nan] = np.nan
        row = cv.coords['row']
        col = cv.coords['col']

        # ----- Disparity map -----
        disp_map = xr.Dataset({'disparity_map': (['row', 'col'], disp)}, coords={'row': row, 'col': col})

        invalid_mc = np.min(indices_nan, axis=2)
        # Pixels where the disparity interval is missing in the right image, have a disparity value invalid_value
        invalid_pixel = np.where(invalid_mc == True)
        disp_map['disparity_map'].data[invalid_pixel] = self._invalid_disparity

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

    @staticmethod
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

    @staticmethod
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


