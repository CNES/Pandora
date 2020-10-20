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
This module contains functions to test the subpixel refinement step.
"""

import json
import logging
import logging.config
import os
import unittest

import numpy as np
import xarray as xr

import pandora.constants as cst
import pandora.refinement as refinement


class TestRefinement(unittest.TestCase):
    """
    TestRefinement class allows to test the refinement module
    """

    def setUp(self):
        """
        Method called to prepare the test fixture

        """
        self.cv = xr.Dataset({'cost_volume': (['row', 'col', 'disp'], np.array([[[39, 32.5, 28, 34.5, 41],
                                                                                 [49, 41.5, 37, 34, 35.5],
                                                                                 [42.5, 40, 45, 40.5, 41],
                                                                                 [22, 30, 45, 50, 31]]]))},
                             coords={'row': [1], 'col': [0, 1, 2, 3], 'disp': [-2, -1, 0, 1, 2]})
        self.cv.attrs['subpixel'] = 1
        self.cv.attrs['measure'] = 'sad'
        self.cv.attrs['type_measure'] = 'min'

        self.disp = xr.Dataset({'disparity_map': (['row', 'col'], np.array([[0, 1, -1, -2]], np.float32)),
                                'validity_mask': (['row', 'col'], np.array([[0, 0, 0, 0]], np.uint16))},
                               coords={'row': [1], 'col': [0, 1, 2, 3]})

    def test_quadratic(self):
        """
        Test the quadratic_curve method

        """
        # Subpixel disparity map ground truth
        gt_sub_disp = np.array([[0 - ((34.5 - 32.5) / (2 * (32.5 + 34.5 - 2 * 28))),
                                 1 - ((35.5 - 37) / (2 * (37 + 35.5 - 2 * 34))),
                                 -1 - ((45 - 42.5) / (2 * (42.5 + 45 - 2 * 40))),
                                 -2]], np.float32)

        # Subpixel cost map ground truth
        x_0 = - ((34.5 - 32.5) / (2 * (32.5 + 34.5 - 2 * 28)))
        x_1 = - ((35.5 - 37) / (2 * (37 + 35.5 - 2 * 34)))
        x_2 = - ((45 - 42.5) / (2 * (42.5 + 45 - 2 * 40)))
        gt_sub_cost = np.array([[((32.5 + 34.5 - 2 * 28) / 2) * x_0 * x_0 + ((34.5 - 32.5) / 2) * x_0 + 28,
                                 ((37 + 35.5 - 2 * 34) / 2) * x_1 * x_1 + ((35.5 - 37) / 2) * x_1 + 34,
                                 ((42.5 + 45 - 2 * 40) / 2) * x_2 * x_2 + ((45 - 42.5) / 2) * x_2 + 40,
                                 22]])

        # Validity mask ground truth
        gt_mask = np.array([[0, 0, 0, cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION]], dtype=np.uint16)

        # -------- Compute the refinement with quadratic by calling subpixel_refinement --------
        quadratic_refinement = refinement.AbstractRefinement(**{'refinement_method': 'quadratic'})
        orig_cv = self.cv.copy()
        quadratic_refinement.subpixel_refinement(self.cv, self.disp)
        # Check if the calculated disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(self.disp['disparity_map'].data, gt_sub_disp)

        # Check if the calculated coefficients is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(self.disp['interpolated_coeff'].data, gt_sub_cost)

        # Check if the calculated validity mask  is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(self.disp['validity_mask'].data, gt_mask)

        # Check if the cost volume is not change
        np.testing.assert_array_equal(self.cv['cost_volume'].data, orig_cv['cost_volume'].data)

    def test_vfit(self):
        """
        Test the vfit method

        """
        # Subpixel disparity map ground truth
        gt_sub_disp = np.array([[0 + ((32.5 - 34.5) / (2 * (34.5 - 28))),
                                 1 + ((37 - 35.5) / (2 * (37 - 34))),
                                 -1 + ((42.5 - 45) / (2 * (45 - 40))),
                                 -2]], np.float32)
        # Subpixel cost map ground truth
        gt_sub_cost = np.array([[34.5 + (((32.5 - 34.5) / (2 * (34.5 - 28))) - 1) * (34.5 - 28),
                                 35.5 + (((37 - 35.5) / (2 * (37 - 34))) - 1) * (37 - 34),
                                 45 + ((42.5 - 45) / (2 * (45 - 40)) - 1) * (45 - 40),
                                 22]])

        # Validity mask ground truth
        gt_mask = np.array([[0, 0, 0, cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION]], dtype=np.uint16)

        # -------- Compute the refinement with vfit by calling subpixel_refinement --------
        vfit_refinement = refinement.AbstractRefinement(**{'refinement_method': 'vfit'})
        orig_cv = self.cv.copy()
        vfit_refinement.subpixel_refinement(self.cv, self.disp)
        # Check if the calculated disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(self.disp['disparity_map'].data, gt_sub_disp)

        # Check if the calculated coefficients is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(self.disp['interpolated_coeff'].data, gt_sub_cost)

        # Check if the calculated validity mask  is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(self.disp['validity_mask'].data, gt_mask)

        # Check if the cost volume is not change
        np.testing.assert_array_equal(self.cv['cost_volume'].data, orig_cv['cost_volume'].data)

    @staticmethod
    def test_vfit_approximate_subpixel_refinement():
        """
        Test the approximate_subpixel_refinement method

        """
        # left cost volume
        cv_left = xr.Dataset({'cost_volume': (['row', 'col', 'disp'], np.array([[[np.nan, np.nan, np.nan, 5, 0, 1],
                                                                                 [np.nan, np.nan, 4, 1, 0, 2],
                                                                                 [np.nan, 2, 3, 2, 0, np.nan],
                                                                                 [0, 5, 4, 2, np.nan, np.nan]]]))},
                             coords={'row': [1], 'col': [0, 1, 2, 3], 'disp': [-3, -2, -1, 0, 1, 2]})
        cv_left.attrs['subpixel'] = 1
        cv_left.attrs['measure'] = 'sad'
        cv_left.attrs['type_measure'] = 'min'

        # right disparity map
        disp_right = xr.Dataset({'disparity_map': (['row', 'col'], np.array([[3, -1, -1, -1]], np.float32)),
                                 'validity_mask': (['row', 'col'], np.array([[0, 0, 0, 0]], np.uint16))},
                                coords={'row': [1], 'col': [0, 1, 2, 3]})

        # Compute the refinement with vfit fast by calling fast_subpixel_refinement
        vfit_refinement = refinement.AbstractRefinement(**{'refinement_method': 'vfit'})
        sub_disp = vfit_refinement.approximate_subpixel_refinement(cv_left, disp_right)

        # Subpixel costs map ground truth
        gt_sub_costs = np.array([[0,
                                  0,
                                  2 + ((1 - 2) / (2 * (2 - 0)) - 1) * (2 - 0),
                                  2 + ((2 - 2) / (2 * (2 - 0)) - 1) * (2 - 0)]], np.float32)

        # Subpixel disparity map ground truth
        gt_sub_disp = np.array([[3,
                                 -1,
                                 -1 + (1 - 2) / (2 * (2 - 0)),
                                 -1 + (2 - 2) / (2 * (2 - 0))]], np.float32)

        # Validity mask ground truth
        gt_mask = np.array([[cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION,
                             cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION, 0, 0]], np.uint16)

        # Check if the calculated disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(sub_disp['disparity_map'].data, gt_sub_disp)

        # Check if the calculated validity mask is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(sub_disp['validity_mask'].data, gt_mask)

        # Check if the calculated coefficients is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(sub_disp['interpolated_coeff'].data, gt_sub_costs)

    @staticmethod
    def test_vfit_with_nan():
        """
        Test the vfit method on a cost volume that contains invalid values ( == np.nan )

        """
        # Cost volume
        cv = xr.Dataset({'cost_volume': (['row', 'col', 'disp'], np.array([[[np.nan, np.nan, np.nan],
                                                                            [np.nan, 2, 4],
                                                                            [3, 1, 4]]]))},
                        coords={'row': [1], 'col': [0, 1, 2], 'disp': [-1, 0, 1]})
        cv.attrs['subpixel'] = 1
        cv.attrs['measure'] = 'sad'
        cv.attrs['type_measure'] = 'min'

        disp = xr.Dataset({'disparity_map': (['row', 'col'], np.array([[0, 0, 0]], np.float32)),
                           'validity_mask': (['row', 'col'], np.array([[0, 0, 0]], np.uint16))},
                          coords={'row': [1], 'col': [0, 1, 2]})

        # Subpixel disparity map ground truth
        gt_sub_disp = np.array([[0, 0, 0 + ((3 - 4) / (2 * (4 - 1)))]], np.float32)

        # Subpixel cost map ground truth
        gt_sub_cost = np.array([[np.nan, 2, 4 + (((3 - 4) / (2 * (4 - 1))) - 1) * (4 - 1)]])

        # Validity mask ground truth
        gt_mask = np.array([[0, cst.PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION, 0]], np.uint16)

        # -------- Compute the refinement with vfit by calling subpixel_refinement --------
        vfit_refinement = refinement.AbstractRefinement(**{'refinement_method': 'vfit'})
        orig_cv = cv.copy()
        vfit_refinement.subpixel_refinement(cv, disp)

        # Check if the calculated disparity map is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(disp['disparity_map'].data, gt_sub_disp)

        # Check if the calculated coefficients is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(disp['interpolated_coeff'].data, gt_sub_cost)

        # Check if the calculated validity mask is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(disp['validity_mask'].data, gt_mask)

        # Check if the cost volume is not change
        np.testing.assert_array_equal(cv['cost_volume'].data, orig_cv['cost_volume'].data)


def setup_logging(path='logging.json', default_level=logging.WARNING, ):
    """
    Setup the logging configuration

    :param path: path to the configuration file
    :type path: string
    :param default_level: default level
    :type default_level: logging level
    """
    if os.path.exists(path):
        with open(path, 'rt') as file_:
            config = json.load(file_)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


if __name__ == '__main__':
    setup_logging()
    unittest.main()
