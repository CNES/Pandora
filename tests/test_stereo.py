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
This module contains functions to test the cost volume measure step.
"""

import unittest
import logging
import logging.config
import os
import json
import numpy as np
import xarray as xr

import pandora.stereo as stereo


class TestStereo(unittest.TestCase):
    """
    TestStereo class allows to test all the methods in the class Stereo,
    and the plugins pixel_wise, zncc
    """
    def setUp(self):
        """
        Method called to prepare the test fixture

        """
        # Create a stereo object
        data = np.array(([1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 2, 1],
                         [1, 1, 1, 4, 3, 1],
                         [1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1]), dtype=np.float64)
        self.ref = xr.Dataset({'im': (['row', 'col'], data)},
                              coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})
        self.ref.attrs = {'valid_pixels': 0, 'no_data_mask': 1}

        data = np.array(([1, 1, 1, 2, 2, 2],
                         [1, 1, 1, 4, 2, 4],
                         [1, 1, 1, 4, 4, 1],
                         [1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1]), dtype=np.float64)
        self.sec = xr.Dataset({'im': (['row', 'col'], data)},
                              coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})
        self.sec.attrs = {'valid_pixels': 0, 'no_data_mask': 1}

    def test_ssd_cost(self):
        """
        Test the sum of squared difference method

        """
        # Squared difference pixel-wise ground truth for the images self.ref, self.sec, with window_size = 1
        sd_ground_truth = np.array(([0, 0, 0, 1, 1, 1],
                                    [0, 0, 0, (1-4)**2, 0, (1-4)**2],
                                    [0, 0, 0, 0, (3-4)**2, 0],
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0]))

        # Computes the sd cost for the whole images
        stereo_matcher = stereo.AbstractStereo(**{'stereo_method': 'ssd', 'window_size': 1, 'subpix': 1})
        ssd = stereo_matcher.compute_cost_volume(img_ref=self.ref, img_sec=self.sec, disp_min=-1, disp_max=1)

        # Check if the calculated sd cost is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(ssd['cost_volume'].sel(disp=0), sd_ground_truth)

        # Sum of squared difference pixel-wise ground truth for the images self.ref, self.sec, with window_size = 5
        ssd_ground_truth = np.array(([[12., 22.]]))

        # Computes the sd cost for the whole images
        stereo_matcher = stereo.AbstractStereo(**{'stereo_method': 'ssd', 'window_size': 5, 'subpix': 1})
        ssd = stereo_matcher.compute_cost_volume(img_ref=self.ref, img_sec=self.sec, disp_min=-1, disp_max=1)
        ssd = stereo_matcher.cv_masked(self.ref, self.sec, ssd, -1, 1)

        # Check if the calculated sd cost is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(ssd['cost_volume'].sel(disp=0), ssd_ground_truth)

    def test_sad_cost(self):
        """
        Test the absolute difference method

        """
        # Absolute difference pixel-wise ground truth for the images self.ref, self.sec
        ad_ground_truth = np.array(([0, 0, 0, 1, 1, 1],
                                    [0, 0, 0, abs(1-4), 0, abs(1-4)],
                                    [0, 0, 0, 0, abs(3-4), 0],
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0]))

        # Computes the ad cost for the whole images
        stereo_matcher = stereo.AbstractStereo(**{'stereo_method': 'sad', 'window_size': 1, 'subpix': 1})
        sad = stereo_matcher.compute_cost_volume(img_ref=self.ref, img_sec=self.sec, disp_min=-1, disp_max=1)

        # Check if the calculated ad cost is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(sad['cost_volume'].sel(disp=0), ad_ground_truth)

        # Sum of absolute difference pixel-wise ground truth for the images self.ref, self.sec with window size 5
        sad_ground_truth = np.array(([[6., 10.]]))

        # Computes the ad cost for the whole images
        stereo_matcher = stereo.AbstractStereo(**{'stereo_method': 'sad', 'window_size': 5, 'subpix': 1})
        sad = stereo_matcher.compute_cost_volume(img_ref=self.ref, img_sec=self.sec, disp_min=-1, disp_max=1)
        sad = stereo_matcher.cv_masked(self.ref, self.sec, sad, -1, 1)

        # Check if the calculated ad cost is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(sad['cost_volume'].sel(disp=0), sad_ground_truth)

    def test_census_cost(self):
        """
        Test the census method

        """
        data = np.array(([1, 1, 1, 3],
                         [1, 2, 1, 0],
                         [2, 1, 0, 1],
                         [1, 1, 1, 1]), dtype=np.float64)
        ref = xr.Dataset({'im': (['row', 'col'], data)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})

        data = np.array(([5, 1, 2, 3],
                         [1, 2, 1, 0],
                         [2, 2, 0, 1],
                         [1, 1, 1, 1]), dtype=np.float64)
        sec = xr.Dataset({'im': (['row', 'col'], data)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})

        # census ground truth for the images ref, sec, window size = 3 and disp = -1
        census_ground_truth_d1 = np.array(([np.nan, 3],
                                           [np.nan, 7]))

        # census ground truth for the images ref, sec, window size = 3 and disp = 0
        census_ground_truth_d2 = np.array(([1, 2],
                                           [2, 0]))

        # census ground truth for the images ref, sec, window size = 3 and disp = 1
        census_ground_truth_d3 = np.array(([4, np.nan],
                                           [5, np.nan]))

        # Computes the census transform for the images with window size = 3
        stereo_matcher = stereo.AbstractStereo(**{'stereo_method': 'census', 'window_size': 3, 'subpix': 1})
        census = stereo_matcher.compute_cost_volume(img_ref=ref, img_sec=sec, disp_min=-1, disp_max=1)
        census = stereo_matcher.cv_masked(ref, sec, census, -1, 1)

        # Check if the calculated census cost is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(census['cost_volume'].sel(disp=-1), census_ground_truth_d1)
        np.testing.assert_array_equal(census['cost_volume'].sel(disp=0), census_ground_truth_d2)
        np.testing.assert_array_equal(census['cost_volume'].sel(disp=1), census_ground_truth_d3)

    def test_point_interval(self):
        """
        Test the point interval method

        """
        stereo_matcher = stereo.AbstractStereo(**{'stereo_method': 'census', 'window_size': 3, 'subpix': 1})

        # Using the two images in self.ref, self.sec,
        # for disparity = 0, the similarity measure will be applied over the whole images
        p_ground_truth_disp = (0, self.ref['im'].shape[1])
        q_ground_truth_disp = (0, self.sec['im'].shape[1])
        calculated_range = stereo_matcher.point_interval(self.ref, self.sec, 0)

        # Check if the calculated range is equal to the ground truth
        np.testing.assert_array_equal(calculated_range[0], p_ground_truth_disp)
        np.testing.assert_array_equal(calculated_range[1], q_ground_truth_disp)

        # for disparity = -2, the similarity measure will be applied over the range
        #          x=2   x=6           x=0   x=4
        #           1 1 1 1             1 1 1 2
        #           1 1 2 1             1 1 1 4
        #           1 4 3 1             1 1 1 4
        #           1 1 1 1             1 1 1 1
        #           1 1 1 1             1 1 1 1
        p_ground_truth_disp = (2, 6)
        q_ground_truth_disp = (0, 4)
        calculated_range = stereo_matcher.point_interval(self.ref, self.sec, -2)
        # Check if the calculated range is equal to the ground truth
        np.testing.assert_array_equal(calculated_range[0], p_ground_truth_disp)
        np.testing.assert_array_equal(calculated_range[1], q_ground_truth_disp)

    def test_cost_volume(self):
        """
        Test the cost volume method

        """
        # Create simple images
        data = np.array(([1, 2, 1, 4],
                         [6, 2, 7, 4],
                         [1, 1, 3, 6]), dtype=np.float64)
        ref = xr.Dataset({'im': (['row', 'col'], data)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})

        data = np.array(([6, 7, 8, 10],
                         [2, 4, 1, 6],
                         [9, 10, 1, 2]), dtype=np.float64)
        sec = xr.Dataset({'im': (['row', 'col'], data)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})

        # Cost Volume ground truth for the stereo image simple_stereo_imgs,
        # with disp_min = -2, disp_max = 1, sad measure and subpixel_offset = 0
        ground_truth = np.array([[[np.nan, np.nan, 48, 35],
                                  [np.nan, 40, 43, np.nan]]])

        # Computes the Cost Volume for the stereo image simple_stereo_imgs,
        # with disp_min = -2, disp_max = 1, sad measure, window_size = 3 and subpix = 1
        stereo_matcher = stereo.AbstractStereo(**{'stereo_method': 'sad', 'window_size': 3, 'subpix': 1})
        cv = stereo_matcher.compute_cost_volume(ref, sec, disp_min=-2, disp_max=1)
        cv = stereo_matcher.cv_masked(ref, sec, cv, -2, 1)

        # Check if the calculated mean is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(cv['cost_volume'].data, ground_truth)

    def test_confidence_measure(self):
        """
        Test the confidence measure at the matching cost computation step
        """
        # load plugins
        stereo_matcher = stereo.AbstractStereo(**{'stereo_method': 'sad', 'window_size': 3, 'subpix': 1})

        # Compute bright standard deviation inside a window of size 3 and create the confidence measure
        std_bright_ground_truth = np.array([[0., np.sqrt(8/9), np.sqrt(10/9), np.sqrt(10/9)],
                                            [0., np.sqrt(8/9), np.sqrt(10/9), np.sqrt(10/9)],
                                            [0., np.sqrt(8/9), np.sqrt(92/81), np.sqrt(92/81)]], dtype=np.float32)
        std_bright_ground_truth = std_bright_ground_truth.reshape(3, 4, 1)

        # compute with compute_cost_volume
        cv = stereo_matcher.compute_cost_volume(self.ref, self.sec, disp_min=-2, disp_max=1)
        cv = stereo_matcher.cv_masked(self.ref, self.sec, cv, -2, 1)

        # Check if the calculated confidence_measure is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(cv['confidence_measure'].data, std_bright_ground_truth)

    def test_popcount32b(self):
        """
        Test the popcount32b method

        """
        stereo_matcher = stereo.AbstractStereo(**{'stereo_method': 'census', 'window_size': 3, 'subpix': 1})

        # Count the number of symbols that are different from the zero
        count_ = stereo_matcher.popcount32b(0b0001000101000)
        # Check if the calculated count_ is equal to the ground truth 3.
        self.assertEqual(count_, 3)

        # Count the number of symbols that are different from the zero
        count_ = stereo_matcher.popcount32b(0b0000000000000000000)
        # Check if the calculated count_ is equal to the ground truth 0.
        self.assertEqual(count_, 0)

    def test_zncc_cost(self):
        """
        Test the zncc_cost method

        """
        # Compute the cost volume for the images self.ref, self.sec,
        # with zncc measure, disp = -1, 1 window size = 5 and subpix = 1
        stereo_matcher = stereo.AbstractStereo(**{'stereo_method': 'zncc', 'window_size': 5, 'subpix': 1})
        cost_volume_zncc = stereo_matcher.compute_cost_volume(self.ref, self.sec, disp_min=-1, disp_max=1)
        cost_volume_zncc = stereo_matcher.cv_masked(self.ref, self.sec, cost_volume_zncc, -1, 1)

        # Ground truth zncc cost for the disparity -1
        x = self.ref['im'].data[:, 1:]
        y = self.sec['im'].data[:, :5]
        ground_truth = np.array(([[np.nan, (np.mean(x * y) - (np.mean(x) * np.mean(y))) / (np.std(x) * np.std(y))]]))

        # Check if the calculated cost volume for the disparity -1 is equal to the ground truth
        np.testing.assert_allclose(cost_volume_zncc['cost_volume'][:, :, 0], ground_truth, rtol=1e-05)

        # Ground truth zncc cost for the disparity 1
        x = self.ref['im'].data[:, :5]
        y = self.sec['im'].data[:, 1:]
        ground_truth = np.array(([[(np.mean(x * y) - (np.mean(x) * np.mean(y))) / (np.std(x) * np.std(y)), np.nan]]))
        # Check if the calculated cost volume for the disparity 1 is equal to the ground truth
        np.testing.assert_allclose(cost_volume_zncc['cost_volume'][:, :, 2], ground_truth, rtol=1e-05)

    def test_subpixel_offset(self):
        """
        Test the cost volume method with 2 subpixel disparity

        """
        # Create a stereo object with simple images
        data = np.array(([7, 8, 1, 0, 2],
                         [4, 5, 2, 1, 0],
                         [8, 9, 10, 0, 0]), dtype=np.float64)
        ref = xr.Dataset({'im': (['row', 'col'], data)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})

        data = np.array(([1, 5, 6, 3, 4],
                         [2, 5, 10, 6, 9],
                         [0, 7, 5, 3, 1]), dtype=np.float64)
        sec = xr.Dataset({'im': (['row', 'col'], data)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})

        # Computes the cost volume for disp min -2 disp max 2 and subpix = 2
        stereo_matcher = stereo.AbstractStereo(**{'stereo_method': 'sad', 'window_size': 3, 'subpix': 2})
        cv_zncc_subpixel = stereo_matcher.compute_cost_volume(ref, sec, disp_min=-2, disp_max=2)
        cv_zncc_subpixel = stereo_matcher.cv_masked(ref, sec, cv_zncc_subpixel, -2, 1)
        # Test the disparity range
        disparity_range_compute = cv_zncc_subpixel.coords['disp'].data
        disparity_range_ground_truth = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
        # Check if the calculated disparity range is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(disparity_range_compute, disparity_range_ground_truth)

        # Cost volume ground truth with subpixel precision 0.5
        cost_volume_ground_truth = np.array([[[np.nan, np.nan, np.nan, np.nan, 39, 32.5, 28, 34.5, 41],
                                              [np.nan, np.nan, 49, 41.5, 34, 35.5, 37, np.nan, np.nan],
                                              [45, 42.5, 40, 40.5, 41, np.nan, np.nan, np.nan, np.nan]]])

        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(cv_zncc_subpixel['cost_volume'].data, cost_volume_ground_truth)

    def test_masks_invalid_pixels(self):
        """
        Test the method masks_invalid_pixels

        """
        # ------------ Test the method with a reference mask ( secondary mask contains valid pixels ) ------------
        # Mask convention
        # cfg['image']['valid_pixels'] = 0
        # cfg['image']['no_data'] = 1
        # invalid_pixels all other values
        data = np.array(([1, 1, 1, 3, 4],
                         [1, 2, 1, 0, 2],
                         [2, 1, 0, 1, 2],
                         [1, 1, 1, 1, 4]), dtype=np.float64)

        mask = np.array(([0, 0, 2, 0, 1],
                         [0, 2, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [1, 0, 0, 0, 2]), dtype=np.int16)

        ref = xr.Dataset({'im': (['row', 'col'], data),
                          'msk': (['row', 'col'], mask)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})

        ref.attrs = {'valid_pixels': 0, 'no_data_mask': 1}

        data = np.array(([5, 1, 2, 3, 4],
                         [1, 2, 1, 0, 2],
                         [2, 2, 0, 1, 4],
                         [1, 1, 1, 1, 2]), dtype=np.float64)
        # Secondary mask contains valid pixels
        mask = np.zeros((4, 5), dtype=np.int16)
        sec = xr.Dataset({'im': (['row', 'col'], data),
                          'msk': (['row', 'col'], mask)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})

        sec.attrs = {'valid_pixels': 0, 'no_data_mask': 1}

        dmin = -1
        dmax = 1

        stereo_ = stereo.AbstractStereo(**{'stereo_method': 'sad', 'window_size': 3, 'subpix': 1})
        # ref_dil, sec_dil = stereo_.masks_dilatation(ref, sec, 1, 3, {'valid_pixels': 0, 'no_data': 1})
        # print ('ref_dil ', ref_dil)
        # exit()
        # Compute the cost volume and invalidate pixels if need
        cv = stereo_.compute_cost_volume(img_ref=ref, img_sec=sec, disp_min=dmin, disp_max=dmax)
        cv = stereo_.cv_masked(img_ref=ref, img_sec=sec, cost_volume=cv, disp_min=dmin, disp_max=dmax)

        # Cost volume before invalidation
        #  disp       -1    0   1
        # Row 1
        # col 1    [[[nan, 6., 8.],
        # col 2      [12., 2., 13.],
        # col 3      [10., 3., nan]],
        #
        #  Row 2
        # col 1    [[nan, 1., 5.],
        # col 2     [7., 1., 10.],
        # col 3     [11., 4., nan]]], dtype=float32)

        # Cost volume ground truth after invalidation
        cv_ground_truth = np.array([[[np.nan, np.nan, np.nan],
                                     [12, 2., 13.],
                                     [np.nan, np.nan, np.nan]],

                                    [[np.nan, np.nan, np.nan],
                                     [7., 1., 10.],
                                     [11., 4., np.nan]]], dtype=np.float32)

        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(cv['cost_volume'], cv_ground_truth)

        # ------------ Test the method with a secondary mask ( reference mask contains valid pixels ) ------------
        # Mask convention
        # cfg['image']['valid_pixels'] = 0
        # cfg['image']['no_data'] = 1
        # invalid_pixels all other values
        data = np.array(([1, 1, 1, 3, 4],
                         [1, 2, 1, 0, 2],
                         [2, 1, 0, 1, 2],
                         [1, 1, 1, 1, 4]), dtype=np.float64)
        # Reference mask contains valid pixels
        mask = np.zeros((4, 5), dtype=np.int16)

        ref = xr.Dataset({'im': (['row', 'col'], data),
                          'msk': (['row', 'col'], mask)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})

        ref.attrs = {'valid_pixels': 0, 'no_data_mask': 1}

        data = np.array(([5, 1, 2, 3, 4],
                         [1, 2, 1, 0, 2],
                         [2, 2, 0, 1, 4],
                         [1, 1, 1, 1, 2]), dtype=np.float64)
        mask = np.array(([0, 0, 0, 0, 2],
                         [0, 1, 0, 0, 0],
                         [0, 2, 0, 2, 0],
                         [1, 0, 0, 0, 0]), dtype=np.int16)
        sec = xr.Dataset({'im': (['row', 'col'], data),
                          'msk': (['row', 'col'], mask)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})

        sec.attrs = {'valid_pixels': 0, 'no_data_mask': 1}

        dmin = -1
        dmax = 1

        stereo_ = stereo.AbstractStereo(**{'stereo_method': 'sad', 'window_size': 3, 'subpix': 1})
        # Compute the cost volume and invalidate pixels if need
        cv = stereo_.compute_cost_volume(img_ref=ref, img_sec=sec, disp_min=dmin, disp_max=dmax)
        cv = stereo_.cv_masked(img_ref=ref, img_sec=sec, cost_volume=cv, disp_min=dmin, disp_max=dmax)
        # Cost volume before invalidation
        #  disp       -1    0   1
        # Row 1
        # col 1    [[[nan, 6., 8.],
        # col 2      [12., 2., 13.],
        # col 3      [10., 3., nan]],
        #
        #  Row 2
        # col 1    [[nan, 1., 5.],
        # col 2     [7., 1., 10.],
        # col 3     [11., 4., nan]]], dtype=float32)

        # Cost volume ground truth after invalidation
        cv_ground_truth = np.array([[[np.nan, np.nan, np.nan],
                                     [np.nan, np.nan, 13.],
                                     [np.nan, 3., np.nan]],

                                    [[np.nan, np.nan, np.nan],
                                     [np.nan, np.nan, np.nan],
                                     [np.nan, np.nan, np.nan]]], dtype=np.float32)

        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(cv['cost_volume'], cv_ground_truth)

        # ------------ Test the method with a reference and secondary mask ------------
        # Mask convention
        # cfg['image']['valid_pixels'] = 0
        # cfg['image']['no_data'] = 1
        # invalid_pixels all other values
        data = np.array(([1, 1, 1, 3, 4],
                         [1, 2, 1, 0, 2],
                         [2, 1, 0, 1, 2],
                         [1, 1, 1, 1, 4]), dtype=np.float64)
        # Reference mask contains valid pixels
        mask = np.array(([1, 0, 0, 2, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 2, 0, 0],
                         [2, 0, 0, 0, 1]), dtype=np.int16)

        ref = xr.Dataset({'im': (['row', 'col'], data),
                          'msk': (['row', 'col'], mask)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})

        ref.attrs = {'valid_pixels': 0, 'no_data_mask': 1}

        data = np.array(([5, 1, 2, 3, 4],
                         [1, 2, 1, 0, 2],
                         [2, 2, 0, 1, 4],
                         [1, 1, 1, 1, 2]), dtype=np.float64)
        mask = np.array(([0, 2, 0, 0, 1],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 2, 0],
                         [1, 0, 2, 0, 0]), dtype=np.int16)
        sec = xr.Dataset({'im': (['row', 'col'], data),
                          'msk': (['row', 'col'], mask)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})

        sec.attrs = {'valid_pixels': 0, 'no_data_mask': 1}

        dmin = -1
        dmax = 1

        stereo_ = stereo.AbstractStereo(**{'stereo_method': 'sad', 'window_size': 3, 'subpix': 1})
        # Compute the cost volume and invalidate pixels if need
        cv = stereo_.compute_cost_volume(img_ref=ref, img_sec=sec, disp_min=dmin, disp_max=dmax)
        cv = stereo_.cv_masked(img_ref=ref, img_sec=sec, cost_volume=cv, disp_min=dmin, disp_max=dmax)
        # Cost volume before invalidation
        #  disp       -1    0   1
        # Row 1
        # col 1    [[[nan, 6., 8.],
        # col 2      [12., 2., 13.],
        # col 3      [10., 3., nan]],
        #
        #  Row 2
        # col 1    [[nan, 1., 5.],
        # col 2     [7., 1., 10.],
        # col 3     [11., 4., nan]]], dtype=float32)

        # Cost volume ground truth after invalidation
        cv_ground_truth = np.array([[[np.nan, np.nan, np.nan],
                                     [12, 2, np.nan],
                                     [10, np.nan, np.nan]],

                                    [[np.nan, np.nan, 5],
                                     [np.nan, np.nan, np.nan],
                                     [np.nan, np.nan, np.nan]]], dtype=np.float32)

        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(cv['cost_volume'], cv_ground_truth)

        # ------------ Test the method with a reference and secondary mask and window size 5 ------------
        # Mask convention
        # cfg['image']['valid_pixels'] = 0
        # cfg['image']['no_data'] = 1
        # invalid_pixels all other values
        data = np.array(([0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 1, 1, 3, 4, 0],
                         [0, 1, 2, 1, 0, 2, 0],
                         [0, 2, 1, 0, 1, 2, 0],
                         [0, 1, 1, 1, 1, 4, 0],
                         [0, 0, 0, 0, 0, 0, 0]), dtype=np.float64)

        mask = np.array(([2, 0, 0, 0, 0, 0, 1],
                         [0, 0, 0, 0, 0, 0, 0],
                         [0, 2, 0, 0, 0, 0, 0],
                         [0, 0, 0, 2, 0, 0, 0],
                         [0, 0, 0, 0, 0, 2, 0],
                         [1, 0, 0, 0, 0, 0, 2]), dtype=np.int16)

        ref = xr.Dataset({'im': (['row', 'col'], data),
                          'msk': (['row', 'col'], mask)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})
        ref.attrs = {'valid_pixels': 0, 'no_data_mask': 1}
        data = np.array(([0, 0, 0, 0, 0, 0, 0],
                         [0, 5, 1, 2, 3, 4, 0],
                         [0, 1, 2, 1, 0, 2, 0],
                         [0, 2, 2, 0, 1, 4, 0],
                         [0, 1, 1, 1, 1, 2, 0],
                         [0, 0, 0, 0, 0, 0, 0]), dtype=np.float64)

        mask = np.array(([1, 0, 0, 0, 0, 0, 2],
                         [0, 0, 0, 0, 0, 0, 0],
                         [2, 0, 2, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 2],
                         [0, 0, 0, 0, 0, 0, 0],
                         [2, 0, 0, 0, 0, 0, 1]), dtype=np.int16)
        sec = xr.Dataset({'im': (['row', 'col'], data),
                          'msk': (['row', 'col'], mask)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})
        sec.attrs = {'valid_pixels': 0, 'no_data_mask': 1}

        dmin = -1
        dmax = 1

        stereo_ = stereo.AbstractStereo(**{'stereo_method': 'sad', 'window_size': 5, 'subpix': 1})
        # Compute the cost volume and invalidate pixels if need
        cv = stereo_.compute_cost_volume(img_ref=ref, img_sec=sec, disp_min=dmin, disp_max=dmax)
        cv = stereo_.cv_masked(img_ref=ref, img_sec=sec, cost_volume=cv, disp_min=dmin, disp_max=dmax)

        # Cost volume ground truth after invalidation
        cv_ground_truth = np.array([[[np.nan, np.nan, 24.],
                                     [np.nan, 10., 27.],
                                     [np.nan, np.nan, np.nan]],

                                    [[np.nan, np.nan, np.nan],
                                     [np.nan, np.nan, np.nan],
                                     [31., np.nan, np.nan]]], dtype=np.float32)

        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(cv['cost_volume'], cv_ground_truth)

        # ------------ Test the method with a reference and secondary mask with window size 1------------
        # Mask convention
        # cfg['image']['valid_pixels'] = 0
        # cfg['image']['no_data'] = 1
        # invalid_pixels all other values
        data = np.array(([1, 1, 1, 3, 4],
                         [1, 1, 1, 1, 4]), dtype=np.float64)
        # Reference mask contains valid pixels
        mask = np.array(([1, 0, 0, 2, 0],
                         [2, 0, 0, 0, 1]), dtype=np.int16)

        ref = xr.Dataset({'im': (['row', 'col'], data),
                          'msk': (['row', 'col'], mask)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})
        ref.attrs = {'valid_pixels': 0, 'no_data_mask': 1}

        data = np.array(([5, 1, 2, 3, 4],
                         [1, 1, 1, 1, 2]), dtype=np.float64)
        mask = np.array(([0, 2, 0, 0, 1],
                         [1, 0, 2, 0, 0]), dtype=np.int16)
        sec = xr.Dataset({'im': (['row', 'col'], data),
                          'msk': (['row', 'col'], mask)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})
        sec.attrs = {'valid_pixels': 0, 'no_data_mask': 1}

        dmin = -1
        dmax = 1

        stereo_ = stereo.AbstractStereo(**{'stereo_method': 'sad', 'window_size': 1, 'subpix': 1})
        # Compute the cost volume and invalidate pixels if need
        cv = stereo_.compute_cost_volume(img_ref=ref, img_sec=sec, disp_min=dmin, disp_max=dmax)
        cv = stereo_.cv_masked(img_ref=ref, img_sec=sec, cost_volume=cv, disp_min=dmin, disp_max=dmax)

        # Cost volume ground truth after invalidation
        cv_ground_truth = np.array([[[np.nan, np.nan, np.nan],
                                     [4, np.nan, 1],
                                     [np.nan, 1, 2],
                                     [np.nan, np.nan, np.nan],
                                     [1, np.nan, np.nan]],

                                    [[np.nan, np.nan, np.nan],
                                     [np.nan, 0, np.nan],
                                     [0, np.nan, 0],
                                     [np.nan, 0, 1],
                                     [np.nan, np.nan, np.nan]]], dtype=np.float32)

        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(cv['cost_volume'], cv_ground_truth)

        # ------------ Test the method with a reference and secondary mask with window size 3 and ZNCC ------------
        # Mask convention
        # cfg['image']['valid_pixels'] = 0
        # cfg['image']['no_data'] = 1
        # invalid_pixels all other values
        data = np.array(([1, 1, 1, 3, 4],
                         [1, 2, 1, 0, 2],
                         [2, 1, 0, 1, 2],
                         [1, 1, 1, 1, 4]), dtype=np.float64)
        # Reference mask contains valid pixels
        mask = np.array(([1, 0, 0, 2, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 2, 0, 0],
                         [2, 0, 0, 0, 1]), dtype=np.int16)

        ref = xr.Dataset({'im': (['row', 'col'], data),
                          'msk': (['row', 'col'], mask)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})

        ref.attrs = {'valid_pixels': 0, 'no_data_mask': 1}

        data = np.array(([5, 1, 2, 3, 4],
                         [1, 2, 1, 0, 2],
                         [2, 2, 0, 1, 4],
                         [1, 1, 1, 1, 2]), dtype=np.float64)
        mask = np.array(([0, 2, 0, 0, 1],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 2, 0],
                         [1, 0, 2, 0, 0]), dtype=np.int16)
        sec = xr.Dataset({'im': (['row', 'col'], data),
                          'msk': (['row', 'col'], mask)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})

        sec.attrs = {'valid_pixels': 0, 'no_data_mask': 1}

        dmin = -1
        dmax = 1

        stereo_ = stereo.AbstractStereo(**{'stereo_method': 'zncc', 'window_size': 3, 'subpix': 1})
        # Compute the cost volume and invalidate pixels if need
        cv = stereo_.compute_cost_volume(img_ref=ref, img_sec=sec, disp_min=dmin, disp_max=dmax)
        cv = stereo_.cv_masked(img_ref=ref, img_sec=sec, cost_volume=cv, disp_min=dmin, disp_max=dmax)

        # Cost volume ground truth after invalidation
        cv_ground_truth = np.array([[[np.nan, np.nan, np.nan],
                                     [0.02146693953705469, 0.8980265101338747, np.nan],
                                     [0.40624999999999994, np.nan, np.nan]],

                                    [[np.nan, np.nan, 0.2941742027072762],
                                     [np.nan, np.nan, np.nan],
                                     [np.nan, np.nan, np.nan]]], dtype=np.float32)

        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(cv['cost_volume'], cv_ground_truth)

    def test_masks_invalid_pixels_subpixel(self):
        """
        Test the method masks_invalid_pixels with subpixel precision

        """
        # ------------ Test the method with a secondary mask with window size 1 subpixel 2 ------------
        # Mask convention
        # cfg['image']['valid_pixels'] = 0
        # cfg['image']['no_data'] = 1
        # invalid_pixels all other values
        data = np.array(([1, 1, 1, 3, 4],
                         [1, 1, 1, 1, 4]), dtype=np.float64)
        # Reference mask contains valid pixels
        mask = np.array(([0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]), dtype=np.int16)

        ref = xr.Dataset({'im': (['row', 'col'], data),
                          'msk': (['row', 'col'], mask)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})

        ref.attrs = {'valid_pixels': 0, 'no_data_mask': 1}

        data = np.array(([5, 1, 2, 3, 4],
                         [1, 1, 1, 1, 2]), dtype=np.float64)
        mask = np.array(([0, 0, 0, 0, 1],
                         [1, 0, 2, 0, 0]), dtype=np.int16)
        sec = xr.Dataset({'im': (['row', 'col'], data),
                          'msk': (['row', 'col'], mask)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})

        sec.attrs = {'valid_pixels': 0, 'no_data_mask': 1}

        dmin = -1
        dmax = 1

        stereo_ = stereo.AbstractStereo(**{'stereo_method': 'sad', 'window_size': 1, 'subpix': 2})
        # Compute the cost volume and invalidate pixels if need
        cv = stereo_.compute_cost_volume(img_ref=ref, img_sec=sec, disp_min=dmin, disp_max=dmax)
        cv = stereo_.cv_masked(img_ref=ref, img_sec=sec, cost_volume=cv, disp_min=dmin, disp_max=dmax)

        # The cost volume before invalidation
        # <xarray.DataArray 'cost_volume' (row: 2, col: 5, disp: 5)>
        # array([[[nan, nan, 4. , 2. , 0. ],
        #         [4. , 2. , 0. , 0.5, 1. ],
        #         [0. , 0.5, 1. , 1.5, 2. ],
        #         [1. , 0.5, 0. , 0.5, 1. ],
        #         [1. , 0.5, 0. , nan, nan]],
        #
        #        [[nan, nan, 0. , 0. , 0. ],
        #         [0. , 0. , 0. , 0. , 0. ],
        #         [0. , 0. , 0. , 0. , 0. ],
        #         [0. , 0. , 0. , 0.5, 1. ],
        #         [3. , 2.5, 2. , nan, nan]]], dtype=float32)
        # Coordinates:
        #   * row      (row) int64 0 1
        #   * col      (col) int64 0 1 2 3 4
        #   * disp     (disp) float64 -1.0 -0.5 0.0 0.5 1.0

        cv_ground_truth = np.array([[[np.nan, np.nan,      4,      2,      0],
                                     [     4,      2,      0,    0.5,      1],
                                     [     0,    0.5,      1,    1.5,      2],
                                     [     1,    0.5,      0, np.nan, np.nan],
                                     [     1, np.nan, np.nan, np.nan, np.nan]],

                                    [[np.nan, np.nan, np.nan, np.nan,      0],
                                     [np.nan, np.nan,      0, np.nan, np.nan],
                                     [     0, np.nan, np.nan, np.nan,      0],
                                     [np.nan, np.nan,      0,    0.5,      1],
                                     [     3,    2.5,      2, np.nan, np.nan]]], dtype=np.float32)

        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(cv['cost_volume'], cv_ground_truth)

        # ------------ Test the method with a secondary mask with window size 1 subpixel 4 ------------
        # Mask convention
        # cfg['image']['valid_pixels'] = 5
        # cfg['image']['no_data'] = 7
        # invalid_pixels all other values
        data = np.array(([1, 1, 1],
                         [1, 1, 1]), dtype=np.float64)
        # Reference mask contains valid pixels
        mask = np.array(([5, 5, 5],
                         [5, 5, 5]), dtype=np.int16)

        ref = xr.Dataset({'im': (['row', 'col'], data),
                          'msk': (['row', 'col'], mask)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})

        ref.attrs = {'valid_pixels': 5, 'no_data_mask': 7}

        data = np.array(([5, 1, 2],
                         [1, 1, 1]), dtype=np.float64)
        mask = np.array(([5, 4, 7],
                         [6, 7, 5]), dtype=np.int16)
        sec = xr.Dataset({'im': (['row', 'col'], data),
                          'msk': (['row', 'col'], mask)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})

        sec.attrs = {'valid_pixels': 5, 'no_data_mask': 7}

        dmin = -1
        dmax = 1

        stereo_ = stereo.AbstractStereo(**{'stereo_method': 'sad', 'window_size': 1, 'subpix': 4})
        # Compute the cost volume and invalidate pixels if need
        cv = stereo_.compute_cost_volume(img_ref=ref, img_sec=sec, disp_min=dmin, disp_max=dmax)
        cv = stereo_.cv_masked(img_ref=ref, img_sec=sec, cost_volume=cv, disp_min=dmin, disp_max=dmax)

        # The cost volume before invalidation
        # <xarray.DataArray 'cost_volume' (row: 2, col: 5, disp: 5)>
        # array([[[ nan,  nan,  nan,  nan, 4.  , 3.  , 2.  , 1.  , 0.  ],
        #         [4.  , 3.  , 2.  , 1.  , 0.  , 0.25, 0.5 , 0.75, 1.  ],
        #         [0.  , 0.25, 0.5 , 0.75, 1.  ,  nan,  nan,  nan,  nan]],
        #
        #        [[ nan,  nan,  nan,  nan, 0.  , 0.  , 0.  , 0.  , 0.  ],
        #         [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
        #         [0.  , 0.  , 0.  , 0.  , 0.  ,  nan,  nan,  nan,  nan]]],
        #       dtype=float32)
        # Coordinates:
        #   * row      (row) int64 0 1
        #   * col      (col) int64 0 1 2
        #   * disp     (disp) float64 -1.0 -0.75 -0.5 -0.25 0.0 0.25 0.5 0.75 1.0

        cv_ground_truth = np.array([[
        [np.nan, np.nan, np.nan, np.nan, 4.    , np.nan, np.nan, np.nan, np.nan],
        [4.    , np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]],

       [[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.    ],
        [np.nan, np.nan, np.nan, np.nan, 0.    , np.nan, np.nan, np.nan, np.nan]]], dtype=np.float32)

        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(cv['cost_volume'], cv_ground_truth)

        # ------------ Test the method with a reference and secondary mask, window size 3, subpixel 2 ------------
        # Mask convention
        # cfg['image']['valid_pixels'] = 5
        # cfg['image']['no_data'] = 7
        # invalid_pixels all other values
        data = np.array(([1, 1, 1, 3, 4],
                         [1, 2, 1, 0, 2],
                         [2, 1, 0, 1, 2],
                         [1, 1, 1, 1, 4]), dtype=np.float64)
        mask = np.array(([5, 56, 5, 12, 5],
                         [5, 5, 5, 5, 5],
                         [5, 5, 5, 5, 5],
                         [3, 5, 4, 5, 7]), dtype=np.int16)

        ref = xr.Dataset({'im': (['row', 'col'], data),
                          'msk': (['row', 'col'], mask)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})

        ref.attrs = {'valid_pixels': 5, 'no_data_mask': 7}

        data = np.array(([5, 1, 2, 3, 4],
                         [1, 2, 1, 0, 2],
                         [2, 2, 0, 1, 4],
                         [1, 1, 1, 1, 2]), dtype=np.float64)
        mask = np.array(([7, 5, 5, 5, 5],
                         [5, 5, 5, 65, 5],
                         [5, 5, 5, 5, 5],
                         [5, 23, 5, 5, 2]), dtype=np.int16)
        sec = xr.Dataset({'im': (['row', 'col'], data),
                          'msk': (['row', 'col'], mask)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})

        sec.attrs = {'valid_pixels': 5, 'no_data_mask': 7}

        dmin = -1
        dmax = 1

        stereo_ = stereo.AbstractStereo(**{'stereo_method': 'sad', 'window_size': 3, 'subpix': 2})
        # Compute the cost volume and invalidate pixels if need
        cv = stereo_.compute_cost_volume(img_ref=ref, img_sec=sec, disp_min=dmin, disp_max=dmax)
        cv = stereo_.cv_masked(img_ref=ref, img_sec=sec, cost_volume=cv, disp_min=dmin, disp_max=dmax)

        # Cost volume before invalidation
        # array([[[ nan,  nan,  6. ,  6. ,  8. ],
        #         [12. ,  7. ,  2. ,  6.5, 13. ],
        #         [10. ,  5.5,  3. ,  nan,  nan]],
        #
        #        [[ nan,  nan,  1. ,  2. ,  5. ],
        #         [ 7. ,  4. ,  1. ,  4.5, 10. ],
        #         [11. ,  6.5,  4. ,  nan,  nan]]], dtype=float32)
        # Coordinates:
        #   * row      (row) int64 1 2
        #   * col      (col) int64 1 2 3
        #   * disp     (disp) float64 -1.0 -0.5 0.0 0.5 1.0

        # Cost volume ground truth after invalidation
        cv_ground_truth = np.array([[[np.nan, np.nan, np.nan, np.nan, 8.    ],
                                     [np.nan, np.nan, 2.    , np.nan, np.nan],
                                     [10.   , np.nan, np.nan, np.nan, np.nan]],

                                    [[np.nan, np.nan, 1.    , 2.    , 5.    ],
                                     [7.    , 4.    , 1.    , 4.5   , 10.   ],
                                     [np.nan, np.nan, np.nan, np.nan, np.nan]]], dtype=np.float32)

        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(cv['cost_volume'], cv_ground_truth)

        # ------------ Test the method with a reference and secondary mask with window size 3 and census ------------
        # Mask convention
        # cfg['image']['valid_pixels'] = 5
        # cfg['image']['no_data'] = 7
        # invalid_pixels all other values
        data = np.array(([1, 1, 1, 3],
                         [1, 2, 1, 0],
                         [2, 1, 0, 1],
                         [1, 1, 1, 1]), dtype=np.float64)
        mask = np.array(([7, 5, 5, 2],
                         [0, 5, 5, 5],
                         [5, 5, 5, 0],
                         [0, 5, 5, 7]), dtype=np.int16)
        ref = xr.Dataset({'im': (['row', 'col'], data),
                          'msk': (['row', 'col'], mask)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})

        ref.attrs = {'valid_pixels': 5, 'no_data_mask': 7}

        data = np.array(([5, 1, 2, 3],
                         [1, 2, 1, 0],
                         [2, 2, 0, 1],
                         [1, 1, 1, 1]), dtype=np.float64)
        mask = np.array(([2, 5, 5, 2],
                         [0, 5, 2, 5],
                         [5, 5, 5, 0],
                         [7, 5, 5, 5]), dtype=np.int16)
        sec = xr.Dataset({'im': (['row', 'col'], data),
                          'msk': (['row', 'col'], mask)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})

        sec.attrs = {'valid_pixels': 5, 'no_data_mask': 7}

        dmin = -1
        dmax = 1

        # Cost volume ground truth after invalidation
        census_ground_truth = np.array([[[np.nan, np.nan, np.nan, np.nan, np.nan],
                                         [3., np.nan, np.nan, np.nan, np.nan]],

                                        [[np.nan, np.nan, np.nan, np.nan, 5.],
                                         [np.nan, np.nan, np.nan, np.nan, np.nan]]], dtype=np.float32)

        # Computes the census transform for the images with window size = 3
        stereo_matcher = stereo.AbstractStereo(**{'stereo_method': 'census', 'window_size': 3, 'subpix': 2})
        census = stereo_matcher.compute_cost_volume(img_ref=ref, img_sec=sec, disp_min=dmin, disp_max=dmax)
        census = stereo_matcher.cv_masked(img_ref=ref, img_sec=sec, cost_volume=census, disp_min=dmin, disp_max=dmax)

        # Check if the calculated census cost is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(census['cost_volume'], census_ground_truth)

        # ------------ Test the method with a reference and secondary mask with window size 3 and ZNCC ------------
        data = np.array(([1, 1, 1, 3, 4],
                         [1, 2, 1, 0, 2],
                         [2, 1, 0, 1, 2],
                         [1, 1, 1, 1, 4]), dtype=np.float64)
        # Reference mask contains valid pixels
        mask = np.array(([1, 0, 0, 2, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 2, 0, 0],
                         [2, 0, 0, 0, 1]), dtype=np.int16)

        ref = xr.Dataset({'im': (['row', 'col'], data),
                          'msk': (['row', 'col'], mask)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})

        ref.attrs = {'valid_pixels': 0, 'no_data_mask': 1}

        data = np.array(([5, 1, 2, 3, 4],
                         [1, 2, 1, 0, 2],
                         [2, 2, 0, 1, 4],
                         [1, 1, 1, 1, 2]), dtype=np.float64)
        mask = np.array(([0, 2, 0, 0, 1],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 2, 0],
                         [1, 0, 2, 0, 0]), dtype=np.int16)
        sec = xr.Dataset({'im': (['row', 'col'], data),
                          'msk': (['row', 'col'], mask)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})

        sec.attrs = {'valid_pixels': 0, 'no_data_mask': 1}

        dmin = -1
        dmax = 1

        stereo_ = stereo.AbstractStereo(**{'stereo_method': 'zncc', 'window_size': 3, 'subpix': 2})
        # Compute the cost volume and invalidate pixels if need
        cv = stereo_.compute_cost_volume(img_ref=ref, img_sec=sec, disp_min=dmin, disp_max=dmax)
        cv = stereo_.cv_masked(img_ref=ref, img_sec=sec, cost_volume=cv, disp_min=dmin, disp_max=dmax)

        # Cost volume ground truth after invalidation
        cv_ground_truth = np.array([[[np.nan, np.nan, np.nan, np.nan, np.nan],
                                     [0.02146693953705469, 0.5486081, 0.8980265101338747, np.nan, np.nan],
                                     [0.40624999999999994, np.nan, np.nan, np.nan, np.nan]],

                                    [[np.nan, np.nan, np.nan, np.nan, 0.2941742027072762],
                                     [np.nan, np.nan, np.nan, np.nan, np.nan],
                                     [np.nan, np.nan, np.nan, np.nan, np.nan]]], dtype=np.float32)

        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(cv['cost_volume'], cv_ground_truth)

    def test_masks_dilatation(self):
        """
        Test the method masks_dilatation

        """
        # Mask convention
        # cfg['image']['valid_pixels'] = 5
        # cfg['image']['no_data'] = 7
        # invalid_pixels all other values
        data = np.array(([1, 1, 1, 3, 4],
                         [1, 2, 1, 0, 2],
                         [2, 1, 0, 1, 2],
                         [1, 1, 1, 1, 4]), dtype=np.float64)
        mask = np.array(([5, 56, 5, 12, 5],
                         [5, 5, 5, 5, 5],
                         [5, 5, 5, 5, 5],
                         [3, 5, 4, 5, 7]), dtype=np.int16)

        ref = xr.Dataset({'im': (['row', 'col'], data),
                          'msk': (['row', 'col'], mask)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})

        ref.attrs = {'valid_pixels': 5, 'no_data_mask': 7}

        data = np.array(([5, 1, 2, 3, 4],
                         [1, 2, 1, 0, 2],
                         [2, 2, 0, 1, 4],
                         [1, 1, 1, 1, 2]), dtype=np.float64)
        mask = np.array(([7, 5, 5, 5, 5],
                         [5, 5, 5, 65, 5],
                         [5, 5, 5, 5, 5],
                         [5, 23, 5, 5, 2]), dtype=np.int16)
        sec = xr.Dataset({'im': (['row', 'col'], data),
                          'msk': (['row', 'col'], mask)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})

        sec.attrs = {'valid_pixels': 5, 'no_data_mask': 7}

        # masks_dilatation(self, img_ref, img_sec, offset_row_col, window_size, subp, cfg)
        stereo_ = stereo.AbstractStereo(**{'stereo_method': 'sad', 'window_size': 3, 'subpix': 4})
        # Compute the dilated / shifted masks
        mask_ref, masks_sec = stereo_.masks_dilatation(img_ref=ref, img_sec=sec, offset_row_col=int((3 - 1) / 2),
                                                       window_size=3, subp=4)
        # Reference mask ground truth
        gt_ref = np.array([[0, 0, 0],
                           [0, 0, np.nan]], dtype=np.float32)
        gt_ref = xr.DataArray(gt_ref, coords=[[1, 2], [1, 2, 3]], dims=['row', 'col'])

        # Check if the calculated reference masks is equal to the ground truth (same dimensions, coordinates and values)
        if not mask_ref.equals(gt_ref):
            raise ValueError('test_masks_dilatation error : reference mask ')

        # Secondary mask ground truth with pixel precision
        gt_sec_pixel = np.array([[np.nan, 0, np.nan],
                                 [0, 0, 0]], dtype=np.float32)
        gt_sec_pixel = xr.DataArray(gt_sec_pixel, coords=[[1, 2], [1, 2, 3]], dims=['row', 'col'])

        if not masks_sec[0].equals(gt_sec_pixel):
            raise ValueError('test_masks_dilatation error : secondary mask ')

        # Secondary mask ground truth with sub-pixel precision
        gt_sec_subpixel = np.array([[np.nan, np.nan],
                                    [0, 0]], dtype=np.float32)
        gt_sec_subpixel = xr.DataArray(gt_sec_subpixel, coords=[[1, 2], [1.5, 2.5]], dims=['row', 'col'])

        if not masks_sec[1].equals(gt_sec_subpixel):
            raise ValueError('test_masks_dilatation error : secondary shifted mask ')

    def test_cmax(self):
        """
        Test the cmax attribute of the cost volume

        """
        # Test cmax for the census mesure
        stereo_matcher = stereo.AbstractStereo(**{'stereo_method': 'census', 'window_size': 3, 'subpix': 1})
        census_cmax_w3 = stereo_matcher.compute_cost_volume(img_ref=self.ref, img_sec=self.sec, disp_min=-1, disp_max=1)
        # Check if the calculated maximal cost is equal to the ground truth
        np.testing.assert_array_equal(census_cmax_w3.attrs['cmax'], 9)
        assert (np.nanmax(census_cmax_w3['cost_volume'].data) <= 9)

        stereo_matcher = stereo.AbstractStereo(**{'stereo_method': 'census', 'window_size': 5, 'subpix': 1})
        census_cmax_w5 = stereo_matcher.compute_cost_volume(img_ref=self.ref, img_sec=self.sec, disp_min=-1, disp_max=1)
        # Check if the calculated maximal cost is equal to the ground truth
        np.testing.assert_array_equal(census_cmax_w5.attrs['cmax'], 25)
        assert (np.nanmax(census_cmax_w5['cost_volume'].data) <= 25)

        # Test cmax for the sad mesure
        stereo_matcher = stereo.AbstractStereo(**{'stereo_method': 'sad', 'window_size': 3, 'subpix': 1})
        sad_cmax_w3 = stereo_matcher.compute_cost_volume(img_ref=self.ref, img_sec=self.sec, disp_min=-1, disp_max=1)
        # Check if the calculated maximal cost is equal to the ground truth
        np.testing.assert_array_equal(sad_cmax_w3.attrs['cmax'], int(abs(4 - 1) * (3**2)))
        assert (np.nanmax(sad_cmax_w3['cost_volume'].data) <= int(abs(4 - 1) * (3**2)))

        stereo_matcher = stereo.AbstractStereo(**{'stereo_method': 'sad', 'window_size': 5, 'subpix': 1})
        sad_cmax_w5 = stereo_matcher.compute_cost_volume(img_ref=self.ref, img_sec=self.sec, disp_min=-1, disp_max=1)
        # Check if the calculated maximal cost is equal to the ground truth
        np.testing.assert_array_equal(sad_cmax_w5.attrs['cmax'], int(abs(4 - 1) * (5**2)))
        assert (np.nanmax(sad_cmax_w3['cost_volume'].data) <= int(abs(4 - 1) * (5**2)))

        # Test cmax for the ssd mesure
        stereo_matcher = stereo.AbstractStereo(**{'stereo_method': 'ssd', 'window_size': 3, 'subpix': 1})
        ssd_cmax_w3 = stereo_matcher.compute_cost_volume(img_ref=self.ref, img_sec=self.sec, disp_min=-1, disp_max=1)
        # Check if the calculated maximal cost is equal to the ground truth
        np.testing.assert_array_equal(ssd_cmax_w3.attrs['cmax'], int(abs(4 - 1)**2 * (3**2)))
        assert (np.nanmax(sad_cmax_w3['cost_volume'].data) <= int(abs(4 - 1)**2 * (3**2)))

        stereo_matcher = stereo.AbstractStereo(**{'stereo_method': 'ssd', 'window_size': 5, 'subpix': 1})
        ssd_cmax_w5 = stereo_matcher.compute_cost_volume(img_ref=self.ref, img_sec=self.sec, disp_min=-1, disp_max=1)
        # Check if the calculated maximal cost is equal to the ground truth
        np.testing.assert_array_equal(ssd_cmax_w5.attrs['cmax'], int(abs(4 - 1)**2 * (5**2)))
        assert (np.nanmax(sad_cmax_w3['cost_volume'].data) <= int(abs(4 - 1)**2 * (5**2)))

        # Test cmax for the zncc mesure
        stereo_matcher = stereo.AbstractStereo(**{'stereo_method': 'zncc', 'window_size': 3, 'subpix': 1})
        zncc_cmax = stereo_matcher.compute_cost_volume(img_ref=self.ref, img_sec=self.sec, disp_min=-1, disp_max=1)
        # Check if the calculated maximal cost is equal to the ground truth
        np.testing.assert_array_equal(zncc_cmax.attrs['cmax'], 1)
        assert (np.nanmax(zncc_cmax['cost_volume'].data) <= 1)

    def test_dmin_dmax(self):
        """
        Test dmin_dmax function which returns the min disparity and the max disparity

        """
        # Load stereo plugin
        stereo_matcher = stereo.AbstractStereo(**{'stereo_method': 'census', 'window_size': 3, 'subpix': 1})

        # dmin and dmax values
        dmin_int = -2

        dmin_grid = np.array([[2, 3, 5, 12],
                              [15, 0, -5, -2],
                              [-4, 5, 10, 1]])

        dmax_int = 20

        dmax_grid = np.array([[18, 12, 8, 25],
                              [16, 7, -1, 0],
                              [5, 10, 20, 11]])

        # Case with dmin and dmax are fixed disparities
        gt_fixed_disp = (-2, 20)
        compute_fixed_disp = stereo_matcher.dmin_dmax(dmin_int, dmax_int)
        self.assertEqual(gt_fixed_disp, compute_fixed_disp)

        # Case with dmin is a fixed disparity and dmax is a variable disparity
        gt_fixed_var_disp = (-2, 25)
        compute_fixed_var_disp = stereo_matcher.dmin_dmax(dmin_int, dmax_grid)
        self.assertEqual(gt_fixed_var_disp, compute_fixed_var_disp)

        # Case with dmin is a variable disparity and dmax is a fixed disparity
        gt_var_fixed_disp = (-5, 20)
        compute_var_fixed_disp = stereo_matcher.dmin_dmax(dmin_grid, dmax_int)
        self.assertEqual(gt_var_fixed_disp, compute_var_fixed_disp)

        # Case with dmin and dmax are variable disparities
        gt_variable_disp = (-5, 25)
        compute_var_disp = stereo_matcher.dmin_dmax(dmin_grid, dmax_grid)
        self.assertEqual(gt_variable_disp, compute_var_disp)

    def test_cv_masked(self):
        """
        Test cv_masked function which masks with nan, the costs which have been computed with disparities outside
        of the range of variable disparities grid

        """
        # Initialize data
        data = np.array(([1, 1, 1, 3, 2, 1, 7, 2, 3, 4, 6],
                         [1, 3, 2, 5, 2, 6, 1, 8, 7, 0, 4],
                         [2, 1, 0, 1, 7, 9, 5, 4, 9, 1, 5],
                         [1, 5, 4, 3, 2, 6, 7, 6, 5, 2, 1]), dtype=np.float64)

        ref = xr.Dataset({'im': (['row', 'col'], data)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})

        ref.attrs = {'valid_pixels': 0, 'no_data_mask': 1}

        data = np.array(([5, 1, 2, 3, 4, 7, 9, 6, 5, 2, 7],
                         [1, 3, 0, 2, 5, 3, 7, 8, 7, 6, 5],
                         [2, 3, 5, 0, 1, 5, 6, 5, 2, 3, 6],
                         [1, 6, 7, 5, 3, 2, 1, 0, 3, 4, 7]), dtype=np.float64)

        sec = xr.Dataset({'im': (['row', 'col'], data)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})

        sec.attrs = {'valid_pixels': 0, 'no_data_mask': 1}

        dmin_grid = np.array([[-0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0],
                              [-0, -8, -8, -5, -8, -4, -6, -7, -9, -8, -0],
                              [-0, -9, -8, -4, -6, -5, -7, -8, -9, -7, -0],
                              [-0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0]])

        dmax_grid = np.array([[-0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0],
                              [-0, -2, -1, -1, -5, -1, -2, -6, -4, -3, -0],
                              [-0, -3, 0, -2, -2, -2, -3, -5, -5, -4, -0],
                              [-0, -0, -0, -0, -0, -0, -0, -0, -0, -0, -0]])

        # Initialization of stereo plugin
        stereo_ = stereo.AbstractStereo(**{'stereo_method': 'census', 'window_size': 3, 'subpix': 1})

        # compute the min disparity of disp_min and the max disparity of disp_max
        dmin_int, dmax_int = stereo_.dmin_dmax(dmin_grid, dmax_grid)

        # ------------ Test the method with disp_min as a grid and disp_max as a grid, subpixel = 1 ------------
        # Compute the cost_volume
        cv = stereo_.compute_cost_volume(img_ref=ref, img_sec=sec, disp_min=dmin_int, disp_max=dmax_int)

        # Compute the masked cost volume
        cv = stereo_.cv_masked(img_ref=ref, img_sec=sec, cost_volume=cv, disp_min=dmin_grid, disp_max=dmax_grid)

        # Cost volume ground truth
        gt_cv_masked = np.array([[[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                                  [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 5., np.nan],
                                  [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1., 8., np.nan],
                                  [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                                  [np.nan, np.nan, np.nan, np.nan, np.nan, 5., 4., 3., 2., np.nan],
                                  [np.nan, np.nan, np.nan, np.nan, 7., 2., 3., 6., np.nan, np.nan],
                                  [np.nan, np.nan, np.nan, 1., np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                                  [np.nan, np.nan, 4., 5., 6., 3., np.nan, np.nan, np.nan, np.nan],
                                  [np.nan, 6., 1., 4., 7., 1., 6., np.nan, np.nan, np.nan]],

                                 [[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                                  [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 5., 6.],
                                  [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 4., np.nan, np.nan],
                                  [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 2., 3., np.nan, np.nan],
                                  [np.nan, np.nan, np.nan, np.nan, np.nan, 3., 2., 7., np.nan, np.nan],
                                  [np.nan, np.nan, np.nan, np.nan, 5., 4., 3., np.nan, np.nan, np.nan],
                                  [np.nan, np.nan, np.nan, 4., 5., np.nan, np.nan, np.nan, np.nan, np.nan],
                                  [np.nan, np.nan, 3., 2., 7., np.nan, np.nan, np.nan, np.nan, np.nan],
                                  [np.nan, np.nan, 4., 3., 3., 4., np.nan, np.nan, np.nan, np.nan]]], dtype=np.float32)

        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(gt_cv_masked, cv['cost_volume'].data)

        # ------------ Test the method with disp_min as a grid and disp_max as a grid, subpixel = 2 ------------
        # Initialization of stereo plugin
        stereo_ = stereo.AbstractStereo(**{'stereo_method': 'census', 'window_size': 3, 'subpix': 2})

        # compute the min disparity of disp_min and the max disparity of disp_max
        dmin_int, dmax_int = stereo_.dmin_dmax(dmin_grid, dmax_grid)

        # Compute the cost_volume
        cv = stereo_.compute_cost_volume(img_ref=ref, img_sec=sec, disp_min=dmin_int, disp_max=dmax_int)

        # Compute the masked cost volume
        cv = stereo_.cv_masked(img_ref=ref, img_sec=sec, cost_volume=cv, disp_min=dmin_grid, disp_max=dmax_grid)

        # Cost volume ground truth
        gt_cv_masked = np.array([[[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                                  [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 5., np.nan, np.nan],
                                  [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1., 5., 8., np.nan, np.nan],
                                  [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                                  [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 5., 3., 4., 4., 3., 3., 2., np.nan, np.nan],
                                  [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 7., 3., 2., 2., 3., 5., 6., np.nan, np.nan, np.nan, np.nan],
                                  [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1., np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                                  [np.nan, np.nan, np.nan, np.nan, 4., 4., 5., 5., 6., 4., 3., np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                                  [np.nan, np.nan, 6., 2., 1., 1., 4., 6., 7., 4., 1., 2., 6., np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]],

                                 [[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                                  [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 5., 6., 6.],
                                  [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 4., np.nan, np.nan, np.nan, np.nan],
                                  [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 2., 3., 3., np.nan, np.nan, np.nan, np.nan],
                                  [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 3., 2., 2., 5., 7., np.nan, np.nan, np.nan, np.nan],
                                  [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 5., 4., 4., 1., 3., np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                                  [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 4., 5., 5., np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                                  [np.nan, np.nan, np.nan, np.nan, 3., 2., 2., 5., 7., np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                                  [np.nan, np.nan, np.nan, np.nan, 4., 3., 3., 2., 3., 3., 4., np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]]], dtype=np.float32)

        # Check if the calculated cost volume is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(gt_cv_masked, cv['cost_volume'].data)


def setup_logging(path='logging.json', default_level=logging.WARNING,):
    """
    Setup the logging configuration

    :param path: path to the configuration file
    :type path: string
    :param default_level: default level
    :type default_level: logging level
    """
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


if __name__ == '__main__':
    setup_logging()
    unittest.main()
