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
This module contains functions to test the confidence module.
"""

import unittest

import numpy as np
import xarray as xr

import pandora
import pandora.confidence as confidence
from pandora.state_machine import PandoraMachine
import pandora.matching_cost as matching_cost


class TestConfidence(unittest.TestCase):
    """
    TestConfidence class allows to test the confidence module
    """

    def setUp(self):
        """
        Method called to prepare the test fixture

        """

    @staticmethod
    def test_ambiguity():
        """
        Test the ambiguity method

        """
        cv_ = np.array([[[np.nan, 1, 3],
                         [4, 1, 1],
                         [1.2, 1, 2]],

                        [[5, np.nan, np.nan],
                        [6.2, np.nan, np.nan],
                        [0, np.nan, 0]]], dtype=np.float32)

        cv_ = xr.Dataset({'cost_volume': (['row', 'col', 'disp'], cv_)},
                         coords={'row': [0, 1], 'col': [0, 1, 2], 'disp': [-1, 0, 1]})

        ambiguity_ = confidence.AbstractConfidence(**{'confidence_method': 'ambiguity', 'eta_max': 0.2,
                                                      'eta_step': 0.1})

        # Apply median filter to the disparity map. Median filter is only applied on valid pixels.
        ambiguity_.confidence_prediction(None, None, None, cv_)

        # Ambiguity integral not normalized
        amb_int = np.array([[2., 4., 3.],
                           [2., 2., 4.]])
        # Normalized ambiguity
        amb_int = np.array([[0., 1., 0.5],
                            [0., 0., 1.]])
        # Ambiguity to confidence measure
        ambiguity_ground_truth = (1 - amb_int)

        # Check if the calculated ambiguity is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(cv_['confidence_measure'].data[:, :, 0], ambiguity_ground_truth)

    @staticmethod
    def test_ambiguity_std_full_pipeline():
        """
        Test the ambiguity and std_intensity methods using the pandora run method

        """
        # Create left and right images
        left_im = np.array([[2, 5, 3, 1],
                            [5, 3, 2, 1],
                            [4, 2, 3, 2],
                            [4, 5, 3, 2]], dtype=np.float32)

        mask_ = np.array([[0, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 1]], dtype=np.int16)

        left_im = xr.Dataset({'im': (['row', 'col'], left_im), 'msk': (['row', 'col'], mask_)},
                             coords={'row': np.arange(left_im.shape[0]), 'col': np.arange(left_im.shape[1])})
        # Add image conf to the image dataset
        left_im.attrs = {'no_data_img': 0,
                         'valid_pixels': 0,  # arbitrary default value
                         'no_data_mask': 1}  # arbitrary default value
        right_im = np.array([[1, 2, 1, 2],
                             [2, 3, 5, 3],
                             [0, 2, 4, 2],
                             [5, 3, 1, 4]], dtype=np.float32)

        mask_ = np.array([[0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]], dtype=np.int16)

        right_im = xr.Dataset({'im': (['row', 'col'], right_im), 'msk': (['row', 'col'], mask_)},
                              coords={'row': np.arange(right_im.shape[0]), 'col': np.arange(right_im.shape[1])})
        # Add image conf to the image dataset
        right_im.attrs = {'no_data_img': 0,
                         'valid_pixels': 0,  # arbitrary default value
                         'no_data_mask': 1}  # arbitrary default value
        user_cfg = {
            'pipeline':
                {
                    'matching_cost': {
                        'matching_cost_method': 'sad',
                        'window_size': 1,
                        'subpix': 1
                    },
                    'confidence': {
                        'confidence_method': 'std_intensity'
                    },
                    'disparity': {
                        'disparity_method': 'wta'
                    },
                    'filter': {
                        'filter_method': 'median'
                    },
                    'confidence.2': {
                        'confidence_method': 'ambiguity',
                        'eta_max': 0.3,
                        'eta_step': 0.25
                    }
                }
        }
        pandora_machine = PandoraMachine()

        # Update the user configuration with default values
        cfg = pandora.json_checker.update_conf(pandora.json_checker.default_short_configuration, user_cfg)

        # Run the pandora pipeline
        left, _ = pandora.run(pandora_machine, left_im, right_im, -1, 1, cfg['pipeline'])

        assert np.sum(left.coords['indicator'].data != ['stereo_pandora_intensityStd', 'ambiguity_confidence']) == 0

        # ----- Check ambiguity results ------

        # Cost volume               Normalized cost volume
        # [[[nan  1.  0.]           [[[ nan 0.25 0.  ]
        #   [ 4.  3.  4.]            [1.   0.75 1.  ]
        #   [ 1.  2.  1.]            [0.25 0.5  0.25]
        #   [ 0.  1. nan]]           [0.   0.25  nan]]
        #  [[nan  3.  2.]           [[ nan 0.75 0.5 ]
        #   [nan nan nan]            [ nan  nan  nan]
        #   [ 1.  3.  1.]            [0.25 0.75 0.25]
        #   [ 4.  2. nan]]           [1.   0.5   nan]]
        #  [[nan  4.  2.]           [[ nan 1.   0.5 ]
        #   [ 2.  0.  2.]            [0.5  0.   0.5 ]
        #   [ 1.  1.  1.]            [0.25 0.25 0.25]
        #   [ 2.  0. nan]]           [0.5  0.    nan]]
        #  [[nan  1.  1.]           [[ nan 0.25 0.25]
        #   [ 0.  2.  4.]            [0.   0.5  1.  ]
        #   [ 0.  2.  1.]            [0.   0.5  0.25]
        #   [nan nan nan]]]          [ nan  nan  nan]]]
        #
        # Ambiguity integral not normalized
        amb_int = np.array([[3., 4., 5., 3.],
                            [3., 0., 4., 2.],
                            [2., 2., 6., 2.],
                            [4., 2., 3., 0.]])
        # Normalized ambiguity
        amb_int = np.array([[3./6., 4./6., 5./6., 3./6.],
                            [3./6., 0./6., 4./6., 2./6.],
                            [2./6., 2./6., 6./6., 2./6.],
                            [4./6., 2./6., 3./6., 0./6.]])
        # Ambiguity to confidence measure
        ambiguity_ground_truth = (1 - amb_int)

        # Check if the calculated ambiguity is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_allclose(left['confidence_measure'].data[:, :, 1], ambiguity_ground_truth, rtol=1e-06)

        # ----- Check std_intensity results ------
        std_intensity_gt = np.array([[0., 0., 0., 0.],
                                     [0., 0., 0., 0.],
                                     [0., 0., 0., 0.],
                                     [0., 0., 0., 0.]])
        # Check if the calculated std_intensity is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(left['confidence_measure'].data[:, :, 0], std_intensity_gt)

    @staticmethod
    def test_std_intensity():
        """
        Test the confidence measure std_intensity
        """
        # Create a stereo object
        left_data = np.array(([1, 1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 2, 1],
                              [1, 1, 1, 4, 3, 1],
                              [1, 1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1, 1]), dtype=np.float64)
        left = xr.Dataset({'im': (['row', 'col'], left_data)},
                          coords={'row': np.arange(left_data.shape[0]), 'col': np.arange(left_data.shape[1])})
        left.attrs = {'valid_pixels': 0, 'no_data_mask': 1}

        right_data = np.array(([1, 1, 1, 2, 2, 2],
                               [1, 1, 1, 4, 2, 4],
                               [1, 1, 1, 4, 4, 1],
                               [1, 1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1, 1]), dtype=np.float64)
        right = xr.Dataset({'im': (['row', 'col'], right_data)},
                           coords={'row': np.arange(right_data.shape[0]), 'col': np.arange(right_data.shape[1])})
        right.attrs = {'valid_pixels': 0, 'no_data_mask': 1}

        # load plugins
        stereo_matcher = matching_cost.AbstractMatchingCost(**{'matching_cost_method': 'sad', 'window_size': 3,
                                                               'subpix': 1})

        # Compute bright standard deviation inside a window of size 3 and create the confidence measure
        std_bright_ground_truth = np.array([[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                                            [np.nan, 0., np.sqrt(8 / 9), np.sqrt(10 / 9), np.sqrt(10 / 9), np.nan],
                                            [np.nan, 0., np.sqrt(8 / 9), np.sqrt(10 / 9), np.sqrt(10 / 9), np.nan],
                                            [np.nan, 0., np.sqrt(8 / 9), np.sqrt(92 / 81), np.sqrt(92 / 81), np.nan],
                                            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]],
                                           dtype=np.float32)

        std_bright_ground_truth = std_bright_ground_truth.reshape((5, 6, 1))

        # compute with compute_cost_volume
        cv = stereo_matcher.compute_cost_volume(left, right, disp_min=-2, disp_max=1)
        stereo_matcher.cv_masked(left, right, cv, -2, 1)

        std_intensity = confidence.AbstractConfidence(**{'confidence_method': 'std_intensity'})

        # Apply median filter to the disparity map. Median filter is only applied on valid pixels.
        _, cv_with_intensity = std_intensity.confidence_prediction(None, left, right, cv)

        # Check if the calculated confidence_measure is equal to the ground truth (same shape and all elements equals)
        assert np.sum(cv_with_intensity.coords['indicator'].data != ['stereo_pandora_intensityStd']) == 0
        np.testing.assert_array_equal(cv_with_intensity['confidence_measure'].data, std_bright_ground_truth)


if __name__ == '__main__':
    unittest.main()
