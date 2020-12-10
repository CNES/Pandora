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
        _, cv_with_ambiguity = ambiguity_.confidence_prediction(None, None, None, cv_)

        # Ambiguity integral not normalized
        amb_int = np.array([[2., 4., 3.],
                           [2., 2., 4.]])
        # Normalized ambiguity
        amb_int = np.array([[0., 1., 0.5],
                            [0., 0., 1.]])
        # Ambiguity to confidence measure
        ambiguity_ground_truth = (1 - amb_int)

        # Check if the calculated ambiguity is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(cv_with_ambiguity['confidence_measure'].data[:, :, 0], ambiguity_ground_truth)

    @staticmethod
    def test_ambiguity_full_pipeline():
        """
        Test the ambiguity method using the pandora run method

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
                    'right_disp_map': {
                        'method': 'accurate'
                    },
                    'stereo': {
                        'stereo_method': 'sad',
                        'window_size': 1,
                        'subpix': 1
                    },
                    'disparity': {
                        'disparity_method': 'wta'
                    },
                    'refinement': {
                        'refinement_method': 'vfit'
                    },
                    'filter': {
                        'filter_method': 'median'
                    },
                    'confidence': {
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
        assert np.sum(left.coords['indicator'].data != ['stereo_pandora_intensityStd', 'ambiguity_confidence']) == 0
        np.testing.assert_allclose(left['confidence_measure'].data[:, :, 1], ambiguity_ground_truth, rtol=1e-06)


if __name__ == '__main__':
    unittest.main()
