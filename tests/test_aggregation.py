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
This module contains functions to test the cost volume aggregation step.
"""

import unittest
import numpy as np
import xarray as xr

import pandora.aggregation as aggregation
import pandora.stereo as stereo
import pandora.aggregation.cbca as cbca


class TestAggregation(unittest.TestCase):
    """
    TestAggregation class allows to test all the methods in the class Aggregation,
    and the plugins
    """
    def setUp(self):
        """
        Method called to prepare the test fixture

        """
        data = np.array(([[5, 1, 15, 7, 3],
                          [10, 9, 11, 9, 6],
                          [1, 18, 4, 5, 9]]), dtype=np.float32)
        self.ref = xr.Dataset({'im': (['row', 'col'], data)},
                              coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})

        data = np.array(([[1, 5, 1, 15, 7],
                          [2, 10, 9, 11, 9],
                          [3, 1, 18, 4, 5]]), dtype=np.float32)
        self.sec = xr.Dataset({'im': (['row', 'col'], data)},
                              coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})

        # Create the matching cost for the images self.ref and self.sec, with disp = [-1, 0, 1] and SAD measuress
        row = np.arange(3)
        col = np.arange(5)
        disp = np.array([-1, 0, 1])
        self.cv = xr.Dataset({'cost_volume': (['row', 'col', 'disp'], np.zeros((3, 5, 3), dtype=np.float32) + np.nan)},
                             coords={'row': row, 'col': col, 'disp': disp})
        self.cv['cost_volume'].loc[:, 1:, -1] = abs(self.ref['im'].data[:, 1:] - self.sec['im'].data[:, :4])
        self.cv['cost_volume'].loc[:, :, 0] = abs(self.ref['im'].data - self.sec['im'].data)
        self.cv['cost_volume'].loc[:, :3, 1] = abs(self.ref['im'].data[:, :4] - self.sec['im'].data[:, 1:])

        self.cv.attrs = {"measure": 'sad', "subpixel": 1, "offset_row_col": 0, "cmax": 18}

    def test_compute_cbca_subpixel(self):
        """
        Test the cross-based cost aggregation method with subpixel precision

        """
        stereo_matcher = stereo.AbstractStereo(**{'stereo_method': 'sad', 'window_size': 1, 'subpix': 2})
        sad = stereo_matcher.compute_cost_volume(img_ref=self.ref, img_sec=self.sec, disp_min=-1, disp_max=1,
                                                 **{'valid_pixels': 0, 'no_data': 1})

        # Computes the cost aggregation with the cross-based cost aggregation method,
        # with cbca_intensity=5 and cbca_distance=3
        cbca_obj = aggregation.AbstractAggregation(**{'aggregation_method': 'cbca',
                                                   'cbca_intensity': 5., 'cbca_distance': 3})

        cv_aggreg = cbca_obj.cost_volume_aggregation(self.ref, self.sec, sad)

        # Aggregate cost volume ground truth with the cross-based cost aggregation method for the stereo image
        aggregated_ground_truth = np.array([[[np.nan, np.nan, 36./6, 25./7, 0.],
                                             [36./6, 46./7, 66./9, 38.5/12, 0.],
                                             [55./9, 64.5/12, 74./12, 32.5/10, 0.],
                                             [55./9, 59.5/10, 52./9, 31.5/9, 0.],
                                             [41./6, 43.5/9, 52./9, np.nan, np.nan]],

                                            [[np.nan, np.nan, 36./6, 25./7, 0.],
                                             [36./6, 46./7, 66./9, 38.5/12, 0.],
                                             [55./9, 64.5/12, 74./12, 32.5/10, 0.],
                                             [55./9, 59.5/10, 52./9, 31.5/9, 0.],
                                             [41./6, 43.5/9, 52./9, np.nan, np.nan]],

                                            [[np.nan, np.nan, 36./6, 25./7, 0.],
                                             [36./6, 46./7, 66./9, 38.5/12, 0.],
                                             [55./9, 64.5/12, 74./12, 32.5/10, 0.],
                                             [55./9, 59.5/10, 52./9, 31.5/9, 0.],
                                             [41./6, 43.5/9, 52./9, np.nan, np.nan]]])

        # Check if the calculated standard deviation is equal ( to desired tolerance 1e-07 ) to the ground truth
        np.testing.assert_allclose(cv_aggreg['cost_volume'].data, aggregated_ground_truth, rtol=1e-07)

    def test_cross_support_region(self):
        """
        Test the method cross support region

        """
        # Computes cross support for the reference image, with cbca_intensity=5 and cbca_distance=3
        csr_ = cbca.cross_support(self.ref['im'].data, 3, 5.)

        # Cross support region top arm ground truth for the reference image self.ref
        csr_ground_truth_top_arm = np.array([[0, 0, 0, 0, 0],
                                             [1, 1, 1, 1, 1],
                                             [1, 1, 1, 2, 1]])
        # Check if the calculated Cross support region top arm is equal to the ground truth
        # (same shape and all elements equals)

        np.testing.assert_array_equal(csr_[:, :, 2], csr_ground_truth_top_arm)

        # Cross support region bottom arm ground truth for the reference image self.ref
        csr_ground_truth_bottom_arm = np.array([[1, 1, 1, 2, 1],
                                                [1, 1, 1, 1, 1],
                                                [0, 0, 0, 0, 0]])
        # Check if the calculated Cross support region bottom arm is equal to the ground truth
        # (same shape and all elements equals)
        np.testing.assert_array_equal(csr_[:, :, 3], csr_ground_truth_bottom_arm)

        # Cross support region left arm ground truth for the reference image self.ref
        csr_ground_truth_left_arm = np.array([[0, 1, 1, 1, 1],
                                              [0, 1, 2, 2, 1],
                                              [0, 1, 1, 1, 1]])
        # Check if the calculated Cross support region left arm is equal to the ground truth
        # (same shape and all elements equals)
        np.testing.assert_array_equal(csr_[:, :, 0], csr_ground_truth_left_arm)

        # Cross support region right arm ground truth for the reference image self.ref
        csr_ground_truth_right_arm = np.array([[1, 1, 1, 1, 0],
                                               [2, 2, 1, 1, 0],
                                               [1, 1, 1, 1, 0]])
        # Check if the calculated Cross support region right arm is equal to the ground truth
        # (same shape and all elements equals)
        np.testing.assert_array_equal(csr_[:, :, 1], csr_ground_truth_right_arm)

    def test_compute_cbca(self):
        """
        Test the cross-based cost aggregation method

        """
        # Computes the cost aggregation with the cross-based cost aggregation method,
        # with cbca_intensity=5 and cbca_distance=3
        cbca_obj = aggregation.AbstractAggregation(**{'aggregation_method': 'cbca',
                                                   'cbca_intensity': 5., 'cbca_distance': 3})

        cv_aggreg = cbca_obj.cost_volume_aggregation(self.ref, self.sec, self.cv)

        # Aggregate cost volume ground truth with the cross-based cost aggregation method for the stereo image

        aggregated_ground_truth = np.array([[[np.nan, 36./6, 0.],
                                             [36./6, 66./9, 0.],
                                             [55./9, 74./12, 0.],
                                             [55./9, 52./9, 0.],
                                             [41./6, 52./9, np.nan]],

                                            [[np.nan, 36./6, 0.],
                                             [36./6, 66./9, 0.],
                                             [55./9, 74./12, 0.],
                                             [55./9, 52./9, 0.],
                                             [41./6, 52./9, np.nan]],

                                            [[np.nan, 36./6, 0.],
                                             [36./6, 66./9, 0.],
                                             [55./9, 74./12, 0.],
                                             [55./9, 52./9, 0.],
                                             [41./6, 52./9, np.nan]]])

        # Check if the calculated standard deviation is equal ( to desired tolerance 1e-07 ) to the ground truth
        np.testing.assert_allclose(cv_aggreg['cost_volume'].data, aggregated_ground_truth, rtol=1e-07)

    def test_cmax(self):
        """
        Test the cmax attribute of the cost volume

        """
        cbca_obj = aggregation.AbstractAggregation(**{'aggregation_method': 'cbca',
                                                   'cbca_intensity': 5., 'cbca_distance': 3})

        cv_aggreg = cbca_obj.cost_volume_aggregation(self.ref, self.sec, self.cv)

        # Check if the calculated maximal cost is equal to the ground truth
        assert (np.nanmax(cv_aggreg['cost_volume'].data) <= (24 * 18))


if __name__ == '__main__':
    import logging
    logging.basicConfig(format="[%(asctime)s][%(levelname)s] %(message)s", level=logging.DEBUG)
    unittest.main()
