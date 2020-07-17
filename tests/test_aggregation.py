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
        aggregated_ground_truth = np.array([[[np.nan, np.nan, (4+4+8+1)/4, (2+2+4+0.5+1)/5, 0.],
                                             [(0+7+10+1)/4, (2+12+3+1.5+1)/5, (4+4+14+8+1+2)/6, (2+2+7+4+0.5+1+1)/7, 0.],
                                             [(0+10+6+7+1+0)/6, (2+12+1+3+1.5+1+4)/7, (14+4+8+1+2+2+3)/7, (2+7+4+4+0.5+1+1)/7, 0.],
                                             [(10+6+12+1+0+5)/6, (12+1+8+3+1.5+1+4+6+5.5+4.5)/10, (14+8+4+2+2+3)/6, (7+4+0.5+1+1)/5, 0.],
                                             [(6+12+0+5)/4, (1+8+1.5+1+4)/5, (8+4+2+3+2)/5, np.nan, np.nan]],

                                            [[np.nan, np.nan, (4+4+8+1+2+17)/6, (2+2+4+0.5+1+1+8.5)/7, 0.],
                                             [(0+10+7+1+15+3)/6, (2+12+3+1.5+1+16+5.5)/7, (4+4+14+8+1+2+2+17+14)/9, (2+2+7+4+0.5+1+1+1+8.5+7)/10, 0.],
                                             [(0+10+6+7+1+0+15+3+13)/9, (2+12+1+3+1.5+1+4+16+5.5+6)/10, (4+14+8+1+2+2+3+17+14+1)/10, (2+7+4+4+0.5+1+1+8.5+7+0.5)/10, 0.],
                                             [(10+6+12+1+0+5+3+13+5)/9, (12+1+8+3+1.5+1+4+5.5+6+4.5)/10, (14+8+4+2+2+3+14+1+4)/9, (7+4+0.5+1+1+7+0.5)/7, 0.],
                                             [(6+12+0+5+13+5)/6, (1+8+1.5+1+4+6+4.5)/7, (2+8+4+2+3+1+4)/7, np.nan, np.nan]],

                                            [[np.nan, np.nan, (2+8+1+17)/4, (4+0.5+1+1+8.5)/5, 0.],
                                             [(7+1+15+3)/4, (3+1.5+1+16+5.5)/5, (8+1+2+2+17+14)/6, (4+0.5+1+1+1+8.5+7)/7, 0.],
                                             [(7+1+0+15+3+13)/6, (3+1.5+1+4+16+5.5+6)/7, (1+2+2+17+14+1+3)/7, (4+0.5+1+1+8.5+7+0.5)/7, 0.],
                                             [(1+0+5+3+13+5)/6, (1+8+3+1.5+1+4+5.5+6+4.5+12)/10, (2+2+3+14+1+4)/6, (0.5+1+1+7+0.5)/5, 0.],
                                             [(0+5+13+5)/4, (1.5+1+4+6+4.5)/5, (2+2+3+1+4)/5, np.nan, np.nan]]])

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

        aggregated_ground_truth = np.array([[[np.nan, (4+4+8+1)/4, 0.],
                                             [(0+7+10+1)/4, (4+4+14+8+1+2)/6, 0.],
                                             [(0+10+6+7+1+0)/6, (14+4+8+1+2+2+3)/7, 0.],
                                             [(10+6+12+1+0+5)/6, (14+8+4+2+2+3)/6, 0.],
                                             [(6+12+0+5)/4, (8+4+2+3+2)/5, np.nan]],

                                            [[np.nan, (4+4+8+1+2+17)/6, 0.],
                                             [(0+10+7+1+15+3)/6, (4+4+14+8+1+2+2+17+14)/9, 0.],
                                             [(0+10+6+7+1+0+15+3+13)/9, (4+14+8+1+2+2+3+17+14+1)/10, 0.],
                                             [(10+6+12+1+0+5+3+13+5)/9, (14+8+4+2+2+3+14+1+4)/9, 0.],
                                             [(6+12+0+5+13+5)/6, (2+8+4+2+3+1+4)/7, np.nan]],

                                            [[np.nan, (2+8+1+17)/4, 0.],
                                             [(7+1+15+3)/4, (8+1+2+2+17+14)/6, 0.],
                                             [(7+1+0+15+3+13)/6, (1+2+2+17+14+1+3)/7, 0.],
                                             [(1+0+5+3+13+5)/6, (2+2+3+14+1+4)/6, 0.],
                                             [(0+5+13+5)/4, (2+2+3+1+4)/5, np.nan]]])

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

    def test_compute_cbca_with_invalid_cost(self):
        """
        Test the cross-based cost aggregation method with invalid cost

        """
        # Invalid pixel in the reference image
        # Mask convention
        # cfg['image']['valid_pixels'] = 0
        # cfg['image']['no_data'] = 1
        # --------------- Without invalid / nodata pixels ----------------
        data = np.array(([[5, 1, 15, 7, 3],
                          [10, 9, 11, 9, 6],
                          [1, 18, 4, 5, 9],
                          [5, 1, 15, 7, 3]]), dtype=np.float32)
        mask = np.array(([[0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0],
                          [3, 0, 0, 0, 0]]))
        ref = xr.Dataset({'im': (['row', 'col'], data), 'msk': (['row', 'col'], mask)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})
        ref.attrs['valid_pixels'] = 0
        ref.attrs['no_data'] = 1

        data = np.array(([[1, 5, 1, 15, 7],
                          [2, 10, 9, 11, 9],
                          [3, 1, 18, 4, 5],
                          [1, 5, 1, 15, 7]]), dtype=np.float32)
        mask = np.array(([[0, 0, 0, 0, 0],
                          [0, 0, 5, 1, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0]]))
        sec = xr.Dataset({'im': (['row', 'col'], data), 'msk': (['row', 'col'], mask)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})
        sec.attrs['valid_pixels'] = 0
        sec.attrs['no_data'] = 1

        stereo_matcher = stereo.AbstractStereo(**{'stereo_method': 'sad', 'window_size': 1, 'subpix': 1})
        sad = stereo_matcher.compute_cost_volume(img_ref=ref, img_sec=sec, disp_min=-1, disp_max=1,
                                                 **{'valid_pixels': 0, 'no_data': 1})

        cbca_obj = aggregation.AbstractAggregation(**{'aggregation_method': 'cbca',
                                                   'cbca_intensity': 5., 'cbca_distance': 3})

        cv_aggreg = cbca_obj.cost_volume_aggregation(ref, sec, sad, **{'valid_pixels': 0, 'no_data': 1})

        # Aggregate cost volume ground truth with the cross-based cost aggregation method for the stereo image disp = 0
        aggregated_ground_truth = np.array([[(4+8+1)/3, np.nan, (14+8)/2, (8+14+4)/3, (4+8+3)/3],
                                            [(8+4+1+2+17)/5, (8+1+2+17+14)/5, np.nan, np.nan, (8+4+3+4+4+8)/6.],
                                            [(2+8+1+17)/4, (8+1+2+17+14+4+14)/7, (17+14+4+14+8)/5, np.nan, (4+3+4+8)/4],
                                            [np.nan, (4+2+17+14+14)/5, (14+17+14+4+8)/5, (14+8+4)/3, (4+4+8)/3]])

        # Check if the calculated aggregated cost volume is equal ( to desired tolerance 1e-07 ) to the ground truth
        np.testing.assert_allclose(cv_aggreg['cost_volume'].data[:, :, 1], aggregated_ground_truth, rtol=1e-07)

    def test_compute_cbca_with_offset(self):
        """
        Test the cross-based cost aggregation method when the window_size is > 1

        """
        data = np.array(([[5, 1, 15, 7, 3],
                          [10, 9, 11, 9, 6],
                          [1, 18, 4, 5, 9],
                          [5, 1, 15, 7, 3]]), dtype=np.float32)
        ref = xr.Dataset({'im': (['row', 'col'], data)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})

        data = np.array(([[1, 5, 1, 15, 7],
                          [2, 10, 9, 11, 9],
                          [3, 1, 18, 4, 5],
                          [1, 5, 1, 15, 7]]), dtype=np.float32)
        sec = xr.Dataset({'im': (['row', 'col'], data)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})

        stereo_matcher = stereo.AbstractStereo(**{'stereo_method': 'sad', 'window_size': 3, 'subpix': 1})
        sad = stereo_matcher.compute_cost_volume(img_ref=ref, img_sec=sec, disp_min=-1, disp_max=1)

        cbca_obj = aggregation.AbstractAggregation(**{'aggregation_method': 'cbca',
                                                   'cbca_intensity': 5., 'cbca_distance': 3})

        cv_aggreg = cbca_obj.cost_volume_aggregation(ref, sec, sad, **{'valid_pixels': 5, 'no_data': 7})

        # Aggregate cost volume ground truth with the cross-based cost aggregation method for the stereo image
        aggregated_ground_truth = np.array([[[np.nan, (66.+63+66+63)/4, 0.],
                                             [55., (66+63+52+66+63+52)/6, 0.],
                                             [55., (63+63+52+52)/4, np.nan]],
                                            [[np.nan, (66.+63+66+63)/4, 0.],
                                             [55., (66+63+52+66+63+52)/6, 0.],
                                             [55., (63+63+52+52)/4, np.nan]]])

        # Check if the calculated aggregated cost volume is equal ( to desired tolerance 1e-07 ) to the ground truth
        np.testing.assert_allclose(cv_aggreg['cost_volume'].data, aggregated_ground_truth, rtol=1e-07)

    def test_computes_cross_support(self):
        """
        Test the method computes_cross_support

        """
        # --------------- Without invalid / nodata pixels ----------------
        data = np.array(([[5, 1, 15, 7, 3],
                          [10, 9, 11, 9, 6],
                          [1, 18, 4, 5, 9]]), dtype=np.float32)
        ref = xr.Dataset({'im': (['row', 'col'], data)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})
        ref.attrs['valid_pixels'] = 0
        ref.attrs['no_data'] = 1

        data = np.array(([[1, 5, 1, 15, 7],
                          [2, 10, 9, 11, 9],
                          [3, 1, 18, 4, 5]]), dtype=np.float32)

        sec = xr.Dataset({'im': (['row', 'col'], data)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})
        sec.attrs['valid_pixels'] = 0
        sec.attrs['no_data'] = 1

        stereo_matcher = stereo.AbstractStereo(**{'stereo_method': 'sad', 'window_size': 1, 'subpix': 1})
        sad = stereo_matcher.compute_cost_volume(img_ref=ref, img_sec=sec, disp_min=-1, disp_max=1,
                                                 **{'valid_pixels': 0, 'no_data': 1})

        cbca_obj = aggregation.AbstractAggregation(**{'aggregation_method': 'cbca',
                                                   'cbca_intensity': 5., 'cbca_distance': 3})
        cross_ref, cross_sec = cbca_obj.computes_cross_supports(ref, sec, sad, {'valid_pixels': 0, 'no_data': 1})

        # Cross support region top arm ground truth for the reference
        top_arm = np.array([[0, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 2, 1]])

        # Cross support region bottom arm ground truth for the reference image
        bottom_arm = np.array([[1, 1, 1, 2, 1],
                               [1, 1, 1, 1, 1],
                               [0, 0, 0, 0, 0]])

        # Cross support region left arm ground truth for the reference image
        left_arm = np.array([[0, 1, 1, 1, 1],
                             [0, 1, 2, 2, 2],
                             [0, 1, 1, 1, 1]])

        # Cross support region right arm ground truth for the reference image
        right_arm = np.array([[1, 1, 1, 1, 0],
                              [2, 2, 2, 1, 0],
                              [1, 1, 1, 1, 0]])

        gt_ref_arms = np.stack((left_arm, right_arm, top_arm, bottom_arm), axis=-1)
        # Check if the calculated reference cross support region is equal to the ground truth
        # (same shape and all elements equals)
        np.testing.assert_array_equal(cross_ref, gt_ref_arms)

        # Cross support region top arm ground truth for the secondary image
        top_arm = np.array([[0, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1],
                            [2, 2, 1, 1, 2]])

        # Cross support region bottom arm ground truth for the secondary image
        bottom_arm = np.array([[2, 2, 1, 1, 2],
                               [1, 1, 1, 1, 1],
                               [0, 0, 0, 0, 0]])

        # Cross support region left arm ground truth for the secondary image
        left_arm = np.array([[0, 1, 2, 1, 1],
                             [0, 1, 1, 1, 2],
                             [0, 1, 1, 1, 1]])

        # Cross support region right arm ground truth for the secondary image
        right_arm = np.array([[2, 1, 1, 1, 0],
                              [1, 1, 2, 1, 0],
                              [1, 1, 1, 1, 0]])

        gt_sec_arms = np.stack((left_arm, right_arm, top_arm, bottom_arm), axis=-1)

        # Check if the calculated secondary cross support region is equal to the ground truth
        # (same shape and all elements equals)
        np.testing.assert_array_equal(cross_sec[0], gt_sec_arms)
        # No subpixel precision
        assert len(cross_sec) == 1

        # --------------- With invalid / nodata pixels ----------------
        data = np.array(([[5, 1, 15, 7, 3],
                          [10, 9, 11, 9, 6],
                          [1, 18, 4, 5, 9]]), dtype=np.float32)
        mask = np.array(([[2, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0],
                          [0, 3, 0, 0, 0]]))
        ref = xr.Dataset({'im': (['row', 'col'], data), 'msk': (['row', 'col'], mask)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})
        ref.attrs['valid_pixels'] = 0
        ref.attrs['no_data'] = 1

        data = np.array(([[1, 5, 1, 15, 7],
                          [2, 10, 9, 11, 9],
                          [3, 1, 18, 4, 5]]), dtype=np.float32)
        mask = np.array(([[0, 0, 0, 0, 0],
                          [0, 1, 0, 3, 0],
                          [0, 0, 0, 0, 0]]))
        sec = xr.Dataset({'im': (['row', 'col'], data), 'msk': (['row', 'col'], mask)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})
        sec.attrs['valid_pixels'] = 0
        sec.attrs['no_data'] = 1

        stereo_matcher = stereo.AbstractStereo(**{'stereo_method': 'sad', 'window_size': 1, 'subpix': 1})
        sad = stereo_matcher.compute_cost_volume(img_ref=ref, img_sec=sec, disp_min=-1, disp_max=1,
                                                 **{'valid_pixels': 0, 'no_data': 1})

        cbca_obj = aggregation.AbstractAggregation(**{'aggregation_method': 'cbca',
                                                   'cbca_intensity': 6., 'cbca_distance': 3})
        cross_ref, cross_sec = cbca_obj.computes_cross_supports(ref, sec, sad, {'valid_pixels': 0, 'no_data': 1})

        # Cross support region top arm ground truth for the reference
        top_arm = np.array([[0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 1],
                            [1, 0, 1, 0, 1]])

        # Cross support region bottom arm ground truth for the reference image
        bottom_arm = np.array([[0, 1, 1, 0, 1],
                               [1, 0, 1, 0, 1],
                               [0, 0, 0, 0, 0]])

        # Cross support region left arm ground truth for the reference image
        left_arm = np.array([[0, 0, 1, 1, 1],
                             [0, 1, 2, 0, 0],
                             [0, 0, 0, 1, 2]])

        # Cross support region right arm ground truth for the reference image
        right_arm = np.array([[0, 1, 1, 1, 0],
                              [2, 1, 0, 0, 0],
                              [0, 0, 2, 1, 0]])

        gt_ref_arms = np.stack((left_arm, right_arm, top_arm, bottom_arm), axis=-1)

        # Check if the calculated reference cross support region is equal to the ground truth
        # (same shape and all elements equals)
        np.testing.assert_array_equal(cross_ref, gt_ref_arms)

        # Cross support region top arm ground truth for the secondary
        top_arm = np.array([[0, 0, 0, 0, 0],
                            [1, 0, 1, 0, 1],
                            [2, 0, 1, 0, 2]])

        # Cross support region bottom arm ground truth for the secondary image
        bottom_arm = np.array([[2, 0, 1, 0, 2],
                               [1, 0, 1, 0, 1],
                               [0, 0, 0, 0, 0]])

        # Cross support region left arm ground truth for the secondary image
        left_arm = np.array([[0, 1, 2, 1, 1],
                             [0, 0, 0, 0, 0],
                             [0, 1, 1, 1, 1]])

        # Cross support region right arm ground truth for the secondary image
        right_arm = np.array([[2, 1, 1, 1, 0],
                              [0, 0, 0, 0, 0],
                              [1, 1, 1, 1, 0]])

        gt_sec_arms = np.stack((left_arm, right_arm, top_arm, bottom_arm), axis=-1)

        # Check if the calculated secondary cross support region is equal to the ground truth
        # (same shape and all elements equals)
        np.testing.assert_array_equal(cross_sec[0], gt_sec_arms)
        # No subpixel precision
        assert len(cross_sec) == 1

    def test_computes_cross_support_with_subpixel(self):
        """
        Test the method computes_cross_support with subpixel precision

        """
        # --------------- Without invalid / nodata pixels ----------------
        data = np.array(([[5, 1, 15, 7, 3],
                          [10, 9, 11, 9, 6],
                          [1, 18, 4, 5, 9]]), dtype=np.float32)
        ref = xr.Dataset({'im': (['row', 'col'], data)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})
        ref.attrs['valid_pixels'] = 0
        ref.attrs['no_data'] = 1

        data = np.array(([[1, 5, 1, 15, 7],
                          [2, 10, 9, 11, 9],
                          [3, 1, 18, 4, 5]]), dtype=np.float32)

        sec = xr.Dataset({'im': (['row', 'col'], data)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})
        sec.attrs['valid_pixels'] = 0
        sec.attrs['no_data'] = 1

        stereo_matcher = stereo.AbstractStereo(**{'stereo_method': 'sad', 'window_size': 1, 'subpix': 2})
        sad = stereo_matcher.compute_cost_volume(img_ref=ref, img_sec=sec, disp_min=-1, disp_max=1,
                                                 **{'valid_pixels': 0, 'no_data': 1})

        cbca_obj = aggregation.AbstractAggregation(**{'aggregation_method': 'cbca',
                                                   'cbca_intensity': 5., 'cbca_distance': 3})
        cross_ref, cross_sec = cbca_obj.computes_cross_supports(ref, sec, sad, {'valid_pixels': 0, 'no_data': 1})

        # Cross support region top arm ground truth for the secondary shifted image
        top_arm = np.array([[0, 0, 0, 0],
                            [1, 1, 1, 1],
                            [2, 1, 2, 1]])

        # Cross support region bottom arm ground truth for the secondary shifted image
        bottom_arm = np.array([[2, 1, 2, 1],
                               [1, 1, 1, 1],
                               [0, 0, 0, 0]])

        # Cross support region left arm ground truth for the secondary shifted image
        left_arm = np.array([[0, 1, 1, 1],
                             [0, 1, 2, 2],
                             [0, 1, 1, 1]])

        # Cross support region right arm ground truth for the secondary shifted image
        right_arm = np.array([[1, 1, 1, 0],
                              [2, 2, 1, 0],
                              [1, 1, 1, 0]])

        gt_sec_arms = np.stack((left_arm, right_arm, top_arm, bottom_arm), axis=-1)

        # Check if the calculated secondary cross support region is equal to the ground truth
        # (same shape and all elements equals)
        np.testing.assert_array_equal(cross_sec[1], gt_sec_arms)

        # --------------- With invalid / nodata pixels ----------------
        data = np.array(([[5, 1, 15, 7, 3],
                          [10, 9, 11, 9, 6],
                          [1, 18, 4, 5, 9]]), dtype=np.float32)
        mask = np.array(([[0, 0, 0, 0, 0],
                          [0, 1, 0, 3, 0],
                          [0, 0, 0, 0, 0]]))
        ref = xr.Dataset({'im': (['row', 'col'], data), 'msk': (['row', 'col'], mask)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})
        ref.attrs['valid_pixels'] = 0
        ref.attrs['no_data'] = 1

        data = np.array(([[1, 5, 1, 15, 7],
                          [2, 10, 9, 11, 9],
                          [3, 1, 18, 4, 5]]), dtype=np.float32)
        mask = np.array(([[2, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0],
                          [0, 3, 0, 0, 0]]))

        sec = xr.Dataset({'im': (['row', 'col'], data), 'msk': (['row', 'col'], mask)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})
        sec.attrs['valid_pixels'] = 0
        sec.attrs['no_data'] = 1

        stereo_matcher = stereo.AbstractStereo(**{'stereo_method': 'sad', 'window_size': 1, 'subpix': 2})
        sad = stereo_matcher.compute_cost_volume(img_ref=ref, img_sec=sec, disp_min=-1, disp_max=1,
                                                 **{'valid_pixels': 0, 'no_data': 1})

        cbca_obj = aggregation.AbstractAggregation(**{'aggregation_method': 'cbca',
                                                   'cbca_intensity': 6., 'cbca_distance': 3})
        cross_ref, cross_sec = cbca_obj.computes_cross_supports(ref, sec, sad, {'valid_pixels': 0, 'no_data': 1})

        # Cross support region top arm ground truth for the secondary shifted image
        top_arm = np.array([[0, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 0]])

        # Cross support region bottom arm ground truth for the secondary shifted image
        bottom_arm = np.array([[0, 1, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0]])

        # Cross support region left arm ground truth for the secondary shifted image
        left_arm = np.array([[0, 0, 1, 1],
                             [0, 1, 0, 0],
                             [0, 0, 0, 1]])

        # Cross support region right arm ground truth for the secondary shifted image
        right_arm = np.array([[0, 1, 1, 0],
                              [1, 0, 0, 0],
                              [0, 0, 1, 0]])

        gt_sec_arms = np.stack((left_arm, right_arm, top_arm, bottom_arm), axis=-1)

        # Check if the calculated secondary cross support region is equal to the ground truth
        # (same shape and all elements equals)
        np.testing.assert_array_equal(cross_sec[1], gt_sec_arms)

    def test_computes_cross_support_with_offset(self):
        """
        Test the method computes_cross_support with window_size > 1

        """
        # --------------- Without invalid / nodata pixels ----------------
        data = np.array(([[5, 1, 15, 7, 3],
                          [10, 9, 11, 9, 6],
                          [1, 18, 4, 5, 9],
                          [5, 1, 15, 7, 3]]), dtype=np.float32)
        ref = xr.Dataset({'im': (['row', 'col'], data)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})
        ref.attrs['valid_pixels'] = 0
        ref.attrs['no_data'] = 1

        data = np.array(([[1, 5, 1, 15, 7],
                          [2, 10, 9, 11, 9],
                          [3, 1, 18, 4, 5],
                          [1, 5, 1, 15, 7]]), dtype=np.float32)

        sec = xr.Dataset({'im': (['row', 'col'], data)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})
        sec.attrs['valid_pixels'] = 0
        sec.attrs['no_data'] = 1

        stereo_matcher = stereo.AbstractStereo(**{'stereo_method': 'sad', 'window_size': 3, 'subpix': 1})
        sad = stereo_matcher.compute_cost_volume(img_ref=ref, img_sec=sec, disp_min=-1, disp_max=1,
                                                 **{'valid_pixels': 0, 'no_data': 1})

        cbca_obj = aggregation.AbstractAggregation(**{'aggregation_method': 'cbca',
                                                      'cbca_intensity': 5., 'cbca_distance': 3})
        cross_ref, cross_sec = cbca_obj.computes_cross_supports(ref, sec, sad, {'valid_pixels': 0, 'no_data': 1})

        # Cross support region top arm ground truth for the secondary shifted image
        top_arm = np.array([[0, 0, 0],
                            [1, 1, 1]])

        # Cross support region bottom arm ground truth for the secondary shifted image
        bottom_arm = np.array([[1, 1, 1],
                               [0, 0, 0]])

        # Cross support region left arm ground truth for the secondary shifted image
        left_arm = np.array([[0, 1, 2],
                             [0, 1, 2]])

        # Cross support region right arm ground truth for the secondary shifted image
        right_arm = np.array([[2, 1, 0],
                              [2, 1, 0]])

        gt_ref_arms = np.stack((left_arm, right_arm, top_arm, bottom_arm), axis=-1)
        # Check if the calculated reference cross support region is equal to the ground truth
        # (same shape and all elements equals)
        np.testing.assert_array_equal(cross_ref, gt_ref_arms)

        # Cross support region top arm ground truth for the secondary image
        top_arm = np.array([[0, 0, 0],
                            [1, 1, 1]])

        # Cross support region bottom arm ground truth for the secondary image
        bottom_arm = np.array([[1, 1, 1],
                               [0, 0, 0]])

        # Cross support region left arm ground truth for the secondary image
        left_arm = np.array([[0, 1, 1],
                             [0, 1, 1]])

        # Cross support region right arm ground truth for the secondary image
        right_arm = np.array([[1, 1, 0],
                              [1, 1, 0]])

        gt_sec_arms = np.stack((left_arm, right_arm, top_arm, bottom_arm), axis=-1)

        # Check if the calculated secondary cross support region is equal to the ground truth
        # (same shape and all elements equals)
        np.testing.assert_array_equal(cross_sec[0], gt_sec_arms)
        # No subpixel precision
        assert len(cross_sec) == 1


if __name__ == '__main__':
    unittest.main()
