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
This module contains functions to test all the methods in img_tools module.
"""

import logging
import logging.config
import json
import os
import unittest

import numpy as np
import xarray as xr

import pandora
import pandora.img_tools as img_tools


class TestImgTools(unittest.TestCase):
    """
    TestImgTools class allows to test all the methods in the module img_tools
    """

    def setUp(self):
        """
        Method called to prepare the test fixture

        """
        data = np.array(([1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 2, 1],
                         [1, 1, 1, 4, 3, 1],
                         [1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1]))

        self.img = xr.Dataset({'im': (['row', 'col'], data)},
                              coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])})

    def test_census_transform(self):
        """
        Test the census transform method

        """
        # Census transform ground truth for the image self.img with window size 3
        census_ground_truth = np.array(([0b000000000, 0b000000001, 0b000001011, 0b000000110],
                                        [0b000000000, 0b000001000, 0b000000000, 0b000100000],
                                        [0b000000000, 0b001000000, 0b011000000, 0b110000000]))
        # Computes the census transform for the image self.img with window size 3
        census_transform = img_tools.census_transform(self.img, 3)
        # Check if the census_transform is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(census_transform['im'].data, census_ground_truth)

        # Census transform ground truth for the image self.img with window size 5
        census_ground_truth = np.array(([[0b0000000001000110000000000, 0b0]]))
        # Computes the census transform for the image self.img with window size 5
        census_transform = img_tools.census_transform(self.img, 5)
        # Check if the census_transform is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(census_transform['im'].data, census_ground_truth)

    def test_compute_mean_raster(self):
        """
        Test the method compute_mean_raster

        """
        # Mean raster ground truth for the image self.img with window size 3
        mean_ground_truth = np.array(([1., 12 / 9., 15 / 9., 15 / 9.],
                                      [1., 12 / 9., 15 / 9., 15 / 9.],
                                      [1., 12 / 9., 14. / 9, 14. / 9]))
        # Computes the mean raster for the image self.img with window size 3
        mean_r = img_tools.compute_mean_raster(self.img, 3)
        # Check if the calculated mean is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(mean_r, mean_ground_truth)

        # Mean raster ground truth for the image self.img with window size 5
        mean_ground_truth = np.array(([[31 / 25., 31 / 25.]]))
        # Computes the mean raster for the image self.img with window size 5
        mean_r = img_tools.compute_mean_raster(self.img, 5)
        # Check if the calculated mean is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(mean_r, mean_ground_truth)

    def test_compute_mean_patch(self):
        """
        Test the method compute_mean_patch

        """
        # Computes the mean for the image self.img with window size 3 centered on col=1, row=1
        mean = img_tools.compute_mean_patch(self.img, 1, 1, 3)
        # Check if the calculated mean is equal to the ground truth 1.
        self.assertEqual(mean, 1.)

        # Computes the mean for the image self.img with window size 5 centered on col=2, row=2
        mean = img_tools.compute_mean_patch(self.img, 2, 2, 5)
        # Check if the calculated mean is equal to the ground truth 31/25.
        self.assertEqual(mean, np.float32(31 / 25.))

    def test_check_inside_image(self):
        """
        Test the method check_inside_image

        """
        # Test that the coordinates row=0,col=0 are in the image self.img
        self.assertTrue(img_tools.check_inside_image(self.img, 0, 0))
        # Test that the coordinates row=-1,col=0 are not in the the image self.img
        self.assertFalse(img_tools.check_inside_image(self.img, -1, 0))
        # Test that the coordinates row=0,col=6 are not in the the image self.img
        # Because shape self.img row=6, col=5
        self.assertFalse(img_tools.check_inside_image(self.img, 0, 6))

    def test_compute_std_raster(self):
        """
        Test the method compute_std_raster

        """
        # standard deviation raster ground truth for the image self.img with window size 3
        std_ground_truth = np.array(
            ([0., np.std(self.img['im'][:3, 1:4]), np.std(self.img['im'][:3, 2:5]), np.std(self.img['im'][:3, 3:])],
             [0., np.std(self.img['im'][1:4, 1:4]), np.std(self.img['im'][1:4, 2:5]), np.std(self.img['im'][1:4, 3:])],
             [0., np.std(self.img['im'][2:5, 1:4]), np.std(self.img['im'][2:5, 2:5]), np.std(self.img['im'][2:5, 3:])]))
        # Computes the standard deviation raster for the image self.img with window size 3
        std_r = img_tools.compute_std_raster(self.img, 3)
        # Check if the calculated standard deviation is equal ( to desired tolerance 1e-07 ) to the ground truth
        np.testing.assert_allclose(std_r, std_ground_truth, rtol=1e-07)

        # standard deviation raster ground truth for the image self.img with window size 5
        std_ground_truth = np.array(([[np.std(self.img['im'][:, :5]), np.std(self.img['im'][:, 1:])]]))
        # Computes the standard deviation raster for the image self.img with window size 5
        std_r = img_tools.compute_std_raster(self.img, 5)
        # Check if the calculated standard deviation is equal ( to desired tolerance 1e-07 ) to the ground truth
        np.testing.assert_allclose(std_r, std_ground_truth, rtol=1e-07)

    @staticmethod
    def test_read_img():
        """
        Test the method read_img

        """
        # Build the default configuration
        default_cfg = pandora.json_checker.default_short_configuration
        # left_img = array([[ -9999.,  1.,  2.,  3.,  -9999.],
        #                  [ 5.,  6.,  7.,  8.,  9.],
        #                  [ -9999.,  -9999., 23.,  5.,  6.],
        #                  [12.,  5.,  6.,  3.,  -9999.]], dtype=float32)

        # Convention 0 is a valid pixel, everything else is considered invalid
        # mask_left = array([[  0,   0,   1,   2,   0],
        #                   [  0,   0,   0,   0,   1],
        #                   [  3,   5,   0,   0,   1],
        #                   [  0,   0, 565,   0,   0]])

        # Computes the dataset image
        dst_left = img_tools.read_img(img='tests/image/left_img.tif', no_data=default_cfg['input']['nodata_left']
                                      , mask='tests/image/mask_left.tif')

        # Mask ground truth
        mask_gt = np.array([[1, 0, 2, 2, 1],
                            [0, 0, 0, 0, 2],
                            [1, 1, 0, 0, 2],
                            [0, 0, 2, 0, 1]])

        # Check if the calculated mask is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(dst_left['msk'].data, mask_gt)

        left_img = np.array([[-9999., 1., 2., 3., -9999.],
                             [5., 6., 7., 8., 9.],
                             [-9999., -9999., 23., 5., 6.],
                             [12., 5., 6., 3., -9999.]], dtype=np.float32)

        # Check the image
        np.testing.assert_array_equal(dst_left['im'].data, left_img)

    @staticmethod
    def test_read_img_with_nan():
        """
        Test the method read_img

        """
        # left_img = array([[ NaN,  1.,  2.,  3.,  NaN],
        #                  [ 5.,  6.,  7.,  8.,  9.],
        #                  [ NaN,  0., 23.,  5.,  6.],
        #                  [12.,  5.,  6.,  3.,  NaN]], dtype=float32)

        # Convention 0 is a valid pixel, everything else is considered invalid
        # mask_left = array([[  0,   0,   1,   2,   0],
        #                   [  0,   0,   0,   0,   1],
        #                   [  3,   5,   0,   0,   1],
        #                   [  0,   0, 565,   0,   0]])

        # Computes the dataset image and use nan as no data,not cfg value
        dst_left = img_tools.read_img(img='tests/image/left_img_nan.tif', no_data=np.nan,
                                      mask='tests/image/mask_left.tif')

        # Mask ground truth
        mask_gt = np.array([[1, 0, 2, 2, 1],
                            [0, 0, 0, 0, 2],
                            [1, 1, 0, 0, 2],
                            [0, 0, 2, 0, 1]])

        # Check if the calculated mask is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(dst_left['msk'].data, mask_gt)

        left_img = np.array([[-9999.0, 1., 2., 3., -9999.0],
                             [5., 6., 7., 8., 9.],
                             [-9999.0, -9999.0, 23., 5., 6.],
                             [12., 5., 6., 3., -9999.0]], dtype=np.float32)

        # Check the image
        np.testing.assert_array_equal(dst_left['im'].data, left_img)

    @staticmethod
    def test_read_img_classif():
        """
        Test the method read_img for the classif

        """
        # Build the default configuration
        default_cfg = pandora.json_checker.default_short_configuration

        # Computes the dataset image
        dst_left = img_tools.read_img(img='tests/image/left_img.tif', no_data=default_cfg['input']['nodata_left'],
                                      classif='tests/image/mask_left.tif')

        # Classif ground truth
        classif_gt = np.array([[0, 0, 1, 2, 0],
                               [0, 0, 0, 0, 1],
                               [3, 5, 0, 0, 1],
                               [0, 0, 255, 0, 0]], )

        # Check if the calculated mask is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(dst_left['classif'].data, classif_gt)

    @staticmethod
    def test_read_img_segm():
        """
        Test the method read_img for the segmentation

        """
        # Build the default configuration
        default_cfg = pandora.json_checker.default_short_configuration

        # Computes the dataset image
        dst_left = img_tools.read_img(img='tests/image/left_img.tif', no_data=default_cfg['input']['nodata_left'],
                                      segm='tests/image/mask_left.tif')

        # Segmentation ground truth
        segm_gt = np.array([[0, 0, 1, 2, 0],
                            [0, 0, 0, 0, 1],
                            [3, 5, 0, 0, 1],
                            [0, 0, 255, 0, 0]],)

        # Check if the calculated mask is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(dst_left['segm'].data, segm_gt)

    @staticmethod
    def test_read_disp():
        """
        Test the method read_disp
        """
        # Ground truth (numpy array of pandora/tests/image/mask_left image)
        gt = np.array([[0, 0, 1, 2, 0],
                       [0, 0, 0, 0, 1],
                       [3, 5, 0, 0, 1],
                       [0, 0, 255, 0, 0]])

        disp_ = img_tools.read_disp('tests/image/mask_left.tif')

        # Check if the calculated disparity is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(disp_, gt)

        # Check with integer disparity
        gt = -60
        disp_ = img_tools.read_disp(-60)

        # Check if the calculated disparity is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(disp_, gt)


def setup_logging(path='logging.json', default_level=logging.WARNING, ):
    """
    Setup the logging configuration
    :param path: path to the configuration file
    :type path: string
    :param default_level: default level
    :type default_level: logging level
    """
    if os.path.exists( path):
        with open(path, 'rt') as file_:
            config = json.load( file_)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level = default_level)


if __name__ == '__main__':
    setup_logging()
    unittest.main()
