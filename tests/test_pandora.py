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
This module contains functions to test the Pandora pipeline.
"""

import unittest
import logging
import logging.config
import os
import json
import rasterio
import numpy as np
import xarray as xr

import pandora
from pandora.img_tools import read_img
from pandora import import_plugin
from tempfile import TemporaryDirectory
import pandora.common as common
from pandora.state_machine import PandoraMachine


class TestPandora(unittest.TestCase):
    """
    TestPandora class allows to test the pandora pipeline
    """

    def setUp(self):
        """
        Method called to prepare the test fixture

        """
        # Build the default configuration
        default_cfg = pandora.JSON_checker.default_short_configuration

        self.ref = read_img('tests/pandora/ref.png', no_data=np.nan, cfg=default_cfg['image'], mask=None)
        self.sec = read_img('tests/pandora/sec.png', no_data=np.nan, cfg=default_cfg['image'], mask=None)
        self.disp_ref = rasterio.open('tests/pandora/disp_ref.tif').read(1)
        self.disp_sec = rasterio.open('tests/pandora/disp_sec.tif').read(1)
        self.occlusion = rasterio.open('tests/pandora/occlusion.png').read(1)

    def error(self, data, gt, threshold, unknown_disparity=0):
        """
        Percentage of bad pixels whose error is > 1

        """
        row, col = self.ref['im'].shape
        nb_error = 0
        for r in range(row):
            for c in range(col):
                if gt[r, c] != unknown_disparity:
                    if abs((data[r, c] + gt[r, c])) > threshold:
                        nb_error += 1

        return nb_error / float(row * col)

    def error_mask(self, data, gt):
        """
        Percentage of bad pixels ( != ground truth ) in the validity mask
        """
        row, col = self.ref['im'].shape
        nb_error = 0
        for r in range(row):
            for c in range(col):
                if data[r, c] != gt[r, c]:
                    nb_error += 1

        return nb_error / float(row * col)

    def test_run(self):
        """
        Test the run method

        """
        user_cfg = {
            "pipeline":
                {
                    "stereo": {
                        "stereo_method": "zncc",
                        "window_size": 5,
                        "subpix": 2
                    },
                    "disparity": "wta",
                    "refinement": {
                        "refinement_method": "vfit"
                    },
                    "filter": {
                        "filter_method": "median"
                    },
                    "validation": {
                        "validation_method": "cross_checking",
                        "right_left_mode": "accurate"
                    }
                }
        }
        pandora_machine = PandoraMachine()

        # Update the user configuration with default values
        cfg = pandora.JSON_checker.update_conf(pandora.JSON_checker.default_short_configuration, user_cfg)

        # Run the pandora pipeline
        ref, sec = pandora.run(pandora_machine, self.ref, self.sec, -60, 0, cfg)

        # Check the reference disparity map
        if self.error(ref['disparity_map'].data, self.disp_ref, 1) > 0.20:
            raise AssertionError

        # Check the reference validity mask cross checking ( bit 8 and 9 )
        occlusion = np.ones((ref['validity_mask'].shape[0], ref['validity_mask'].shape[1]))
        occlusion[ref['validity_mask'].data >= 512] = 0

        if self.error_mask(occlusion, self.occlusion) > 0.16:
            raise AssertionError

        # Check the secondary disparity map
        if self.error(-1 * sec['disparity_map'].data, self.disp_sec, 1) > 0.20:
            raise AssertionError

    def test_confidence_measure(self):
        """
        Test Pandora run method on confidence_measure map
        """

        # Create ref and sec images
        data_ref = np.array([[2, 5, 3, 1, 6, 1, 3, 3],
                             [5, 3, 2, 1, 4, 3, 3, 2],
                             [4, 2, 3, 2, 2, 3, 4, 6],
                             [4, 5, 3, 2, 0, 1, 0, 1],
                             [1, 3, 2, 1, 0, 2, 1, 3],
                             [5, 2, 1, 0, 1, 2, 3, 5],
                             [3, 3, 2, 3, 0, 4, 1, 2]], dtype=np.float32)

        img_ref = xr.Dataset({'im': (['row', 'col'], data_ref)},
                             coords={'row': np.arange(data_ref.shape[0]), 'col': np.arange(data_ref.shape[1])})

        data_sec = np.array([[1, 2, 1, 2, 5, 3, 1, 6],
                             [2, 3, 5, 3, 2, 1, 4, 3],
                             [0, 2, 4, 2, 3, 2, 2, 3],
                             [5, 3, 1, 4, 5, 3, 2, 0],
                             [2, 1, 3, 2, 1, 0, 2, 1],
                             [5, 5, 5, 2, 1, 0, 1, 2],
                             [1, 2, 2, 3, 3, 2, 3, 0]], dtype=np.float32)
        img_sec = xr.Dataset({'im': (['row', 'col'], data_sec)},
                             coords={'row': np.arange(data_sec.shape[0]), 'col': np.arange(data_sec.shape[1])})

        # Load a configuration
        user_cfg = {
            "input": {
                "disp_min": -2,
                "disp_max": 2
            },
            "pipeline": {
                "stereo": {
                    "stereo_method": "census",
                    "window_size": 5,
                    "subpix": 1
                },
                "disparity": "wta",
                "validation": {
                    "validation_method": "cross_checking",
                    "right_left_mode": "accurate"
                }
            }
        }

        pandora_machine = PandoraMachine()

        cfg = pandora.JSON_checker.update_conf(pandora.JSON_checker.default_short_configuration, user_cfg)
        import_plugin()

        # Run the Pandora pipeline
        ref, sec = pandora.run(pandora_machine,img_ref, img_sec, cfg['input']['disp_min'],
                               cfg['input']['disp_max'], cfg)

        # Ground truth confidence measure
        gt_ref_indicator_stereo = np.array([[1.57175062, 1.46969385, 1.39484766, 1.6],
                                            [1.51578363, 1.2, 1.1892855, 1.54712637],
                                            [1.43331783, 1.24835892, 1.21720992, 1.58694675]], dtype=np.float32)

        gt_ref_indicator_validation = np.array([[0, 0, 2, 3],
                                                [0, 0, 0, 2],
                                                [0, 0, 0, 1]], dtype=np.float32)

        gt_ref_confidence_measure = np.full((7, 8, 2), np.nan, dtype=np.float32)
        gt_ref_confidence_measure[2:-2, 2:-2, 0] = gt_ref_indicator_stereo
        gt_ref_confidence_measure[2:-2, 2:-2, 1] = gt_ref_indicator_validation

        gt_sec_indicator_stereo = np.array([[1.4164745, 1.33026313, 1.36, 1.47295621],
                                            [1.5147277, 1.49986666, 1.44222051, 1.24835892],
                                            [1.48916084, 1.38794813, 1.28747816, 1.24835892]], dtype=np.float32)

        gt_sec_indicator_validation = np.array([[2, 1, 0, 0],
                                                [0, 1, 0, 0],
                                                [0, 1, 0, 0]], dtype=np.float32)

        gt_sec_confidence_measure = np.full((7, 8, 2), np.nan, dtype=np.float32)
        gt_sec_confidence_measure[2:-2, 2:-2, 0] = gt_sec_indicator_stereo
        gt_sec_confidence_measure[2:-2, 2:-2, 1] = gt_sec_indicator_validation

        # assert equal on ref confidence_measure
        np.testing.assert_array_equal(gt_ref_confidence_measure, ref['confidence_measure'].data)

        # assert equal on sec confidence_measure
        np.testing.assert_array_equal(gt_sec_confidence_measure, sec['confidence_measure'].data)

    def test_main(self):
        """
        Test the main method ( read and write products )

        """
        # Create temporary directory
        with TemporaryDirectory() as tmp_dir:
            pandora.main('tests/pandora/cfg.json', tmp_dir, verbose=False)

            # Check the reference disparity map
            if self.error(rasterio.open(tmp_dir + '/ref_disparity.tif').read(1), self.disp_ref, 1) > 0.20:
                raise AssertionError

            # Check the secondary disparity map
            if self.error(-1 * rasterio.open(tmp_dir +'/sec_disparity.tif').read(1), self.disp_sec, 1) > 0.20:
                raise AssertionError

            # Check the reference validity mask cross checking ( bit 8 and 9 )
            out_occlusion = rasterio.open(tmp_dir + '/ref_validity_mask.tif').read(1)
            occlusion = np.ones((out_occlusion.shape[0], out_occlusion.shape[1]))
            occlusion[out_occlusion >= 512] = 0
    
    def test_dataset_image(self):
        """
        Test pandora with variable coordinate in dataset image

        """

        user_cfg = {
            "pipeline":{
                "stereo": {
                    "stereo_method": "census",
                    "window_size": 5,
                    "subpix": 2
                },
                "disparity": "wta",
                "refinement": {
                    "refinement_method": "vfit"
                },
                "filter": {
                    "filter_method": "median"
                },
                "validation": {
                    "validation_method": "cross_checking",
                    "right_left_mode": "accurate"
                }
            }
        }

        pandora_machine = PandoraMachine()

        # Update the user configuration with default values
        cfg = pandora.JSON_checker.update_conf(pandora.JSON_checker.default_short_configuration, user_cfg)

        ref_img = read_img('tests/pandora/ref.png', no_data=np.nan, cfg=cfg['image'], mask=None)
        sec_img = read_img('tests/pandora/sec.png', no_data=np.nan, cfg=cfg['image'], mask=None)

        # Run the pandora pipeline on images without modified coordinates
        ref_origin, sec_origin = pandora.run(pandora_machine, ref_img, sec_img, -60, 0, cfg)

        row_c = ref_img.coords['row'].data
        row_c += 41
        col_c = ref_img.coords['col'].data
        col_c += 45
        # Changes the coordinate images
        ref_img.assign_coords(row=row_c, col=col_c)
        sec_img.assign_coords(row=row_c, col=col_c)

        # Run the pandora pipeline on images with modified coordinates
        ref_modified, sec_modified = pandora.run(pandora_machine, ref_img, sec_img, -60, 0, cfg)

        # check if the disparity maps are equals
        np.testing.assert_array_equal(ref_origin['disparity_map'].values, ref_modified['disparity_map'].values)
        np.testing.assert_array_equal(sec_origin['disparity_map'].values, sec_modified['disparity_map'].values)

    def test_variable_range_of_disp(self):
        """
        Test that variable range of disparities (grids of local disparities) are well taken into account in Pandora

        """
        cfg = {
            "input": {
                "img_ref": "tests/pandora/ref.png",
                "img_sec": "tests/pandora/sec.png",
                "disp_min": "tests/pandora/disp_min_grid.tif",
                "disp_max": "tests/pandora/disp_max_grid.tif",
                "disp_min_sec": "tests/pandora/sec_disp_min_grid.tif",
                "disp_max_sec": "tests/pandora/sec_disp_max_grid.tif",
            },
            "pipeline":
                {
                    "stereo": {
                        "stereo_method": "zncc",
                        "window_size": 5,
                        "subpix": 2
                    },
                    "disparity": "wta",
                    "refinement": {
                        "refinement_method": "vfit"
                    },
                    "filter": {
                        "filter_method": "median"
                    },
                    "validation": {
                        "validation_method": "cross_checking"
                    }
                }
        }

        # Create temporary directory
        with TemporaryDirectory() as tmp_dir:

            with open(os.path.join(tmp_dir, 'config.json'), 'w') as f:
                json.dump(cfg, f, indent=2)

            # Run Pandora pipeline
            pandora.main(tmp_dir + '/config.json', tmp_dir, verbose=False)

            # Check the reference disparity map
            if self.error(rasterio.open(tmp_dir + '/ref_disparity.tif').read(1), self.disp_ref, 1) > 0.20:
                raise AssertionError

            # Check the secondary disparity map
            if self.error(-1 * rasterio.open(tmp_dir +'/sec_disparity.tif').read(1), self.disp_sec, 1) > 0.20:
                raise AssertionError


def setup_logging(path='logging.json', default_level=logging.WARNING, ):
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
