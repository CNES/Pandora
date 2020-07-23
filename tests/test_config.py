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
This module contains functions to test the configuration
"""

import unittest
import json_checker

import pandora.JSON_checker as JSON_checker


class TestConfig(unittest.TestCase):
    """
    TestConfig class allows to test the configuration
    """

    def setUp(self):
        pass

    def test_check_complete_disparities(self):
        """
        Test the method check_complete_disparities
        """
        # Check the secondary disparities
        sec_min, sec_max = JSON_checker.check_complete_disparities(-60, 0, None, None, 'tests/pandora/ref.png')

        if (sec_min != 0) and (sec_max != 60):
            raise AssertionError

        # Check the secondary disparities with negative disparity range
        sec_min, sec_max = JSON_checker.check_complete_disparities(-60, -10, None, None, 'tests/pandora/ref.png')

        if (sec_min != 10) and (sec_max != 60):
            raise AssertionError

        # Check the secondary disparities with positive disparity range
        sec_min, sec_max = JSON_checker.check_complete_disparities(10, 60, None, None, 'tests/pandora/ref.png')

        if (sec_min != -60) and (sec_max != -10):
            raise AssertionError

        # Check the secondary disparities with reference disparity grids
        sec_min, sec_max = JSON_checker.check_complete_disparities("tests/pandora/disp_min_grid.tif",
                                                                   "tests/pandora/disp_max_grid.tif", None,
                                                                   None, 'tests/pandora/ref.png')

        if (sec_min is not None) and (sec_max is not None):
            raise AssertionError

        # Check the secondary disparities with reference and secondary disparity grids
        sec_min, sec_max = JSON_checker.check_complete_disparities("tests/pandora/disp_min_grid.tif",
                                                                   "tests/pandora/disp_max_grid.tif",
                                                                   "tests/pandora/disp_min_grid.tif",
                                                                   "tests/pandora/disp_max_grid.tif",
                                                                   'tests/pandora/ref.png')

        if (sec_min != "tests/pandora/disp_min_grid.tif") and (sec_max != "tests/pandora/disp_max_grid.tif"):
            raise AssertionError

    def test_check_input_section(self):
        """
        Test the method check_input_section
        """

        # Test configuration with reference disparity grids and secondary disparity = none
        cfg = {
            "input": {
                "img_ref": "tests/pandora/ref.png",
                "img_sec": "tests/pandora/sec.png",
                "disp_min": "tests/pandora/disp_min_grid.tif",
                "disp_max": "tests/pandora/disp_max_grid.tif"
            }
        }
        cfg_return = JSON_checker.check_input_section(cfg)
        if (cfg_return['input']['disp_min_sec'] is not None) and (cfg_return['input']['disp_max_sec'] is not None):
            raise AssertionError

        # Test configuration with reference disparity as integer
        cfg = {
            "input": {
                "img_ref": "tests/pandora/ref.png",
                "img_sec": "tests/pandora/sec.png",
                "disp_min": -60,
                "disp_max": 0
            }
        }
        cfg_return = JSON_checker.check_input_section(cfg)
        if (cfg_return['input']['disp_min_sec'] != 0) and (cfg_return['input']['disp_max_sec'] != 60):
            raise AssertionError

        # Test configuration with reference and secondary disparity grids
        cfg = {
            "input": {
                "img_ref": "tests/pandora/ref.png",
                "img_sec": "tests/pandora/sec.png",
                "disp_min": "tests/pandora/disp_min_grid.tif",
                "disp_max": "tests/pandora/disp_max_grid.tif",
                "disp_min_sec": "tests/pandora/disp_min_grid.tif",
                "disp_max_sec": "tests/pandora/disp_max_grid.tif"
            }
        }
        cfg_return = JSON_checker.check_input_section(cfg)
        if (cfg_return['input']['disp_min_sec'] != "tests/pandora/disp_min_grid.tif") and \
                (cfg_return['input']['disp_max_sec'] != "tests/pandora/disp_max_grid.tif"):
            raise AssertionError

    def test_check_input_section_with_error(self):
        """
        Test the method check_input_section that must raise an error
        """
        # Test configuration with reference disparity min as grids and reference disparity max as integer
        cfg = {
            "input": {
                "img_ref": "tests/pandora/ref.png",
                "img_sec": "tests/pandora/sec.png",
                "disp_min": "tests/pandora/disp_min_grid.tif",
                "disp_max": 45,
            }
        }
        # Json checker must raise an error
        self.assertRaises(json_checker.core.exceptions.DictCheckerError, JSON_checker.check_input_section, cfg)

        # Test configuration with reference disparity grids and secondary disparity max as integer
        cfg = {
            "input": {
                "img_ref": "tests/pandora/ref.png",
                "img_sec": "tests/pandora/sec.png",
                "disp_min": "tests/pandora/disp_min_grid.tif",
                "disp_max": "tests/pandora/disp_max_grid.tif",
                "disp_max_sec": -4
            }
        }
        # Json checker must raise an error
        self.assertRaises(json_checker.core.exceptions.DictCheckerError, JSON_checker.check_input_section, cfg)

        # Test configuration with reference disparity grids and secondary disparity max as integer
        cfg = {
            "input": {
                "img_ref": "tests/pandora/ref.png",
                "img_sec": "tests/pandora/sec.png",
                "disp_min": "tests/pandora/disp_min_grid.tif",
                "disp_max": "tests/pandora/disp_max_grid.tif",
                "disp_max_sec": -4,
                "disp_min_sec": "tests/pandora/disp_max_grid.tif"
            }
        }
        # Json checker must raise an error
        self.assertRaises(json_checker.core.exceptions.DictCheckerError, JSON_checker.check_input_section, cfg)

        # Test configuration with reference disparity grids and secondary disparity min as integer
        cfg = {
            "input": {
                "img_ref": "tests/pandora/ref.png",
                "img_sec": "tests/pandora/sec.png",
                "disp_min": "tests/pandora/disp_min_grid.tif",
                "disp_max": "tests/pandora/disp_max_grid.tif",
                "disp_min_sec": -4
            }
        }
        # Json checker must raise an error
        self.assertRaises(json_checker.core.exceptions.DictCheckerError, JSON_checker.check_input_section, cfg)

        # Test configuration with reference disparity grids and secondary disparities as integer
        cfg = {
            "input": {
                "img_ref": "tests/pandora/ref.png",
                "img_sec": "tests/pandora/sec.png",
                "disp_min": "tests/pandora/disp_min_grid.tif",
                "disp_max": "tests/pandora/disp_max_grid.tif",
                "disp_min_sec": -4,
                "disp_max_sec": 0
            }
        }
        # Json checker must raise an error
        self.assertRaises(json_checker.core.exceptions.DictCheckerError, JSON_checker.check_input_section, cfg)

    def test_check_conf(self):
        """
        Test the method check_conf
        """
        # Check the configuration returned with reference disparity grids
        cfg = {
            "input": {
                "img_ref": "tests/pandora/ref.png",
                "img_sec": "tests/pandora/sec.png",
                "disp_min": "tests/pandora/disp_min_grid.tif",
                "disp_max": "tests/pandora/disp_max_grid.tif"
            },
            "stereo": {
                "stereo_method": "zncc",
                "window_size": 5,
                "subpix": 2
            }
        }
        cfg_return = JSON_checker.check_conf(cfg)
        cfg_gt = {
            "image": {
                "nodata1": 0,
                "nodata2": 0,
                "valid_pixels": 0,
                "no_data": 1
            },
            "input": {
                "ref_mask": None,
                "sec_mask": None,
                "disp_min_sec": None,
                "disp_max_sec": None,
                "img_ref": "tests/pandora/ref.png",
                "img_sec": "tests/pandora/sec.png",
                "disp_min": "tests/pandora/disp_min_grid.tif",
                "disp_max": "tests/pandora/disp_max_grid.tif"
            },
            "invalid_disparity": -9999,
            "stereo": {
                "stereo_method": "zncc",
                "window_size": 5,
                "subpix": 2
            },
            "aggregation": {
                "aggregation_method": "none"
            },
            "optimization": {
                "optimization_method": "none"
            },
            "refinement": {
                "refinement_method": "none"
            },
            "filter": {
                "filter_method": "none"
            },
            "validation": {
                "validation_method": "none",
                "interpolated_disparity": "none"
            }
        }
        assert (cfg_return == cfg_gt)

        # Check the configuration returned with reference and secondary disparity grids
        cfg = {
            "input": {
                "img_ref": "tests/pandora/ref.png",
                "img_sec": "tests/pandora/sec.png",
                "disp_min": "tests/pandora/disp_min_grid.tif",
                "disp_max": "tests/pandora/disp_max_grid.tif",
                "disp_min_sec": "tests/pandora/disp_min_grid.tif",
                "disp_max_sec": "tests/pandora/disp_max_grid.tif"
            },
            "stereo": {
                "stereo_method": "zncc",
                "window_size": 5,
                "subpix": 2
            }
        }
        cfg_return = JSON_checker.check_conf(cfg)
        cfg_gt = {
            "image": {
                "nodata1": 0,
                "nodata2": 0,
                "valid_pixels": 0,
                "no_data": 1
            },
            "input": {
                "ref_mask": None,
                "sec_mask": None,
                "disp_min_sec": "tests/pandora/disp_min_grid.tif",
                "disp_max_sec": "tests/pandora/disp_max_grid.tif",
                "img_ref": "tests/pandora/ref.png",
                "img_sec": "tests/pandora/sec.png",
                "disp_min": "tests/pandora/disp_min_grid.tif",
                "disp_max": "tests/pandora/disp_max_grid.tif"
            },
            "invalid_disparity": -9999,
            "stereo": {
                "stereo_method": "zncc",
                "window_size": 5,
                "subpix": 2
            },
            "aggregation": {
                "aggregation_method": "none"
            },
            "optimization": {
                "optimization_method": "none"
            },
            "refinement": {
                "refinement_method": "none"
            },
            "filter": {
                "filter_method": "none"
            },
            "validation": {
                "validation_method": "none",
                "interpolated_disparity": "none"
            }
        }
        assert (cfg_return == cfg_gt)

        # Check the configuration returned with reference disparity grids and cross checking method
        cfg = {
            "input": {
                "img_ref": "tests/pandora/ref.png",
                "img_sec": "tests/pandora/sec.png",
                "disp_min": "tests/pandora/disp_min_grid.tif",
                "disp_max": "tests/pandora/disp_max_grid.tif"
            },
            "stereo": {
                "stereo_method": "zncc",
                "window_size": 5,
                "subpix": 2
            },
            "validation": {
                "validation_method": "cross_checking"
            }
        }

        # When reference disparities are grids and secondary are none, cross checking method cannot be used : the program exits
        self.assertRaises(SystemExit, JSON_checker.check_conf, cfg)

        # Check the configuration returned with reference and secondary disparity grids and cross checking method
        cfg = {
            "input": {
                "img_ref": "tests/pandora/ref.png",
                "img_sec": "tests/pandora/sec.png",
                "disp_min": "tests/pandora/disp_min_grid.tif",
                "disp_max": "tests/pandora/disp_max_grid.tif",
                "disp_min_sec": "tests/pandora/disp_min_grid.tif",
                "disp_max_sec": "tests/pandora/disp_max_grid.tif",
            },
            "stereo": {
                "stereo_method": "zncc",
                "window_size": 5,
                "subpix": 2
            },
            "validation": {
                "validation_method": "cross_checking"
            }
        }
        # When reference and secondary disparities are grids, cross checking method can be used
        cfg_return = JSON_checker.check_conf(cfg)
        cfg_gt = {
            "image": {
                "nodata1": 0,
                "nodata2": 0,
                "valid_pixels": 0,
                "no_data": 1
            },
            "input": {
                "ref_mask": None,
                "sec_mask": None,
                "disp_min_sec": "tests/pandora/disp_min_grid.tif",
                "disp_max_sec": "tests/pandora/disp_max_grid.tif",
                "img_ref": "tests/pandora/ref.png",
                "img_sec": "tests/pandora/sec.png",
                "disp_min": "tests/pandora/disp_min_grid.tif",
                "disp_max": "tests/pandora/disp_max_grid.tif"
            },
            "invalid_disparity": -9999,
            "stereo": {
                "stereo_method": "zncc",
                "window_size": 5,
                "subpix": 2
            },
            "aggregation": {
                "aggregation_method": "none"
            },
            "optimization": {
                "optimization_method": "none"
            },
            "refinement": {
                "refinement_method": "none"
            },
            "filter": {
                "filter_method": "none"
            },
            "validation": {
                "validation_method": "cross_checking",
                'cross_checking_threshold': 1.0,
                'right_left_mode': 'accurate',
                "interpolated_disparity": "none",
                'filter_interpolated_disparities': True
            }
        }
        assert (cfg_return == cfg_gt)


if __name__ == '__main__':
    unittest.main()
