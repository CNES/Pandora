# type:ignore
#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of PANDORA
#
#     https://github.com/CNES/Pandora
#
# Licensed under the Apache License, Version 2.0 (the 'License'');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
This module contains functions to test the configuration
"""
import unittest
import copy
import json_checker
import pytest
from transitions.core import MachineError
import numpy as np
from tests import common
from pandora import check_configuration
from pandora.state_machine import PandoraMachine


class TestConfig(unittest.TestCase):
    """
    TestConfig class allows to test the configuration
    """

    def setUp(self):
        """
        Method called to prepare the test fixture

        """

    def test_check_input_section(self):
        """
        Test the method check_input_section
        """

        # Test configuration with left disparity grids and right disparity = none
        cfg = {"input": copy.deepcopy(common.input_cfg_left_grids)}
        cfg_return = check_configuration.check_input_section(cfg)

        self.assertIsNone(cfg_return["input"]["right"]["disp"])

        # Test configuration with left disparity as list of integers
        cfg = {"input": copy.deepcopy(common.input_cfg_basic)}
        cfg_return = check_configuration.check_input_section(cfg)

        self.assertIsNone(cfg_return["input"]["right"]["disp"])

        # Test configuration with left and right disparity grids
        cfg = {"input": copy.deepcopy(common.input_cfg_left_right_grids)}
        cfg_return = check_configuration.check_input_section(cfg)

        self.assertEqual(cfg_return["input"]["right"]["disp"], "tests/pandora/right_disparity_grid.tif")

        # Test configuration with left disparity grids and right disparity = none and classif and segm = none
        cfg = {"input": copy.deepcopy(common.input_cfg_left_grids)}
        cfg_return = check_configuration.check_input_section(cfg)
        inputs_to_check = ("classif", "segm")
        self.assertTrue(all(cfg_return["input"]["left"][input_key] is None for input_key in inputs_to_check))
        self.assertTrue(all(cfg_return["input"]["right"][input_key] is None for input_key in inputs_to_check))

        # Test configuration with left disparity grids and right disparity = none and left classif filled
        #           and segm filled with artificial data from left and right images
        cfg = {"input": copy.deepcopy(common.input_cfg_left_grids)}
        cfg["input"]["left"]["classif"] = "tests/pandora/right.png"
        cfg["input"]["left"]["segm"] = "tests/pandora/left.png"

        cfg_return = check_configuration.check_input_section(cfg)
        self.assertTrue(
            (cfg_return["input"]["left"]["classif"] == "tests/pandora/right.png")
            and (cfg_return["input"]["left"]["segm"] == "tests/pandora/left.png")
            and (cfg_return["input"]["right"]["classif"] is None)
            and (cfg_return["input"]["right"]["segm"] is None)
        )

        # Test configuration with left disparity grids and right disparity = none and right classif filled
        #           and segm filled with artificial data from left and right images
        cfg = {"input": copy.deepcopy(common.input_cfg_left_grids)}
        cfg["input"]["right"]["classif"] = "tests/pandora/right.png"
        cfg["input"]["right"]["segm"] = "tests/pandora/left.png"

        cfg_return = check_configuration.check_input_section(cfg)
        self.assertTrue(
            (cfg_return["input"]["right"]["classif"] == "tests/pandora/right.png")
            and (cfg_return["input"]["right"]["segm"] == "tests/pandora/left.png")
            and (cfg_return["input"]["left"]["classif"] is None)
            and (cfg_return["input"]["left"]["segm"] is None)
        )

    def test_check_input_section_with_error(self):
        """
        Test the method check_input_section that must raise an error
        """
        # ~~~~~~~~~~~~~~~~~~
        # Test type validity
        # ~~~~~~~~~~~~~~~~~~
        # Test left only disparity fails if it is a single value integer instead of tuple
        cfg = {
            "input": {
                "left": {"img": "tests/pandora/left.png", "disp": 45},
                "right": {
                    "img": "tests/pandora/right.png",
                },
            }
        }
        # Json checker must raise an error
        self.assertRaises(json_checker.core.exceptions.DictCheckerError, check_configuration.check_input_section, cfg)

        # Test right disparity fails if it is a single value integer instead of tuple
        cfg = {
            "input": {
                "left": {
                    "img": "tests/pandora/left.png",
                    "disp": [0, 45],
                },
                "right": {
                    "img": "tests/pandora/right.png",
                    "disp": 32,
                },
            }
        }
        # Json checker must raise an error
        self.assertRaises(json_checker.core.exceptions.DictCheckerError, check_configuration.check_input_section, cfg)

        # ~~~~~~~~~~~~~~~~~~~~~~~
        # Test type compatibility
        # ~~~~~~~~~~~~~~~~~~~~~~~
        # Test left disparity as grid and right disparity as ints fails
        cfg = {"input": copy.deepcopy(common.input_cfg_left_grids)}
        cfg["input"]["right"]["disp"] = [0, 45]

        # Json checker must raise an error
        self.assertRaises(json_checker.core.exceptions.DictCheckerError, check_configuration.check_input_section, cfg)

        # Test left disparity as ints and right disparity as grid fails
        cfg = {
            "input": {
                "left": {
                    "img": "tests/pandora/left.png",
                    "disp": [-4, 0],
                },
                "right": {
                    "img": "tests/pandora/right.png",
                    "disp": "tests/pandora/right_disparity_grid.tif",
                },
            }
        }
        # Json checker must raise an error
        self.assertRaises(json_checker.core.exceptions.DictCheckerError, check_configuration.check_input_section, cfg)

        # ~~~~~~~~~~~~~~~~~~
        # Test values order
        # ~~~~~~~~~~~~~~~~~~
        # Test configuration with left disparity inverted
        cfg = {
            "input": {
                "left": {
                    "img": "tests/pandora/left.png",
                    "disp": [0, -4],
                },
                "right": {
                    "img": "tests/pandora/right.png",
                },
            }
        }
        # Json checker must raise an error
        self.assertRaises(ValueError, check_configuration.check_input_section, cfg)

        # Test configuration with right disparity inverted
        cfg = {
            "input": {
                "left": {
                    "img": "tests/pandora/left.png",
                    "disp": [-4, 0],
                },
                "right": {
                    "img": "tests/pandora/right.png",
                    "disp": [0, -4],
                },
            }
        }
        # Json checker must raise an error
        self.assertRaises(json_checker.core.exceptions.DictCheckerError, check_configuration.check_input_section, cfg)

    def test_multiband_pipeline(self):
        """
        Test the method check_conf for multiband images
        """
        pandora_machine = PandoraMachine()
        cfg = {
            "input": copy.deepcopy(common.input_multiband_cfg),
            "pipeline": {
                "matching_cost": {"matching_cost_method": "zncc", "window_size": 5, "subpix": 2, "band": "r"},
                "disparity": {"disparity_method": "wta"},
            },
        }

        cfg_return = check_configuration.check_conf(cfg, pandora_machine)

        cfg_gt = {
            "input": {
                "left": {
                    "nodata": -9999,
                    "mask": None,
                    "classif": None,
                    "segm": None,
                    "disp": [-60, 0],
                    "img": "tests/pandora/left_rgb.tif",
                },
                "right": {
                    "nodata": -9999,
                    "mask": None,
                    "classif": None,
                    "segm": None,
                    "disp": None,
                    "img": "tests/pandora/right_rgb.tif",
                },
            },
            "pipeline": copy.deepcopy(common.basic_pipeline_cfg),
        }
        # correct band for correlation
        cfg_gt["pipeline"]["matching_cost"]["band"] = "r"

        del cfg_gt["pipeline"]["refinement"]
        del cfg_gt["pipeline"]["filter"]

        assert cfg_return == cfg_gt

    def test_failed_multiband_pipeline(self):
        """
        Test the method check_conf for multiband images with errors
        """

        pandora_machine = PandoraMachine()

        # config with wrong band parameters
        cfg = {
            "input": copy.deepcopy(common.input_multiband_cfg),
            "pipeline": {
                "matching_cost": {"matching_cost_method": "zncc", "window_size": 5, "subpix": 2, "band": "n"},
                "disparity": {"disparity_method": "wta"},
            },
        }
        img_left = check_configuration.get_metadata(cfg["input"]["left"]["img"], cfg["input"]["left"]["disp"])
        img_right = check_configuration.get_metadata(cfg["input"]["right"]["img"], None)
        # Check that the check_conf function raises an error
        with pytest.raises(MachineError, match="A problem occurs during Pandora checking. Be sure of your sequencing"):
            check_configuration.check_conf(cfg, pandora_machine)
        # Check that the check_band_pipeline raises an error (this shall be the source of check_conf's error)
        with pytest.raises(AttributeError, match="Wrong band instantiate on zncc step: n not in input image"):
            pandora_machine.check_band_pipeline(
                img_left.coords["band_im"].data,
                cfg["pipeline"]["matching_cost"]["matching_cost_method"],
                cfg["pipeline"]["matching_cost"]["band"],
            )
        with pytest.raises(AttributeError, match="Wrong band instantiate on zncc step: n not in input image"):
            pandora_machine.check_band_pipeline(
                img_right.coords["band_im"].data,
                cfg["pipeline"]["matching_cost"]["matching_cost_method"],
                cfg["pipeline"]["matching_cost"]["band"],
            )

        # config with missing band parameters
        cfg = {
            "input": copy.deepcopy(common.input_multiband_cfg),
            "pipeline": {
                "matching_cost": {"matching_cost_method": "zncc", "window_size": 5, "subpix": 2},
                "disparity": {"disparity_method": "wta"},
            },
        }
        img_left = check_configuration.get_metadata(cfg["input"]["left"]["img"], cfg["input"]["left"]["disp"])
        img_right = check_configuration.get_metadata(cfg["input"]["right"]["img"], None)
        pandora_machine = PandoraMachine()

        # Check that the check_conf function raises an error
        with pytest.raises(MachineError, match="A problem occurs during Pandora checking. Be sure of your sequencing"):
            check_configuration.check_conf(cfg, pandora_machine)
        # Check that the check_band_pipeline raises an error (this shall be the source of check_conf's error)
        with pytest.raises(AttributeError, match="Missing band instantiate on zncc step : input image is multiband"):
            # We add the band argument ad None because normally it is completed in the check_conf function,
            # which then calls check_band_pipeline
            pandora_machine.check_band_pipeline(
                img_left.coords["band_im"].data,
                cfg["pipeline"]["matching_cost"]["matching_cost_method"],
                band_used=None,
            )
        with pytest.raises(AttributeError, match="Missing band instantiate on zncc step : input image is multiband"):
            pandora_machine.check_band_pipeline(
                img_right.coords["band_im"].data,
                cfg["pipeline"]["matching_cost"]["matching_cost_method"],
                band_used=None,
            )

    @staticmethod
    def test_update_conf():
        """
        Test the method update_conf
        """

        # Test configuration with nodata_left and nodata_right as NaN
        user_cfg = {
            "input": {
                "left": {
                    "img": "tests/pandora/left.png",
                    "disp": [-4, 0],
                    "nodata": "NaN",
                },
                "right": {"img": "tests/pandora/right.png", "nodata": "NaN"},
            }
        }

        cfg_return = check_configuration.update_conf(check_configuration.default_short_configuration_input, user_cfg)

        if (not np.isnan(cfg_return["input"]["left"]["nodata"])) or (
            not np.isnan(cfg_return["input"]["right"]["nodata"])
        ):
            raise AssertionError

        # Test configuration with nodata_left and nodata_right as inf
        user_cfg = {
            "input": {
                "left": {
                    "img": "tests/pandora/left.png",
                    "disp": [-4, 0],
                    "nodata": "inf",
                },
                "right": {
                    "img": "tests/pandora/right.png",
                    "nodata": "inf",
                },
            }
        }

        cfg_return = check_configuration.update_conf(check_configuration.default_short_configuration_input, user_cfg)

        if not cfg_return["input"]["left"]["nodata"] == np.inf or not cfg_return["input"]["right"]["nodata"] == np.inf:
            raise AssertionError

        # Test configuration with nodata_left and nodata_right as -inf
        user_cfg = {
            "input": {
                "left": {
                    "img": "tests/pandora/left.png",
                    "disp": [-4, 0],
                    "nodata": "-inf",
                },
                "right": {
                    "img": "tests/pandora/right.png",
                    "nodata": "-inf",
                },
            }
        }

        cfg_return = check_configuration.update_conf(check_configuration.default_short_configuration_input, user_cfg)

        if (
            not cfg_return["input"]["left"]["nodata"] == -np.inf
            or not cfg_return["input"]["right"]["nodata"] == -np.inf
        ):
            raise AssertionError

        # Test configuration with nodata_left and nodata_right as int
        user_cfg = {
            "input": {
                "left": {
                    "img": "tests/pandora/left.png",
                    "disp": [-4, 0],
                    "nodata": 3,
                },
                "right": {
                    "img": "tests/pandora/right.png",
                    "nodata": -7,
                },
            }
        }

        cfg_return = check_configuration.update_conf(check_configuration.default_short_configuration_input, user_cfg)

        if not cfg_return["input"]["left"]["nodata"] == 3 or not cfg_return["input"]["right"]["nodata"] == -7:
            raise AssertionError

        # Test configuration with nodata_left as NaN and nodata_right not defined
        user_cfg = {
            "input": {
                "left": {
                    "img": "tests/pandora/left.png",
                    "disp": [-4, 0],
                    "nodata": "NaN",
                },
                "right": {
                    "img": "tests/pandora/right.png",
                },
            }
        }

        cfg_return = check_configuration.update_conf(check_configuration.default_short_configuration_input, user_cfg)

        if not np.isnan(cfg_return["input"]["left"]["nodata"]) or not cfg_return["input"]["right"]["nodata"] == -9999:
            raise AssertionError

        # Test configuration with nodata_left not defined and nodata_right as NaN
        user_cfg = {
            "input": {
                "left": {
                    "img": "tests/pandora/left.png",
                    "disp": [-4, 0],
                },
                "right": {
                    "img": "tests/pandora/right.png",
                    "nodata": "NaN",
                },
            }
        }

        cfg_return = check_configuration.update_conf(check_configuration.default_short_configuration_input, user_cfg)

        if not (cfg_return["input"]["left"]["nodata"] == -9999) or not np.isnan(cfg_return["input"]["right"]["nodata"]):
            raise AssertionError

    def test_check_conf(self):
        """
        Test the method check_conf
        """

        # Check the configuration returned with left disparity grids

        pandora_machine = PandoraMachine()
        cfg = {
            "input": copy.deepcopy(common.input_cfg_left_grids),
            "pipeline": {
                "matching_cost": {"matching_cost_method": "zncc", "window_size": 5, "subpix": 2},
                "disparity": {"disparity_method": "wta"},
            },
        }

        cfg_return = check_configuration.check_conf(cfg, pandora_machine)
        cfg_gt = {
            "input": {
                "left": {
                    "nodata": -9999,
                    "mask": None,
                    "classif": None,
                    "segm": None,
                    "disp": "tests/pandora/left_disparity_grid.tif",
                    "img": "tests/pandora/left.png",
                },
                "right": {
                    "nodata": -9999,
                    "mask": None,
                    "classif": None,
                    "segm": None,
                    "disp": None,
                    "img": "tests/pandora/right.png",
                },
            },
            "pipeline": copy.deepcopy(common.basic_pipeline_cfg),
        }

        del cfg_gt["pipeline"]["refinement"]
        del cfg_gt["pipeline"]["filter"]

        assert cfg_return == cfg_gt

        # Check the configuration returned with left and right disparity grids
        pandora_machine = PandoraMachine()
        cfg = {
            "input": copy.deepcopy(common.input_cfg_left_right_grids),
            "pipeline": {
                "matching_cost": {"matching_cost_method": "zncc", "window_size": 5, "subpix": 2},
                "disparity": {"disparity_method": "wta"},
            },
        }
        cfg_return = check_configuration.check_conf(cfg, pandora_machine)

        cfg_gt = {
            "input": {
                "left": {
                    "nodata": -9999,
                    "mask": None,
                    "classif": None,
                    "segm": None,
                    "disp": "tests/pandora/left_disparity_grid.tif",
                    "img": "tests/pandora/left.png",
                },
                "right": {
                    "nodata": -9999,
                    "mask": None,
                    "classif": None,
                    "segm": None,
                    "disp": "tests/pandora/right_disparity_grid.tif",
                    "img": "tests/pandora/right.png",
                },
            },
            "pipeline": copy.deepcopy(common.basic_pipeline_cfg),
        }

        del cfg_gt["pipeline"]["refinement"]
        del cfg_gt["pipeline"]["filter"]

        assert cfg_return == cfg_gt

        # Check the configuration returned with left disparity grids and cross checking method
        pandora_machine = PandoraMachine()
        cfg = {
            "input": copy.deepcopy(common.input_cfg_left_grids),
            "pipeline": copy.deepcopy(common.validation_pipeline_cfg),
        }

        # When left disparities are grids and right are none, cross checking method cannot be used : the program exits
        self.assertRaises(MachineError, check_configuration.check_conf, cfg, pandora_machine)

        # Check the configuration returned with left and right disparity grids and cross checking method
        cfg = {
            "input": copy.deepcopy(common.input_cfg_left_right_grids),
            "pipeline": copy.deepcopy(common.validation_pipeline_cfg),
        }
        # When left and right disparities are grids, cross checking method can be used
        pandora_machine = PandoraMachine()
        cfg_return = check_configuration.check_conf(cfg, pandora_machine)
        cfg_gt = {
            "input": {
                "left": {
                    "nodata": -9999,
                    "mask": None,
                    "classif": None,
                    "segm": None,
                    "disp": "tests/pandora/left_disparity_grid.tif",
                    "img": "tests/pandora/left.png",
                },
                "right": {
                    "nodata": -9999,
                    "mask": None,
                    "classif": None,
                    "segm": None,
                    "disp": "tests/pandora/right_disparity_grid.tif",
                    "img": "tests/pandora/right.png",
                },
            },
            "pipeline": common.validation_pipeline_cfg,
        }
        cfg_gt["pipeline"]["cost_volume_confidence"]["indicator"] = ""

        assert cfg_return == cfg_gt

        # Check the configuration returned with left disparity grids and multiscale processing
        pandora_machine = PandoraMachine()
        cfg = {
            "input": copy.deepcopy(common.input_cfg_left_grids),
            "pipeline": copy.deepcopy(common.multiscale_pipeline_cfg),
        }

        # When left disparities are grids and multiscale processing cannot be used : the program exits
        self.assertRaises(TypeError, check_configuration.check_conf, cfg, pandora_machine)

        # Check the configuration returned with left disparity integer and multiscale processing
        pandora_machine = PandoraMachine()
        cfg = {
            "input": copy.deepcopy(common.input_cfg_basic),
            "pipeline": copy.deepcopy(common.multiscale_pipeline_cfg),
        }

        cfg_return = check_configuration.check_conf(cfg, pandora_machine)

        cfg_gt = {
            "input": {
                "left": {
                    "nodata": -9999,
                    "mask": None,
                    "classif": None,
                    "segm": None,
                    "disp": [-60, 0],
                    "img": "tests/pandora/left.png",
                },
                "right": {
                    "nodata": -9999,
                    "mask": None,
                    "classif": None,
                    "segm": None,
                    "disp": None,
                    "img": "tests/pandora/right.png",
                },
            },
            "pipeline": copy.deepcopy(common.multiscale_pipeline_cfg),
        }

        assert cfg_return == cfg_gt

    def test_check_pipeline_section_with_error(self):
        """
        Test the method check_input_section that must raise an error from PandoraMachine
        """

        cfg = {
            "input": copy.deepcopy(common.input_cfg_basic),
            "pipeline": {
                "matching_cost": {"matching_cost_method": "census", "window_size": 5, "subpix": 2},
                "filter": {"filter_method": "median"},
                "disparity": {"disparity_method": "wta", "invalid_disparity": -9999},
                "validation": {"validation_method": "cross_checking_accurate"},
            },
        }

        pandora_machine = PandoraMachine()
        img_left = check_configuration.get_metadata(cfg["input"]["left"]["img"], cfg["input"]["left"]["disp"])
        img_right = check_configuration.get_metadata(cfg["input"]["right"]["img"], None)

        self.assertRaises(
            MachineError, check_configuration.check_pipeline_section, cfg, img_left, img_right, pandora_machine
        )

    @staticmethod
    def test_memory_consumption_estimation():
        """
        Test the method test_memory_consumption_estimation
        """

        # Most consuming function is to_disp
        cv_size = 450 * 375 * 61
        m_line = 8.68e-06
        n_line = 243
        # Compute memory consumption in GiB with a marge of +-10%
        consumption_vt = (
            ((cv_size * m_line + n_line) * (1 - 0.1)) / 1024,
            ((cv_size * m_line + n_line) * (1 + 0.1)) / 1024,
        )

        # Run memory_consumption_estimation giving the input parameters directly
        img_left_path = "tests/pandora/left.png"
        disp_min = -60
        disp_max = 0
        pandora_machine = PandoraMachine()
        pipeline_cfg = {"pipeline": copy.deepcopy(common.basic_pipeline_cfg)}

        min_mem_consump, max_mem_consump = check_configuration.memory_consumption_estimation(
            pipeline_cfg, (img_left_path, disp_min, disp_max), pandora_machine
        )
        np.testing.assert_allclose((min_mem_consump, max_mem_consump), consumption_vt, rtol=1e-02)

        # Run memory_consumption_estimation giving the input parameters in a dict
        pandora_machine = PandoraMachine()
        input_cfg = {"input": copy.deepcopy(common.input_cfg_basic)}
        min_mem_consump, max_mem_consump = check_configuration.memory_consumption_estimation(
            pipeline_cfg, input_cfg, pandora_machine
        )
        np.testing.assert_allclose((min_mem_consump, max_mem_consump), consumption_vt, rtol=1e-02)

    @staticmethod
    def test_memory_consumption_estimation_with_already_check_pipeline():
        """
        Test the method test_memory_consumption_estimation with an already checked input pipeline configuration.
        """

        # Most consuming function is to_disp
        cv_size = 450 * 375 * 61
        m_line = 8.68e-06
        n_line = 243
        # Compute memory consumption in GiB with a marge of +-10%
        consumption_vt = (
            ((cv_size * m_line + n_line) * (1 - 0.1)) / 1024,
            ((cv_size * m_line + n_line) * (1 + 0.1)) / 1024,
        )

        # Run memory_consumption_estimation giving the input parameters directly
        img_left_path = "tests/pandora/left.png"
        disp_min = -60
        disp_max = 0
        pandora_machine = PandoraMachine()
        cfg = {"input": copy.deepcopy(common.input_cfg_basic), "pipeline": copy.deepcopy(common.basic_pipeline_cfg)}
        img_left = check_configuration.get_metadata(cfg["input"]["left"]["img"], cfg["input"]["left"]["disp"])
        img_right = check_configuration.get_metadata(cfg["input"]["right"]["img"], None)

        # check pipeline before memory_consumption_estimation
        pipeline_cfg = check_configuration.check_pipeline_section(cfg, img_left, img_right, pandora_machine)

        min_mem_consump, max_mem_consump = check_configuration.memory_consumption_estimation(
            pipeline_cfg, (img_left_path, disp_min, disp_max), pandora_machine, True
        )
        np.testing.assert_allclose((min_mem_consump, max_mem_consump), consumption_vt, rtol=1e-02)

        # Run memory_consumption_estimation giving the input parameters in a dict
        pandora_machine = PandoraMachine()
        input_cfg = {"input": copy.deepcopy(common.input_cfg_basic)}
        min_mem_consump, max_mem_consump = check_configuration.memory_consumption_estimation(
            pipeline_cfg, input_cfg, pandora_machine, True
        )
        np.testing.assert_allclose((min_mem_consump, max_mem_consump), consumption_vt, rtol=1e-02)

    @staticmethod
    def test_memory_consumption_estimation_with_grid_disparity():
        """
        Test the method test_memory_consumption_estimation when disparity is given as a grid
        """

        # Most consuming function is to_disp
        cv_size = 450 * 375 * 75
        m_line = 8.68e-06
        n_line = 243
        # Compute memory consumption in GiB with a marge of +-10%
        consumption_vt = (
            ((cv_size * m_line + n_line) * (1 - 0.1)) / 1024,
            ((cv_size * m_line + n_line) * (1 + 0.1)) / 1024,
        )

        # Run memory_consumption_estimation giving the input parameters directly
        img_left_path = "tests/pandora/left.png"
        disp_left_path = "tests/pandora/left_disparity_grid.tif"
        pandora_machine = PandoraMachine()
        pipeline_cfg = {"pipeline": copy.deepcopy(common.basic_pipeline_cfg)}

        min_mem_consump, max_mem_consump = check_configuration.memory_consumption_estimation(
            pipeline_cfg, (img_left_path, disp_left_path), pandora_machine
        )
        np.testing.assert_allclose((min_mem_consump, max_mem_consump), consumption_vt, rtol=1e-02)

        # Run memory_consumption_estimation giving the input parameters in a dict
        pandora_machine = PandoraMachine()

        input_cfg = {"input": copy.deepcopy(common.input_cfg_left_grids)}

        min_mem_consump, max_mem_consump = check_configuration.memory_consumption_estimation(
            pipeline_cfg, input_cfg, pandora_machine
        )
        np.testing.assert_allclose((min_mem_consump, max_mem_consump), consumption_vt, rtol=1e-02)

    def test_memory_consumption_estimation_with_wrong_input(self):
        """
        Test the method test_memory_consumption_estimation when input is given in the wrong type
        """
        # Run memory_consumption_estimation giving the input parameters as a list
        img_left_path = "tests/pandora/left.png"
        disp_min = -60
        disp_max = 0
        pandora_machine = PandoraMachine()
        pipeline_cfg = {"pipeline": copy.deepcopy(common.basic_pipeline_cfg)}

        self.assertRaises(
            TypeError,
            check_configuration.memory_consumption_estimation,
            pipeline_cfg,
            [img_left_path, disp_min, disp_max],
            pandora_machine,
        )

    def test_step_in_matching_cost(self):
        """
        Test that user get a warning if he instantiates a step parameter in matching cost configuration
        """
        # Check the configuration returned with left and right disparity grids
        pandora_machine = PandoraMachine()
        cfg = {
            "input": copy.deepcopy(common.input_cfg_left_right_grids),
            "pipeline": {
                "matching_cost": {"matching_cost_method": "zncc", "window_size": 5, "subpix": 2, "step": 2},
                "disparity": {"disparity_method": "wta"},
            },
        }

        # When left disparities are grids and multiscale processing cannot be used : the program exits
        self.assertRaises(ValueError, check_configuration.check_conf, cfg, pandora_machine)


if __name__ == "__main__":
    common.setup_logging()
    unittest.main()
