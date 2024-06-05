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
# Licensed under the Apache License, Version 2.0 (the 'License');
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
This module contains functions to test the Pandora pipeline.
"""

# pylint: disable=redefined-outer-name

import json
import unittest
import copy

import numpy as np
import pytest
import xarray as xr
from rasterio import Affine

from tests import common
import pandora
from pandora import import_plugin
from pandora.img_tools import create_dataset_from_inputs, rasterio_open, add_disparity
from pandora.state_machine import PandoraMachine


def error(
    data: np.ndarray,
    ground_truth: np.ndarray,
    threshold: int,
    unknown_disparity: int = 0,
) -> float:
    """
    Ratio of bad pixels whose absolute sum with ground truth is above threshold.
    :param data: data to test.
    :type data: np.ndarray
    :param ground_truth: ground_truth
    :type ground_truth: np.ndarray
    :param threshold: threshold
    :type threshold: int
    :param unknown_disparity: unknown disparity
    :type unknown_disparity: int
    :return: ratio
    :rtype: float
    """
    mask = ground_truth != unknown_disparity
    selected_data = data[mask]
    selected_ground_truth = ground_truth[mask]
    error_mask = abs(selected_data + selected_ground_truth) > threshold
    nb_of_errors = error_mask.sum()
    return nb_of_errors / data.size


@pytest.fixture()
def disp_left():
    return rasterio_open("tests/pandora/disp_left.tif").read(1)


@pytest.fixture()
def disp_right():
    return rasterio_open("tests/pandora/disp_right.tif").read(1)


class TestMain:
    """Test Main."""

    def test_left_disparity(self, tmp_path, disp_left):
        """
        Test the main method for the left disparity computation( read and write products )

        """
        cfg = {
            "input": copy.deepcopy(common.input_cfg_left_grids),
            "pipeline": copy.deepcopy(common.basic_pipeline_cfg),
        }

        config_path = tmp_path / "config.json"
        with open(config_path, "w", encoding="utf-8") as file_:
            json.dump(cfg, file_, indent=2)

        # Run Pandora pipeline
        pandora.main(config_path, str(tmp_path), verbose=False)

        # Check the left disparity map
        assert (
            error(
                rasterio_open(str(tmp_path / "left_disparity.tif")).read(1),
                disp_left,
                1,
            )
            <= 0.20
        )

        # Check the crs & transform properties
        left_im_prop = rasterio_open("tests/pandora/left.png").profile
        left_disp_prop = rasterio_open(str(tmp_path / "left_disparity.tif")).profile
        assert left_im_prop["crs"] == left_disp_prop["crs"]
        assert left_im_prop["transform"] == left_disp_prop["transform"]

    def test_left_right_disparity(self, tmp_path, disp_left, disp_right):
        """
        Test the main method for the left and right disparity computation ( read and write products )

        """
        cfg = {
            "input": copy.deepcopy(common.input_cfg_left_right_grids),
            "pipeline": copy.deepcopy(common.validation_pipeline_cfg),
        }

        # Create temporary directory
        config_path = tmp_path / "config.json"
        with open(config_path, "w", encoding="utf-8") as file_:
            json.dump(cfg, file_, indent=2)

        # Run Pandora pipeline
        pandora.main(config_path, str(tmp_path), verbose=False)

        # Check the left disparity map
        assert error(rasterio_open(str(tmp_path / "left_disparity.tif")).read(1), disp_left, 1) <= 0.20

        # Check the right disparity map
        assert error(-1 * rasterio_open(str(tmp_path / "right_disparity.tif")).read(1), disp_right, 1) <= 0.20

        # Check the left validity mask cross checking ( bit 8 and 9 )
        out_occlusion = rasterio_open(str(tmp_path / "left_validity_mask.tif")).read(1)
        occlusion = np.ones((out_occlusion.shape[0], out_occlusion.shape[1]))
        occlusion[out_occlusion >= 512] = 0

        # Check the crs & transform properties
        left_im_prop = rasterio_open("tests/pandora/left.png").profile
        left_disp_prop = rasterio_open(str(tmp_path / "left_disparity.tif")).profile
        right_im_prop = rasterio_open("tests/pandora/right.png").profile
        right_disp_prop = rasterio_open(str(tmp_path / "right_disparity.tif")).profile
        assert left_im_prop["crs"] == left_disp_prop["crs"]
        assert left_im_prop["transform"] == left_disp_prop["transform"]
        assert right_im_prop["crs"] == right_disp_prop["crs"]
        assert right_im_prop["transform"] == right_disp_prop["transform"]
        assert left_disp_prop != right_disp_prop

    def test_variable_range_of_disp(self, tmp_path, disp_left, disp_right):
        """
        Test that variable range of disparities (grids of local disparities) are well taken into account in Pandora

        """
        cfg = {
            "input": copy.deepcopy(common.input_cfg_left_right_grids),
            "pipeline": copy.deepcopy(common.validation_pipeline_cfg),
        }

        # Create temporary directory
        config_path = tmp_path / "config.json"
        with open(config_path, "w", encoding="utf-8") as file_:
            json.dump(cfg, file_, indent=2)

        # Run Pandora pipeline
        pandora.main(config_path, str(tmp_path), verbose=False)

        # Check the left disparity map
        assert error(rasterio_open(str(tmp_path / "left_disparity.tif")).read(1), disp_left, 1) <= 0.20

        # Check the right disparity map
        assert error(-1 * rasterio_open(str(tmp_path / "right_disparity.tif")).read(1), disp_right, 1) <= 0.20

    def test_margins_are_stored_in_final_output_configuration(self, tmp_path):
        """We expect to find information about used margins in the output configuration."""
        cfg = {
            "input": copy.deepcopy(common.input_cfg_left_grids),
            "pipeline": copy.deepcopy(common.basic_pipeline_cfg),
        }

        config_path = tmp_path / "config.json"
        with open(config_path, "w", encoding="utf-8") as file_:
            json.dump(cfg, file_, indent=2)

        # Run Pandora pipeline
        pandora.main(config_path, str(tmp_path), verbose=False)

        with (tmp_path / "cfg" / "config.json").open() as file_path:
            result = json.load(file_path)

        assert result["margins"] == {
            "cumulative margins": {
                "matching_cost": {"left": 2, "up": 2, "right": 2, "down": 2},
                "disparity": {"left": 0, "up": 0, "right": 0, "down": 0},
                "refinement": {"down": 0, "left": 0, "right": 0, "up": 0},
            },
            "non-cumulative margins": {
                "filter": {"left": 3, "up": 3, "right": 3, "down": 3},
            },
            "global margins": {"left": 3, "up": 3, "right": 3, "down": 3},
        }


class TestPandora(unittest.TestCase):
    """
    TestPandora class allows to test the pandora pipeline
    """

    def setUp(self):
        """
        Method called to prepare the test fixture

        """

        input_config = {
            "left": {
                "img": "tests/pandora/left.png",
                "nodata": np.nan,
                "disp": [-60, 0],
            },
            "right": {
                "img": "tests/pandora/right.png",
                "nodata": np.nan,
                "disp": [0, 60],
            },
        }
        self.left = create_dataset_from_inputs(input_config=input_config["left"])
        self.right = create_dataset_from_inputs(input_config=input_config["right"])
        self.disp_left = rasterio_open("tests/pandora/disp_left.tif").read(1)
        self.disp_right = rasterio_open("tests/pandora/disp_right.tif").read(1)
        self.occlusion = rasterio_open("tests/pandora/occlusion.png").read(1)

    def error(self, data, gt, threshold, unknown_disparity=0):
        """
        Percentage of bad pixels whose error is > 1

        """
        n_row, n_col = self.left["im"].shape
        nb_error = 0
        for row in range(n_row):
            for col in range(n_col):
                if gt[row, col] != unknown_disparity:
                    if abs((data[row, col] + gt[row, col])) > threshold:
                        nb_error += 1

        return nb_error / float(n_row * n_col)

    def error_mask(self, data, gt):
        """
        Percentage of bad pixels ( != ground truth ) in the validity mask
        """
        n_row, n_col = self.left["im"].shape
        nb_error = 0
        for row in range(n_row):
            for col in range(n_col):
                if data[row, col] != gt[row, col]:
                    nb_error += 1

        return nb_error / float(n_row * n_col)

    def test_run_with_validation(self):
        """
        Test the run method

        """
        user_cfg = {
            "input": copy.deepcopy(common.input_cfg_basic),
            "pipeline": copy.deepcopy(common.validation_pipeline_cfg),
        }
        pandora_machine = PandoraMachine()

        # Update the user configuration with default values
        cfg = pandora.check_configuration.update_conf(pandora.check_configuration.default_short_configuration, user_cfg)
        # Run the pandora pipeline
        left, right = pandora.run(pandora_machine, self.left, self.right, cfg)

        # Check the left disparity map
        if self.error(left["disparity_map"].data, self.disp_left, 1) > 0.20:
            raise AssertionError

        # Check the left validity mask cross checking ( bit 8 and 9 )
        occlusion = np.ones((left["validity_mask"].shape[0], left["validity_mask"].shape[1]))
        occlusion[left["validity_mask"].data >= 512] = 0

        if self.error_mask(occlusion, self.occlusion) > 0.16:
            raise AssertionError

        # Check the right disparity map
        if self.error(-1 * right["disparity_map"].data, self.disp_right, 1) > 0.20:
            raise AssertionError

    def test_run_without_validation(self):
        """
        Test the run method

        """
        user_cfg = {
            "input": copy.deepcopy(common.input_cfg_basic),
            "pipeline": copy.deepcopy(common.basic_pipeline_cfg),
        }
        pandora_machine = PandoraMachine()

        # Update the user configuration with default values
        cfg = pandora.check_configuration.update_conf(pandora.check_configuration.default_short_configuration, user_cfg)

        # Run the pandora pipeline
        left, right = pandora.run(pandora_machine, self.left, self.right, cfg)  # pylint: disable = unused-variable

        # Check the left disparity map
        if self.error(left["disparity_map"].data, self.disp_left, 1) > 0.20:
            raise AssertionError

        # Check the left validity mask cross checking ( bit 8 and 9 )
        occlusion = np.ones((left["validity_mask"].shape[0], left["validity_mask"].shape[1]))
        occlusion[left["validity_mask"].data >= 512] = 0

        if self.error_mask(occlusion, self.occlusion) > 0.16:
            raise AssertionError

    def test_run_2_scales(self):
        """
        Test the run method for 2 scales

        """
        user_cfg = {
            "input": copy.deepcopy(common.input_cfg_basic),
            "pipeline": copy.deepcopy(common.multiscale_pipeline_cfg),
        }
        user_cfg["pipeline"]["validation"] = {"validation_method": "cross_checking_accurate"}

        pandora_machine = PandoraMachine()

        # Update the user configuration with default values
        cfg = pandora.check_configuration.update_conf(pandora.check_configuration.default_short_configuration, user_cfg)

        # Run the pandora pipeline
        left, right = pandora.run(pandora_machine, self.left, self.right, cfg)

        # Check the left disparity map
        if self.error(left["disparity_map"].data, self.disp_left, 1) > 0.20:
            raise AssertionError

        # Check the left validity mask cross checking ( bit 8 and 9 )
        occlusion = np.ones((left["validity_mask"].shape[0], left["validity_mask"].shape[1]))
        occlusion[left["validity_mask"].data >= 512] = 0

        if self.error_mask(occlusion, self.occlusion) > 0.16:
            raise AssertionError

        # Check the right disparity map
        if self.error(-1 * right["disparity_map"].data, self.disp_right, 1) > 0.20:
            raise AssertionError

    def test_run_3_scales(self):
        """
        Test the run method for 3 scales

        """
        user_cfg = {
            "input": copy.deepcopy(common.input_cfg_basic),
            "pipeline": copy.deepcopy(common.multiscale_pipeline_cfg),
        }
        user_cfg["pipeline"]["multiscale"]["num_scales"] = 3
        user_cfg["pipeline"]["validation"] = {"validation_method": "cross_checking_accurate"}

        pandora_machine = PandoraMachine()

        # Update the user configuration with default values
        cfg = pandora.check_configuration.update_conf(pandora.check_configuration.default_short_configuration, user_cfg)

        # Run the pandora pipeline
        left, right = pandora.run(pandora_machine, self.left, self.right, cfg)
        # Check the left disparity map
        if self.error(left["disparity_map"].data, self.disp_left, 1) > 0.20:
            raise AssertionError

        # Check the left validity mask cross checking ( bit 8 and 9 )
        occlusion = np.ones((left["validity_mask"].shape[0], left["validity_mask"].shape[1]))
        occlusion[left["validity_mask"].data >= 512] = 0

        if self.error_mask(occlusion, self.occlusion) > 0.16:
            raise AssertionError

        # Check the right disparity map
        if self.error(-1 * right["disparity_map"].data, self.disp_right, 1) > 0.20:
            raise AssertionError

    def test_multiechelles_and_confidence(self):
        """
        Test the run method for 2 scales

        """
        user_cfg = {
            "input": copy.deepcopy(common.input_cfg_basic),
            "pipeline": {
                "matching_cost": {"matching_cost_method": "zncc", "window_size": 5, "subpix": 2},
                "cost_volume_confidence": {"confidence_method": "ambiguity"},
                "disparity": {"disparity_method": "wta", "invalid_disparity": -9999},
                "refinement": {"refinement_method": "vfit"},
                "filter": {"filter_method": "median", "filter_size": 3},
                "multiscale": {
                    "multiscale_method": "fixed_zoom_pyramid",
                    "num_scales": 2,
                    "scale_factor": 2,
                    "marge": 1,
                },
                "validation": {"validation_method": "cross_checking_accurate"},
            },
        }

        pandora_machine = PandoraMachine()

        # Update the user configuration with default values
        cfg = pandora.check_configuration.update_conf(pandora.check_configuration.default_short_configuration, user_cfg)

        # Run the pandora pipeline
        left, right = pandora.run(pandora_machine, self.left, self.right, cfg)

        # Check the left disparity map
        if self.error(left["disparity_map"].data, self.disp_left, 1) > 0.20:
            raise AssertionError

        # Check the left validity mask cross checking ( bit 8 and 9 )
        occlusion = np.ones((left["validity_mask"].shape[0], left["validity_mask"].shape[1]))
        occlusion[left["validity_mask"].data >= 512] = 0

        if self.error_mask(occlusion, self.occlusion) > 0.16:
            raise AssertionError

        # Check the right disparity map
        if self.error(-1 * right["disparity_map"].data, self.disp_right, 1) > 0.20:
            raise AssertionError

    @staticmethod
    def test_confidence_measure():
        """
        Test Pandora run method on confidence_measure map
        """

        # Create left and right images
        data_left = np.array(
            [
                [2, 5, 3, 1, 6, 1, 3, 3],
                [5, 3, 2, 1, 4, 3, 3, 2],
                [4, 2, 3, 2, 2, 3, 4, 6],
                [4, 5, 3, 2, 0, 1, 0, 1],
                [1, 3, 2, 1, 0, 2, 1, 3],
                [5, 2, 1, 0, 1, 2, 3, 5],
                [3, 3, 2, 3, 0, 4, 1, 2],
            ],
            dtype=np.float32,
        )
        img_left = xr.Dataset(
            {"im": (["row", "col"], data_left)},
            coords={"row": np.arange(data_left.shape[0]), "col": np.arange(data_left.shape[1])},
        )
        img_left.attrs["crs"] = None
        img_left.attrs["transform"] = Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        img_left.pipe(add_disparity, disparity=[-2, 2], window=None)

        data_right = np.array(
            [
                [1, 2, 1, 2, 5, 3, 1, 6],
                [2, 3, 5, 3, 2, 1, 4, 3],
                [0, 2, 4, 2, 3, 2, 2, 3],
                [5, 3, 1, 4, 5, 3, 2, 0],
                [2, 1, 3, 2, 1, 0, 2, 1],
                [5, 5, 5, 2, 1, 0, 1, 2],
                [1, 2, 2, 3, 3, 2, 3, 0],
            ],
            dtype=np.float32,
        )
        img_right = xr.Dataset(
            {"im": (["row", "col"], data_right)},
            coords={"row": np.arange(data_right.shape[0]), "col": np.arange(data_right.shape[1])},
        )
        img_right.attrs["crs"] = None
        img_right.attrs["transform"] = Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        img_right.pipe(add_disparity, disparity=[-2, 2], window=None)

        # Load a configuration
        user_cfg = {
            "input": copy.deepcopy(common.input_cfg_basic_with_none_right_disp),
            "pipeline": copy.deepcopy(common.validation_pipeline_cfg),
        }
        user_cfg["input"]["left"]["disp"] = (-2, 2)
        user_cfg["pipeline"]["matching_cost"]["matching_cost_method"] = "census"
        user_cfg["pipeline"]["matching_cost"]["subpix"] = 1
        user_cfg["pipeline"]["disparity"]["invalid_disparity"] = -10
        del user_cfg["pipeline"]["refinement"]
        del user_cfg["pipeline"]["filter"]

        pandora_machine = PandoraMachine()

        cfg = pandora.check_configuration.update_conf(pandora.check_configuration.default_short_configuration, user_cfg)
        import_plugin()

        # Run the Pandora pipeline
        left, right = pandora.run(pandora_machine, img_left, img_right, cfg)

        # Ground truth confidence measure
        gt_left_indicator_stereo = np.array(
            [
                [1.57175062, 1.46969385, 1.39484766, 1.6],
                [1.51578363, 1.2, 1.1892855, 1.54712637],
                [1.43331783, 1.24835892, 1.21720992, 1.58694675],
            ],
            dtype=np.float32,
        )

        gt_left_indicator_validation = np.array([[0, 0, 2, 3], [0, 0, 0, 2], [0, 0, 0, 1]], dtype=np.float32)

        gt_left_confidence_measure = np.full((7, 8, 2), np.nan, dtype=np.float32)
        gt_left_confidence_measure[2:-2, 2:-2, 0] = gt_left_indicator_stereo
        gt_left_confidence_measure[2:-2, 2:-2, 1] = gt_left_indicator_validation

        gt_right_indicator_stereo = np.array(
            [
                [1.4164745, 1.33026313, 1.36, 1.47295621],
                [1.5147277, 1.49986666, 1.44222051, 1.24835892],
                [1.48916084, 1.38794813, 1.28747816, 1.24835892],
            ],
            dtype=np.float32,
        )

        gt_right_indicator_validation = np.array([[2, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]], dtype=np.float32)

        gt_right_confidence_measure = np.full((7, 8, 2), np.nan, dtype=np.float32)
        gt_right_confidence_measure[2:-2, 2:-2, 0] = gt_right_indicator_stereo
        gt_right_confidence_measure[2:-2, 2:-2, 1] = gt_right_indicator_validation

        # assert equal on left confidence_measure
        np.testing.assert_array_equal(gt_left_confidence_measure, left["confidence_measure"].data)

        # assert equal on right confidence_measure
        np.testing.assert_array_equal(gt_right_confidence_measure, right["confidence_measure"].data)

    @staticmethod
    def test_dataset_image():
        """
        Test pandora with variable coordinate in dataset image

        """

        user_cfg = {
            "input": copy.deepcopy(common.input_cfg_basic_with_none_right_disp),
            "pipeline": copy.deepcopy(common.validation_pipeline_cfg),
        }
        user_cfg["input"]["left"]["nodata"] = np.nan
        user_cfg["input"]["right"]["nodata"] = np.nan
        user_cfg["pipeline"]["matching_cost"]["matching_cost_method"] = "census"

        pandora_machine = PandoraMachine()

        # Update the user configuration with default values
        cfg = pandora.check_configuration.update_conf(pandora.check_configuration.default_short_configuration, user_cfg)

        left_img = create_dataset_from_inputs(input_config=user_cfg["input"]["left"])
        right_img = create_dataset_from_inputs(input_config=user_cfg["input"]["right"])

        # Run the pandora pipeline on images without modified coordinates
        left_origin, right_origin = pandora.run(pandora_machine, left_img, right_img, cfg)

        row_c = left_img.coords["row"].data
        row_c += 41
        col_c = left_img.coords["col"].data
        col_c += 45
        # Changes the coordinate images
        left_img.assign_coords(row=row_c, col=col_c)
        right_img.assign_coords(row=row_c, col=col_c)

        # Run the pandora pipeline on images with modified coordinates
        left_modified, right_modified = pandora.run(pandora_machine, left_img, right_img, cfg)

        # check if the disparity maps are equals
        np.testing.assert_array_equal(left_origin["disparity_map"].values, left_modified["disparity_map"].values)
        np.testing.assert_array_equal(right_origin["disparity_map"].values, right_modified["disparity_map"].values)

    def test_main_with_rgb_image(self):
        """
        Test the basic pipeline with an rgb data input
        """

        user_cfg = {
            "input": copy.deepcopy(common.input_multiband_cfg),
            "pipeline": copy.deepcopy(common.basic_pipeline_cfg),
        }
        user_cfg["input"]["left"]["nodata"] = np.nan
        user_cfg["input"]["right"]["nodata"] = np.nan
        user_cfg["input"]["right"]["disp"] = None
        # working on green band
        user_cfg["pipeline"]["matching_cost"]["band"] = "g"

        left_rgb = create_dataset_from_inputs(input_config=user_cfg["input"]["left"])
        right_rgb = create_dataset_from_inputs(input_config=user_cfg["input"]["right"])

        pandora_machine = PandoraMachine()
        # Update the user configuration with default values
        cfg = pandora.check_configuration.update_conf(pandora.check_configuration.default_short_configuration, user_cfg)

        # Run the pandora pipeline
        left, _ = pandora.run(pandora_machine, left_rgb, right_rgb, cfg)

        # Check the left disparity map
        if self.error(left["disparity_map"].data, self.disp_left, 1) > 0.25:
            raise AssertionError

    def test_main_with_rgb_image_and_mask(self):
        """
        Test the basic pipeline with an rgb data input and masks
        """

        user_cfg = {
            "input": copy.deepcopy(common.input_multiband_cfg),
            "pipeline": copy.deepcopy(common.basic_pipeline_cfg),
        }
        # add masks
        user_cfg["input"]["left"]["mask"] = "tests/pandora/occlusion.png"

        # working on green band
        user_cfg["pipeline"]["matching_cost"]["band"] = "g"

        input_config = user_cfg["input"]
        input_config["left"]["mask"] = None
        input_config["right"]["mask"] = None
        input_config["left"]["nodata"] = np.nan
        input_config["right"]["nodata"] = np.nan
        input_config["right"]["disp"] = None
        left_rgb = create_dataset_from_inputs(input_config=input_config["left"])
        right_rgb = create_dataset_from_inputs(input_config=input_config["right"])

        pandora_machine = PandoraMachine()
        # Update the user configuration with default values
        cfg = pandora.check_configuration.update_conf(pandora.check_configuration.default_short_configuration, user_cfg)

        # Run the pandora pipeline
        left, _ = pandora.run(pandora_machine, left_rgb, right_rgb, cfg)

        # Check the left disparity map
        if self.error(left["disparity_map"].data, self.disp_left, 1) > 0.25:
            raise AssertionError

    def test_multiscale_with_rgb_image_and_mask(self):
        """
        Test the multiscale pipeline with an rgb data input and masks
        """

        user_cfg = {
            "input": copy.deepcopy(common.input_multiband_cfg),
            "pipeline": copy.deepcopy(common.multiscale_pipeline_cfg),
        }
        # add masks
        user_cfg["input"]["left"]["mask"] = "tests/pandora/occlusion.png"

        # working on green band
        user_cfg["pipeline"]["matching_cost"]["band"] = "g"

        input_config = user_cfg["input"]
        input_config["left"]["mask"] = None
        input_config["right"]["mask"] = None
        input_config["left"]["nodata"] = np.nan
        input_config["right"]["nodata"] = np.nan
        input_config["right"]["disp"] = None
        left_rgb = create_dataset_from_inputs(input_config=input_config["left"])
        right_rgb = create_dataset_from_inputs(input_config=input_config["right"])

        pandora_machine = PandoraMachine()
        # Update the user configuration with default values
        cfg = pandora.check_configuration.update_conf(pandora.check_configuration.default_short_configuration, user_cfg)

        # Run the pandora pipeline
        left, _ = pandora.run(pandora_machine, left_rgb, right_rgb, cfg)

        # Check the left disparity map
        if self.error(left["disparity_map"].data, self.disp_left, 1) > 0.25:
            raise AssertionError

    def test_validation_with_rgb_image_and_mask(self):
        """
        Test the validation pipeline with an rgb data input and masks
        """

        user_cfg = {
            "input": copy.deepcopy(common.input_multiband_cfg),
            "pipeline": copy.deepcopy(common.validation_pipeline_cfg),
        }
        # add masks
        user_cfg["input"]["left"]["mask"] = "tests/pandora/occlusion.png"

        # working on green band
        user_cfg["pipeline"]["matching_cost"]["band"] = "g"
        user_cfg["input"]["left"]["nodata"] = np.nan
        user_cfg["input"]["right"]["nodata"] = np.nan

        # Update the user configuration with default values
        cfg = pandora.check_configuration.update_conf(pandora.check_configuration.default_short_configuration, user_cfg)

        new_input_config = copy.deepcopy(cfg["input"])
        new_input_config["left"]["mask"] = None
        left_rgb = create_dataset_from_inputs(input_config=new_input_config["left"])
        right_rgb = create_dataset_from_inputs(input_config=new_input_config["right"])

        pandora_machine = PandoraMachine()

        # Run the pandora pipeline
        left, _ = pandora.run(pandora_machine, left_rgb, right_rgb, cfg)

        # Check the left disparity map
        if self.error(left["disparity_map"].data, self.disp_left, 1) > 0.25:
            raise AssertionError


if __name__ == "__main__":
    common.setup_logging()
    unittest.main()
