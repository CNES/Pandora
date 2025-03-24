# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of PANDORA
#
#     https://github.com/CNES/Pandora
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
Module with functional tests.
"""

# Make pylint happy with tests:
# pylint: disable=redefined-outer-name,too-few-public-methods,too-many-positional-arguments

import json
import sys

import numpy as np
import pytest

import pandora
from pandora.img_tools import rasterio_open

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points


plugin_set = entry_points(group="pandora.plugin").names


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
def left_png_path(root_dir):
    return root_dir / "tests/pandora/left.png"


@pytest.fixture()
def right_png_path(root_dir):
    return root_dir / "tests/pandora/right.png"


@pytest.fixture()
def left_disparity_path(root_dir):
    return root_dir / "tests/pandora/disp_left.tif"


@pytest.fixture()
def left_disparity(left_disparity_path):
    return rasterio_open(str(left_disparity_path)).read(1)


@pytest.fixture()
def right_disparity_path(root_dir):
    return root_dir / "tests/pandora/disp_right.tif"


@pytest.fixture()
def right_disparity(right_disparity_path):
    return rasterio_open(str(right_disparity_path)).read(1)


@pytest.fixture()
def left_disparity_grid_path(root_dir):
    return root_dir / "tests/pandora/left_disparity_grid.tif"


@pytest.fixture()
def input_cfg_left_grids(left_png_path, right_png_path, left_disparity_grid_path):
    return {
        "left": {
            "img": str(left_png_path),
            "disp": str(left_disparity_grid_path),
        },
        "right": {
            "img": str(right_png_path),
        },
    }


@pytest.fixture()
def basic_pipeline_cfg():
    return {
        "matching_cost": {"matching_cost_method": "census", "window_size": 5, "subpix": 1},
        "optimization": {
            "optimization_method": "sgm",
            "overcounting": False,
            "penalty": {"P1": 8, "P2": 32, "p2_method": "constant", "penalty_method": "sgm_penalty"},
        },
        "disparity": {"disparity_method": "wta", "invalid_disparity": "NaN"},
        "refinement": {"refinement_method": "vfit"},
        "filter": {"filter_method": "median", "filter_size": 3},
    }


@pytest.mark.functional_tests
@pytest.mark.skipif(not any("sgm" in name for name in plugin_set), reason="SGM plugin not installed")
class TestMain:
    """Test Main."""

    def test_left_disparity(self, tmp_path, left_png_path, left_disparity, input_cfg_left_grids, basic_pipeline_cfg):
        """
        Test the main method for the left disparity computation( read and write products )

        """
        cfg = {
            "input": input_cfg_left_grids,
            "pipeline": basic_pipeline_cfg,
        }

        config_path = tmp_path / "config.json"
        with open(config_path, "w", encoding="utf-8") as file_:
            json.dump(cfg, file_, indent=2)

        # Run Pandora pipeline
        pandora.main(config_path, str(tmp_path), verbose=False)

        result = rasterio_open(str(tmp_path / "left_disparity.tif")).read(1)

        # Check the left disparity map
        assert error(result, left_disparity, 1) <= 0.20

        # Check the crs & transform properties
        left_im_prop = rasterio_open(str(left_png_path)).profile
        left_disp_prop = rasterio_open(str(tmp_path / "left_disparity.tif")).profile
        assert left_im_prop["crs"] == left_disp_prop["crs"]
        assert left_im_prop["transform"] == left_disp_prop["transform"]
