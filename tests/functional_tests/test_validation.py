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
from pandora import validation
from pandora.validation.validation import CrossCheckingAccurate
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
    Ratio of pixels whose value differ by more than threshold.

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
    error_mask = abs(selected_data - selected_ground_truth) > threshold
    nb_of_errors = error_mask.sum()
    return nb_of_errors / data.size


@pytest.fixture()
def left_png_path(root_dir):
    return root_dir / "tests/pandora/left.png"


@pytest.fixture()
def right_png_path(root_dir):
    return root_dir / "tests/pandora/right.png"


@pytest.fixture
def left_classif_path(request, root_dir):
    """
    Fixture returning the path to the classification file if asked to, or None
    """
    if request.param == "classif_file":
        return root_dir / "tests/pandora/mask_from_occlusion_left.tif"
    return None


@pytest.fixture()
def user_cfg(left_png_path, right_png_path, left_disparity, left_classif_path, matching_cost_method):
    return {
        "input": {
            "left": {
                "img": str(left_png_path),
                "disp": left_disparity,
                "classif": str(left_classif_path) if left_classif_path else left_classif_path,
            },
            "right": {
                "img": str(right_png_path),
            },
        },
        "pipeline": {
            "matching_cost": {
                "matching_cost_method": matching_cost_method,
            },
            "disparity": {"disparity_method": "wta", "invalid_disparity": "NaN"},
            "validation": {"validation_method": "TO_SET_IN_TEST"},
        },
    }


@pytest.mark.functional_tests
class TestMain:
    """Test Main."""

    @pytest.mark.parametrize("left_disparity", [[-60, 0]])
    @pytest.mark.parametrize("left_classif_path", [None, "classif_file"], indirect=["left_classif_path"])
    @pytest.mark.parametrize("matching_cost_method", ["ssd", "sad", "zncc", "census"])
    def test_validation_fast(self, tmp_path, user_cfg):
        """
        Test the validation fast method by comparing its results to the accurate method.
        Runs the pipeline twice (with fast & accurate) then compare the results
        and assert they are striclty equal.
        """

        if user_cfg["pipeline"]["matching_cost"]["matching_cost_method"] == "mc_cnn":
            if not any("mc_cnn" in x for x in plugin_set):
                pytest.skip(reason="MCCNN plugin not installed")

        config_fast_path = tmp_path / "config_fast.json"
        out_fast_path = tmp_path / "out_fast"
        user_cfg_fast = user_cfg
        user_cfg_fast["pipeline"]["validation"]["validation_method"] = "cross_checking_fast"

        with open(config_fast_path, "w", encoding="utf-8") as file_:
            json.dump(user_cfg_fast, file_, indent=2)

        config_accurate_path = tmp_path / "config_accurate.json"
        out_accurate_path = tmp_path / "out_accurate"
        user_cfg_accurate = user_cfg
        user_cfg_accurate["pipeline"]["validation"]["validation_method"] = "cross_checking_accurate"

        with open(config_accurate_path, "w", encoding="utf-8") as file_:
            json.dump(user_cfg_accurate, file_, indent=2)

        # run both configs
        pandora.main(config_fast_path, out_fast_path, verbose=False)
        pandora.main(config_accurate_path, out_accurate_path, verbose=False)

        result_fast = rasterio_open(str(out_fast_path / "left_disparity.tif")).read(1)
        result_accurate = rasterio_open(str(out_accurate_path / "left_disparity.tif")).read(1)

        # Check they are *strictly* equal
        assert error(result_fast, result_accurate, threshold=0) == 0
