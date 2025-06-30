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
import os

import numpy as np
import pytest

import pandora

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points


plugin_set = entry_points(group="pandora.plugin").names


@pytest.fixture()
def left_png_path(root_dir):
    return root_dir / "tests/pandora/left.png"


@pytest.fixture()
def right_png_path(root_dir):
    return root_dir / "tests/pandora/right.png"


@pytest.fixture()
def user_cfg(left_png_path, right_png_path, profiling_conf):
    return {
        "profiling": profiling_conf,
        "input": {
            # incoherent disp, we just want a fast run
            "left": {"img": str(left_png_path), "disp": [-2, 2]},
            "right": {
                "img": str(right_png_path),
            },
        },
        "pipeline": {
            "matching_cost": {
                "matching_cost_method": "census",
            },
            "disparity": {"disparity_method": "wta", "invalid_disparity": "NaN"},
        },
    }


@pytest.mark.functional_tests
class TestMain:
    """Test Main."""

    @pytest.mark.parametrize(
        "profiling_conf",
        [
            {"save_raw_data": True, "save_graphs": True},
            {"save_raw_data": True, "save_graphs": False},
            {"save_raw_data": False, "save_graphs": True},
            {"save_raw_data": False, "save_graphs": False},
        ],
    )
    def test_profiling_saves_data(self, tmp_path, user_cfg):
        """
        Test the validation fast method by comparing its results to the accurate method.
        Runs the pipeline twice (with fast & accurate) then compare the results
        and assert they are striclty equal.
        """
        s_rd = user_cfg["profiling"]["save_raw_data"]
        s_gr = user_cfg["profiling"]["save_graphs"]

        config_prof = tmp_path / "config_prof.json"
        out_prof = tmp_path / "out_profiling"

        with open(config_prof, "w", encoding="utf-8") as file_:
            json.dump(user_cfg, file_, indent=2)

        # run both configs
        pandora.main(config_prof, out_prof, verbose=False)

        if s_rd or s_gr:
            assert os.path.isdir(os.path.join(out_prof, "profiling"))

            if s_rd:
                assert os.path.isfile(os.path.join(out_prof, "profiling", "raw_data.pickle"))
            else:
                assert not os.path.isfile(os.path.join(out_prof, "profiling", "raw_data.pickle"))

            if s_gr:
                assert os.path.isfile(os.path.join(out_prof, "profiling", "time_graph.html"))
                assert os.path.isfile(os.path.join(out_prof, "profiling", "memory_main.run.html"))
            else:
                assert not os.path.isfile(os.path.join(out_prof, "profiling", "time_graph.html"))
                assert not os.path.isfile(os.path.join(out_prof, "profiling", "memory_main.run.html"))

        else:
            assert not os.path.isdir(os.path.join(out_prof, "profiling"))
