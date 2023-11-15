# Copyright (c) 2020 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of PANDORA
#
#     https://github.com/CNES/Pandora_pandora
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

"""This module contains tests of the pandora machine."""

# pylint with pytest's fixtures compatibility:
# pylint: disable=redefined-outer-name

import copy

import numpy as np
import pytest
import xarray as xr

from pandora.descriptors.margins import Margins
from pandora import PandoraMachine


@pytest.fixture()
def monoband_image():
    """Return a 2D dataset builder of a given shape."""

    def inner(shape):
        return xr.Dataset(
            {},
            coords={"band_im": [None], "row": np.arange(shape[0]), "col": np.arange(shape[1])},
            attrs={"disparity_source": None},
        )

    return inner


@pytest.fixture()
def multiband_image():
    """Return a 3D dataset builder of a given shape."""

    def inner(shape):
        return xr.Dataset(
            {},
            coords={"band_im": ["R", "G", "B"], "row": np.arange(shape[0]), "col": np.arange(shape[1])},
            attrs={"disparity_source": None},
        )

    return inner


@pytest.fixture()
def pandora_machine_builder():
    """Return a pandora machine builder which expects an image shape."""

    def builder(image_shape, image_builder):
        machine = PandoraMachine()
        machine.left_img = image_builder(image_shape)
        machine.right_img = image_builder(image_shape)
        return machine

    return builder


class TestFilterCheckConf:
    """Test filter_check_conf method."""

    @pytest.fixture()
    def user_pipeline(self):
        """Pipeline given by user."""
        user_cfg = {
            "filter": {"filter_method": "bilateral", "sigma_color": 4.0, "sigma_space": 6.0},
        }
        return user_cfg

    @pytest.mark.parametrize(
        "image_shape",
        [
            (3, 4),
            (10, 7),
        ],
    )
    @pytest.mark.parametrize(
        "image_builder_name",
        [
            "monoband_image",
            "multiband_image",
        ],
    )
    def test_filter_check_conf(self, request, image_builder_name, image_shape, pandora_machine_builder, user_pipeline):
        """Test image_shape is added to bilateralfilter config."""
        expected = copy.deepcopy(user_pipeline["filter"])
        expected["image_shape"] = image_shape
        pandora_machine = pandora_machine_builder(image_shape, request.getfixturevalue(image_builder_name))

        pandora_machine.filter_check_conf(user_pipeline, "filter")

        assert pandora_machine.pipeline_cfg["pipeline"]["filter"] == expected


class TestGlobalMargins:
    """Test global margins of the pandora machine."""

    @pytest.fixture(autouse=True)
    def patch_null_margins(self, mocker):
        """In order to know if null margins are taken into account, we need to make them non-null."""
        mocker.patch("pandora.state_machine.disparity.AbstractDisparity.margins", Margins(1, 1, 1, 1))
        mocker.patch("pandora.state_machine.refinement.AbstractRefinement.margins", Margins(1, 1, 1, 1))
        mocker.patch("pandora.state_machine.aggregation.AbstractAggregation.margins", Margins(1, 1, 1, 1))

    @pytest.mark.parametrize(
        ["image_shape", "configuration", "expected"],
        [
            pytest.param((10, 10), {"pipeline": {}}, Margins(0, 0, 0, 0), id="Empty pipeline gives null margins"),
            pytest.param(
                (10, 10),
                {
                    "pipeline": {
                        "matching_cost": {"matching_cost_method": "zncc", "window_size": 11},
                    }
                },
                Margins(5, 5, 5, 5),
                id="Only matching_cost",
            ),
            pytest.param(
                (10, 10),
                {
                    "pipeline": {
                        "matching_cost": {
                            "matching_cost_method": "zncc",
                            "window_size": 11,
                            "subpix": 2,
                            "band": None,
                            "step": 1,
                        },  # Margins(5, 5, 5, 5)
                        "disparity": {
                            "disparity_method": "wta",
                            "invalid_disparity": -9999,
                        },  # Margins(1, 1, 1, 1) because of patch_margins fixture
                        "refinement": {
                            "refinement_method": "vfit"
                        },  # Margins(1, 1, 1, 1) because of patch_margins fixture
                    },
                },
                Margins(7, 7, 7, 7),
                id="Only cumulative margins 1",
            ),
            pytest.param(
                (10, 10),
                {
                    "pipeline": {
                        "matching_cost": {
                            "matching_cost_method": "zncc",
                            "window_size": 11,
                            "subpix": 2,
                            "band": None,
                            "step": 1,
                        },  # Margins(5, 5, 5, 5)
                        "aggregation": {
                            "aggregation_method": "cbca",
                            "cbca_intensity": 5.0,
                            "cbca_distance": 3,
                        },  # Margins(1, 1, 1, 1) because of patch_margins fixture
                    },
                },
                Margins(6, 6, 6, 6),
                id="Only cumulative margins 2",
            ),
            pytest.param(
                (20, 20),
                {
                    "pipeline": {
                        "matching_cost": {
                            "matching_cost_method": "zncc",
                            "window_size": 15,
                            "subpix": 2,
                            "band": None,
                            "step": 1,
                        },  # Margins(7, 7, 7, 7)
                        "disparity": {
                            "disparity_method": "wta",
                            "invalid_disparity": -9999,
                        },  # Margins(1, 1, 1, 1) because of patch_margins fixture
                        "filter": {
                            "filter_method": "bilateral",
                            "sigma_color": 4.0,
                            "sigma_space": 2.0,
                        },  # Margins(7, 7, 7, 7)
                    },
                },
                Margins(8, 8, 8, 8),
                id="With non-cumulative margins and filter lose",
            ),
            pytest.param(
                (10, 10),
                {
                    "pipeline": {
                        "matching_cost": {
                            "matching_cost_method": "zncc",
                            "window_size": 11,
                            "subpix": 2,
                            "band": None,
                            "step": 1,
                        },  # Margins(5, 5, 5, 5)
                        "disparity": {
                            "disparity_method": "wta",
                            "invalid_disparity": -9999,
                        },  # Margins(1, 1, 1, 1) because of patch_margins fixture
                        "filter": {
                            "filter_method": "bilateral",
                            "sigma_color": 4.0,
                            "sigma_space": 2.0,
                        },  # Margins(7, 7, 7, 7)
                    },
                },
                Margins(7, 7, 7, 7),
                id="With non-cumulative margins and filter wins",
            ),
            pytest.param(
                (10, 10),
                {
                    "pipeline": {
                        "matching_cost": {
                            "matching_cost_method": "zncc",
                            "window_size": 11,
                            "subpix": 2,
                            "band": None,
                            "step": 1,
                        },  # Margins(5, 5, 5, 5)
                        "disparity": {
                            "disparity_method": "wta",
                            "invalid_disparity": -9999,
                        },  # Margins(1, 1, 1, 1) because of patch_margins fixture
                        "filter": {
                            "filter_method": "bilateral",
                            "sigma_color": 4.0,
                            "sigma_space": 2.0,
                        },  # Margins(7, 7, 7, 7)
                        "filter.1": {
                            "filter_method": "median",
                            "filter_size": 11,
                        },  # Margins(11, 11, 11, 11)
                    },
                },
                Margins(11, 11, 11, 11),
                id="With non-cumulative margins, multiple filters and filter.1 wins",
            ),
        ],
    )
    def test_each_steps_margins_are_taken_into_account(
        self, pandora_machine_builder, image_shape, monoband_image, configuration, expected
    ):
        """
        Given a pipeline with steps, each step with margins should contribute to global margins.
        """
        pandora_machine = pandora_machine_builder(image_shape, monoband_image)
        pandora_machine.check_conf(configuration, pandora_machine.left_img, pandora_machine.right_img)

        assert pandora_machine.margins == expected
