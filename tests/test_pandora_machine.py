# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
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
# pylint: disable=unused-argument

import sys

import numpy as np
import pytest
import xarray as xr
from pytest_mock import MockerFixture
from pandora import check_configuration
from pandora.margins import Margins
from pandora import PandoraMachine


@pytest.fixture()
def bypass_matching_cost_step_verification(mocker: MockerFixture):
    fake_modules = {**sys.modules.copy(), "pandora2d": mocker.ANY}
    mocker.patch("pandora.matching_cost.matching_cost.sys.modules", fake_modules)


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

        assert pandora_machine.margins.global_margins == expected


class TestGlobalMarginsWithStep:
    """Test global margins of the pandora machine with step > 1."""

    @pytest.fixture(autouse=True)
    def patch_null_margins(self, mocker):
        """In order to know if null margins are taken into account, we need to make them non-null."""
        mocker.patch("pandora.state_machine.disparity.AbstractDisparity.margins", Margins(1, 1, 1, 1))
        mocker.patch("pandora.state_machine.refinement.AbstractRefinement.margins", Margins(1, 1, 1, 1))
        mocker.patch("pandora.state_machine.aggregation.AbstractAggregation.margins", Margins(1, 1, 1, 1))

    @pytest.mark.parametrize(
        ["image_shape", "configuration", "expected"],
        [
            pytest.param(
                (10, 10),
                {
                    "pipeline": {
                        "matching_cost": {
                            "matching_cost_method": "zncc",
                            "window_size": 11,
                            "step": 2,
                        },
                    }
                },
                Margins(5, 5, 5, 5),
                id="Only matching_cost: not affected by step",
            ),
            pytest.param(
                (20, 20),
                {
                    "pipeline": {
                        "matching_cost": {
                            "matching_cost_method": "zncc",
                            "window_size": 31,
                            "subpix": 2,
                            "band": None,
                            "step": 2,
                        },  # Margins(15, 15, 15, 15)
                        "disparity": {
                            "disparity_method": "wta",
                            "invalid_disparity": -9999,
                        },  # Margins(1, 1, 1, 1) because of patch_margins fixture
                        "filter": {
                            "filter_method": "bilateral",
                            "sigma_color": 4.0,
                            "sigma_space": 1.0,
                        },  # Margins(14, 14, 14, 14)
                    },
                },
                Margins(16, 16, 16, 16),
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
                            "step": 2,
                        },  # Margins(5, 5, 5, 5)
                        "disparity": {
                            "disparity_method": "wta",
                            "invalid_disparity": -9999,
                        },  # Margins(1, 1, 1, 1) because of patch_margins fixture
                        "filter": {
                            "filter_method": "bilateral",
                            "sigma_color": 4.0,
                            "sigma_space": 2.0,
                        },  # Margins(14, 14, 14, 14)
                    },
                },
                Margins(14, 14, 14, 14),
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
                            "step": 2,
                        },  # Margins(5, 5, 5, 5)
                        "disparity": {
                            "disparity_method": "wta",
                            "invalid_disparity": -9999,
                        },  # Margins(1, 1, 1, 1) because of patch_margins fixture
                        "filter": {
                            "filter_method": "bilateral",
                            "sigma_color": 4.0,
                            "sigma_space": 2.0,
                        },  # Margins(14, 14, 14, 14)
                        "filter.1": {
                            "filter_method": "median",
                            "filter_size": 11,
                        },  # Margins(22, 22, 22, 22)
                    },
                },
                Margins(22, 22, 22, 22),
                id="With non-cumulative margins, multiple filters and filter.1 wins",
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
                            "step": 2,
                        },  # Margins(5, 5, 5, 5)
                        "disparity": {
                            "disparity_method": "wta",
                            "invalid_disparity": -9999,
                        },  # Margins(1, 1, 1, 1) because of patch_margins fixture
                        "filter": {
                            "filter_method": "bilateral",
                            "sigma_color": 4.0,
                            "sigma_space": 2.0,
                        },  # Margins(14, 14, 14, 14)
                        "filter.1": {
                            "filter_method": "median_for_intervals",
                            "filter_size": 11,
                        },  # Margins(22, 22, 22, 22)
                    },
                },
                Margins(22, 22, 22, 22),
                id="With non-cumulative margins, multiple filters and filter.1 (median_for_intervals) wins",
            ),
        ],
    )
    def test_with_step_of_2(
        self,
        bypass_matching_cost_step_verification,
        pandora_machine_builder,
        image_shape,
        monoband_image,
        configuration,
        expected,
    ):
        """
        Given a pipeline with steps, each step with margins should contribute to global margins.
        """
        pandora_machine = pandora_machine_builder(image_shape, monoband_image)
        pandora_machine.check_conf(configuration, pandora_machine.left_img, pandora_machine.right_img)

        assert pandora_machine.margins.global_margins == expected

    @pytest.mark.parametrize(
        ["image_shape", "configuration", "expected"],
        [
            pytest.param(
                (10, 10),
                {
                    "pipeline": {
                        "matching_cost": {
                            "matching_cost_method": "zncc",
                            "window_size": 11,
                            "step": 5,
                        },
                    }
                },
                Margins(5, 5, 5, 5),
                id="Only matching_cost: not affected by step",
            ),
            pytest.param(
                (51, 51),
                {
                    "pipeline": {
                        "matching_cost": {
                            "matching_cost_method": "zncc",
                            "window_size": 51,
                            "subpix": 2,
                            "band": None,
                            "step": 5,
                        },  # Margins(25, 25, 25, 25)
                        "disparity": {
                            "disparity_method": "wta",
                            "invalid_disparity": -9999,
                        },  # Margins(1, 1, 1, 1) because of patch_margins fixture
                        "filter": {
                            "filter_method": "bilateral",
                            "sigma_color": 4.0,
                            "sigma_space": 1.0,
                        },  # Margins(20, 20, 20, 20)
                    },
                },
                Margins(26, 26, 26, 26),
                id="With non-cumulative margins and filter lose",
            ),
            pytest.param(
                (51, 51),
                {
                    "pipeline": {
                        "matching_cost": {
                            "matching_cost_method": "zncc",
                            "window_size": 11,
                            "subpix": 2,
                            "band": None,
                            "step": 5,
                        },  # Margins(5, 5, 5, 5)
                        "disparity": {
                            "disparity_method": "wta",
                            "invalid_disparity": -9999,
                        },  # Margins(1, 1, 1, 1) because of patch_margins fixture
                        "filter": {
                            "filter_method": "bilateral",
                            "sigma_color": 4.0,
                            "sigma_space": 2.0,
                        },  # Margins(35, 35, 35, 35)
                    },
                },
                Margins(35, 35, 35, 35),
                id="With non-cumulative margins and filter wins",
            ),
            pytest.param(
                (50, 50),
                {
                    "pipeline": {
                        "matching_cost": {
                            "matching_cost_method": "zncc",
                            "window_size": 11,
                            "subpix": 2,
                            "band": None,
                            "step": 5,
                        },  # Margins(5, 5, 5, 5)
                        "disparity": {
                            "disparity_method": "wta",
                            "invalid_disparity": -9999,
                        },  # Margins(1, 1, 1, 1) because of patch_margins fixture
                        "filter": {
                            "filter_method": "bilateral",
                            "sigma_color": 4.0,
                            "sigma_space": 2.0,
                        },  # Margins(14, 14, 14, 14)
                        "filter.1": {
                            "filter_method": "median",
                            "filter_size": 11,
                        },  # Margins(55, 55, 55, 55)
                    },
                },
                Margins(55, 55, 55, 55),
                id="With non-cumulative margins, multiple filters and filter.1 wins",
            ),
            pytest.param(
                (50, 50),
                {
                    "pipeline": {
                        "matching_cost": {
                            "matching_cost_method": "zncc",
                            "window_size": 11,
                            "subpix": 2,
                            "band": None,
                            "step": 5,
                        },  # Margins(5, 5, 5, 5)
                        "disparity": {
                            "disparity_method": "wta",
                            "invalid_disparity": -9999,
                        },  # Margins(1, 1, 1, 1) because of patch_margins fixture
                        "filter": {
                            "filter_method": "bilateral",
                            "sigma_color": 4.0,
                            "sigma_space": 2.0,
                        },  # Margins(14, 14, 14, 14)
                        "filter.1": {
                            "filter_method": "median_for_intervals",
                            "filter_size": 11,
                        },  # Margins(55, 55, 55, 55)
                    },
                },
                Margins(55, 55, 55, 55),
                id="With non-cumulative margins, multiple filters and filter.1 (median_for_intervals) wins",
            ),
        ],
    )
    def test_with_step_of_5(
        self,
        bypass_matching_cost_step_verification,
        pandora_machine_builder,
        image_shape,
        monoband_image,
        configuration,
        expected,
    ):
        """
        Given a pipeline with steps, each step with margins should contribute to global margins.
        """
        pandora_machine = pandora_machine_builder(image_shape, monoband_image)
        pandora_machine.check_conf(configuration, pandora_machine.left_img, pandora_machine.right_img)

        assert pandora_machine.margins.global_margins == expected


# pylint: disable=too-few-public-methods
class TestStepMatchingCost:
    """Test step in the matching cost"""

    def test_update_pandora_machine_step(
        self, bypass_matching_cost_step_verification, pandora_machine_builder, monoband_image
    ):
        """
        Test that pandora_machine.step is updated after matching_cost_check_conf()

        """

        pipeline_cfg = {
            "pipeline": {
                "matching_cost": {"matching_cost_method": "zncc", "window_size": 5, "subpix": 2, "step": 2},
                "disparity": {"disparity_method": "wta", "invalid_disparity": -9999},
                "refinement": {"refinement_method": "vfit"},
                "filter": {"filter_method": "median", "filter_size": 3},
                "validation": {"validation_method": "cross_checking_accurate", "cross_checking_threshold": 1.0},
            }
        }

        # Update the user configuration with default values
        cfg = check_configuration.update_conf(check_configuration.default_short_configuration, pipeline_cfg)

        pandora_machine = pandora_machine_builder((10, 10), monoband_image)

        # pandora_machine.matching_cost_._step_col = 2 # pylint: disable=protected-access
        pandora_machine.check_conf(cfg, pandora_machine.left_img, pandora_machine.right_img)

        assert pandora_machine.step == cfg["pipeline"]["matching_cost"]["step"]


# pylint: disable=too-few-public-methods
class TestStepSGM:
    """Test that the SGM optimisation cannot be run with a step different from 1"""

    @staticmethod
    def test_sgm_step_different_one():
        """
        Test the optimization_check_conf method error message with sgm optimization step when the step value is 1

        """

        pipeline_cfg = {
            "pipeline": {
                "matching_cost": {"matching_cost_method": "zncc", "window_size": 5, "subpix": 2},
                "optimization": {
                    "optimization_method": "sgm",
                    "penalty": {"penalty_method": "sgm_penalty", "P1": 8, "P2": 32, "p2_method": "constant"},
                },
                "disparity": {"disparity_method": "wta", "invalid_disparity": -9999},
                "refinement": {"refinement_method": "vfit"},
                "filter": {"filter_method": "median", "filter_size": 3},
                "validation": {"validation_method": "cross_checking_accurate", "cross_checking_threshold": 1.0},
            }
        }

        pandora_machine = PandoraMachine()

        # In this test we force the use of a step equals to 2.
        # This allows to check that we raise an exception when we try to use SGM optimization
        # with a step different from 1
        pandora_machine.step = 2

        with pytest.raises(
            AttributeError, match="For performing the SGM optimization step, step attribute must be equal to 1"
        ):
            pandora_machine.optimization_check_conf(pipeline_cfg, "optimization")
