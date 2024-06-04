# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
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
Test margins.
"""
import re

import pytest

from pandora.margins import (
    GlobalMargins,
    Margins,
    MarginDict,
    max_margins,
)


@pytest.mark.parametrize(
    ["augend", "addend", "expected"],
    [
        [Margins(0, 0, 0, 0), Margins(0, 0, 0, 0), Margins(0, 0, 0, 0)],
        [Margins(1, 2, 4, 1), Margins(2, 4, 1, 2), Margins(3, 6, 5, 3)],
    ],
)
def test_margins_are_summable(augend, addend, expected):
    """Do element wise addition instead of extending left tuple with right one."""
    assert augend + addend == expected


def test_margins_can_be_converted_to_dict():
    """Margins should have a method to convert it to dict."""
    result = Margins(1, 2, 3, 4).asdict()
    assert result == {"left": 1, "up": 2, "right": 3, "down": 4}


@pytest.mark.parametrize("values", [(-1, 2, 2, 2), (2, -1, 2, 2), (2, 2, -1, 2), (2, 2, 2, -1)])
def test_margins_are_positive(values):
    """Margins can not be negatives."""
    with pytest.raises(ValueError, match=re.escape(f"Margins values should be positive. Got {values}")):
        Margins(*values)


class TestMarginDict:
    """We test the behavior of some methods that extend a regular dict containing Margins."""

    def test_only_accept_margins_values_at_init(self):
        """MarginDict accept values at init."""
        with pytest.raises(
            ValueError, match="MarginDict only accept values of type Margins. Got <class 'tuple'> instead."
        ):
            MarginDict(step1=(0, 0, 0, 0))

    def test_only_accept_margins_values_at_update(self):
        """MarginDict accept values at update."""
        margin_dict = MarginDict(step1=Margins(0, 0, 0, 0))
        with pytest.raises(
            ValueError, match="MarginDict only accept values of type Margins. Got <class 'tuple'> instead."
        ):
            margin_dict.update({"step2": (1, 1, 1, 1)})

    def test_only_accept_margins_values_at_insertion(self):
        """MarginDict accept values at insertion."""
        margin_dict = MarginDict(step1=Margins(0, 0, 0, 0))
        with pytest.raises(
            ValueError, match="MarginDict only accept values of type Margins. Got <class 'tuple'> instead."
        ):
            # ignoring type as it is what we want to test
            margin_dict["step2"] = (1, 1, 1, 1)  # type: ignore[assignment]

    def test_sum(self):
        """We test sum method."""
        margin_dict = MarginDict(step1=Margins(0, 0, 0, 0), step2=Margins(1, 1, 1, 1), step3=Margins(2, 2, 2, 2))
        assert margin_dict.sum() == Margins(3, 3, 3, 3)


class TestGlobalMargins:
    """We test GlobalMargins class."""

    def test_can_not_add_non_cumulative_margins_if_already_present_in_cumulative(self):
        """Test add margins if already present in dict."""
        global_margins = GlobalMargins()
        global_margins.add_cumulative("step1", Margins(1, 1, 1, 1))
        with pytest.raises(
            KeyError,
            match=(
                "step1 is already a cumulative margins. Cumulative margins and non-cumulative margins are exclusive."
            ),
        ):
            global_margins.add_non_cumulative("step1", Margins(2, 2, 2, 2))

    def test_can_not_add_cumulative_margins_if_already_present_in_non_cumulative(self):
        """Test add margins if already present in dict."""
        global_margins = GlobalMargins()
        global_margins.add_non_cumulative("step1", Margins(1, 1, 1, 1))
        with pytest.raises(
            KeyError,
            match=(
                "step1 is already a non-cumulative margins. "
                "Cumulative margins and non-cumulative margins are exclusive."
            ),
        ):
            global_margins.add_cumulative("step1", Margins(2, 2, 2, 2))

    def test_cumulatives_cannot_be_modified_from_outside(self):
        """Test dict cannot be updated from outside."""
        global_margins = GlobalMargins()
        global_margins.add_cumulative("step1", Margins(1, 1, 1, 1))

        global_margins.cumulatives["step1"] = Margins(0, 0, 0, 0)

        assert global_margins.cumulatives == MarginDict(step1=Margins(1, 1, 1, 1))

    def test_non_cumulatives_cannot_be_modified_from_outside(self):
        """Test dict cannot be updated from outside."""
        global_margins = GlobalMargins()
        global_margins.add_non_cumulative("step1", Margins(1, 1, 1, 1))

        global_margins.non_cumulatives["step1"] = Margins(0, 0, 0, 0)

        assert global_margins.non_cumulatives == MarginDict(step1=Margins(1, 1, 1, 1))

    def test_can_remove_cumulative_margins(self):
        """Test dict can remove item."""
        global_margins = GlobalMargins()
        global_margins.add_cumulative("step1", Margins(1, 1, 1, 1))

        global_margins.remove_cumulative("step1")

        assert global_margins.cumulatives == MarginDict()

    def test_can_remove_non_cumulative_margins(self):
        """Test dict can remove item."""
        global_margins = GlobalMargins()
        global_margins.add_non_cumulative("step1", Margins(1, 1, 1, 1))

        global_margins.remove_non_cumulative("step1")

        assert global_margins.cumulatives == MarginDict()

    @pytest.mark.parametrize(
        ["cumulatives", "non_cumulatives", "expected"],
        [
            pytest.param(
                {"step1": Margins(1, 1, 1, 1), "step2": Margins(1, 1, 1, 1)},
                {"step3": Margins(3, 3, 3, 3)},
                Margins(3, 3, 3, 3),
                id="cumulatives win",
            ),
            pytest.param(
                {"step1": Margins(1, 1, 1, 1), "step2": Margins(4, 1, 1, 1)},
                {"step3": Margins(3, 3, 3, 3)},
                Margins(5, 3, 3, 3),
                id="Non cumulative win",
            ),
        ],
    )
    def test_global_property(self, cumulatives, non_cumulatives, expected):
        """Test add_cumulative and add_non_cumulative method."""
        global_margins = GlobalMargins()
        for step, margins in cumulatives.items():
            global_margins.add_cumulative(step, margins)
        for step, margins in non_cumulatives.items():
            global_margins.add_non_cumulative(step, margins)

        assert global_margins.global_margins == expected

    def test_to_dict(self):
        """Test to_disp method."""
        global_margins = GlobalMargins()
        global_margins.add_cumulative("matching_cost", Margins(2, 2, 2, 2))
        global_margins.add_cumulative("disparity", Margins(0, 0, 0, 0))
        global_margins.add_cumulative("refinement", Margins(0, 0, 0, 0))

        global_margins.add_non_cumulative("filter", Margins(3, 3, 3, 3))

        assert global_margins.to_dict() == {
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

    @pytest.mark.parametrize(
        ["cumulatives", "non_cumulatives", "key", "expected"],
        [
            pytest.param(
                {"step1": Margins(1, 1, 1, 1), "step2": Margins(1, 1, 1, 1)},
                {"step3": Margins(3, 3, 3, 3)},
                "step1",
                Margins(1, 1, 1, 1),
                id="cumulative key",
            ),
            pytest.param(
                {"step1": Margins(1, 1, 1, 1), "step2": Margins(4, 1, 1, 1)},
                {"step3": Margins(3, 3, 3, 3)},
                "step3",
                Margins(3, 3, 3, 3),
                id="Non cumulative key",
            ),
            pytest.param(
                {"step1": Margins(1, 1, 1, 1), "step2": Margins(4, 1, 1, 1)},
                {"step3": Margins(3, 3, 3, 3)},
                "step4",
                None,
                id="Key not exists",
            ),
        ],
    )
    def test_get(self, cumulatives, non_cumulatives, key, expected):
        """Test get method."""
        global_margins = GlobalMargins()
        for step, margins in cumulatives.items():
            global_margins.add_cumulative(step, margins)
        for step, margins in non_cumulatives.items():
            global_margins.add_non_cumulative(step, margins)

        assert global_margins.get(key) == expected


@pytest.mark.parametrize(
    ["margin_list", "expected"],
    [
        pytest.param([Margins(1, 2, 3, 0)], Margins(1, 2, 3, 0), id="One margins"),
        pytest.param([Margins(0, 0, 0, 0), Margins(0, 0, 0, 0)], Margins(0, 0, 0, 0), id="Two null margins"),
        pytest.param([Margins(1, 2, 4, 1), Margins(2, 4, 1, 2)], Margins(2, 4, 4, 2), id="Two margins"),
        pytest.param(
            [Margins(1, 2, 4, 9), Margins(2, 4, 1, 2), Margins(6, 1, 0, 3)],
            Margins(6, 4, 4, 9),
            id="More than Two margins",
        ),
        pytest.param(
            iter([Margins(1, 2, 4, 9), Margins(2, 4, 1, 2), Margins(6, 1, 0, 3)]),
            Margins(6, 4, 4, 9),
            id="With iterator",
        ),
    ],
)
def test_max_margins(margin_list, expected):
    """max_margins should return element wise maximum of a Margins list."""
    assert max_margins(margin_list) == expected
