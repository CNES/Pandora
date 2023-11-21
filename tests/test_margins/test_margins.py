# Copyright (c) 2023 Centre National d'Etudes Spatiales (CNES).
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
    Margins,
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
