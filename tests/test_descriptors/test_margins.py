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
Test margin descriptors
"""

import pytest
from pandora.descriptors.margins import (
    FixedMargins,
    UniformMargins,
    NullMargins,
    HalfWindowMargins,
    ReadOnlyDescriptor,
    UniformMarginsFromAttribute,
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
    """max_margins should returns element wise maximum of a Margins list."""
    assert max_margins(margin_list) == expected


@pytest.mark.parametrize("attribute_name", ["margin", "nawak"])
def test_readonlydescriptor(mocker, attribute_name):
    """Test we can not override a descriptor."""

    # We need to add the descriptor at class declaration in order to let descriptor to know the attribute name it's
    # assigned to. As attribute_name is a string we need to proceed as this:
    ParentClass = type("ParentClass", (), {attribute_name: ReadOnlyDescriptor()})
    parent = ParentClass()

    with pytest.raises(AttributeError, match="Read-Only attribute") as exc_info:
        setattr(parent, attribute_name, mocker.sentinel.new_value)
    assert exc_info.value.args[1] == attribute_name


class TestFixedMargins:
    """Test FixedMargins."""

    def test_is_read_only(self):
        assert issubclass(FixedMargins, ReadOnlyDescriptor)

    @pytest.mark.parametrize(
        ["left", "up", "right", "down"],
        [
            [0, 0, 0, 0],
            [1, 2, 3, 4],
        ],
    )
    def test_returns_margins(self, mocker, left, up, right, down):
        """We expect to get a Margins object."""
        parent_class = mocker.Mock()

        descriptor = FixedMargins(left, up, right, down)
        # See https://github.com/pylint-dev/pylint/issues/8265 for why we disable pylint
        # We ignore typing because mypy does not seem to understand the call to __get__
        margin = descriptor.__get__(parent_class)  # type: ignore  # pylint:disable=unnecessary-dunder-call

        assert margin.astuple() == (left, up, right, down)
        assert margin.left == left
        assert margin.up == up
        assert margin.right == right
        assert margin.down == down


class TestUniformMargins:
    """Test UniformMargins."""

    def test_is_read_only(self):
        assert issubclass(UniformMargins, ReadOnlyDescriptor)

    def test_returns_margins(self, mocker):
        """We expect to get a Margins object."""
        parent_class = mocker.Mock()

        descriptor = UniformMargins(40)
        # See https://github.com/pylint-dev/pylint/issues/8265 for why we disable pylint
        # We ignore typing because mypy does not seem to understand the call to __get__
        margin = descriptor.__get__(parent_class)  # type: ignore  # pylint:disable=unnecessary-dunder-call

        assert margin.astuple() == (40, 40, 40, 40)
        assert margin.left == 40
        assert margin.up == 40
        assert margin.right == 40
        assert margin.down == 40


class TestNullMargins:
    """Test NullMargins."""

    def test_is_read_only(self):
        assert issubclass(NullMargins, ReadOnlyDescriptor)

    def test_returns_margins(self, mocker):
        """We expect to get a Margins object."""
        parent_class = mocker.Mock()

        descriptor = NullMargins()
        # See https://github.com/pylint-dev/pylint/issues/8265 for why we disable pylint
        # We ignore typing because mypy does not seem to understand the call to __get__
        margin = descriptor.__get__(parent_class)  # type: ignore  # pylint:disable=unnecessary-dunder-call

        assert margin.astuple() == (0, 0, 0, 0)
        assert margin.left == 0
        assert margin.up == 0
        assert margin.right == 0
        assert margin.down == 0


class TestUniformMarginsFromAttribute:
    """Test UniformMarginsFromAttribute."""

    def test_is_read_only(self):
        assert issubclass(NullMargins, ReadOnlyDescriptor)

    @pytest.mark.parametrize("value", [3, 5])
    @pytest.mark.parametrize("reference_attribute", ["_FILTER_SIZE ", "nawak"])
    def test_returns_margins(self, mocker, reference_attribute, value):
        """We expect to get a Margins object."""
        parent_class = mocker.Mock()
        setattr(parent_class, reference_attribute, value)

        descriptor = UniformMarginsFromAttribute(reference_attribute)
        # See https://github.com/pylint-dev/pylint/issues/8265 for why we disable pylint
        # We ignore typing because mypy does not seem to understand the call to __get__
        margin = descriptor.__get__(parent_class)  # type: ignore  # pylint:disable=unnecessary-dunder-call

        assert margin.astuple() == (value, value, value, value)
        assert margin.left == value
        assert margin.up == value
        assert margin.right == value
        assert margin.down == value


class TestHalfWindowMargins:
    """Test HalfWindowMargins."""

    def test_is_read_only(self):
        assert issubclass(NullMargins, ReadOnlyDescriptor)

    @pytest.mark.parametrize(
        ["window_size", "left", "up", "right", "down"],
        [
            [11, 5, 5, 5, 5],
            [13, 6, 6, 6, 6],
            [0, 0, 0, 0, 0],
        ],
    )
    def test_returns_margins(self, mocker, window_size, left, up, right, down):
        """We expect to get a Margins object."""
        parent_class = mocker.Mock()
        parent_class._window_size = window_size  # pylint:disable=protected-access

        descriptor = HalfWindowMargins()
        # See https://github.com/pylint-dev/pylint/issues/8265 for why we disable pylint
        # We ignore typing because mypy does not seem to understand the call to __get__
        margin = descriptor.__get__(parent_class)  # type: ignore  # pylint:disable=unnecessary-dunder-call

        assert margin.astuple() == (left, up, right, down)
        assert margin.left == left
        assert margin.up == up
        assert margin.right == right
        assert margin.down == down
