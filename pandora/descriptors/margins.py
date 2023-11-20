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
Margin descriptors
"""
from __future__ import annotations

from dataclasses import dataclass, astuple, asdict
import operator
from typing import overload, Sequence, Tuple, Dict


@dataclass(order=True, frozen=True)
class Margins:
    """Tuple of margins."""

    left: int
    up: int
    right: int
    down: int

    def __post_init__(self):
        if any(m < 0 for m in self.astuple()):
            raise ValueError(f"Margins values should be positive. Got {self.astuple()}")

    def __add__(self, other: Margins) -> Margins:
        return Margins(*map(operator.add, self.astuple(), other.astuple()))

    def astuple(self) -> Tuple:
        """Convert self to a tuple of (left, up, right, down)."""
        return astuple(self)

    def asdict(self) -> Dict:
        """Convert self to a dictionary."""
        return asdict(self)


def max_margins(margins: Sequence[Margins]) -> Margins:
    """
    Return a Margins which is the max of margins element wise.

    :param margins: sequence of Margins to compute max of.
    :type margins: Sequence[Margins]
    :return: Maximum Margins
    :rtype: Margins
    """
    as_tuple_margins = list(map(astuple, margins))
    if len(as_tuple_margins) == 1:
        return Margins(*as_tuple_margins[0])
    return Margins(*map(max, *as_tuple_margins))


class ReadOnlyDescriptor:
    """Descriptor that can not be reassigned."""

    # pylint:disable=too-few-public-methods

    def __set_name__(self, owner: type[object], name: str) -> None:
        self.attribute_name = name  # pylint:disable=attribute-defined-outside-init

    def __set__(self, instance: object, value: object) -> None:
        raise AttributeError("Read-Only attribute", self.attribute_name)


class FixedMargins(ReadOnlyDescriptor):
    """Margins with fixed values."""

    # pylint:disable=too-few-public-methods

    def __init__(self, left: int, up: int, right: int, down: int) -> None:
        self.value = Margins(left, up, right, down)

    @overload
    def __get__(self, instance: None, owner: None) -> FixedMargins:
        ...

    @overload
    def __get__(self, instance: object, owner: type[object]) -> Margins:
        ...

    def __get__(self, instance: object | None, owner: type[object] | None = None) -> FixedMargins | Margins:
        if instance is None:
            return self
        return self.value


class UniformMargins(FixedMargins):
    """Margins with same fixed values in all directions."""

    # pylint:disable=too-few-public-methods

    def __init__(self, value: int) -> None:
        super().__init__(value, value, value, value)


class NullMargins(UniformMargins):
    """Margins with null values in all directions."""

    # pylint:disable=too-few-public-methods

    def __init__(self) -> None:
        super().__init__(0)


class UniformMarginsFromAttribute(ReadOnlyDescriptor):
    """Margins with same fixed values in all directions read from another attribute."""

    # pylint:disable=too-few-public-methods

    def __init__(self, reference_attribute: str) -> None:
        self.reference_attribute = reference_attribute

    @overload
    def __get__(self, instance: None, owner: None) -> UniformMarginsFromAttribute:
        ...

    @overload
    def __get__(self, instance: object, owner: type[object]) -> Margins:
        ...

    def __get__(
        self, instance: object | None, owner: type[object] | None = None
    ) -> UniformMarginsFromAttribute | Margins:
        if instance is None:
            return self
        value = instance.__dict__[self.reference_attribute]
        return Margins(value, value, value, value)


class HalfWindowMargins(ReadOnlyDescriptor):
    """Margins corresponding to half window.

    Expects instance object has a `_window_size` member.
    """

    # pylint:disable=too-few-public-methods

    @overload
    def __get__(self, instance: None, owner: None) -> HalfWindowMargins:
        ...

    @overload
    def __get__(self, instance: object, owner: type[object]) -> Margins:
        ...

    def __get__(self, instance: object | None, owner: type[object] | None = None) -> HalfWindowMargins | Margins:
        if instance is None:
            return self
        # We call __dict__ because mypy says "object" has no attribute "_window_size"
        value = int((instance.__dict__["_window_size"] - 1) / 2)
        return Margins(value, value, value, value)
