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
This module contains descriptors which returns Margins.

Descriptors are a kind of properties object that can be reused in several classes.
They meant to be used as class attributes.

"""
from __future__ import annotations

from typing import overload

from pandora.margins import Margins


# In order to make a descriptor instance aware of the name it is affected to,
# we can use the dunder method `__set_name__` which is called at affectation.

# When the attribute the descriptor is affected to is called directly from the
# class and not from an instance of this class, the instance argument passed to
# the `__get__` method is `None`. In this case, we want to return the
# descriptor itself and not a value. In order to tell mypy that depending on
# the type of the argument a different type is returned, we declare an
# overload. Thus, the use of overload is only for typing purpose.


class ReadOnlyDescriptor:
    """Descriptor that can not be reassigned."""

    # pylint:disable=too-few-public-methods

    def __set_name__(self, owner: type[object], name: str) -> None:
        self.attribute_name = name  # pylint:disable=attribute-defined-outside-init

    def __set__(self, instance: object, value: object) -> None:
        raise AttributeError("Read-Only attribute", self.attribute_name)


class FixedMargins(ReadOnlyDescriptor):
    """Getter returns Margins with fixed values."""

    # pylint:disable=too-few-public-methods

    def __init__(self, left: int, up: int, right: int, down: int) -> None:
        self.value = Margins(left, up, right, down)

    @overload
    def __get__(self, instance: None, owner: None) -> FixedMargins: ...

    @overload
    def __get__(self, instance: object, owner: type[object]) -> Margins: ...

    def __get__(self, instance: object | None, owner: type[object] | None = None) -> FixedMargins | Margins:
        if instance is None:
            return self
        return self.value


class UniformMargins(FixedMargins):
    """Getter returns Margins with same fixed values in all directions."""

    # pylint:disable=too-few-public-methods

    def __init__(self, value: int) -> None:
        super().__init__(value, value, value, value)


class NullMargins(UniformMargins):
    """Margins with null values in all directions."""

    # pylint:disable=too-few-public-methods

    def __init__(self) -> None:
        super().__init__(0)


class HalfWindowMargins(ReadOnlyDescriptor):
    """Getter returns Margins corresponding to half window.

    Expects instance object has a `_window_size` member.
    """

    # pylint:disable=too-few-public-methods

    @overload
    def __get__(self, instance: None, owner: None) -> HalfWindowMargins: ...

    @overload
    def __get__(self, instance: object, owner: type[object]) -> Margins: ...

    def __get__(self, instance: object | None, owner: type[object] | None = None) -> HalfWindowMargins | Margins:
        if instance is None:
            return self
        # We call __dict__ because mypy says "object" has no attribute "_window_size"
        value = int((instance.__dict__["_window_size"] - 1) / 2)
        return Margins(value, value, value, value)
