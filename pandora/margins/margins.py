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
Margin margins
"""

from __future__ import annotations

from dataclasses import dataclass, astuple, asdict
import operator
from typing import Sequence, Tuple, Dict

__all__ = ["Margins", "max_margins"]


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
