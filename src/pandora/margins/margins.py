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
Margin margins
"""

from __future__ import annotations

from collections import UserDict
from dataclasses import dataclass, astuple, asdict
import operator
from functools import reduce
from typing import Sequence, Tuple, Dict

__all__ = ["Margins", "max_margins", "MarginDict", "GlobalMargins"]


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


class MarginDict(UserDict):
    """A dictionary that stores Margins."""

    def __setitem__(self, key, value: Margins):
        if not isinstance(value, Margins):
            raise ValueError(f"MarginDict only accept values of type Margins. Got {type(value)} instead.")
        super().__setitem__(key, value)

    def sum(self) -> Margins:
        """Compute the sum of margins on each direction."""
        return reduce(operator.add, self.data.values(), Margins(0, 0, 0, 0))


class GlobalMargins:
    """Class to store Margins and compute the global."""

    def __init__(self):
        # Margins that cumulates:
        self._cumulatives = MarginDict()
        # Margins that takes maximum values
        self._non_cumulatives = MarginDict()

    def add_cumulative(self, key, value):
        """Add a margins that cumulates."""
        if key in self._non_cumulatives:
            raise KeyError(
                f"{key} is already a non-cumulative margins. "
                "Cumulative margins and non-cumulative margins are exclusive."
            )
        self._cumulatives[key] = value

    def add_non_cumulative(self, key, value):
        """Add a margins that does not cumulate."""
        if key in self._cumulatives:
            raise KeyError(
                f"{key} is already a cumulative margins. "
                "Cumulative margins and non-cumulative margins are exclusive."
            )
        self._non_cumulatives[key] = value

    def remove_cumulative(self, key):
        """Remove a margin that cumulates."""
        try:
            del self._cumulatives[key]
        except KeyError:
            raise KeyError(key)

    def remove_non_cumulative(self, key):
        """Remove a margin that does not cumulate."""
        try:
            del self._non_cumulatives[key]
        except KeyError:
            raise KeyError(key)

    @property
    def cumulatives(self):
        """MarginDict of margins that cumulates."""
        return MarginDict(self._cumulatives)

    @property
    def non_cumulatives(self):
        """MarginDict of margins that does not cumulate."""
        return MarginDict(self._non_cumulatives)

    @property
    def global_margins(self):
        """Computed global margins."""
        return max_margins([self._cumulatives.sum(), *self.non_cumulatives.values()])

    def to_dict(self):
        """Convert self to a dictionary in order to be json serializable."""
        return {
            "cumulative margins": {s: m.asdict() for s, m in self._cumulatives.items()},
            "non-cumulative margins": {s: m.asdict() for s, m in self._non_cumulatives.items()},
            "global margins": self.global_margins.asdict(),
        }

    def get(self, key):
        """Find key in cumulative margins or non-cumulative margins and return corresponding margins"""
        if key in self._cumulatives:
            return self._cumulatives[key]
        if key in self._non_cumulatives:
            return self._non_cumulatives[key]
        return None


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
