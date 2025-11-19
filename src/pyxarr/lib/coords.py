#
# This file is part of Python package: `pyxarr`
#
#     https://github.com/rmvanhees/pyxarr.git
#
# Copyright (c) 2025 SRON
#    All rights reserved.
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
"""Definition of the pyxarr class: `Coords`."""

from __future__ import annotations

__all__ = ["Coords"]

from copy import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

# - global parameters -------------------------


# - class _Coord (local) ----------------------
@dataclass(frozen=True, slots=True)
class _Coord:
    """Define one coordinate of a labeled array.

    Parameters
    ----------
    name :  str, optional
      name of the coordinate
    values :  ArrayLike, optional
      values or labels of the coordinate
    attrs :  dict[str, Any], optional
      attributes providing meta-data of the array

    """

    name: str | None = None
    values: ArrayLike | None = None
    attrs: dict = field(default_factory=dict)

    def __bool__(self: _Coord) -> bool:
        """Return False if object is empty."""
        return self.name is not None

    def __eq__(self: _Coord, other: _Coord) -> bool:
        """Return True if both objects are equal."""
        return (
            self.name == other.name
            and (self.attrs == other.attrs)
            and np.array_equal(self.values, other.values)
        )

    def __len__(self: _Coord) -> int:
        """Return length of coordinate."""
        return len(self.values) if self else 0

    def __getitem__(self: _Coord, key: int | NDArray[bool]) -> _Coord:
        """Return selected elements."""
        if isinstance(key, np.ndarray):
            return _Coord(
                name=self.name, values=np.asarray(self.values)[key], attrs=self.attrs
            )
        return _Coord(name=self.name, values=[self.values[key]], attrs=self.attrs)

    def __setitem__(self: _Coord, key: int | NDArray[bool], values: ArrayLike) -> None:
        """Return selected elements."""
        self.values[key] = values

    def copy(self: _Coord) -> _Coord:
        """Return deep copy."""
        return _Coord(
            name=self.name, values=self.values.copy(), attrs=self.attrs.copy()
        )


# - class Coords -----------------------------------
@dataclass(slots=True)
class Coords:
    """Define one coordinate of a labeled array.

    Parameters
    ----------
    coords :  dict, list, optional
      a list or dictionary of coordinates. If a list, it should be a list of tuples
      where the first element is the dimension name and the second element is the
      corresponding coordinate array_like object.

    """

    coords: tuple[_Coord] = field(default_factory=tuple)

    def __post_init__(self: Coords) -> None:
        """Convert tuple parameters to class _Coord coordinates."""
        if self.coords is None or all(isinstance(x, _Coord) for x in self.coords):
            return

        coords_in = copy(self.coords)
        self.coords = ()
        if isinstance(coords_in, dict):
            for res in coords_in.items():
                self += res
        else:
            for res in coords_in:
                self += res

    def __repr__(self: Coords) -> str:
        """Convert object to string representation."""
        msg = f"<pyxarr.Coords ({', '.join([x.name for x in self])})>"
        with np.printoptions(threshold=5, floatmode="maxprec"):
            for coord in self.coords:
                msg += f"\n   {coord.name:8s} {coord.values.dtype} {coord.values}"
                if coord.attrs:
                    msg += "\nAttributes:"
                    for key, val in coord.attrs.items():
                        msg += f"\n    {key}:\t{val}"
        return msg

    def __bool__(self: Coords) -> bool:
        """Return False if object is empty."""
        return bool(self.coords)

    def __contains__(self: Coords, name: str) -> bool:
        """Return True when item is member of object."""
        return any(name == coord.name for coord in self.coords)

    def __eq__(self: Coords, other: Coords) -> bool:
        """Return True if both objects are equal."""
        for co_self, co_other in zip(self, other, strict=True):
            if not co_self == co_other:
                return False

        return True

    def __getitem__(self: Coords, name: str) -> _Coord | None:
        """Select coordinate given its dimension name."""
        for coord in self.coords:
            if name == coord.name:
                return coord

        return None

    def __iter__(self: Coords) -> Coords:
        """Return an iterator object."""
        return iter(self.coords)

    def __len__(self: Coords) -> int:
        """Return number of coordinates."""
        return len(self.coords)

    def __add__(self: Coords, coord: _Coord | tuple[str, ArrayLike]) -> Coords:
        """Add a coordinate to object."""
        if isinstance(coord, _Coord):
            if coord.name in self:
                raise ValueError("do not try to overwrite a coordinate")
            self.coords += (coord,)
        else:
            name, val = coord
            if name in self:
                raise ValueError("do not try to overwrite a coordinate")
            self.coords += (_Coord(name, np.asarray(val)),)
        return self
