#
# This file is part of Python package: `pyxarr`
#
#     https://github.com/rmvanhees/pyxarr.git
#
# Copyright (c) 2025-2026 - R.M. van Hees (SRON)
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
    name :  str
      name of the coordinate
    values :  ArrayLike
      values or labels of the coordinate
    dim_ref: str, optional
    attrs :  dict[str, Any], optional
      attributes providing meta-data of the array

    """

    name: str
    values: ArrayLike
    dim_ref: str | None = None
    attrs: dict = field(default_factory=dict)

    def __repr__(self: Coords) -> str:  # pragma: no cover
        """Convert object to string representation."""
        with np.printoptions(edgeitems=2, threshold=4, floatmode="maxprec"):
            msg = (
                f"  {'*' if self.name == self.dim_ref else ' '} {self.name}"
                f" ({'' if self.dim_ref is None else self.dim_ref})"
                f" {self.values.dtype} {self.values}"
            )
        return msg

    def __eq__(self: _Coord, other: _Coord) -> bool:
        """Return True if both objects are equal."""
        return (
            self.name == other.name
            and np.array_equal(self.values, other.values)
            and (self.attrs == other.attrs)
        )

    def __len__(self: _Coord) -> int:
        """Return length of coordinate."""
        return len(self.values)

    def __getitem__(self: _Coord, key: int | slice | NDArray[bool]) -> _Coord:
        """Return selected elements."""
        if isinstance(key, np.ndarray):
            return _Coord(
                name=self.name,
                values=np.asarray(self.values)[key],
                dim_ref=self.dim_ref,
                attrs=self.attrs
            )
        if isinstance(key, int):
            return _Coord(
                name=self.name,
                values=np.asarray([self.values[key]]),
                dim_ref=self.dim_ref,
                attrs=self.attrs
            )
        return _Coord(
            name=self.name,
            values=self.values[key],
            dim_ref=self.dim_ref,
            attrs=self.attrs,
        )

    def __setitem__(self: _Coord, key: int | NDArray[bool], values: ArrayLike) -> None:
        """Return selected elements."""
        self.values[key] = values

    def copy(self: _Coord) -> _Coord:
        """Return deep copy."""
        return _Coord(
            name=self.name,
            values=self.values.copy(),
            dim_ref=self.dim_ref,
            attrs=self.attrs.copy()
        )

    @property
    def is_dimension(self: _Coord) -> bool:
        """Check if dimension coordinate."""
        return self.name == self.dim_ref


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

    def __repr__(self: Coords) -> str:  # pragma: no cover
        """Convert object to string representation."""
        msg = "Coordinates:"
        for coord in self.coords:
            msg += f"\n{coord.__repr__()}"
            if coord.attrs:
                msg += "\nAttributes:"
                for key, val in coord.attrs.items():
                    msg += f"\n    {key}:\t{val}"
        return msg

    def __bool__(self: Coords) -> bool:
        """Return False if object is empty."""
        return bool(self.coords)

    def __contains__(self: Coords, key: str | _Coord) -> bool:
        """Return True when item is member of object.

        Parameters
        ----------
        key :  str | _Coord
           Check only on coordinate name or check on the _Coord object

        Returns
        -------
        True when coordinate is present in the Coordinates
        """
        if isinstance(key, str):
            return any(key == coord.name for coord in self.coords)

        return any(key == coord for coord in self.coords)

    def __eq__(self: Coords, other: Coords) -> bool:
        """Return True if both objects are equal."""
        if len(self) != len(other):
            return False

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

    def __setitem__(
        self: Coords, name: str, value: ArrayLike | tuple[str, ArrayLike]
    ) -> None:
        """Select coordinate given its dimension name."""
        if name in self:
            raise KeyError("Can not modify existing dimension")

        self += (name, value)

    def __iter__(self: Coords) -> Coords:
        """Return an iterator object."""
        return iter(self.coords)

    def __len__(self: Coords) -> int:
        """Return number of coordinates."""
        return len(self.coords)

    def __add__(self: Coords, coord: _Coord | tuple[str, ArrayLike]) -> Coords:
        """Add a coordinate to object."""
        # print(f"function __add__: {coord}")
        if isinstance(coord, _Coord):
            if coord.name in self:
                raise ValueError("do not try to overwrite a coordinate")
            self.coords += (coord,)
        else:
            name, val = coord
            if name in self:
                raise ValueError("do not try to overwrite a coordinate")
            if (
                len(val) == 2
                and isinstance(val[0], str)
                and not isinstance(val[1], str)
            ):
                self.coords += (_Coord(name, np.asarray(val[1]), val[0]),)
            else:
                self.coords += (_Coord(name, np.asarray(val), name),)
        return self

    @property
    def ndim(self: Coords) -> int:
        """Return number of dimension coordinates, ignoring auxiliary coordinates."""
        num = 0
        for coord in self.coords:
            if coord.is_dimension:
                num += 1

        return num
