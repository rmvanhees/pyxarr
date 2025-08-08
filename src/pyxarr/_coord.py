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
"""Definition of the pyxarr class `Coords`."""

from __future__ import annotations

__all__ = ["Coords"]

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

# - global parameters ------------------------------
MAX_DIMS = 3


# - class Coord ------------------------------------
@dataclass(frozen=True, slots=True)
class Coord:
    """Define one coordinate of a labeled array.

    Parameters
    ----------
    name :  str, optional
      name of the coordinate
    values :  ArrayLike, optional
      values or labels of the coordinate

    """

    name: str | None = None
    values: NDArray | None = None

    def __bool__(self: Coord) -> bool:
        """Return False if Coord is empty."""
        return self.name is not None

    def copy(self: Coord) -> Coord:
        """Return copy."""
        return Coord(name=self.name, values=self.values.copy())


# - class Coords -----------------------------------
class Coords:
    """Define one coordinate of a labeled array.

    Parameters
    ----------
    coords :  dict, list, optional
      a list or dictionary of coordinates. If a list, it should be a list of tuples
      where the first element is the dimension name and the second element is the
      corresponding coordinate array_like object.


    """

    def __init__(
        self: Coords,
        coords: dict[str, ArrayLike] | list[tuple[str, ArrayLike]] | None = None,
    ) -> None:
        self.data = ()
        if coords is None:
            return
        if isinstance(coords, dict):
            for key, val in coords.items():
                self.__add__(Coord(key, np.asarray(val)))
        else:
            for key, val in coords:
                self.__add__(Coord(key, np.asarray(val)))

    def __repr__(self: Coord) -> str:
        msg = ""
        with np.printoptions(threshold=5, floatmode="maxprec"):
            for coord in self.data:
                msg += f"\n  * {coord.name:8s} {coord.values.dtype} {coord.values}"
        return msg

    def __bool__(self: Coord) -> bool:
        return bool(self.data)

    def __contains__(self: Coord, name: str) -> bool:
        for coord in self.data:
            if name == coord.name:
                return True
        return False

    def __len__(self: Coord) -> int:
        """Return number of coordinates."""
        return len(self.data)

    def __getitem__(self: Coords, name: str) -> Coord | None:
        """Select coordinate on its dimension name."""
        for coord in self.data:
            if name == coord.name:
                return coord

        return None

    def __setitem__(self: Coords, coord: Coord) -> Coord | None:
        """Add coordinate to Coords."""
        if not self.__contains__(coord.name):
            self.__add__(coord)

    def __iter__(self: Coords) -> Coords:
        return iter(self.data)

    def __add__(self: Coords, coord: Coord) -> Coords:
        if not isinstance(coord, Coord):
            raise ValueError("Invalid coordinate (not of type Coord)")
        self.data += (coord,)
        return self


def tests() -> None:
    """..."""
    coords = Coords()
    print(coords)

    coords += Coord(name="x", values=np.arange(5))
    print(coords)
    print("x" in coords)

    coords += Coord(name="y", values=np.arange(7))
    print(coords)
    print("y" in coords)
    print(coords["x"])
    print(coords["y"])



if __name__ == "__main__":
    tests()
