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

from copy import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

# - global parameters ------------------------------
MAX_DIMS = 3


# - class _Coord (local) --------------------
@dataclass(frozen=True, slots=True)
class _Coord:
    """Define one coordinate of a labeled array.

    Parameters
    ----------
    name :  str, optional
      name of the coordinate
    values :  ArrayLike, optional
      values or labels of the coordinate

    """

    name: str | None = None
    values: ArrayLike | None = None

    def __bool__(self: _Coord) -> bool:
        """Return False if _Coord is empty."""
        return self.name is not None

    def __len__(self: _Coord) -> int:
        """Return length of coordinate."""
        return len(self.values) if self else 0

    def __getitem__(self: _Coord, key: int | NDArray[bool]) -> _Coord:
        """Return selected elements."""
        if isinstance(key, np.ndarray):
            return _Coord(name=self.name, values=np.asarray(self.values)[key])

        return _Coord(name=self.name, values=self.values[key])

    def __setitem__(self: _Coord, key: int | NDArray[bool], values: ArrayLike) -> None:
        """Return selected elements."""
        self.values[key] = values

    def copy(self: _Coord) -> _Coord:
        """Return deep copy."""
        return _Coord(name=self.name, values=self.values.copy())


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

    def __bool__(self: Coords) -> bool:
        return bool(self.coords)

    def __contains__(self: Coords, name: str) -> bool:
        return any(name == coord.name for coord in self.coords)

    def __len__(self: Coords) -> int:
        """Return number of coordinates."""
        return len(self.coords)

    def __getitem__(self: Coords, name: str) -> _Coord | None:
        """Select coordinate given its dimension name."""
        for coord in self.coords:
            if name == coord.name:
                return coord

        return None

    def __iter__(self: Coords) -> Coords:
        return iter(self.coords)

    def __add__(self: Coords, coord: tuple[str, ArrayLike]) -> Coords:
        name, val = coord
        if name in self:
            raise ValueError("You can not overwrite a coordinate")
        self.coords += (_Coord(name, np.asarray(val)),)
        return self


def tests_coordinate() -> None:
    """Run tests on local class _Coord."""
    # - empty coordinate
    print("Create an empty coordinate:", _Coord())
    print("- boolean test:", bool(_Coord()))
    print("- size:", len(_Coord()))

    # - time coordinate
    coord = _Coord("time", np.arange("2025-02", "2025-03", dtype="datetime64[D]"))
    print("\nCreate a time coordinate:", coord)
    print("- boolean test:", bool(coord))
    print("- size:", len(coord))
    print("- index:", coord[1])
    print("- slice:", coord[10:15])
    mask = np.ones(len(coord), dtype=bool)
    print("- mask all True:", len(coord[mask]))
    print("- mask all False:", len(coord[~mask]))
    mask[:4] = False
    print("- mask skip first 4:", len(coord[mask]))
    coord[4] = np.datetime64("1970-01-01")
    print("- change fifth item:", coord)

    # - integer coordinate
    coord = _Coord("column", list(range(500, 600)))
    print("\nCreate a integer coordinate:", coord)
    print("- boolean test:", bool(coord))
    print("- size:", len(coord))
    print("- index:", coord[1])
    print("- slice:", coord[10:15])
    mask = np.ones(len(coord), dtype=bool)
    print("- mask all True:", len(coord[mask]))
    print("- mask all False:", len(coord[~mask]))
    mask[:4] = False
    print("- mask skip first 4:", len(coord[mask]))
    coord[4] = -999
    print("- change fifth item:", coord)


def tests_coords() -> None:
    """Run tests on local class Coords."""
    # - empty class Coords
    print("\nCreate an empty Coords:", Coords())
    print("- boolean test:", bool(Coords()))
    print("- size:", len(Coords()))

    print("\nFill a Coords instance")
    co_dict = {
        "time": np.arange("2025-02", "2025-03", dtype="datetime64[D]"),
        "row": np.arange(5),
        "column": list(range(11)),
    }
    coords = Coords(co_dict)
    print(
        "- add time-coordinate:",
        _Coord("time", np.arange("2025-02", "2025-03", dtype="datetime64[D]")),
    )
    print("- add Y-coordinate:", _Coord(name="row", values=np.arange(5)))
    print("- add X-coordinate:", _Coord(name="column", values=np.arange(11)))
    print("- check if 'row' in Coords:", "row" in coords)
    print("- check if 'y' in Coords:", "y" in coords)
    print("- existing dimension:", coords["time"])
    print("- non-existing dimension:", coords["y"])
    co_list = [
        ("time", np.arange("2025-02", "2025-03", dtype="datetime64[D]")),
        ("row", list(range(5))),
        ("column", np.arange(11)),
    ]
    coords = Coords(co_list)
    print("- perform loop iteration:")
    for coord in coords:
        print(coord)


def tests() -> None:
    """..."""
    # Run tests on local class Coord
    tests_coordinate()
    tests_coords()


if __name__ == "__main__":
    tests()
