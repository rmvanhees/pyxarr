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
"""Definition of the pyxarr class `DataArray`."""

from __future__ import annotations

__all__ = ["Coord", "DataArray"]

from dataclasses import KW_ONLY, dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

# - global parameters ------------------------------
MAX_DIMS = 3


# - class Coord ------------------------------------
@dataclass(frozen=True)
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
    values: ArrayLike | None = None

    def __bool__(self: Coord) -> bool:
        """Return False if Coord is empty."""
        return self.name is not None


# - class DataArray --------------------------------
@dataclass()
class DataArray:
    """Define multi-dimensional labeled array.

    Parameters
    ----------
    values :  ArrayLike, optional
      multi-dimensional array
    coords : tuple[Coord, ...], optional
      coordinates for each dimension of the array
    dims :  tuple[str, ...], optional
      dimension names for each dimension of the array
    attrs :  dict[str, Any], optional
      attributes providing meta-data of the array
    name :  str, optional
      name of the array

    """

    values: ArrayLike | None = None
    _: KW_ONLY
    coords: tuple[Coord, ...] = field(default_factory=tuple)
    dims: tuple[str, ...] = field(default_factory=tuple)
    attrs: dict = field(default_factory=dict)
    name: str | None = None

    def __post_init__(self: DataArray) -> None:
        """Initialize undefined class variables of DataArray."""
        if self.values is None:
            return

        # make sure that values are a numpy array
        self.values = np.asarray(self.values)

        # define coordinates
        if not self.coords:
            if self.values.ndim > MAX_DIMS:
                raise KeyError("Please provide coordinates of N-dim > 3")
            if self.dims and len(self.dims) != self.values.ndim:
                raise KeyError("Please provide dimension names for each dimension")

            # use dimension names or use default dimension names
            _keys = (
                self.dims
                if self.dims
                else ("time", "row", "column")[MAX_DIMS - self.values.ndim :]
            )
            for ii in range(self.values.ndim):
                self.coords += (Coord(_keys[ii], list(range(self.values.shape[ii]))),)
        else:
            if len(self.coords) != self.values.ndim:
                raise KeyError("Please provide coordinates for each dimension")

        # define dimensions
        if not self.dims:
            for coord in self.coords:
                self.dims += (coord.name,)

        # add attributes units and long_name
        # for key in ["units", "long_name"]:
        #    if key not in self.attrs:
        #        self.attrs[key] = ""

    def __bool__(self: DataArray) -> bool:
        """Return False if DataArray is empty."""
        return self.values is not None


def tests() -> None:
    """..."""
    attrs = {
        "long_name": "my dataset",
        "units": "m/s",
    }

    print("# --- empty data array ---")
    xarr = DataArray(attrs=attrs)
    if xarr:
        print(xarr.values)
        print(xarr.coords)
        print(xarr.dims)
    print(xarr.attrs)

    print("# --- 1-D array without coordinates ---")
    xarr = DataArray(list(range(11)), attrs=attrs)
    if xarr:
        print(xarr.values)
        print(xarr.coords)
        print(xarr.dims)
        print(xarr.attrs)

    print("\n# --- 2-D array without coordinates ---")
    xarr = DataArray(np.arange(3 * 7).reshape(3, 7))
    print(xarr.values)
    print(xarr.coords)
    print(xarr.dims)
    print(xarr.attrs)

    print("\n# --- 3-D array without coordinates ---")
    xarr = DataArray(np.arange(5 * 3 * 7).reshape(5, 3, 7))
    print(xarr.values)
    print(xarr.coords)
    print(xarr.dims)
    print(xarr.attrs)

    print("\n# --- 2-D array without coordinates with dimension names ---")
    xarr = DataArray(np.arange(3 * 7).reshape(3, 7), dims=("x", "y"))
    print(xarr.values)
    print(xarr.coords)
    print(xarr.dims)
    print(xarr.attrs)

    print("\n# --- 2-D array with coordinates ---")
    xarr = DataArray(
        np.arange(3 * 7).reshape(3, 7),
        coords=(Coord("x", [1, 2, 3]), Coord("y", np.arange(7))),
    )
    print(xarr.values)
    print(xarr.coords)
    print(xarr.dims)
    print(xarr.attrs)


if __name__ == "__main__":
    tests()
