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

__all__ = ["DataArray"]

from dataclasses import KW_ONLY, dataclass, field
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
                msg += f"  * {coord.name:8s} {coord.values.dtype} {coord.values}\n"
        return msg

    def __len__(self: Coord) -> int:
        """Return number of coordinates."""
        return len(self.data)

    def __getitem__(self: Coords, name: str) -> Coord | None:
        """Select coordinate on its dimension name."""
        if not self.data:
            return None

        for coord in self.data:
            if coord.name == name:
                return coord
        raise ValueError(f"no coordinate: {name}")

    def __iter__(self: Coords) -> Coords:
        return iter(self.data)

    # ToDo: shouldn't we add a dimension as well? How to tricker this?
    def __add__(self: Coords, coord: Coord) -> Coords:
        if not isinstance(coord, Coord):
            raise ValueError("Invalid coordinate (not of type Coord)")
        self.data += (coord,)
        return self

    # ToDo: do we need a __sub__ as well?


# - class DataArray --------------------------------
@dataclass(slots=True)
class DataArray:
    """Define multi-dimensional labeled array.

    Parameters
    ----------
    values :  ArrayLike, optional
      multi-dimensional array
    coords : dict[str, ArrayLike] | list[tuple[str, ArrayLike]], optional
      coordinates for each dimension of the array
    dims :  tuple[str, ...], optional
      dimension names for each dimension of the array
    attrs :  dict[str, Any], optional
      attributes providing meta-data of the array
    name :  str, optional
      name of the array

    """

    values: NDArray | None = None
    _: KW_ONLY
    coords: Coords = field(default_factory=Coords)
    dims: tuple[str, ...] = field(default_factory=tuple)
    attrs: dict = field(default_factory=dict)
    name: str | None = None

    def __post_init__(self: DataArray) -> None:
        """Check and or convert class attributes of DataArray.

        The following attributes are checked:
          - values: converted to an numpy array
          - coords & dims: converted to instance of class Coords
            if coords is a dictionary then dims overwritten as list of keys
            if coords is list with dimension names and coords
            if coords and dims are provided and of equal length then zip(dims, coords)

        """
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
            _val = [list(range(x)) for x in self.values.shape]
            if not self.dims:
                self.dims = _keys
            self.coords = Coords(list(zip(_keys, _val, strict=True)))
        elif isinstance(self.coords, dict):
            if not self.dims:
                self.dims = tuple(x for x in self.coords)
            self.coords = Coords(self.coords)
        elif self.dims:
            self.coords = Coords(list(zip(self.dims, self.coords, strict=True)))
        else:
            self.dims = tuple(x for x, _ in self.coords)
            self.coords = Coords(self.coords)

    def __bool__(self: DataArray) -> bool:
        """Return False if DataArray is empty."""
        return self.values is not None

    def __repr__(self: DataArray) -> str:
        list_dims = [f"{self.dims[i]}: {x}" for i, x in enumerate(self.values.shape)]
        msg = f"<pyxarr.DataArray ({', '.join(list_dims)})>"
        if self.name is not None:
            msg += f"\n{self.name}"
        with np.printoptions(threshold=5, floatmode="maxprec"):
            msg += f"\n{self.values}"
        msg += "\nCoordinates:"
        msg += f"\n{self.coords}"
        if self.attrs:
            msg += "\nAttributes:"
            msg += f"\n{self.attrs}"
        return msg

    def mean(
        self: DataArray,
        dim: str | None = None,
        *,
        skipna: bool = False,
    ) -> DataArray:
        """Reduce this DataArray's data by applying `mean` along some axis.

        Parameters
        ----------
        dim :  str, optional
           Name of the dimension along which to apply `mean`.
        skipna :  bool, default=False
           If True, skip missing values (as marked by NaN)

        """
        if dim is None:
            return DataArray(
                np.nanmean(self.values) if skipna else self.values.mean(),
                attrs=self.attrs.copy(),
                coords=(),
                name=self.name,
            )

        try:
            indx = self.dims.index(dim)
        except ValueError as exc:
            raise ValueError("invalid dimension") from exc

        return DataArray(
            np.nanmean(self.values, axis=indx)
            if skipna
            else self.values.mean(axis=indx),
            attrs=self.attrs.copy(),
            dims=[x.name for x in self.coords if x.name != dim],
            coords=[x.values.copy() for x in self.coords if x.name != dim],
            name=self.name,
        )

    def median(
        self: DataArray,
        dim: str | None = None,
        *,
        skipna: bool = False,
    ) -> DataArray:
        """Reduce this DataArray's data by applying `median` along some axis.

        Parameters
        ----------
        dim :  str, optional
           Name of the dimension along which to apply `mean`.
        skipna :  bool, default=False
           If True, skip missing values (as marked by NaN)

        """
        if dim is None:
            return DataArray(
                np.nanmedian(self.values) if skipna else np.median(self.values),
                attrs=self.attrs.copy(),
                coords=(),
                name=self.name,
            )

        try:
            indx = self.dims.index(dim)
        except ValueError as exc:
            raise ValueError("invalid dimension") from exc

        return DataArray(
            np.nanmedian(self.values, axis=indx)
            if skipna
            else np.median(self.values, axis=indx),
            attrs=self.attrs.copy(),
            dims=[x.name for x in self.coords if x.name != dim],
            coords=[x.values.copy() for x in self.coords if x.name != dim],
            name=self.name,
        )

    def std(
        self: DataArray,
        dim: str | None = None,
        *,
        ddof: int = 0,
        skipna: bool = False,
    ) -> DataArray:
        """Reduce this DataArray's data by applying `std` along some axis.

        Parameters
        ----------
        dim :  str, optional
           Name of the dimension along which to apply `mean`.
        ddof :  int, default=0
           Delta degree of freedom
        skipna :  bool, default=False
           If True, skip missing values (as marked by NaN)

        """
        if dim is None:
            return DataArray(
                np.nanstd(self.values, ddof=ddof)
                if skipna
                else self.values.std(ddof=ddof),
                attrs=self.attrs.copy(),
                coords=(),
                name=self.name,
            )

        try:
            indx = self.dims.index(dim)
        except ValueError as exc:
            raise ValueError("invalid dimension") from exc

        return DataArray(
            np.nanstd(self.values, ddof=ddof, axis=indx)
            if skipna
            else self.values.std(ddof=ddof, axis=indx),
            attrs=self.attrs.copy(),
            dims=[x.name for x in self.coords if x.name != dim],
            coords=[x.values.copy() for x in self.coords if x.name != dim],
            name=self.name,
        )


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
    xarr = DataArray(np.arange(3 * 7).reshape(3, 7), dims=("y", "x"))
    print(xarr.values)
    print(xarr.coords)
    print(xarr.dims)
    print(xarr.attrs)

    print("\n# --- 2-D array with coordinates (dict) ---")
    xarr = DataArray(
        np.arange(3 * 7).reshape(3, 7),
        coords={"y": [1, 2, 3], "x": list(range(7))},
    )
    print(xarr.values)
    print(xarr.coords)
    print(xarr.dims)
    print(xarr.attrs)

    print("\n# --- 2-D array with coordinates (list[tuple]) ---")
    xarr = DataArray(
        np.arange(3 * 7).reshape(3, 7),
        coords=[("y", [1, 2, 3]), ("x", list(range(7)))],
    )
    print(xarr.values)
    print(xarr.coords)
    print(xarr.dims)
    print(xarr.attrs)

    print("\n# --- 2-D array with dims & coordinates ---")
    xarr = DataArray(
        np.arange(3 * 7).reshape(3, 7),
        dims=("y", "x"),
        coords=([1, 2, 3], list(range(7))),
    )
    print(xarr.values)
    print(xarr.coords)
    print(xarr.dims)
    print(xarr.attrs)

    xarr = DataArray(np.arange(5 * 3 * 7).reshape(5, 3, 7))
    print("\n# --- median of 3-D array, dim='time' ---")
    print(xarr.median(dim="time"))
    print("\n# --- median of 3-D array, dim='row' ---")
    print(xarr.median(dim="row"))
    print("\n# --- median of 3-D array, dim='column' ---")
    print(xarr.median(dim="column"))


if __name__ == "__main__":
    tests()
