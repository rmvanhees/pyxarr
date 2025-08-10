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

from pyxarr import Coords

if TYPE_CHECKING:
    from numpy.typing import NDArray

# - global parameters ------------------------------
MAX_DIMS = 3  # maximum number of automatically generated dimensions


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
        elif isinstance(self.coords, Coords) and not self.dims:
            self.dims = tuple(x.name for x in self.coords)
        elif isinstance(self.coords, dict):
            if not self.dims:
                self.dims = tuple(x for x in self.coords)
            self.coords = Coords(self.coords)
        elif self.dims:
            self.coords = Coords(list(zip(self.dims, self.coords, strict=True)))
        else:
            self.dims = tuple(x for x, _ in self.coords)
            self.coords = Coords(self.coords)

    def __repr__(self: DataArray) -> str:
        list_dims = []
        if self.values is not None:
            list_dims = [
                f"{self.dims[i]}: {x}" for i, x in enumerate(self.values.shape)
            ]
        msg = " ".join([
            "<pyxarr.DataArray",
            '\b' if self.name is None else f"{self.name!r}",
            f"({', '.join(list_dims)})>"
        ])
        with np.printoptions(threshold=5, floatmode="maxprec"):
            msg += f"\narray({self.values})"
        if self.coords:
            msg += "\nCoordinates:"
            msg += f"{self.coords}"
        if self.attrs:
            msg += "\nAttributes:"
            for key, val in self.attrs.items():
                msg += f"\n    {key}:\t{val}"
        return msg

    def __bool__(self: DataArray) -> bool:
        """Return False if DataArray is empty."""
        return self.values is not None

    def __len__(self: DataArray) -> int:
        """Return length of first dimension."""
        # consistent with numpy and xarray!
        try:
            return len(self.values)
        except TypeError:
            return 0

    def __getitem__(self: DataArray, keys: int | NDArray[bool]) -> DataArray:
        """Return selected elements."""
        # perform loop over dims and perform selection on each coordinate
        print("# --- keys:", keys)
        if not keys or keys is Ellipsis:
            return self

        new_coords = []
        for name, key in zip(self.dims, keys, strict=True):
            new_coords.append((name, self.coords[name][key].values))
            
        return DataArray(
            self.values[keys],
            coords=new_coords,
            name=self.name,
            attrs=self.attrs,
        )

    @property
    def shape(self: DataArray) -> tuple[int, ...] | None:
        """Return lengths of all dimensions."""
        try:
            return self.values.shape
        except TypeError:
            return ()
        except AttributeError:
            return None

    @property
    def size(self: DataArray) -> int | None:
        """Return total size of array."""
        try:
            return self.values.size
        except TypeError:
            return 0
        except AttributeError:
            return None

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
    """Run tests on class DataArray."""
    # - empty class Coords
    xda = DataArray()
    print("Create an empty DataArray:", xda)
    print("- boolean test:", bool(xda))
    print("- length:", len(xda))
    print("- shape:", xda.shape)
    print("- size:", xda.size)

    xda = DataArray(1)
    print("Create a DataArray with scalar:", xda)
    print("- boolean test:", bool(xda))
    print("- length:", len(xda))
    print("- shape:", xda.shape)
    print("- size:", xda.size)

    xda = DataArray(list(range(11)))
    print("Create a DataArray with list:", xda)
    print("- boolean test:", bool(xda))
    print("- length:", len(xda))
    print("- shape:", xda.shape)
    print("- size:", xda.size)


    xda = DataArray(np.arange(3 * 5).reshape(3, 5))
    print("Create a DataArray with numpy:", xda)
    print("- boolean test:", bool(xda))
    print("- length:", len(xda))
    print("- shape:", xda.shape)
    print("- size:", xda.size)
    print("- select one element:", xda[1, 3])
    print("- select one X-element:", xda[:, 3])
    print("- select one Y-element:", xda[1, :])
    print("- select all elements:", xda[()])
    print("- select all elements:", xda[:, :])
    print("- select all elements:", xda[...])
    return

    attrs = {
        "long_name": "my dataset",
        "units": "m/s",
    }

    print("# --- empty data array ---")
    xarr = DataArray(attrs=attrs)
    print(xarr)

    print("# --- 1-D array without coordinates ---")
    xarr = DataArray(list(range(11)), attrs=attrs)
    print(xarr)

    print("\n# --- 2-D array without coordinates ---")
    xarr = DataArray(np.arange(3 * 7).reshape(3, 7))
    print(xarr)

    print("\n# --- 3-D array without coordinates ---")
    xarr = DataArray(np.arange(5 * 3 * 7).reshape(5, 3, 7))
    print(xarr)

    print("\n# --- 2-D array without coordinates with dimension names ---")
    xarr = DataArray(np.arange(3 * 7).reshape(3, 7), dims=("y", "x"))
    print(xarr)

    print("\n# --- 2-D array with coordinates (dict) ---")
    xarr = DataArray(
        np.arange(3 * 7).reshape(3, 7),
        coords={"y": [1, 2, 3], "x": list(range(7))},
    )
    print(xarr)

    print("\n# --- 2-D array with coordinates (list[tuple]) ---")
    xarr = DataArray(
        np.arange(3 * 7).reshape(3, 7),
        coords=[("y", [1, 2, 3]), ("x", list(range(7)))],
    )
    print(xarr)

    print("\n# --- 2-D array with dims & coordinates ---")
    xarr = DataArray(
        np.arange(3 * 7).reshape(3, 7),
        dims=("y", "x"),
        coords=([1, 2, 3], list(range(7))),
    )
    print(xarr)

    xarr = DataArray(np.arange(5 * 3 * 7).reshape(5, 3, 7))
    print("\n# --- median of 3-D array, dim='time' ---")
    print(xarr.median(dim="time"))
    print("\n# --- median of 3-D array, dim='row' ---")
    print(xarr.median(dim="row"))
    print("\n# --- median of 3-D array, dim='column' ---")
    print(xarr.median(dim="column"))


if __name__ == "__main__":
    tests()
