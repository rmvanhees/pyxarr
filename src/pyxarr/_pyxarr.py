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
"""Definition of the pyxarr classes: `Coords`, `DataArray` and `Dataset`."""

from __future__ import annotations

__all__ = ["Coords", "DataArray", "Dataset"]

from copy import copy
from dataclasses import KW_ONLY, dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

# - global parameters -------------------------
MAX_DIMS = 3  # maximum number of automatically generated dimensions


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
        """Return False if _Coord is empty."""
        return self.name is not None

    def __len__(self: _Coord) -> int:
        """Return length of coordinate."""
        return len(self.values) if self else 0

    def __getitem__(self: _Coord, key: int | NDArray[bool]) -> _Coord:
        """Return selected elements."""
        if isinstance(key, np.ndarray):
            return _Coord(
                name=self.name, values=np.asarray(self.values)[key], attrs=self.attrs
            )
        return _Coord(name=self.name, values=self.values[key], attrs=self.attrs)

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

    def __add__(self: Coords, coord: _Coord | tuple[str, ArrayLike]) -> Coords:
        if isinstance(coord, _Coord):
            if coord.name in self:
                raise ValueError("You can not overwrite a coordinate")
            self.coords += (coord,)
        else:
            name, val = coord
            if name in self:
                raise ValueError("You can not overwrite a coordinate")
            self.coords += (_Coord(name, np.asarray(val)),)
        return self


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
        name_str = "\b" if self.name is None else f"{self.name!r}"
        msg = f"<pyxarr.DataArray {name_str} ({', '.join(list_dims)})>"
        with np.printoptions(threshold=5, floatmode="maxprec"):
            msg += f"\narray({self.values})"
            if self.coords:
                msg += "\nCoordinates:"
                for coord in self.coords:
                    msg += (
                        f"\n {'*' if coord.name in self.dims else ' '} "
                        f"{coord.name:8s} {coord.values.dtype} {coord.values}"
                    )
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

    def __getitem__(self: DataArray, keys: int | slice | NDArray[bool]) -> DataArray:
        """Return selected elements."""
        # perform loop over dims and perform selection on each coordinate
        if keys is Ellipsis:
            return self

        new_coords = []
        for name, key in zip(
            self.dims,
            (keys,) if isinstance(keys, np.ndarray) else keys,
            strict=True,
        ):
            new_coords.append((name, self.coords[name].values[key]))

        return DataArray(
            self.values[keys],
            coords=new_coords,
            name=self.name,
            attrs=self.attrs,
        )

    def __add__(self: DataArray, other: DataArray | NDArray) -> DataArray:
        """Return DataArray with values divided by other."""
        return DataArray(
            self.values + other if isinstance(other, np.ndarray) else other.values,
            coords=self.coords,
            name=self.name,
            attrs=self.attrs.copy(),
        )

    def __sub__(self: DataArray, other: DataArray | NDArray) -> DataArray:
        """Return DataArray with values divided by other."""
        return DataArray(
            self.values - other if isinstance(other, np.ndarray) else other.values,
            coords=self.coords,
            name=self.name,
            attrs=self.attrs.copy(),
        )

    def __mull__(self: DataArray, other: DataArray | NDArray) -> DataArray:
        """Return DataArray with values divided by other."""
        return DataArray(
            self.values * other if isinstance(other, np.ndarray) else other.values,
            coords=self.coords,
            name=self.name,
            attrs=self.attrs.copy(),
        )

    def __truediv__(self: DataArray, other: DataArray | NDArray) -> DataArray:
        """Return DataArray with values divided by other."""
        return DataArray(
            self.values / other if isinstance(other, np.ndarray) else other.values,
            coords=self.coords,
            name=self.name,
            attrs=self.attrs.copy(),
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

    def swap_dims(self: DataArray, aux_dim: str, co_dim: str) -> None:
        """Promote an auxiliary coordinate to a dimension coordinate.

        Parameters
        ----------
        aux_dim: str
           Name of the auxiliary coordinate
        co_dim: str
           Name of the dimension coordinate

        """
        if aux_dim not in self.coords:
            raise KeyError("auxiliary coordinate '{aux_dim}' does not exists")
        if co_dim not in self.dims:
            raise KeyError("dimensional coordinate '{co_dim}' does not exists")
        if len(self.coords[co_dim]) != len(self.coords[aux_dim]):
            raise ValueError(
                "coordinates '{aux_dim}' and '{co_dim}' should be equal in size"
            )
        self.dims = tuple(aux_dim if x == co_dim else x for x in self.dims)

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


# - class Dataset ----------------------------------
@dataclass(slots=True)
class Dataset:
    """Define a set of multi-dimensional labeled arrays.

    Parameters
    ----------
    group :  dict[str, DataArray], optional
      dictionary of a set multi-dimensional arrays and their names
    coords :  Coords, optional
      common coordinates of each of the DataArrays
    attrs :  dict[str, Any], optional
      attributes providing meta-data of the array

    """

    group: dict[str, DataArray] = field(default_factory=dict)
    _: KW_ONLY
    coords: Coords = field(default_factory=Coords)
    attrs: dict = field(default_factory=dict)

    # def __post_init__(self: Dataset) -> None:
    #    """Check and or convert class attributes of DataArray."""
    #    return

    def __repr__(self: Dataset) -> str:
        msg = "<pyxarr.Dataset>"
        msg += (
            f"\nDimensions:  "
            f"({', '.join([f'{x.name}: {len(x.values)}' for x in self.coords])})"
        )
        if self.coords:
            with np.printoptions(threshold=5, floatmode="maxprec"):
                msg += "\nCoordinates:"
                for coord in self.coords:
                    msg += f"\n  * {coord.name:8s} {coord.values.dtype} {coord.values}"
        msg += "\nData variables:"
        if self.group:
            with np.printoptions(threshold=5, floatmode="maxprec"):
                for key, dset in self.group.items():
                    msg += f"\n    {key}:\t{dset.dims} {dset.values}"
        else:
            msg += "\n    *empty*"
        if self.attrs:
            msg += "\nAttributes:"
            for key, val in self.attrs.items():
                msg += f"\n    {key}:\t{val}"
        return msg

    def __bool__(self: Dataset) -> bool:
        """Return False if Dataset is empty."""
        return bool(self.group)

    def __len__(self: Dataset) -> int:
        """Return number of DataArrays."""
        return len(self.group)

    def __contains__(self: Dataset, name: str) -> bool:
        return name in self.group

    def __iter__(self: Dataset) -> dict:
        return iter(self.group)

    def __getitem__(self: Dataset, name: str) -> DataArray | None:
        """Return DataArray with given name."""
        if name in self.group:
            return self.group[name]
        return None

    def __setitem__(self: Dataset, name: str, xda: DataArray) -> None:
        """Add DataArray to Dataset."""
        if not isinstance(xda, DataArray):
            raise ValueError("you can only add DataArrays to a Dataset")

        for coord in xda.coords:
            if coord.name not in self.coords:
                self.coords += (coord.name, coord.values)

        self.group[name] = xda
