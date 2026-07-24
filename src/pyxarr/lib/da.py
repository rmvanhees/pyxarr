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
"""Definition of the pyxarr class: `DataArray`."""

from __future__ import annotations

__all__ = ["DataArray"]

from dataclasses import KW_ONLY, dataclass, field, make_dataclass
from pathlib import PosixPath
from typing import TYPE_CHECKING

import netCDF4
import numpy as np
from h5yaml.template_nc import TemplateNc

from .coords import Coords

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import ArrayLike, NDArray


# - global parameters -------------------------
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
    coords: tuple | list | dict = field(default_factory=Coords)
    dims: tuple[str, ...] = field(default_factory=tuple)
    attrs: dict = field(default_factory=dict)
    name: str | None = None
    _coords: Coords = field(default_factory=Coords, init=False)

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
        if not self.values.shape:
            return

        # define coordinates
        if self.coords:
            if isinstance(self.coords, Coords):
                self._coords = self.coords
            else:
                if self.dims:
                    try:
                        self._coords = Coords(
                            list(zip(self.dims, self.coords, strict=True))
                        )
                    except ValueError as exc:
                        raise ValueError(
                            "coords and dims must be of equal length"
                        ) from exc
                else:
                    self._coords = Coords(self.coords)
            if self._coords.ndim != self.values.ndim:
                raise ValueError("No coordinates or dimensions for each data dimension")
            self.dims = tuple(x.name for x in self._coords)
        elif self.dims and len(self.dims) == self.values.ndim:
            _val = [list(range(x)) for x in self.values.shape]
            self._coords = Coords(list(zip(self.dims, _val, strict=True)))
        else:
            raise ValueError("No coordinates or dimensions for each data dimension")

    def __repr__(self: DataArray) -> str:  # pragma: no cover
        """Convert object to string representation."""
        list_dims = []
        if self.values is not None:
            list_dims = [
                f"{self.dims[i]}: {x}" for i, x in enumerate(self.values.shape)
            ]
        name_str = "\b" if self.name is None else f"{self.name!r}"
        msg = f"<pyxarr.DataArray {name_str} ({', '.join(list_dims)})>"
        msg += f"\n{self.values!r})"
        if self._coords:
            msg += f"\n{self._coords.__repr__()}"
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

    def __eq__(self: DataArray, other: DataArray) -> bool:
        """Return True if both objects are equal."""
        return (
            self.attrs.keys() == other.attrs.keys()  # ToDo: perform full compare
            and self._coords == other._coords
            and self.dims == other.dims
            and np.array_equal(self.values, other.values)
        )

    def __getitem__(self: DataArray, data_sel: int | slice) -> DataArray | None:
        """Return selected elements.

        Parameters
        ----------
        data_sel :  int | slice
          data slicing

        """
        # perform loop over dims and perform selection on each coordinate
        new_coords = []
        if data_sel in (Ellipsis, slice(None, None, None)):
            return self

        new_sel = ()
        for name, isel in zip(
            self.dims[: self._coords.ndim],
            (data_sel,) if self.values.ndim == 1 else data_sel,
            strict=True,
        ):
            # We need to rebuild new_sel from data_sel,
            # because data_sel may contain an integer which breaks its coordinate.
            # This maybe fixed in a next release
            if isinstance(isel, (int | np.integer)):
                isel = np.s_[isel : isel + 1]
            new_sel += (isel,)

            new_coords.append((name, (name, self._coords[name].values[isel])))
            for co in self._coords:
                if not co.is_dimension and co.dim_ref == name:
                    co_vals = self._coords[co.name].values
                    new_coords.append(
                        (
                            co.name,
                            (co.dim_ref, None if co_vals is None else co_vals[isel]),
                        )
                    )

        return DataArray(
            self.values[new_sel],
            coords=new_coords,
            name=self.name,
            attrs=self.attrs,
        )

    def __add__(self: DataArray, other: DataArray | NDArray) -> DataArray:
        """Return new DataArray with values of other added to current."""
        return DataArray(
            self.values + (other if isinstance(other, np.ndarray) else other.values),
            coords=self._coords,
            name=self.name,
            attrs=self.attrs.copy(),
        )

    def __sub__(self: DataArray, other: DataArray | NDArray) -> DataArray:
        """Return new DataArray with values of other subtracted from current."""
        return DataArray(
            self.values - (other if isinstance(other, np.ndarray) else other.values),
            coords=self._coords,
            name=self.name,
            attrs=self.attrs.copy(),
        )

    def __mul__(self: DataArray, other: DataArray | NDArray) -> DataArray:
        """Return new DataArray with values of current multiplied by other."""
        return DataArray(
            self.values * (other if isinstance(other, np.ndarray) else other.values),
            coords=self._coords,
            name=self.name,
            attrs=self.attrs.copy(),
        )

    def __truediv__(self: DataArray, other: DataArray | NDArray) -> DataArray:
        """Return new DataArray with values of current divided by other."""
        return DataArray(
            self.values / (other if isinstance(other, np.ndarray) else other.values),
            coords=self._coords,
            name=self.name,
            attrs=self.attrs.copy(),
        )

    @property
    def shape(self: DataArray) -> tuple[int, ...] | None:
        """Return lengths of all dimensions."""
        try:
            return self.values.shape
        except AttributeError:
            return None

    @property
    def size(self: DataArray) -> int | None:
        """Return total size of array."""
        try:
            return self.values.size
        except AttributeError:
            return None

    @property
    def sizes(self: DataArray) -> dataclass:
        """Ordered mapping from dimension names to lengths."""
        return make_dataclass(
            "Sizes",
            zip(self.dims, len(self.dims) * (np.uint32,), strict=True),
            frozen=True,
        )(*self.values.shape)

    def asdict(self: DataArray, group: str | None = None) -> dict:
        """Return DataArray as dictionary.

        Parameters
        ----------
        group : str, default=None
           Store data in a netCDF4 group

        """
        res = {}
        if not bool(self):
            return res

        grp_path = PosixPath("") if group is None else PosixPath("/", group)
        if group is not None:
            res["groups"] = [str(grp_path)]

        res["dimensions"] = {
            str(grp_path / coord.name): {
                "_dtype": coord.values.dtype.str,
                "_size": coord.values.size,
            }
            for coord in self._coords
        }

        # check if the data-type is compound
        var_name = str(grp_path / self.name)
        if self.values.dtype.names is not None:
            cmp_name = f"{var_name}_dtype"
            res["compounds"] = {
                cmp_name: {v[0]: v[1:] for v in self.values.dtype.descr}
            }
            res["variables"] = {
                var_name: {
                    "_dtype": cmp_name,
                    "_dims": (
                        self.dims
                        if group is None
                        else [str(grp_path / x) for x in self.dims]
                    ),
                    "_values": self.values,
                }
            }
        else:
            res["variables"] = {
                var_name: {
                    "_dtype": self.values.dtype.str,
                    "_dims": (
                        self.dims
                        if group is None
                        else [str(grp_path / x) for x in self.dims]
                    ),
                    "_values": self.values,
                }
            }

        res["variables"][var_name] |= self.attrs

        return res

    def add_coord(self: DataArray, co_name: str, co_val: list[str, ArrayLike]) -> None:
        """Add an auxiliary coordinate."""
        self._coords[co_name] = co_val
        self.dims += (co_name,)

    @property
    def get_coords(self: DataArray) -> Coords:
        """Return coordinate."""
        return self._coords.copy()

    def isel(self: DataArray, **kwargs: dict[str, slice | NDArray[bool]]) -> DataArray:
        """Select data along one or more axes using a slice or boolean array.

        Limitations
        -----------
        Works currently only on dimension coordinates.

        """
        data_sel = ()
        new_coords = []
        values = self.values.copy()
        for ix, name in enumerate(self.dims):
            dim_ref = (
                name if self._coords[name].is_dimension else self._coords[name].dim_ref
            )
            new_slice = slice(None, None, None)
            if dim_ref in kwargs:
                if isinstance(kwargs[dim_ref], slice):
                    new_coords.append(
                        (name, [dim_ref, self._coords[name].values[kwargs[dim_ref]]])
                    )
                    new_slice = kwargs[dim_ref]
                elif (
                    isinstance(kwargs[dim_ref], np.ndarray)
                    and kwargs[dim_ref].dtype == np.bool
                ):
                    mask = kwargs[dim_ref]
                    new_coords.append(
                        (name, [dim_ref, self._coords[name].values[mask]])
                    )
                    if name == dim_ref:  # only for dimension coordinates
                        values = np.take(values, mask.nonzero()[0], axis=ix)
                else:
                    raise ValueError(f"{dim_ref}' should be slice or boolean array")
            else:
                new_coords.append((name, [dim_ref, self._coords[name].values]))
            if name == dim_ref:  # only for dimension coordinates
                data_sel += (new_slice,)

        return DataArray(
            values[data_sel],
            coords=new_coords,
            name=self.name,
            attrs=self.attrs,
        )

    def sel(self: DataArray, **kwargs: dict[str, slice]) -> DataArray:
        """Select data along one or more axes using a coordinate-data range."""
        data_sel = ()
        new_coords = len(self.dims) * [None]
        values = self.values.copy()
        for ix, name in enumerate(self.dims):
            if name in kwargs:
                if isinstance(kwargs[name], slice):
                    mask = (self._coords[name].values >= kwargs[name].start) & (
                        self._coords[name].values <= kwargs[name].stop
                    )
                    new_coords[ix] = (
                        name,
                        [self._coords[name].dim_ref, self._coords[name].values[mask]],
                    )
                    if self._coords[name].is_dimension:
                        values = np.take(values, mask.nonzero()[0], axis=ix)

                        # check for auxiliary coordinate which refers to this one
                        for iy, coord in enumerate(self._coords):
                            if name != coord.name:
                                new_coords[iy] = (
                                    coord.name,
                                    [
                                        self._coords[coord.name].dim_ref,
                                        self._coords[coord.name].values[mask],
                                    ],
                                )
                    else:  # auxiliary coordinates
                        dim_ref = self._coords[name].dim_ref
                        iy = self.dims.index(dim_ref)
                        new_coords[iy] = (
                            dim_ref,
                            [dim_ref, self._coords[dim_ref].values[mask]],
                        )
                        values = np.take(values, mask.nonzero()[0], axis=iy)
                else:
                    raise ValueError(f"{dim_ref}' should be slice or boolean array")
            elif new_coords[ix] is None:
                new_coords[ix] = (
                    name,
                    [self._coords[name].dim_ref, self._coords[name].values],
                )

            if self._coords[name].is_dimension:
                data_sel += (slice(None, None, None),)

        return DataArray(
            values[data_sel],
            coords=new_coords,
            name=self.name,
            attrs=self.attrs,
        )

    def sortby(self: DataArray, dim_name: str) -> DataArray:
        """Sort data along one dimension."""
        if dim_name not in self._coords:
            raise KeyError(f"{dim_name} not found in coordinates of DataArray")

        new_coords = []
        sort_indx = np.argsort(self._coords[dim_name].values)
        if self.values.ndim == 1:
            values = self.values[sort_indx]
            for name in self.dims:
                dim_ref = self._coords[name].dim_ref
                new_coords.append(
                    (name, [dim_ref, self._coords[name].values[sort_indx]])
                )
        else:
            dim_indx = self.dims.index(dim_name)
            if dim_indx >= self.values.ndim:
                dim_indx = self.dims.index(self._coords[dim_name].dim_ref)

            aa = np.zeros(self.shape, dtype=int)
            aa_shape = [1, 1, 1]
            aa_shape[dim_indx] = self.shape[dim_indx]
            aa += sort_indx.reshape(aa_shape)
            values = np.take_along_axis(self.values, aa, axis=dim_indx)
            for name in self.dims:
                dim_ref = self._coords[name].dim_ref
                if name in (dim_name, self._coords[dim_name].dim_ref):
                    new_coords.append(
                        (name, [dim_ref, self._coords[name].values[sort_indx]])
                    )
                else:
                    new_coords.append((name, [dim_ref, self._coords[name].values]))

        return DataArray(
            values,
            coords=new_coords,
            name=self.name,
            attrs=self.attrs,
        )

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
                coords=(),
                attrs=self.attrs.copy(),
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
            coords=tuple(x for x in self._coords if x.name != dim),
            attrs=self.attrs.copy(),
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
                coords=(),
                attrs=self.attrs.copy(),
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
            coords=tuple(x for x in self._coords if x.name != dim),
            attrs=self.attrs.copy(),
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
                coords=(),
                attrs=self.attrs.copy(),
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
            coords=tuple(x for x in self._coords if x.name != dim),
            attrs=self.attrs.copy(),
            name=self.name,
        )

    def to_netcdf(
        self: DataArray,
        path: str | Path | None = None,
        mode: str = "w",
        group: str | None = None,
        attrs_group: dict | None = None,
    ) -> None:
        """Write DataArray contents to netCDF4 file.

        Parameters
        ----------
        path :  Path, default=None
           Path to store the DataArray as variable and its dimensions
        mode :  {"w", "a"}, default="w"
           Currently appand mode is not supported
        group : str, default=None
           Store data in a netCDF4 group
        attrs_group :  dict, default=None
           Provide attributes of the netCDF4 group, each attribute must be defined
           relative to root e.g. f'/{group}/{attr}'

        """
        if mode != "w":
            raise NotImplementedError("Append mode not implemented")

        da_dict = self.asdict(group)
        if group is not None and attrs_group is not None:
            da_dict[str(PosixPath(group) / "attrs_groups")] = attrs_group

        TemplateNc(nc_dict=da_dict).create(path)
        # pylint: disable=no-member
        with netCDF4.Dataset(path, "r+") as fid:
            var_name = self.name if group is None else str(PosixPath(group) / self.name)
            fid[var_name][:] = self.values
