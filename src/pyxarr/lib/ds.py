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
"""Definition of the pyxarr class: `Dataset`."""

from __future__ import annotations

__all__ = ["Dataset"]

from dataclasses import KW_ONLY, dataclass, field
from pathlib import PosixPath
from typing import TYPE_CHECKING

import netCDF4
import numpy as np
from h5yaml.template_nc import TemplateNc

from .coords import Coords
from .da import DataArray

if TYPE_CHECKING:
    from pathlib import Path

# - global parameters -------------------------


# - class Dataset ----------------------------------
@dataclass(slots=True)
class Dataset:
    """Define a set of multi-dimensional labeled arrays.

    Parameters
    ----------
    data_vars :  dict[str, DataArray], optional
      dictionary of a set multi-dimensional arrays and their names
    coords :  Coords, optional
      common coordinates of each of the DataArrays
    attrs :  dict[str, Any], optional
      attributes providing meta-data of the array

    """

    data_vars: dict[str, DataArray] = field(default_factory=dict)
    _: KW_ONLY
    attrs: dict = field(default_factory=dict)
    coords: Coords = field(default_factory=Coords, init=False)
    dims: tuple[str, ...] = field(default_factory=tuple, init=False)

    def __post_init__(self: Dataset) -> None:
        """Define dimensions and coordinates of the new Dataset."""
        self.coords = Coords()
        self.dims = ()
        for da in self.data_vars.values():
            for coord in da.get_coords:
                if coord not in self.coords:
                    self.coords += coord
                    self.dims += (coord.name,)

    def __repr__(self: Dataset) -> str:  # pragma: no cover
        """Convert object to string representation."""
        msg = "<pyxarr.Dataset>"
        msg += (
            f"\nDimensions:  "
            f"({', '.join([f'{x.name}: {len(x.values)}' for x in self.coords])})"
        )
        if self.coords:
            msg += "\nCoordinates:"
            with np.printoptions(threshold=5, floatmode="maxprec", linewidth=110):
                for coord in self.coords:
                    msg += f"\n  * {coord.name:8s} {coord.values.dtype} {coord.values}"
        msg += "\nData variables:"
        if self.data_vars:
            for key, dset in self.data_vars.items():
                if key in self.coords:
                    continue
                nbytes = dset.values.nbytes
                for prefix in ["", "k", "M", "G", "T"]:
                    if nbytes < 1000:
                        str_size = f"{round(nbytes):g}{prefix}B"
                        break
                    nbytes /= 1000
                else:
                    str_size = f"{round(nbytes):g}PB"
                msg += f"\n    {key}:\t{dset.dims} {dset.values.dtype} {str_size}"
        else:
            msg += "\n    *empty*"
        if self.attrs:
            msg += "\nAttributes:"
            for key, val in self.attrs.items():
                msg += f"\n    {key}:\t{val}"
        return msg

    def __bool__(self: Dataset) -> bool:
        """Return False if Dataset is empty."""
        return bool(self.data_vars)

    def __len__(self: Dataset) -> int:
        """Return number of DataArrays."""
        return len(self.data_vars)

    def __contains__(self: Dataset, name: str) -> bool:
        """Return True when item exists."""
        return name in self.data_vars

    def __iter__(self: Dataset) -> dict:
        """Return an iterator object."""
        return iter(self.data_vars)

    def __eq__(self: Dataset, other: Dataset) -> bool:
        """Return True if both objects are equal."""
        if self.data_vars.keys() != other.data_vars.keys():
            return False

        for key in self.data_vars:
            if not self[key] == other[key]:
                return False

        return self.attrs == other.attrs and self.coords == other.coords

    def __getitem__(self: Dataset, name: str) -> DataArray | None:
        """Return DataArray with given name."""
        if name in self.data_vars:
            return self.data_vars[name]
        return None

    def __setitem__(self: Dataset, name: str, xda: DataArray) -> None:
        """Add DataArray to Dataset."""
        if not isinstance(xda, DataArray):
            raise ValueError("you can only add DataArrays to a Dataset")

        for coord in xda.get_coords:
            if coord not in self.coords:
                self.coords += coord
                self.dims += (coord.name,)

        self.data_vars[name] = xda

    def asdict(self: Dataset, group: None | str = None) -> dict:
        """Return Dataset as dictionary.

        Parameters
        ----------
        group : str, default=None
           Store data in a netCDF4 group

        """
        if not bool(self):
            return {}

        res = {
            "dimensions": {},
            "compounds": {},
            "variables": {},
            "attrs_global": self.attrs,
        }

        grp_path = PosixPath("") if group is None else PosixPath("/", group)
        if group is not None:
            res["groups"] = [str(grp_path)]

        for name, darr in self.data_vars.items():
            da_dict = darr.asdict(group)
            res["dimensions"] |= da_dict["dimensions"]
            if "compounds" in da_dict:
                res["compounds"] |= da_dict["compounds"]
            res["variables"][str(grp_path / name)] = next(
                iter(da_dict["variables"].values())
            )

        if not res["compounds"]:
            del res["compounds"]
        return res

    def to_netcdf(
        self: Dataset,
        path: None | str | Path = None,
        mode: str = "w",
        group: None | str = None,
        attrs_group: None | dict = None,
    ) -> None:
        """Write Dataset contents to netCDF4 file.

        Parameters
        ----------
        path :  Path, default=None
           Path to store the Dataset as variables and its dimensions
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

        ds_dict = self.asdict(group)
        if group is not None and attrs_group is not None:
            ds_dict["attrs_groups"] = attrs_group

        TemplateNc(nc_dict=ds_dict).create(path)
        # pylint: disable=no-member
        with netCDF4.Dataset(path, "r+") as fid:
            if group is None:
                for key, xarr in self.data_vars.items():
                    fid[key][:] = next(iter(xarr.values))
            else:
                for key, xarr in self.data_vars.items():
                    fid[group][key][:] = next(iter(xarr.values))
