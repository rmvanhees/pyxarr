#
# This file is part of Python package: `pyxarr`
#
#     https://github.com/rmvanhees/pyxarr.git
#
# Copyright (c) 2025 - R.M. van Hees (SRON)
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

import numpy as np

from .coords import Coords
from .da import DataArray

# - global parameters -------------------------


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
    attrs: dict = field(default_factory=dict)
    coords: Coords = field(default_factory=Coords, init=False)
    dims: tuple[str, ...] = field(default_factory=tuple, init=False)

    def __post_init__(self: Dataset) -> None:
        """Define dimensions and coordinates of the new Dataset."""
        self.coords = Coords()
        self.dims = ()
        for da in self.group.values():
            for coord in da.coords:
                if coord not in self.coords:
                    self.coords += (coord,)
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
        if self.group:
            for key, dset in self.group.items():
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
        return bool(self.group)

    def __len__(self: Dataset) -> int:
        """Return number of DataArrays."""
        return len(self.group)

    def __contains__(self: Dataset, name: str) -> bool:
        """Return True when item exists."""
        return name in self.group

    def __iter__(self: Dataset) -> dict:
        """Return an iterator object."""
        return iter(self.group)

    def __eq__(self: Dataset, other: Dataset) -> bool:
        """Return True if both objects are equal."""
        if self.group.keys() != other.group.keys():
            return False

        for key in self.group:
            if not self[key] == other[key]:
                return False

        return self.attrs == other.attrs and self.coords == other.coords

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
                self.coords += coord
                self.dims += (coord.name,)

        self.group[name] = xda
