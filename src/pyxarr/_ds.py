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
"""Definition of the pyxarr class `Dataset`."""

from __future__ import annotations

__all__ = ["Dataset"]

from dataclasses import KW_ONLY, dataclass, field

import numpy as np
from _coord import Coords
from _da import DataArray

# - global parameters ------------------------------


# - class DataArray --------------------------------
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


def tests() -> None:
    """..."""
    xarr1 = DataArray(
        np.arange(3 * 7).reshape(3, 7),
        coords=[("y", [1, 2, 3]), ("x", list(range(7)))],
    )
    xarr2 = DataArray(
        np.arange(3 * 7).reshape(3, 7),
        coords=[("y", [1, 2, 3]), ("x", list(range(7)))],
    )

    # - empty class Dataset
    xds = Dataset()
    print(f"\nShow empty Dataset:\n{xds}")
    print("- boolean test:", bool(xds))
    print("- length:", len(xds))

    # - non-empty class Dataset
    xds["foo"] = xarr1
    xds["bar"] = xarr2
    xds.attrs["title"] = "test example"
    print(f"\nShow non-empty Dataset:\n{xds}")
    print("- boolean test:", bool(xds))
    print("- length:", len(xds))


if __name__ == "__main__":
    tests()
