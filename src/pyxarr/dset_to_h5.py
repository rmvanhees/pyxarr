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
"""Definition of pyxarr function `dset_to_h5`."""

from __future__ import annotations

__all__ = ["dset_from_h5"]

from pathlib import Path
from typing import TYPE_CHECKING

from . import DataArray

if TYPE_CHECKING:
    import h5py


# - local functions --------------------------------
def write_data_array(xda: DataArray, gid: h5py.Group) -> None:
    """..."""
    for coord in xda:
        if coord.name in gid:
            continue

        dset = gid.create_dataset(coord.name, data=coord.values)
        dset.make_scale(
            Path(coord.name).name
            + " This is a netCDF dimension but not a netCDF variable.",
        )


# - main function ----------------------------------
def dset_from_h5(
    xarr: DataArray,
    fid: h5py.File,
    *,
    dest_group: str | None = None,
) -> None:
    """Write content of DataArray or Dataset to a HDF4 file.

    Parameters
    ----------
    fid :  h5py.File | h5py.Group
       blah blah blah
    xarr :  DataArray | Dataset
       blah blah blah
    dest_group :  str, optional
       blah blah blah

    """
    if not isinstance(xarr, DataArray):
        raise ValueError()

    if fid.mode not in ("r+", "w"):
        raise PermissionError()

    _ = fid if dest_group is None else fid.require_group(dest_group)
