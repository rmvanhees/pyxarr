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

__all__ = ["dset_to_h5"]

# from typing import TYPE_CHECKING
from pathlib import Path

import h5py
import numpy as np
from _da import Coords, DataArray
from dset_from_h5 import dset_from_h5

# if TYPE_CHECKING:


# - local functions --------------------------------
def write_coordinates(gid: h5py.Group, coords: Coords) -> None:
    """Copy coordinates from DataArray/Dataset to HDF5 file/group."""
    for coord in coords:
        if coord.name in gid:
            continue

        if np.issubdtype(coord.values.dtype, np.datetime64):
            ref_day = coord.values[0].astype("datetime64[D]")
            coord.attrs["units"] = f"seconds since {ref_day} 00:00:00"
            data = coord.values - ref_day

            if coord.values.dtype == np.dtype("<M8[ns]"):
                coord.attrs["numpy_dtype"] = "datetime64[ns]"
                data = data.astype(float) / 1e9
            elif coord.values.dtype == np.dtype("<M8[us]"):
                coord.attrs["numpy_dtype"] = "datetime64[us]"
                data = data.astype(float) / 1e6
            elif coord.values.dtype == np.dtype("<M8[ms]"):
                coord.attrs["numpy_dtype"] = "datetime64[ms]"
                data = data.astype(float) / 1e3
            elif coord.values.dtype == np.dtype("<M8[s]"):
                coord.attrs["numpy_dtype"] = "datetime64[s]"
                data = data.astype("i4")
            elif coord.values.dtype == np.dtype("<M8[D]"):
                coord.attrs["numpy_dtype"] = "datetime64[D]"
                coord.attrs["units"] = f"days since {coord.values[0]}"
                data = data.astype("i4")
            else:
                raise KeyError("unknown dtype of np.datetime64")

            dset = gid.create_dataset(
                coord.name,
                coord.values.shape,
                data.dtype,
            )
            dset[:] = data
        else:
            dset = gid.create_dataset(
                coord.name,
                coord.values.shape,
                coord.values.dtype,
            )
            dset[:] = coord.values
        dset.make_scale(
            f"{Path(coord.name).name}"
            " This is a netCDF dimension but not a netCDF variable."
        )
        for key, val in coord.attrs.items():
            dset.attrs[key] = val


def write_data_array(gid: h5py.Group, xda: DataArray) -> None:
    """Write the data of a DataArray to the HDF5 file."""
    # First create the dimension scales
    write_coordinates(gid, xda.coords)

    # Then write the variable
    # ToDo: what if the DataArray has no name?
    dset = gid.create_dataset(xda.name, data=xda.values)
    for ii, dim in enumerate(xda.dims):
        dset.dims[ii].attach_scale(gid[dim])

    # Finaly, attach the attributes of the variable
    for key, val in xda.attrs.items():
        dset.attrs[key] = val


# - main function ----------------------------------
def dset_to_h5(
    fid: h5py.File,
    xarr: DataArray,
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
        raise PermissionError("File not opened in write or append mode")

    gid = fid if dest_group is None else fid.require_group(dest_group)
    write_data_array(gid, xarr)


def test_wr() -> None:
    """..."""
    xda = DataArray(
        np.arange(11 * 23, dtype="f4").reshape(11, 23),
        dims=("row", "column"),
        coords=(np.arange(11, dtype="u2"), np.arange(23, dtype="u2")),
        name="image",
        attrs={
            "long_name": "Just some data",
            "units": "1",
        },
    )
    co_dict = {
        "time": (
            np.datetime64("2025-07-28T07:30:00").astype("datetime64[ns]")
            + (1e9 * np.linspace(0, 120, 361)).astype("timedelta64[ns]")
        ),
    }
    xdb = DataArray(
        np.arange(361),
        coords=co_dict,
        name="position",
    )

    Path.unlink("test_wr_1.h5", missing_ok=True)
    try:
        with h5py.File("test_wr_1.h5") as fid:
            dset_to_h5(fid, xda)
    except FileNotFoundError as exc:
        print(f"an expected error: {exc}")

    with h5py.File("test_wr_1.h5", "w") as fid:
        dset_to_h5(fid, xda)
    with h5py.File("test_wr_1.h5", "r+") as fid:
        dset_to_h5(fid, xdb)
        print(dset_from_h5(fid[xdb.name]))

    with h5py.File("test_wr_2.h5", "w") as fid:
        dset_to_h5(fid, xda, dest_group="group_01")

    try:
        with h5py.File("test_wr_1.h5") as fid:
            dset_to_h5(fid, xda)
    except PermissionError as exc:
        print(f"an expected error: {exc}")


if __name__ == "__main__":
    test_wr()
