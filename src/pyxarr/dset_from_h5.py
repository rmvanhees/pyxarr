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

__all__ = ["dset_from_h5"]

from pathlib import PurePath
from typing import TYPE_CHECKING

import numpy as np

from .xarr import Coord, DataArray

if TYPE_CHECKING:
    import h5py


# - main function ----------------------------------
def dset_from_h5(
    h5_dset: h5py.Dataset,
    data_sel: tuple[slice | int] | None = None,
) -> DataArray:
    r"""Use a hdf5/netCDF4 dataset to initialize a DataArray instance.

    Parameters
    ----------
    h5_dset : h5py.Dataset
       h5py dataset from which the data is read
    data_sel: tuple[slice | int], optional
       a numpy slice generated for example `numpy.s\_`

    """
    ds_name = PurePath(h5_dset.name).name
    ds_coords = ()
    ds_attrs = {}

    # read dataset
    if data_sel is None:
        ds_values = h5_dset[()]
    else:
        # we need to keep all dimensions to get the dimensions
        # of the output data right
        ds_values = h5_dset[data_sel]
        if isinstance(data_sel, tuple):
            for ii, elmnt in enumerate(data_sel):
                if isinstance(elmnt, int | np.int64):
                    ds_values = np.expand_dims(ds_values, axis=ii)

    # copy all dimensions with size longer then 1
    for ii in range(h5_dset.ndim):
        if ds_values.shape[ii] == 1:
            continue

        if len(h5_dset.dims[ii]) != 1:  # bug in some KMNI HDF5 files
            print(f"INFO: h5_dset [{ds_name}] has no dimensions")
            continue

        key = PurePath(h5_dset.dims[ii][0].name).name
        if len(key.split()) > 1:
            key = key.split()[0]

        if h5_dset.dims[ii][0][:].size == h5_dset.shape[ii]:
            buff = h5_dset.dims[ii][0][:]
            if np.all(buff == 0):
                buff = np.arange(buff.size)
        else:  # bug in some KMNI HDF5 files
            print(f"WARNING: h5_dset [{ds_name}] has incorrect dimensions")
            continue

        if ds_values.shape[ii] != h5_dset.shape[ii]:
            if isinstance(data_sel, slice):
                buff = buff[data_sel]
            elif len(data_sel) == h5_dset.ndim:
                buff = buff[data_sel[ii]]
            elif not isinstance(data_sel, tuple):
                buff = buff[data_sel]
            elif ii > len(data_sel):
                buff = buff[data_sel[-1]]
            else:
                buff = buff[data_sel[ii]]

        dset = h5_dset.dims[ii][0]
        if "units" in dset.attrs:
            print(dset.attrs["units"])
            print(dset.attrs["units"].startswith("seconds since"))
            if dset.attrs["units"].startswith("seconds since"):
                if np.issubdtype(dset.dtype, np.floating):
                    buff = (
                        np.array(dset.attrs["units"][14:], dtype="datetime64[us]")
                        + (1e6 * buff).astype("timedelta64[us]")
                    )
                else:
                    buff = np.datetime64(dset.attrs["units"][14:]) + buff

        ds_coords += (Coord(key, buff),)

    # remove all dimensions with size equal 1 from value (and error)
    ds_values = np.squeeze(ds_values)

    # collect all attributes
    ds_attrs = {}
    for key, val in h5_dset.attrs.items():
        if key in ["DIMENSION_LIST", "name"]:
            continue
        ds_attrs[key] = val.decode() if isinstance(val, bytes) else val

    return DataArray(ds_values, coords=ds_coords, attrs=ds_attrs, name=ds_name)
