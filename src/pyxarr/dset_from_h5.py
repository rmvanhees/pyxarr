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
"""Definition of pyxarr function `dset_from_h5`."""

from __future__ import annotations

__all__ = ["dset_from_h5"]

from pathlib import PurePath
from typing import TYPE_CHECKING

import numpy as np

from . import Coords, DataArray

if TYPE_CHECKING:
    import h5py
    from numpy.typeing import ArrayLike


# - local functions --------------------------------
def __get_attrs(dset: h5py.Dataset, field: str) -> dict:
    """Return attributes of the HDF5 dataset.

    Parameters
    ----------
    dset :  h5py.Dataset
       HDF5 dataset from which the attributes are read
    field : str
       Name of field in compound dataset

    Returns
    -------
    dict with numpy arrays

    """
    _field = None
    if field is not None:
        try:
            _field = {
                "name": field,
                "oneof": len(dset.dtype.names),
                "index": dset.dtype.names.index(field),
            }
        except Exception as exc:
            raise RuntimeError(f"field {field} not in dataset {dset.name}") from exc
        # print('_field ', _field)

    attrs = {}
    for key in dset.attrs:
        if key in (
            "CLASS",
            "DIMENSION_LIST",
            "NAME",
            "REFERENCE_LIST",
            "_Netcdf4Dimid",
            "_Netcdf4Coordinates",
        ):
            continue

        attr_value = dset.attrs[key]
        # print('# ----- ', key, type(attr_value), attr_value)
        if isinstance(attr_value, np.ndarray):
            if len(attr_value) == 1:
                attr_value = attr_value[0]
                # print('# ----- ', key, type(attr_value), attr_value)
            elif _field is not None and len(attr_value) == _field["oneof"]:
                attr_value = attr_value[_field["index"]]
                # elif isinstance(attr_value, np.void):
                #    attr_value = attr_value[0]

        attrs[key] = (
            attr_value.decode("ascii") if isinstance(attr_value, bytes) else attr_value
        )

    return attrs


def __get_coords(
    dset: h5py.Dataset,
    data_sel: tuple[slice | int],
    *,
    time_units: str | None = "[D]",
) -> list[tuple[str, ArrayLike]]:
    r"""Return coordinates of the HDF5 dataset with dimension scales.

    Parameters
    ----------
    dset :  h5py.Dataset
       HDF5 dataset from which the data is read
    data_sel :  tuple of slice or int
       A numpy slice generated for example `numpy.s\_`
    time_units: str, optional
       Convert time axis to numpy.datetime64

    Returns
    -------
    A sequence of tuples [(name, data), ...]

    """
    if len(dset.dims) != dset.ndim:
        print("Oeps: what happens here?")

    coords = Coords()
    for ii, dim in enumerate(dset.dims):
        try:
            # get name of dimension
            name = PurePath(dim[0].name).name
            if name.startswith("row") or name.startswith("column"):
                name = name.split(" ")[0]

            # determine coordinate
            values = None
            if dim[0].size > 0:
                if np.all(dim[0][()] == 0):
                    d_type = "u2" if ((dset.shape[ii] - 1) >> 16) == 0 else "u4"
                    values = np.arange(dset.shape[ii], dtype=d_type)
                else:
                    values = dim[0][()]

            if not (values is None or data_sel is None):
                values = values[data_sel[ii]]

        except RuntimeError as exc:
            raise RuntimeError(
                f"failed to collect coordinates of dataset {dset.name}"
            ) from exc

        # find dimension scale and obtain it attributes
        co_grp = dset.parent
        while name not in co_grp:
            if co_grp == co_grp.parent:
                raise RuntimeError("can't find dimension scale")
            co_grp = co_grp.parent

        # do we need to convert this coordinate?
        coord_attrs = __get_attrs(co_grp[name], None)
        if coord_attrs["units"].startswith(("days since", "seconds since")):
            ref_date = np.datetime64(
                coord_attrs["units"][11:]
                if coord_attrs["units"][10] == " "
                else coord_attrs["units"][14:]
            )
            values = ref_date + values.astype(f"timedelta64{time_units}")

        # add this coordinate
        coords += (name, values)
        for key, val in coord_attrs.items():
            coords[name].attrs[key] = val

    print(coords)
    return coords


def __set_coords(
    dset: h5py.Dataset | np.ndarray,
    data_sel: tuple[slice | int] | None,
    dim_names: list | None,
) -> list[tuple[str, ArrayLike]]:
    r"""Set coordinates of the HDF5 dataset.

    Parameters
    ----------
    dset :  h5py.Dataset or np.ndarray
       HDF5 dataset from which the data is read, or numpy array
    data_sel :  tuple of slice or int
       A numpy slice generated for example `numpy.s\_`
    dim_names : list of strings
       Alternative names for the dataset dimensions if not attached to dataset
       Default coordinate names are ['time', ['row', ['column']]]

    Returns
    -------
    A sequence of tuples [(name, data), ...]

    """
    if dim_names is None:
        if dset.ndim > 3:
            raise ValueError("not implemented for ndim > 3")

        dim_names = ["time", "row", "column"][-dset.ndim :]

    coords = []
    for ii in range(dset.ndim):
        co_dtype = "u2" if ((dset.shape[ii] - 1) >> 16) == 0 else "u4"
        values = np.arange(dset.shape[ii], dtype=co_dtype)
        if data_sel is not None:
            values = values[data_sel[ii]]
        coords.append((dim_names[ii], values))

    return coords


def __get_data(
    dset: h5py.Dataset, data_sel: tuple[slice | int] | None, field: str
) -> np.ndarray:
    r"""Return data of the HDF5 dataset.

    Parameters
    ----------
    dset :  h5py.Dataset
       HDF5 dataset from which the data is read
    data_sel :  tuple of slice or int
       A numpy slice generated for example `numpy.s\_`
    field : str
       Name of field in compound dataset or None

    Returns
    -------
    Numpy array

    Notes
    -----
    Read floats always as doubles

    """
    if data_sel is None:
        data_sel = ()

    if np.issubdtype(dset.dtype, np.floating):
        data = dset.astype(float)[data_sel]
        data[data == float.fromhex("0x1.ep+122")] = np.nan
        return data

    if field is None:
        return dset[data_sel]

    data = dset.fields(field)[data_sel]
    if np.issubdtype(data.dtype, np.floating):
        data = data.astype(float)
        data[data == float.fromhex("0x1.ep+122")] = np.nan
    return data


def __check_selection(data_sel: slice | tuple | int, ndim: int) -> slice | tuple | None:
    r"""Check and correct user provided data selection.

    Notes
    -----
    If data_sel is used to select data from a dataset then the number of
    dimensions of data_sel should agree with the HDF5 dataset or one and
    only one Ellipsis has to be used.
    Thus allowed values for data_sel are:
    * [always]: (), np.s\_[:], np.s\_[...]
    * [1-D dataset]: np.s\_[:-1], np.s\_[0]
    * [2-D dataset]: np.s\_[:-1, :], np.s\_[0, :], np.s\_[:-1, 0]
    * [3-D dataset]: np.s\_[:-1, :, 2:4], np.s\_[0, :, :], np.s\_[:-1, 0, 2:4]
    * [Ellipsis] np.s\_[0, ...], np.s\_[..., 4], np.s\_[0, ..., 4]

    """
    if data_sel in (np.s_[:], np.s_[...], np.s_[()]):
        return None

    if np.isscalar(data_sel):
        return np.s_[data_sel : data_sel + 1]

    buff = ()
    for val in data_sel:
        if val == Ellipsis:
            for _ in range(ndim - len(data_sel) + 1):
                buff += np.index_exp[:]
        elif np.isscalar(val):
            buff += (np.s_[val : val + 1],)
        else:
            buff += (val,)

    return buff


# - main function ----------------------------------
def dset_from_h5(
    h5_dset: h5py.Dataset,
    data_sel: tuple[slice | int] | None = None,
    *,
    dim_names: list[str] | None = None,
    field: str | None = None,
) -> DataArray:
    r"""Create pyxarr.DataArray from a HDF5 dataset (with dimension scales).

    Implements a lite interface with the pyxarr.DataArray, should work for all
    2-D detector images, sequences of detector measurements and trend data.

    Parameters
    ----------
    h5_dset :  h5py.Dataset
       Data, dimensions, coordinates and attributes are read for this dataset
    data_sel :  tuple of slice or int, optional
       A numpy slice generated for example `numpy.s\_`
    dim_names :  list of strings, optional
       Alternative names for the dataset dimensions if not attached to dataset
    field : str, optional
       Name of field in compound dataset or None

    Returns
    -------
    pyxarr.DataArray

    Notes
    -----
    All floating datasets are converted to Python type 'float'

    Dimensions and Coordinates:

    * The functions in this module should work with netCDF4 and HDF5 files.
    * In a HDF5 file the 'coordinates' of a dataset can be defined using \
      dimension scales.
    * In a netCDF4 file this is required: all variables have dimensions, \
      which can have coordinates. But under the hood also netCDF4 uses \
      dimension scales.
    * The DataArray structure will have as dimensions, the names of \
      the dimension scales and as coordinates the names and data of the \
      dimensions scales, except when the data only contains zero's.
    * The default dimensions of an image are 'row' and 'column' with evenly \
      spaced values created with np.arange(len(dim), dtype=uint).

    Examples
    --------
    Read HDF5 dataset 'signal' from file::

    > fid = h5py.File(flname, 'r')        # flname is a HDF5/netCDF4 file
    > xdata = dset_from_h5(fid['signal'])
    > fid.close()

    """
    # Check data selection
    if data_sel is not None:
        data_sel = __check_selection(data_sel, h5_dset.ndim)
    # print(f"data_sel: {data_sel}")

    # get coordinates of the dataset
    coords = []
    if dim_names is None:
        coords = __get_coords(h5_dset, data_sel)
    if not coords:
        coords = __set_coords(h5_dset, data_sel, dim_names)
    # print(f"dim_names: {dim_names}")
    # print(f"coords: {coords}, {len(coords)}")

    # get values for the dataset
    data = __get_data(h5_dset, data_sel, field)
    # print(f"data: {data.ndim}, {data.shape}")

    # - check if dimension of dataset and coordinates agree
    if data.ndim < len(coords):
        for ii in reversed(range(len(coords))):
            if np.isscalar(coords[ii][1]):
                del coords[ii]

    # - remove empty coordinates
    # coords = [(key, val) for key, val in coords if val is not None]
    # print(f"coords: {coords}")

    # get dataset attributes
    attrs = __get_attrs(h5_dset, field)

    # get name of the dataset
    name = PurePath(h5_dset.name).name if field is None else field

    return DataArray(data, coords=coords, name=name, attrs=attrs)
