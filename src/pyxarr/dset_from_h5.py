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
"""Read a dataset from a HDF5 file, including dimensions and attributes."""

from __future__ import annotations

__all__ = ["dset_from_h5"]

from pathlib import PosixPath
from typing import TYPE_CHECKING, Any

import numpy as np

from pyxarr import Coords, DataArray
from pyxarr.lib.coords import _Coord

if TYPE_CHECKING:
    import h5py
    from numpy.typeing import NDArray


# - local functions --------------------------------
def __get_attrs(dset: h5py.Dataset, field: str) -> dict[str, Any]:
    """Return attributes of the HDF5 dataset.

    Parameters
    ----------
    dset :  h5py.Dataset
       HDF5 dataset from which the attributes are read
    field : str
       Name of field in compound dataset

    Returns
    -------
    dict with attributes

    """
    _field = None
    if field is not None:
        # Note field should exist, else get_data() should complain
        _field = {
            "name": field,
            "oneof": len(dset.dtype.names),
            "index": dset.dtype.names.index(field),
        }

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


def convert_timestamps(
    co_data: NDArray[float], co_attrs: str, co_units: str
) -> NDArray[np.datetime64]:
    """Convert timestamps to numpy.datetime64."""
    if (
        co_units is not None
        and "units" in co_attrs
        and co_attrs["units"].startswith(("days since", "seconds since"))
    ):
        # obtain reference date from attribute 'units'
        ref_time = np.datetime64(
            co_attrs["units"][11:]
            if co_attrs["units"][10] == " "
            else co_attrs["units"][14:]
        )
        match co_units:
            case "ns":
                co_data *= 1e9
            case "us":
                co_data *= 1e6
            case "ms":
                co_data *= 1e3
            case "s":
                pass
            case "D" if co_attrs["units"].startswith("days since"):
                pass
            case _:
                raise KeyError(f"unknown np.timedelta64 unit: '{co_units}'")

        return ref_time + np.rint(co_data).astype(f"timedelta64[{co_units}]")

    return co_data


def __get_coords(
    dset: h5py.Dataset,
    data_sel: tuple[slice | int] | None = None,
    dim_names: list[str, ...] | None = None,
    time_units: str | None = None,
) -> Coords | None:
    r"""Return coordinates of the HDF5 dataset from its dimension scales.

    Parameters
    ----------
    dset :  h5py.Dataset
       HDF5 dataset from which the data is read
    data_sel :  tuple of slice or int, optional
       A numpy slice generated for example `numpy.s\_`
    dim_names : list of strings, optional
       Alternative names for the dataset dimensions if not attached to dataset.
       Default coordinate names are ['time', 'row', 'column'] for 3-D arrays.
    time_units: str, optional
       Convert time axis to numpy.datetime64

    Returns
    -------
    Coords: coordinates of the dataset

    """
    if data_sel is None:
        data_sel = dset.ndim * (np.s_[:],)
    if dset.ndim == 1:
        data_sel = (data_sel,)

    coords = Coords()
    # loop over dimension_scales of dataset
    for ii, dim_scale in enumerate(dset.dims):
        # loop over dataset of each dimension scale
        for dim_dset in dim_scale.values():
            name = PosixPath(dim_dset.name).name
            # Tropomi CKD: remove specification of spectral band
            if name.startswith("row") or name.startswith("column"):
                name = name.split(" ")[0]
            # print(name, dim_dset.shape, dim_dset.dtype)

            # check dimension attributes
            attrs = __get_attrs(dim_dset, None)
            # convert timestamps to np.datetime64
            dim_data = convert_timestamps(dim_dset[data_sel[ii]], attrs, time_units)

            # add _Coord
            coords += _Coord(name, dim_data, dim_ref=name, attrs=attrs)

    if coords:
        return coords

    # assign coordinates to DataArray
    if dim_names is None:
        if dset.ndim > 3:
            raise ValueError("not implemented for ndim > 3")
        dim_names = ["time", "row", "column"][-dset.ndim :]
    else:
        if dset.ndim != len(dim_names):
            raise ValueError(
                "number of dimension names not equal to dataset dimensions"
            )

    coords = []
    for ii in range(dset.ndim):
        co_dtype = "u2" if ((dset.shape[ii] - 1) >> 16) == 0 else "u4"
        co_data = np.arange(dset.shape[ii], dtype=co_dtype)
        coords.append((dim_names[ii], co_data[data_sel[ii]]))

    return Coords(coords)


def __get_data(
    dset: h5py.Dataset, data_sel: tuple[slice | int] | None, field: str | None
) -> NDArray:
    r"""Return data of the HDF5 dataset.

    Parameters
    ----------
    dset :  h5py.Dataset
       HDF5 dataset from which the data is read
    data_sel :  tuple of slice or int, optional
       A numpy slice generated for example `numpy.s\_`
    field : str, optional
       Name of field in compound dataset or None

    Returns
    -------
    Numpy array

    Notes
    -----
    Read floats always as doubles

    """
    if field is None:
        return dset[() if data_sel is None else data_sel]

    return dset.fields(field)[() if data_sel is None else data_sel]


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
    # print(f"data_sel: {data_sel}")
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
    time_units: str | None = None,
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
    time_units :  str, optional
       Convert time axis to numpy datetime64 array, accepted are:
       'ns', 'us', 'ms', 's', 'D'.

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
    try:
        coords = __get_coords(h5_dset, data_sel, dim_names, time_units)
    except ValueError as exc:
        raise RuntimeError("Failed to read dataset dimensions") from exc
    except KeyError as exc:
        raise RuntimeError("Failed to convert timestamps") from exc
    # print(f"1: coords: {coords}")

    # get values for the dataset
    data = __get_data(h5_dset, data_sel, field)

    # - check if dimension of dataset and coordinates agree
    # RvH: I have disabled the code below because the if-condition cannot be met.
    # if len(coords) > 0 and data.ndim < len(coords):
    #    for ii in reversed(range(len(coords))):
    #        if np.isscalar(coords[ii][1]):
    #            del coords[ii]

    # get dataset attributes
    attrs = __get_attrs(h5_dset, field)

    # get name of the dataset
    name = PosixPath(h5_dset.name).name if field is None else field

    return DataArray(data, coords=coords, name=name, attrs=attrs)
