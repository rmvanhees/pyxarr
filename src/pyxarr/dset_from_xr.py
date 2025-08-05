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
"""Definition of pyxarr function `dset_from_xr`."""

from __future__ import annotations

__all__ = ["dset_from_xr"]

from typing import TYPE_CHECKING

from .xarr import Coord, DataArray

if TYPE_CHECKING:
    import xarray as xr


# - local functions --------------------------------


# - main function ----------------------------------
def dset_from_xr(xda: xr.DataArray) -> DataArray:
    """Copy content of xarray.DataArray to a pyxarr DataArray.

    Parameters
    ----------
    xda : xr.DataArray
       the xarray.DataArray to be copied to a pyxarr.DataArray

    """
    ds_coords = tuple(Coord(x, xda.coords[x]) for x in xda.coords)

    return DataArray(xda.values, coords=ds_coords, attrs=xda.attrs, name=xda.name)
