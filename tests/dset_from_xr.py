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


import numpy as np
import pandas as pd
import xarray as xr

from pyxarr import dset_from_xr

# - local functions --------------------------------


# - main function ----------------------------------
def tests() -> None:
    """..."""
    # generate data for xarray.DataArray
    rng = np.random.default_rng()
    data = rng.random((4, 3))
    locs = ["IA", "IL", "IN"]
    times = pd.date_range("2000-01-01", periods=4)

    # generate xarray.DataArray
    xda = xr.DataArray(
        data,
        coords={
            "time": times,
            "space": locs,
            "const": 42,
            "ranking": (("time", "space"), np.arange(12).reshape(4, 3)),
        },
        dims=["time", "space"],
    )
    print("# -- xarray --")
    print(xda)
    print("# -- pyxarr --")
    print(dset_from_xr(xda))


if __name__ == "__main__":
    tests()
