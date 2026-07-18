#
# This file is part of Python package: `pyxarr`
#
#     https://github.com/rmvanhees/pyxarr.git
#
# Copyright (c) 2026 - R.M. van Hees (SRON)
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
"""Unit-tests for pyxarr module `dset_from_xr.py`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from pyxarr.dset_from_xr import dset_from_xda

if TYPE_CHECKING:
    from pyxarr import DataArray


class TestFromXda:
    """Class to test pyxarr.dset_from_xda."""

    def test_empty(self: TestFromXda) -> None:
        """Unit-test show that an empty xarray.DataArray gives an empty DataArray."""
        da = dset_from_xda(xr.DataArray())
        # check that da is an empty DataArray
        assert not bool(da)
        assert da.size is None
        assert da.shape is None

    def test_scalar(self: TestFromXda) -> None:
        """Unit-test show that a scalar xarray.DataArray gives a scalar DataArray."""
        da = dset_from_xda(xr.DataArray(3.14))
        # check that da is a scalar DataArray
        assert bool(da)
        assert da.size == 1
        assert da.shape == ()

    def test_full(self: TestFromXda, da_full: DataArray) -> None:
        """..."""
        da = dset_from_xda(
            xr.DataArray(
                np.arange(5 * 11 * 17, dtype=float).reshape(5, 11, 17),
                dims=("orbit", "Y", "X"),
                name="range_arr_float",
                attrs={
                    "long_name": "numpy.arange",
                    "units": "1",
                    "A": 1,
                    "A0": 1.0,
                    "B": (1, 2, 3, 4, 5),
                    "B0": (1, 2, 4, 5),
                    "C": np.array(["aapje", "teun"]),
                    "C0": np.array([1, 2], dtype="u2"),
                    "D": "dit is een tekst",
                    "D0": "Dit is een tekst",
                },
            )
        )
        assert da == da_full
