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
"""Test module for pyxarr class `DataArray`."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pyxarr import DataArray


class TestDataArray:
    """Class to test pyxarr.DataArray."""

    def test_empty(self: TestDataArray) -> None:
        """..."""
        assert not bool(DataArray())
        assert len(DataArray()) == 0
        assert DataArray().shape is None
        assert DataArray().size is None

    def test_scalar(self: TestDataArray, da_scalar: DataArray) -> None:
        """..."""
        assert bool(da_scalar)
        assert len(da_scalar) == 0
        assert da_scalar.shape == ()
        assert da_scalar.size == 1

    def test_full(self: TestDataArray, da_full: DataArray) -> None:
        """..."""
        assert bool(da_full)
        assert len(da_full) == 5
        assert da_full.shape == (5, 3, 7)
        assert da_full.size == 5 * 3 * 7

    def test_creation(
        self: TestDataArray,
        da_from_coords: DataArray,
        da_from_dict: DataArray,
        da_from_tuple: DataArray,
        da_from_dims: DataArray,
    ) -> None:
        """..."""
        assert da_from_coords == da_from_dict
        assert da_from_coords == da_from_tuple
        assert da_from_coords == da_from_dims
        assert DataArray(np.ones((24, 5, 7))) == DataArray(
            np.ones((24, 5, 7)), dims=("Z", "Y", "X")
        )
        with pytest.raises(KeyError, match=r".* each dimension") as excinfo:
            _ = DataArray(np.ones((24, 5, 7)), dims=("column", "row"))
        assert "Provide a dimension name for each dimension" in str(excinfo.value)
        with pytest.raises(KeyError, match="N-dim > 3") as excinfo:
            _ = DataArray(np.ones((4, 24, 5, 7)))
        assert "Please provide coordinates if N-dim > 3" in str(excinfo.value)

    def test_swap_dims(self: TestDataArray, da_from_coords: DataArray) -> None:
        """..."""
        xda = da_from_coords
        xda.coords += ("orbit", 2230 + np.arange(xda.values.shape[0]))
        assert "orbit" in xda.coords
        assert "orbit" not in xda.dims
        assert xda.dims == ("time", "row", "column")
        xda.swap_dims("orbit", "time")
        assert xda.dims == ("orbit", "row", "column")
        with pytest.raises(KeyError, match="auxiliary coordinate") as excinfo:
            xda.swap_dims("Z", "time")
        assert "auxiliary coordinate 'Z' does not exists" in str(excinfo.value)
        with pytest.raises(KeyError, match="dimensional coordinate") as excinfo:
            xda.swap_dims("orbit", "Z")
        assert "dimensional coordinate 'Z' does not exists" in str(excinfo.value)
        with pytest.raises(ValueError, match=r".* are not equal") as excinfo:
            xda.swap_dims("orbit", "row")
        assert "length coordinates 'orbit' and 'row' are not equal" in str(
            excinfo.value
        )

    def test_add(self: TestDataArray, da_full: DataArray, da_ones: DataArray) -> None:
        """..."""
        # Add numpy array to DataArray
        xda = da_full + da_ones.values
        assert np.array_equal(xda.values, da_full.values + da_ones.values)
        # Add values of DataArray array to DataArray
        xda = da_full + da_ones
        assert np.array_equal(xda.values, da_full.values + da_ones.values)

    def test_sub(self: TestDataArray, da_full: DataArray, da_ones: DataArray) -> None:
        """..."""
        # Subtract numpy array from DataArray
        xda = da_full - da_ones.values
        assert np.array_equal(xda.values, da_full.values - da_ones.values)
        # Subtract values of DataArray from DataArray
        xda = da_full - da_ones
        assert np.array_equal(xda.values, da_full.values - da_ones.values)

    def test_mul(self: TestDataArray, da_full: DataArray) -> None:
        """..."""
        # Multiply numpy array with DataArray
        xda = da_full * da_full.values
        assert np.array_equal(xda.values, da_full.values * da_full.values)
        # Multiply values of DataArray with DataArray
        xda = da_full * da_full
        assert np.array_equal(xda.values, da_full.values * da_full.values)

    def test_div(self: TestDataArray, da_full: DataArray) -> None:
        """..."""
        res = np.ones(da_full.shape, dtype=float)
        res[0, 0, 0] = np.nan

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Divide numpy array with DataArray
            xda = da_full / da_full.values
            assert np.array_equal(xda.values, res, equal_nan=True)
            # Divide values of DataArray with DataArray
            xda = da_full / da_full
            assert np.array_equal(xda.values, res, equal_nan=True)

    def test_repr(self: TestDataArray, da_full: DataArray) -> None:
        """..."""
        return

    def test_getitem(self: TestDataArray, da_full: DataArray) -> None:
        """..."""
        return

    def test_mean(self: TestDataArray, da_full: DataArray) -> None:
        """..."""
        return

    def test_median(self: TestDataArray, da_full: DataArray) -> None:
        """..."""
        return

    def test_std(self: TestDataArray, da_full: DataArray) -> None:
        """..."""
        return
