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


@pytest.fixture
def empty_da() -> DataArray:
    """Return empty instance of class DataArray."""
    return DataArray()


@pytest.fixture
def scalar_da() -> DataArray:
    """Return scalar instance of class DataArray."""
    return DataArray(3.14)


@pytest.fixture
def test_da() -> DataArray:
    """Return instance of class DataArray with 3 dimensions."""
    return DataArray(
        np.arange(5 * 3 * 7, dtype=float).reshape(5, 3, 7),
        dims=("orbit", "y", "x"),
    )


@pytest.fixture
def test_da_ones() -> DataArray:
    """Return instance of class DataArray with 3 dimensions."""
    return DataArray(
        np.ones((5, 3, 7), dtype=float),
        dims=("orbit", "y", "x"),
    )


class TestDataArray:
    """Class to test pyxarr.DataArray."""

    def test_empty(self: TestDataArray, empty_da: DataArray) -> None:
        """..."""
        assert not bool(empty_da)
        assert len(empty_da) == 0
        assert empty_da.shape is None
        assert empty_da.size is None

    def test_scalar(self: TestDataArray, scalar_da: DataArray) -> None:
        """..."""
        assert bool(scalar_da)
        assert len(scalar_da) == 0
        assert isinstance(scalar_da.shape, tuple)
        assert not scalar_da.shape
        assert scalar_da.size == 1

    def test_da(self: TestDataArray, test_da: DataArray) -> None:
        """..."""
        assert bool(test_da)
        assert len(test_da) == 5
        assert test_da.shape == (5, 3, 7)
        assert test_da.size == 5 * 3 * 7

    def test_swap_dims(self: TestDataArray, test_da: DataArray) -> None:
        """..."""
        xda = test_da
        xda.coords += (
            "time",
            np.arange("2025-07-01", "2025-07-06", dtype="datetime64[D]"),
        )
        assert "time" in xda.coords
        assert "time" not in xda.dims
        assert xda.dims == ("orbit", "y", "x")
        xda.swap_dims("time", "orbit")
        assert xda.dims == ("time", "y", "x")

    def test_add(
        self: TestDataArray, test_da: DataArray, test_da_ones: DataArray
    ) -> None:
        """..."""
        # Add numpy array to DataArray
        xda = test_da + test_da_ones.values
        assert np.array_equal(xda.values, test_da.values + test_da_ones.values)
        # Add values of DataArray array to DataArray
        xda = test_da + test_da_ones
        assert np.array_equal(xda.values, test_da.values + test_da_ones.values)

    def test_sub(
        self: TestDataArray, test_da: DataArray, test_da_ones: DataArray
    ) -> None:
        """..."""
        # Subtract numpy array from DataArray
        xda = test_da - test_da_ones.values
        assert np.array_equal(xda.values, test_da.values - test_da_ones.values)
        # Subtract values of DataArray from DataArray
        xda = test_da - test_da_ones
        assert np.array_equal(xda.values, test_da.values - test_da_ones.values)

    def test_mul(self: TestDataArray, test_da: DataArray) -> None:
        """..."""
        # Multiply numpy array with DataArray
        xda = test_da * test_da.values
        assert np.array_equal(xda.values, test_da.values * test_da.values)
        # Multiply values of DataArray with DataArray
        xda = test_da * test_da
        assert np.array_equal(xda.values, test_da.values * test_da.values)

    def test_div(self: TestDataArray, test_da: DataArray) -> None:
        """..."""
        res = np.ones(test_da.shape, dtype=float)
        res[0, 0, 0] = np.nan

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Divide numpy array with DataArray
            xda = test_da / test_da.values
            assert np.array_equal(xda.values, res, equal_nan=True)
            # Divide values of DataArray with DataArray
            xda = test_da / test_da
            assert np.array_equal(xda.values, res, equal_nan=True)

    def test_mean(self: TestDataArray, test_da: DataArray) -> None:
        """..."""
        return

    def test_median(self: TestDataArray, test_da: DataArray) -> None:
        """..."""
        return

    def test_std(self: TestDataArray, test_da: DataArray) -> None:
        """..."""
        return
