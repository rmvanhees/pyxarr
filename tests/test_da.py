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
"""Test module for pyxarr class `DataArray`."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pytest

from pyxarr import DataArray


class TestDataArray:
    """Class to test pyxarr.DataArray."""

    def test_empty(self: TestDataArray) -> None:
        """Unit-test for empty DataArray."""
        assert not bool(DataArray())
        assert len(DataArray()) == 0
        assert DataArray().shape is None
        assert DataArray().size is None
        assert not DataArray().asdict()

    def test_scalar(self: TestDataArray, da_scalar: DataArray) -> None:
        """Unit-test for DataArray with scalar."""
        assert bool(da_scalar)
        assert len(da_scalar) == 0
        assert da_scalar.shape == ()
        assert da_scalar.size == 1
        assert not da_scalar.asdict()["dimensions"]
        assert "compounds" not in da_scalar.asdict()
        assert "scalar" in da_scalar.asdict()["variables"]
        assert "groups" not in da_scalar.asdict()
        assert "scalar" in da_scalar.asdict()["variables"]
        assert "/GROUP" in da_scalar.asdict("/GROUP")["groups"]
        assert "/GROUP/scalar" in da_scalar.asdict("/GROUP")["variables"]

    def test_cmp(self: TestDataArray, da_compound: DataArray) -> None:
        """Unit-test for DataArray with structured array."""
        assert bool(da_compound)
        assert len(da_compound) == 3
        assert da_compound.shape == (3,)
        assert da_compound.size == 3
        assert "orbit" in da_compound.asdict()["dimensions"]
        assert "compound_arr_dtype" in da_compound.asdict()["compounds"]
        assert "groups" not in da_compound.asdict()
        assert "compound_arr" in da_compound.asdict()["variables"]
        assert "/GROUP" in da_compound.asdict("/GROUP")["groups"]
        assert "/GROUP/compound_arr" in da_compound.asdict("/GROUP")["variables"]

    def test_full(self: TestDataArray, da_full: DataArray) -> None:
        """Unit-test for DataArray with 3-D dataset."""
        assert bool(da_full)
        assert len(da_full) == 5
        assert da_full.shape == (5, 11, 17)
        assert da_full.size == 5 * 11 * 17
        assert da_full.sizes.orbit == 5
        assert da_full.sizes.Y == 11
        assert da_full.sizes.X == 17
        temp_dir = Path(os.getenv("RUNNER_TEMP", "."))
        da_full.to_netcdf(temp_dir / "test_full.nc")
        da_full.to_netcdf(temp_dir / "test_full.nc", group="/GROUP")
        da_full.to_netcdf(
            temp_dir / "test_full.nc",
            group="/GROUP",
            attrs_group={"title": "data from DataArray 'da_full'"},
        )

    def test_creation(
        self: TestDataArray,
        da_from_coords: DataArray,
        da_from_dict: DataArray,
        da_from_tuple: DataArray,
        da_from_list_dims: DataArray,
        da_from_dims: DataArray,
    ) -> None:
        """Unit-test for different DataArray creation settings."""
        assert da_from_coords == da_from_dict
        assert da_from_coords == da_from_tuple
        assert da_from_list_dims == da_from_dims
        with pytest.raises(ValueError, match=r"No coordinates .*") as excinfo:
            _ = DataArray(np.ones((24, 5, 7)), dims=("column", "row"))
        assert "No coordinates or dimensions" in str(excinfo.value)
        with pytest.raises(ValueError, match=r".* equal length") as excinfo:
            _ = DataArray(
                np.ones((24, 5, 7)),
                dims=("column", "row"),
                coords=[np.arange(24), np.arange(5), list(range(11))],
            )
        assert "coords and dims must be of equal length" in str(excinfo.value)
        with pytest.raises(ValueError, match=r"No coordinates .*") as excinfo:
            _ = DataArray(
                np.ones((24, 5, 7)),
                dims=("column", "row"),
                coords=[np.arange(24), np.arange(5)],
            )
        assert "No coordinates or dimensions" in str(excinfo.value)

    def test_getitem(self: TestDataArray, da_full: DataArray) -> None:
        """Unit-test for method DataArray.__getitem__()."""
        assert da_full[...] == da_full
        assert da_full[:] == da_full
        assert np.array_equal(
            da_full[:, 3, 5].values,
            np.array([[[56.0]], [[243.0]], [[430.0]], [[617.0]], [[804.0]]]),
        )
        # add an auxiliary coordinate
        da_full.add_coord("T", ["orbit", list(range(5, 10))])
        assert "T" in da_full.get_coords
        assert da_full[:, 3, 5].shape == (5, 1, 1)
        assert da_full[2:, 3:, :5].shape == (3, 8, 5)
        assert da_full[2, :, :].shape == (1, 11, 17)

    def test_sel(self: TestDataArray, da_full: DataArray) -> None:
        """Unit-test for method DataArray.sel()."""
        mask_y = np.zeros(11, dtype=bool)
        mask_y[3] = True
        mask_x = np.zeros(17, dtype=bool)
        mask_x[5] = True
        assert np.array_equal(
            da_full.sel(Y=mask_y, X=mask_x).values,
            np.array([[[56.0]], [[243.0]], [[430.0]], [[617.0]], [[804.0]]]),
        )
        # add auxilary coordinate
        da_full.add_coord("T", ["Y", list(range(10, 21))])
        assert da_full.sel(Y=mask_y, X=mask_x).get_coords["Y"].values == np.array([3])
        assert da_full.sel(Y=mask_y, X=mask_x).get_coords["T"].values == np.array([13])

    def test_sortby(self: TestDataArray) -> None:
        """Unit-test for sortby method."""
        da_01 = DataArray(
            np.arange(13),
            coords={
                "X": [0, 3, 4, 5, 6, 7, 8, 9, 2, 1, 10, 12, 11],
                "orbit": (
                    "X",
                    [
                        2300,
                        2309,
                        2308,
                        2301,
                        2302,
                        2303,
                        2304,
                        2305,
                        2306,
                        2307,
                        2310,
                        2312,
                        2311,
                    ],
                ),
            },
        )
        da_02 = da_01.sortby("X")
        aa = da_02.get_coords["X"].values
        assert np.all(aa[:-1] <= aa[1:])
        da_02 = da_01.sortby("orbit")
        aa = da_02.get_coords["orbit"].values
        assert np.all(aa[:-1] <= aa[1:])
        aa = np.array([0, 3, 4, 5, 6, 7, 8, 9, 2, 1, 10, 12, 11])
        assert np.array_equal(aa, da_02.values)

        da_01 = DataArray(
            np.arange(13 * 5 * 3).reshape(13, 5, 3),
            coords={
                "Z": [0, 3, 4, 5, 6, 7, 8, 9, 2, 1, 10, 12, 11],
                "Y": list(range(5)),
                "X": list(range(3)),
                "orbit": (
                    "Z",
                    [
                        2300,
                        2309,
                        2308,
                        2301,
                        2302,
                        2303,
                        2304,
                        2305,
                        2306,
                        2307,
                        2310,
                        2312,
                        2311,
                    ],
                ),
            },
        )
        _ = da_01.sortby("Z")
        _ = da_01.sortby("orbit")
        _ = da_01.sortby("Y")

    def test_add(self: TestDataArray, da_full: DataArray, da_ones: DataArray) -> None:
        """Unit-test for add method."""
        # Add numpy array to DataArray
        xda = da_full + da_ones.values
        assert np.array_equal(xda.values, da_full.values + da_ones.values)
        # Add values of DataArray array to DataArray
        xda = da_full + da_ones
        assert np.array_equal(xda.values, da_full.values + da_ones.values)

    def test_sub(self: TestDataArray, da_full: DataArray, da_ones: DataArray) -> None:
        """Unit-test for subtract method."""
        # Subtract numpy array from DataArray
        xda = da_full - da_ones.values
        assert np.array_equal(xda.values, da_full.values - da_ones.values)
        # Subtract values of DataArray from DataArray
        xda = da_full - da_ones
        assert np.array_equal(xda.values, da_full.values - da_ones.values)

    def test_mul(self: TestDataArray, da_full: DataArray) -> None:
        """Unit-test for multiply method."""
        # Multiply numpy array with DataArray
        xda = da_full * da_full.values
        assert np.array_equal(xda.values, da_full.values * da_full.values)
        # Multiply values of DataArray with DataArray
        xda = da_full * da_full
        assert np.array_equal(xda.values, da_full.values * da_full.values)

    def test_div(self: TestDataArray, da_full: DataArray) -> None:
        """Unit-test for division method."""
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

    def test_mean(self: TestDataArray, da_full: DataArray) -> None:
        """Unit-test for mean method."""
        assert da_full.mean().values == np.mean(da_full.values)
        assert np.array_equal(
            da_full.mean(dim="orbit").values, np.mean(da_full.values, axis=0)
        )
        da_full.values[:, 3, 5] = np.nan
        assert da_full.mean(skipna=True).values == np.nanmean(da_full.values)
        with pytest.raises(ValueError, match="invalid dimension") as excinfo:
            _ = da_full.mean(dim="Z").values, np.mean(da_full.values, axis=0)
        assert "invalid dimension" in str(excinfo.value)

    def test_median(self: TestDataArray, da_full: DataArray) -> None:
        """Unit-test for median method."""
        assert da_full.median().values == np.median(da_full.values)
        assert np.array_equal(
            da_full.median(dim="Y").values, np.median(da_full.values, axis=1)
        )
        da_full.values[:, 3, 5] = np.nan
        assert da_full.median(skipna=True).values == np.nanmedian(da_full.values)
        with pytest.raises(ValueError, match="invalid dimension") as excinfo:
            _ = da_full.median(dim="Z").values, np.median(da_full.values, axis=0)
        assert "invalid dimension" in str(excinfo.value)

    def test_std(self: TestDataArray, da_full: DataArray) -> None:
        """Unit-test for standard-deviation method."""
        assert da_full.std(ddof=0).values == np.std(da_full.values, ddof=0)
        assert np.array_equal(
            da_full.std(dim="X").values, np.std(da_full.values, axis=2)
        )
        da_full.values[:, 3, 5] = np.nan
        assert da_full.std(skipna=True, ddof=1).values == np.nanstd(
            da_full.values, ddof=1
        )
        with pytest.raises(ValueError, match="invalid dimension") as excinfo:
            _ = da_full.std(dim="Z").values, np.std(da_full.values, axis=0)
        assert "invalid dimension" in str(excinfo.value)
