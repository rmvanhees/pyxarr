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

from pathlib import Path

import numpy as np
import pytest

from pyxarr import Coords, DataArray, Dataset


class TestDataset:
    """Class to test pyxarr.Dataset."""

    def test_empty(self: TestDataset) -> None:
        """Unit-test for empty Dataset."""
        assert not bool(Dataset())
        assert len(Dataset()) == 0
        assert "scalar" not in Dataset()
        assert not Dataset().asdict()

    def test_scalar(self: TestDataset, da_scalar: DataArray) -> None:
        """Unit-test for Dataset with scalar DataArray."""
        ds_scalar = Dataset({"scalar": da_scalar})
        assert bool(ds_scalar)
        assert len(ds_scalar) == 1
        assert "scalar" in ds_scalar
        assert not ds_scalar.asdict()["dimensions"]
        assert "compounds" not in ds_scalar.asdict()
        assert "scalar" in ds_scalar.asdict()["variables"]
        assert "groups" not in ds_scalar.asdict()
        assert "scalar" in ds_scalar.asdict()["variables"]
        assert "/GROUP" in ds_scalar.asdict("/GROUP")["groups"]
        assert "/GROUP/scalar" in ds_scalar.asdict("/GROUP")["variables"]

    def test_cmp(self: TestDataset, da_compound: DataArray) -> None:
        """Unit-test for DataArray with structured array."""
        ds_compound = Dataset({"foo": da_compound})
        assert bool(ds_compound)
        assert len(ds_compound) == 1
        assert "foo" in ds_compound
        assert "orbit" in ds_compound.asdict()["dimensions"]
        assert "compound_arr_dtype" in ds_compound.asdict()["compounds"]
        assert "groups" not in ds_compound.asdict()
        assert "foo" in ds_compound.asdict()["variables"]
        assert "/GROUP" in ds_compound.asdict("/GROUP")["groups"]
        assert "/GROUP/foo" in ds_compound.asdict("/GROUP")["variables"]

    def test_two(self: TestDataset, da_full: DataArray, da_ones: DataArray) -> None:
        """Unit-test for Dataset with two 3-D DataArray."""
        ds_two = Dataset({"foo": da_full, "bar": da_ones})
        assert bool(ds_two)
        assert len(ds_two) == 2
        assert "bar" in ds_two
        assert ds_two["bar"] == da_ones
        assert ds_two["scalar"] is None
        assert "orbit" in ds_two.asdict()["dimensions"]
        assert "foo" in ds_two.asdict()["variables"]
        assert "bar" in ds_two.asdict()["variables"]
        assert "groups" not in ds_two.asdict()
        assert "/GROUP" in ds_two.asdict("/GROUP")["groups"]
        assert "/GROUP/orbit" in ds_two.asdict("/GROUP")["dimensions"]
        assert "/GROUP/foo" in ds_two.asdict("/GROUP")["variables"]
        ds_two.to_netcdf("test_two1.nc")
        Path("test_two1.nc").unlink()
        ds_two.to_netcdf("test_two2.nc", group="/GROUP")
        Path("test_two2.nc").unlink()
        ds_two.to_netcdf(
            "test_two3.nc",
            group="/GROUP",
            attrs_group={
                "/GROUP/title": "data from DataArrays 'da_full' and 'da_ones'"
            },
        )
        Path("test_two3.nc").unlink()

    def test_creation(
        self: TestDataset, da_full: DataArray, da_ones: DataArray
    ) -> None:
        """Unit-test for Dataset with two 3-D DataArray and __eq__ method."""
        ds_add = Dataset()
        ds_add["foo"] = da_full
        ds_add["bar"] = da_ones
        ds_dict = Dataset({"foo": da_full, "bar": da_ones})
        assert isinstance(ds_add.coords, Coords)
        assert isinstance(ds_dict.coords, Coords)
        assert ds_add == ds_dict
        ds_add = Dataset()
        ds_add["fff"] = da_full
        ds_add["bar"] = da_ones
        assert ds_add != ds_dict
        ds_add = Dataset()
        ds_add["foo"] = da_ones
        ds_add["bar"] = da_ones
        assert ds_add != ds_dict
        ds_add = Dataset()
        with pytest.raises(ValueError, match=r".* only add DataArrays .*") as excinfo:
            ds_add["foo"] = np.ones((24, 1, 17))
        assert "you can only add DataArrays to a Dataset" in str(excinfo.value)
