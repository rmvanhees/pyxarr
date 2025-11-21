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

from pyxarr import DataArray, Dataset


class TestDataset:
    """Class to test pyxarr.Dataset."""

    def test_empty(self: TestDataset) -> None:
        """Unit-test for empty Dataset."""
        assert not bool(Dataset())
        assert len(Dataset()) == 0
        assert "scalar" not in Dataset()

    def test_scalar(self: TestDataset, da_scalar: DataArray) -> None:
        """Unit-test for Dataset with scalar DataArray."""
        ds_scalar = Dataset({"scalar": da_scalar})
        print(ds_scalar)
        assert bool(ds_scalar)
        assert len(ds_scalar) == 1
        assert "scalar" in ds_scalar

    def test_one(self: TestDataset, da_full: DataArray) -> None:
        """Unit-test for Dataset with one 3-D DataArray."""
        ds_full = Dataset({"foo": da_full})
        assert bool(ds_full)
        assert len(ds_full) == 1
        assert "foo" in ds_full

    def test_two(self: TestDataset, da_full: DataArray, da_ones: DataArray) -> None:
        """Unit-test for Dataset with two 3-D DataArray."""
        ds_full = Dataset({"foo": da_full, "bar": da_ones})
        assert bool(ds_full)
        assert len(ds_full) == 2
        assert "bar" in ds_full
        assert ds_full["bar"] == da_ones
        assert ds_full["scalar"] is None
