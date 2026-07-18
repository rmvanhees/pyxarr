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
"""Unit-tests for pyxarr module `dset_from_h5.py`."""

from __future__ import annotations

from importlib.resources import files

import h5py
import numpy as np
import pytest
from h5yaml.template_h5 import TemplateH5

from pyxarr.dset_from_h5 import dset_from_h5


class TestFromH5:
    """Class with unit-tests for pyxarr.dset_from_h5."""

    _res = TemplateH5(
        [
            files("h5yaml.Data") / "h5_testing.yaml",
            files("h5yaml.Data") / "h5_global_attrs.yaml",
        ]
    )
    _res.set_dims({"number_of_images": 10})
    H5_DEF = _res.asdict
    FID_H5 = _res.diskless()

    def test_full(self: TestFromH5) -> None:
        """Unit-test ..."""
        for key in self.H5_DEF["variables"]:
            _ = dset_from_h5(self.FID_H5[key])
            # try time_units
            for val in ["ns", "us", "ms", "s"]:
                _ = dset_from_h5(self.FID_H5[key], time_units=val)
            if "number_of_images" in self.H5_DEF["variables"][key]["_dims"]:
                _ = dset_from_h5(self.FID_H5[key], data_sel=np.s_[3:7, ...])
            if (
                self.H5_DEF["variables"][key]["_dtype"] == "stats_dtype"
                and "_vlen" not in self.H5_DEF["variables"][key]
            ):
                _ = dset_from_h5(self.FID_H5[key], field="index")

        with pytest.raises(ValueError, match=r".*does not appear.*") as excinfo:
            _ = dset_from_h5(self.FID_H5["/group_01/stats"], field="notFound")
        assert "Field notFound does not appear in this type" in str(excinfo.value)

        with pytest.raises(RuntimeError, match=r".* convert timestamps") as excinfo:
            _ = dset_from_h5(self.FID_H5["/group_01/stats"], time_units="ps")
        assert "Failed to convert timestamps" in str(excinfo.value)

    def test_close(self: TestFromH5) -> None:
        """Close the in-memory HDF5 file."""
        self.FID_H5.close()


class TestCoordFromH5:
    """Class with unit-tests for pyxarr.dset_from_h5 (dimension scales)."""

    with h5py.File.in_memory() as fid:
        dset = fid.create_dataset("row", data=np.arange(11, dtype="u2"))
        dset.make_scale("X")
        dset = fid.create_dataset("column", data=np.arange(13, dtype="u2"))
        dset.make_scale("Y")
        # define dataset without dimension scales
        dset = fid.create_dataset(
            "ds_00",
            data=np.arange(11 * 13, dtype="u2").reshape(11, 13),
        )
        dset.dims[0].label = "Y"
        dset.dims[1].label = "X"
        # define dataset with dimension scales and attribute
        dset = fid.create_dataset(
            "ds_01",
            data=np.arange(11 * 13, dtype="u2").reshape(11, 13),
        )
        dset.dims[0].attach_scale(fid["row"])
        dset.dims[1].attach_scale(fid["column"])
        dset.attrs["list_one"] = (3.14,)
        # define dataset with 4 dimensions
        dset = fid.create_dataset(
            "ds_02",
            data=np.arange(3 * 11 * 13 * 7, dtype="u2").reshape(3, 11, 13, 7),
        )
        # define dataset with timescale dimension scale
        dset = fid.create_dataset("time", data=np.arange(31, dtype="u2"))
        dset.attrs["units"] = "days since 2027-03-01T00:00:00"
        dset = fid.create_dataset(
            "ds_03",
            data=np.arange(31, dtype="f4"),
        )
        dset.dims[0].attach_scale(fid["time"])

        res = dset_from_h5(fid["ds_00"])
        assert res.shape == (11, 13)
        res = dset_from_h5(fid["ds_00"], data_sel=np.s_[3, 5])
        assert res.shape == (1, 1)
        res = dset_from_h5(fid["ds_00"], dim_names=["row", "column"])
        res = dset_from_h5(fid["ds_01"])
        res = dset_from_h5(fid["ds_02"], dim_names=["T", "Z", "Y", "X"])
        with pytest.raises(RuntimeError, match=r"Failed .*") as excinfo:
            _ = dset_from_h5(fid["ds_02"])
        assert "Failed to read dataset dimensions" in str(excinfo.value)
        # perform tests on time_units="D" and data_sel
        res = dset_from_h5(fid["ds_03"], time_units="D", data_sel=np.s_[:])
        res = dset_from_h5(fid["ds_03"], time_units="D", data_sel=np.s_[15])


def main() -> None:
    """..."""
    obj = TestFromH5()
    obj.test_full()


if __name__ == "__main__":
    main()
