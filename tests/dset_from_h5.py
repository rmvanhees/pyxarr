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
"""Unit tests for the pyxarr function `dset_from_h5`."""

from __future__ import annotations

import h5py
import numpy as np

from pyxarr import dset_from_h5


def test_1d() -> None:
    """Check read of a 1-D dataset."""
    rng = np.random.default_rng()

    flname = "xarr_test_1d.h5"
    # ++++++++++++++++++++++++++++++++++++++++++++++++++
    with h5py.File(flname, "w") as fid:
        fid.create_dataset("array_1d", data=rng.random(128))

    with h5py.File(flname, "r") as fid:
        print("# Test 1-D, no dims", dset_from_h5(fid["array_1d"]))

    # ++++++++++++++++++++++++++++++++++++++++++++++++++
    with h5py.File(flname, "w") as fid:
        dset = fid.create_dataset("time", data=np.arange(128).astype("f8"))
        dset.attrs["long_name"] = "timestamp of housekeeping data"
        dset.attrs["standard_name"] = "time"
        dset.attrs["calendar"] = "proleptic_gregorian"
        dset.attrs["coverage_content_type"] = "coordinate"
        dset.attrs["units"] = "seconds since 2025-07-28 00:00:00"
        dset.attrs.create("valid_min", 0, dtype="f8")
        dset.attrs.create("valid_max", 92400, dtype="f8")
        dset = fid.create_dataset("array_1d", data=rng.random(128))
        dset.dims[0].attach_scale(fid["time"])

    with h5py.File(flname, "r") as fid:
        print("\n# Test 1-D, with dims", dset_from_h5(fid["array_1d"]))

    # ++++++++++++++++++++++++++++++++++++++++++++++++++
    with h5py.File(flname, "w") as fid:
        dset = fid.create_dataset("time", data=np.arange(128).astype("u4"))
        dset.attrs["long_name"] = "timestamp of housekeeping data"
        dset.attrs["standard_name"] = "time"
        dset.attrs["calendar"] = "proleptic_gregorian"
        dset.attrs["coverage_content_type"] = "coordinate"
        dset.attrs["units"] = "days since 2025-07-01"
        dset = fid.create_dataset("array_1d", data=rng.random(128))
        dset.dims[0].attach_scale(fid["time"])

    with h5py.File(flname, "r") as fid:
        print("\n# Test 1-D, with dims", dset_from_h5(fid["array_1d"]))

    flname = "xarr_test_3d.h5"
    # ++++++++++++++++++++++++++++++++++++++++++++++++++
    with h5py.File(flname, "w") as fid:
        dset = fid.create_dataset("time", data=np.arange(128).astype("u4"))
        dset.attrs["long_name"] = "timestamp of housekeeping data"
        dset.attrs["standard_name"] = "time"
        dset.attrs["calendar"] = "proleptic_gregorian"
        dset.attrs["coverage_content_type"] = "coordinate"
        dset.attrs["units"] = "days since 2025-07-01"
        dset = fid.create_dataset("row", data=np.arange(11).astype("u2"))
        dset = fid.create_dataset("column", data=np.arange(17).astype("u2"))
        dset = fid.create_dataset("array_3d", data=rng.random((128, 11, 17)))
        dset.dims[0].attach_scale(fid["time"])
        dset.dims[1].attach_scale(fid["row"])
        dset.dims[2].attach_scale(fid["column"])

    with h5py.File(flname, "r") as fid:
        print("\n# Test 3-D, with dims", dset_from_h5(fid["array_3d"]))


if __name__ == "__main__":
    test_1d()
