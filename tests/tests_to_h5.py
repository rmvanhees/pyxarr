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
"""Test module for `dset_to_h5`."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from pyxarr import DataArray
from pyxarr.dset_from_h5 import dset_from_h5
from pyxarr.dset_to_h5 import dset_to_h5


def test_wr() -> None:
    """..."""
    xda = DataArray(
        np.arange(11 * 23, dtype="f4").reshape(11, 23),
        dims=("row", "column"),
        coords=(np.arange(11, dtype="u2"), np.arange(23, dtype="u2")),
        name="image",
        attrs={
            "long_name": "Just some data",
            "units": "1",
        },
    )
    co_dict = {
        "time": (
            np.datetime64("2025-07-28T07:30:00").astype("datetime64[ns]")
            + (1e9 * np.linspace(0, 120, 361)).astype("timedelta64[ns]")
        ),
    }
    xdb = DataArray(
        np.arange(361),
        coords=co_dict,
        name="position",
    )

    Path.unlink("test_wr_1.h5", missing_ok=True)
    try:
        with h5py.File("test_wr_1.h5") as fid:
            dset_to_h5(fid, xda)
    except FileNotFoundError as exc:
        print(f"an expected error: {exc}")

    with h5py.File("test_wr_1.h5", "w") as fid:
        dset_to_h5(fid, xda)
    with h5py.File("test_wr_1.h5", "r+") as fid:
        dset_to_h5(fid, xdb)
        print(dset_from_h5(fid[xdb.name]))

    with h5py.File("test_wr_2.h5", "w") as fid:
        dset_to_h5(fid, xda, dest_group="group_01")

    try:
        with h5py.File("test_wr_1.h5") as fid:
            dset_to_h5(fid, xda)
    except PermissionError as exc:
        print(f"an expected error: {exc}")


if __name__ == "__main__":
    test_wr()
