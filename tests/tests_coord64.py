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
"""Definition of pyxarr function `dset_from_h5`."""

from __future__ import annotations

__all__ = ["dset_from_h5"]


import h5py
import numpy as np

from pyxarr.dset_from_h5 import dset_from_h5

#    tstamp = (
#        np.datetime64("2025-07-28T07:30:00.000")
#        + (1e3 * np.linspace(0, 120, 361)).astype("timedelta64[ms]")
#    )


def test_01() -> None:
    """..."""
    # with h5py.File.in_memory() as fid:
    with h5py.File("test_01.h5", "w") as fid:
        days = np.arange(31, dtype="u4")
        dset = fid.create_dataset("time", data=days)
        dset.attrs["standard_name"] = "time"
        dset.attrs["units"] = "days since 2024-07-01"
        dset.make_scale()

        values = np.arange(100, 100 + len(days), dtype="f4")
        dset = fid.create_dataset("values", data=values)
        for dim in dset.dims:
            dim.attach_scale(fid["time"])

    with h5py.File("test_01.h5") as fid:
        dset = fid["values"]
        xda = dset_from_h5(dset)
        print(xda)


if __name__ == "__main__":
    test_01()
