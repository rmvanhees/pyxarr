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

import numpy as np

from pyxarr import DataArray


# - test module ------------------------------------
def tests() -> None:
    """Run tests on class DataArray."""
    # - empty class Coords
    xda = DataArray()
    print(f"\nCreate an empty DataArray:\n{xda}")
    print("- boolean test:", bool(xda))
    print("- length:", len(xda))
    print("- shape:", xda.shape)
    print("- size:", xda.size)

    xda = DataArray(1)
    print(f"\nCreate a DataArray with scalar:\n{xda}")
    print("- boolean test:", bool(xda))
    print("- length:", len(xda))
    print("- shape:", xda.shape)
    print("- size:", xda.size)

    xda = DataArray(list(range(11)))
    print(f"\nCreate a DataArray with list:\n{xda}")
    print("- boolean test:", bool(xda))
    print("- length:", len(xda))
    print("- shape:", xda.shape)
    print("- size:", xda.size)

    xda = DataArray(np.arange(3 * 5).reshape(3, 5))
    print(f"\nCreate a DataArray with numpy:\n{xda}")
    print("- boolean test:", bool(xda))
    print("- length:", len(xda))
    print("- shape:", xda.shape)
    print("- size:", xda.size)
    print(f"- select one element [1, 3]:\n{xda[1, 3]}")
    print(f"- select one X-element:[:, 3]\n{xda[:, 3]}")
    print(f"- select one Y-element [1, :]:\n{xda[1, :]}")
    print(f"- select all elements [()]:\n{xda[()]}")
    print(f"- select all elements [:, :]:\n{xda[:, :]}")
    print(f"- select all elements Elipis:\n{xda[...]}")

    xda = DataArray(np.arange(5 * 3 * 7).reshape(5, 3, 7), dims=("orbit", "y", "x"))
    xda.coords += ("time", np.arange("2025-07-01", "2025-07-06", dtype="datetime64[D]"))
    print(f"\nCreate a DataArray (3-D) and add coordinate 'time':\n{xda}")
    xda.swap_dims("time", "orbit")
    print(f"- change dims to 'time', 'y', 'x':\n{xda}")


if __name__ == "__main__":
    tests()
