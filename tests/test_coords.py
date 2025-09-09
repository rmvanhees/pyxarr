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
"""Test module for pyxarr class `Coords`."""

from __future__ import annotations

import numpy as np

from pyxarr import Coords


def tests_coords() -> None:
    """Run tests on local class Coords."""
    # - empty class Coords
    print("\nCreate an empty Coords:", Coords())
    print("- boolean test:", bool(Coords()))
    print("- size:", len(Coords()))

    print("\nFill a Coords instance")
    co_dict = {
        "time": np.arange("2025-02", "2025-03", dtype="datetime64[D]"),
        "row": np.arange(5),
        "column": list(range(11)),
    }
    coords = Coords(co_dict)
    # print(
    #    "- add time-coordinate:",
    #    _Coord("time", np.arange("2025-02", "2025-03", dtype="datetime64[D]")),
    # )
    # print("- add Y-coordinate:", _Coord(name="row", values=np.arange(5)))
    # print("- add X-coordinate:", _Coord(name="column", values=np.arange(11)))
    print("- check if 'row' in Coords:", "row" in coords)
    print("- check if 'y' in Coords:", "y" in coords)
    print("- existing dimension:", coords["time"])
    print("- non-existing dimension:", coords["y"])
    co_list = [
        ("time", np.arange("2025-02", "2025-03", dtype="datetime64[D]")),
        ("row", list(range(5))),
        ("column", np.arange(11)),
    ]
    coords = Coords(co_list)
    print("- perform loop iteration:")
    for coord in coords:
        print(coord)


def tests() -> None:
    """..."""
    # Run tests on local class Coord
    tests_coords()


if __name__ == "__main__":
    tests()
