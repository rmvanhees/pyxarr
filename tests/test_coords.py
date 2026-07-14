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
"""Test module for pyxarr class `Coords`."""

from __future__ import annotations

import numpy as np
import pytest

from pyxarr.lib.coords import Coords, _Coord


class TestCoord:
    """Class to test _Coord from pyxarr.lib.coords."""

    def test_bool(self: TestCoord, co_struct: _Coord) -> None:
        """Unit-test for bool method."""
        assert bool(co_struct)

    def test_len(self: TestCoord, co_struct: _Coord) -> None:
        """Unit-test for len method."""
        assert len(co_struct) == 13

    def test_getitem(self: TestCoord, co_struct: _Coord) -> None:
        """Unit-test for getitem method."""
        indx = 11
        assert len(co_struct[indx]) == 1
        assert len(co_struct[3:7]) == 4
        assert np.sum(co_struct[3:7].values) == 18
        mask = np.array(3 * [False] + 3 * [True] + 7 * [False], dtype=bool)
        assert len(co_struct[mask]) == 3
        assert np.sum(co_struct[mask].values) == 12

    def test_setitem(self: TestCoord, co_struct: _Coord) -> None:
        """Unit-test for setitem method."""
        assert co_struct.values[-1] == 12
        co_struct[-1] = 15
        assert co_struct.values[-1] == 15
        assert co_struct == co_struct.copy()

    def test_is_dim(self: TestCoord) -> None:
        """Unit-test for method _Coord.is_dimension()."""
        assert _Coord("time", list(range(17)), "time").is_dimension
        assert not _Coord("orbit", list(range(17)), "time").is_dimension


class TestCoords:
    """Class to test Coords from pyxarr.lib.coords."""

    def test_creation(
        self: TestCoords,
        co_from_dict: Coords,
        co_from_tuple: Coords,
        co_from_obj: Coords,
    ) -> None:
        """Unit-test to compare different creation options."""
        assert co_from_dict == co_from_tuple
        assert co_from_dict == co_from_obj

    def test_bool(self: TestCoords, co_from_dict: Coords) -> None:
        """Unit-test for bool method."""
        assert not bool(Coords())
        assert bool(co_from_dict)

    def test_contains(self: TestCoords, co_from_dict: Coords) -> None:
        """Unit-test for contains method."""
        assert "x" not in Coords()
        assert "x" not in co_from_dict
        assert "time" in co_from_dict
        assert "row" in co_from_dict
        assert "column" in co_from_dict
        assert co_from_dict["column"] in co_from_dict

    def test_eq(self: TestCoords, co_from_dict: Coords) -> None:
        """Unit-test for len method."""
        co_tuple = (
            _Coord(
                "time", np.arange("2025-02-02", "2025-02-26", dtype="datetime64[D]")
            ),
        )
        assert co_from_dict != Coords(co_tuple)
        co_tuple = (
            _Coord(
                "time", np.arange("2025-02-02", "2025-02-26", dtype="datetime64[D]")
            ),
            _Coord("Y", np.arange(5)),
            _Coord("Z", list(range(11))),
        )
        assert co_from_dict != Coords(co_tuple)

    def test_len(self: TestCoords) -> None:
        """Unit-test for len method."""
        coords = Coords(
            [
                ("time", list(range(7))),
                ("Y", list(range(3))),
                ("X", list(range(5))),
            ]
        )
        assert len(Coords()) == 0
        assert len(coords) == 3

    def test_add(self: TestCoords, co_from_dict: Coords) -> None:
        """Unit-test for method Coords.__add__()."""
        empty_co = Coords()
        empty_co += ("row", np.arange(5))
        assert "row" in empty_co
        assert len(empty_co["row"]) == 5
        empty_co += co_from_dict["column"]

        with pytest.raises(KeyError, match=r"can't modify .*") as excinfo:
            empty_co += ("row", np.arange(5))
        assert "can't modify existing coordinate" in str(excinfo.value)
        with pytest.raises(KeyError, match=r"can't modify .*") as excinfo:
            empty_co += co_from_dict["row"]
        assert "can't modify existing coordinate" in str(excinfo.value)

    def test_getitem(self: TestCoords, co_from_dict: Coords) -> None:
        """Unit-test for getitem method."""
        assert Coords()["x"] is None
        assert co_from_dict["x"] is None
        assert np.array_equal(co_from_dict["column"].values, np.arange(11))

    def test_setitem(self: TestCoords) -> None:
        """Unit-test for method Coords.setitem()."""
        coords = Coords(
            [
                ("time", list(range(7))),
                ("Y", list(range(3))),
                ("X", list(range(5))),
            ]
        )
        assert "time" in coords
        assert "Y" in coords
        assert "X" in coords
        assert "orbit" not in coords
        coords["orbit"] = ["time", list(range(2222, 2229))]
        assert "orbit" in coords
        with pytest.raises(KeyError, match=r"can't modify .*") as excinfo:
            coords["orbit"] = ["time", list(range(2222, 2229))]
        assert "can't modify existing coordinate" in str(excinfo.value)

    def test_ndim(self: TestCoords) -> None:
        """Unit-test for method Coords.__ndim__()."""
        coords = Coords(
            [
                ("time", list(range(7))),
                ("Y", list(range(3))),
                ("X", list(range(5))),
            ]
        )
        assert coords.ndim == 3
        coords["orbit"] = ["time", list(range(2222, 2229))]
        assert coords.ndim == 3
