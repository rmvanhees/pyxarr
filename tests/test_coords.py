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
import pytest

from pyxarr.lib.coords import Coords, _Coord


@pytest.fixture
def column() -> _Coord:
    """Return instance of class _Coord."""
    return _Coord("column", list(range(13)), {"units": "px"})


@pytest.fixture
def three_co() -> Coords:
    """Return instance of class Coords with 3 dimensions."""
    co_dict = {
        "time": np.arange("2025-02", "2025-03", dtype="datetime64[D]"),
        "row": np.arange(5),
        "column": list(range(11)),
    }
    return Coords(co_dict)


class TestCoord:
    """Class to test _Coord from pyxarr.lib.coords."""

    def test_bool(self: TestCoord, column: _Coord) -> None:
        """Unit-test for bool method."""
        assert not bool(_Coord())
        assert bool(column)

    def test_len(self: TestCoord, column: _Coord) -> None:
        """Unit-test for len method."""
        assert len(_Coord()) == 0
        assert len(column) == 13

    def test_getitem(self: TestCoord, column: _Coord) -> None:
        """Unit-test for getitem method."""
        indx = 11
        assert len(column[indx]) == 1
        mask = np.array(3 * [False] + 3 * [True] + 7 * [False], dtype=bool)
        assert len(column[mask]) == 3

    def test_setitem(self: TestCoord, column: _Coord) -> None:
        """Unit-test for setitem method."""
        assert column.values[-1] == 12
        column[-1] = 15
        assert column.values[-1] == 15
        assert column == column.copy()


class TestCoords:
    """Class to test Coords from pyxarr.lib.coords."""

    def test_bool(self: TestCoords, three_co: Coords) -> None:
        """Unit-test for bool method."""
        assert not bool(Coords())
        assert bool(three_co)

    def test_contains(self: TestCoords, three_co: Coords) -> None:
        """Unit-test for contains method."""
        assert "x" not in Coords()
        assert "time" in three_co
        assert "row" in three_co
        assert "column" in three_co
        assert "x" not in three_co

    def test_getitem(self: TestCoords, three_co: Coords) -> None:
        """Unit-test for getitem method."""
        assert Coords()["x"] is None
        assert three_co["x"] is None
        assert np.array_equal(three_co["column"].values, np.arange(11))

    def test_len(self: TestCoords, three_co: Coords) -> None:
        """Unit-test for len method."""
        assert len(Coords()) == 0
        assert len(three_co) == 3

    def test_add(self: TestCoords, three_co: Coords) -> None:
        """Unit-test for add method."""
        empty_co = Coords()
        empty_co += ("row", np.arange(5))
        assert "row" in empty_co
        assert len(empty_co["row"]) == 5
        empty_co += three_co["column"]
        with pytest.raises(ValueError, match="overwrite") as excinfo:
            empty_co += ("row", np.arange(5))
        assert "do not try to overwrite a coordinate" in str(excinfo.value)
        with pytest.raises(ValueError, match="overwrite") as excinfo:
            empty_co += three_co["row"]
        assert "do not try to overwrite a coordinate" in str(excinfo.value)
