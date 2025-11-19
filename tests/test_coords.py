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


class TestCoord:
    """Class to test _Coord from pyxarr.lib.coords."""

    def test_bool(self: TestCoord, co_struct: _Coord) -> None:
        """Unit-test for bool method."""
        assert not bool(_Coord())
        assert bool(co_struct)

    def test_len(self: TestCoord, co_struct: _Coord) -> None:
        """Unit-test for len method."""
        assert len(_Coord()) == 0
        assert len(co_struct) == 13

    def test_getitem(self: TestCoord, co_struct: _Coord) -> None:
        """Unit-test for getitem method."""
        indx = 11
        assert len(co_struct[indx]) == 1
        mask = np.array(3 * [False] + 3 * [True] + 7 * [False], dtype=bool)
        assert len(co_struct[mask]) == 3

    def test_setitem(self: TestCoord, co_struct: _Coord) -> None:
        """Unit-test for setitem method."""
        assert co_struct.values[-1] == 12
        co_struct[-1] = 15
        assert co_struct.values[-1] == 15
        assert co_struct == co_struct.copy()


class TestCoords:
    """Class to test Coords from pyxarr.lib.coords."""

    def test_bool(self: TestCoords, co_from_dict: Coords) -> None:
        """Unit-test for bool method."""
        assert not bool(Coords())
        assert bool(co_from_dict)

    def test_contains(self: TestCoords, co_from_dict: Coords) -> None:
        """Unit-test for contains method."""
        assert "x" not in Coords()
        assert "time" in co_from_dict
        assert "row" in co_from_dict
        assert "column" in co_from_dict
        assert "x" not in co_from_dict

    def test_getitem(self: TestCoords, co_from_dict: Coords) -> None:
        """Unit-test for getitem method."""
        assert Coords()["x"] is None
        assert co_from_dict["x"] is None
        assert np.array_equal(co_from_dict["column"].values, np.arange(11))

    def test_len(self: TestCoords, co_from_dict: Coords) -> None:
        """Unit-test for len method."""
        assert len(Coords()) == 0
        assert len(co_from_dict) == 3

    def test_add(self: TestCoords, co_from_dict: Coords) -> None:
        """Unit-test for add method."""
        empty_co = Coords()
        empty_co += ("row", np.arange(5))
        assert "row" in empty_co
        assert len(empty_co["row"]) == 5
        empty_co += co_from_dict["column"]
        with pytest.raises(ValueError, match="overwrite") as excinfo:
            empty_co += ("row", np.arange(5))
        assert "do not try to overwrite a coordinate" in str(excinfo.value)
        with pytest.raises(ValueError, match="overwrite") as excinfo:
            empty_co += co_from_dict["row"]
        assert "do not try to overwrite a coordinate" in str(excinfo.value)
