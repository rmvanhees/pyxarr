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

from pyxarr import Coords


@pytest.fixture
def empty_co() -> Coords:
    """Return empty instance of class Coords."""
    return Coords()


@pytest.fixture
def three_co() -> Coords:
    """Return instance of class Coords with 3 dimensions."""
    co_dict = {
        "time": np.arange("2025-02", "2025-03", dtype="datetime64[D]"),
        "row": np.arange(5),
        "column": list(range(11)),
    }
    return Coords(co_dict)


class TestCoords:
    """Class to test pyxarr.Coords."""

    def test_empty(self: TestCoords, empty_co: Coords) -> None:
        """..."""
        assert not bool(empty_co)
        assert "x" not in empty_co
        assert len(empty_co) == 0

    def test_3d(self: TestCoords, three_co: Coords) -> None:
        """..."""
        assert "time" in three_co
        assert "row" in three_co
        assert "column" in three_co
        assert "x" not in three_co
        assert len(three_co) == 3

    def test_add(self: TestCoords, empty_co: Coords) -> None:
        """..."""
        empty_co += ("row", np.arange(5))
        assert "row" in empty_co
        assert len(empty_co["row"]) == 5
