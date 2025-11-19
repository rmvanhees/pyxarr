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
"""Test module for for all pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from pyxarr.lib.coords import Coords, _Coord
from pyxarr.lib.da import DataArray

# from pyxarr.lib.ds import Dataset


@pytest.fixture
def co_struct() -> _Coord:
    """Return instance of class _Coord."""
    return _Coord("column", list(range(13)), {"units": "px"})


@pytest.fixture
def co_from_dict() -> Coords:
    """Return instance of class Coords with 3 dimensions."""
    co_dict = {
        "time": np.arange("2025-02-02", "2025-02-26", dtype="datetime64[D]"),
        "row": np.arange(5),
        "column": list(range(11)),
    }
    return Coords(co_dict)


@pytest.fixture
def da_scalar() -> DataArray:
    """Return scalar instance of class DataArray."""
    return DataArray(3.14)


@pytest.fixture
def da_full() -> DataArray:
    """Return instance of class DataArray with 3 dimensions."""
    return DataArray(
        np.arange(5 * 3 * 7, dtype=float).reshape(5, 3, 7),
        dims=("orbit", "y", "x"),
    )


@pytest.fixture
def da_ones() -> DataArray:
    """Return instance of class DataArray with 3 dimensions."""
    return DataArray(
        np.ones((5, 3, 7), dtype=float),
        dims=("orbit", "y", "x"),
    )


@pytest.fixture
def attrs_for_da() -> dict:
    """Generate attributes for DataArray."""
    return {
        "long_name": "numpy.arange",
        "units": "1",
    }


@pytest.fixture
def da_from_coords(co_from_dict: Coords, attrs_for_da: dict) -> DataArray:
    """Create DataArray, coords initialized with coords."""
    return DataArray(
        np.arange(24 * 5 * 11, dtype=float).reshape(24, 5, 11),
        coords=co_from_dict,
        attrs=attrs_for_da,
    )


@pytest.fixture
def da_from_dict(attrs_for_da: dict) -> DataArray:
    """Create DataArray, coords initialized with dict."""
    return DataArray(
        np.arange(24 * 5 * 11, dtype=float).reshape(24, 5, 11),
        coords={
            "time": np.arange("2025-02-02", "2025-02-26", dtype="datetime64[D]"),
            "row": np.arange(5),
            "column": list(range(11)),
        },
        attrs=attrs_for_da,
    )


@pytest.fixture
def da_from_tuple(attrs_for_da: dict) -> DataArray:
    """Create DataArray, coords initialized with tuple[list[str, ArrayLike]]."""
    return DataArray(
        np.arange(24 * 5 * 11, dtype=float).reshape(24, 5, 11),
        coords=(
            ("time", np.arange("2025-02-02", "2025-02-26", dtype="datetime64[D]")),
            ("row", np.arange(5)),
            ("column", np.arange(11)),
        ),
        attrs=attrs_for_da,
    )


@pytest.fixture
def da_from_dims(co_from_dict: Coords, attrs_for_da: dict) -> DataArray:
    """Create DataArray, coords initialized with coords."""
    return DataArray(
        np.arange(24 * 5 * 11, dtype=float).reshape(24, 5, 11),
        coords=(
            np.arange("2025-02-02", "2025-02-26", dtype="datetime64[D]"),
            np.arange(5),
            np.arange(11),
        ),
        dims=("time", "row", "column"),
        attrs=attrs_for_da,
    )
