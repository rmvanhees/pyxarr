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
"""A minimal and light-weight class to work with multi-dimensional labeled arrays."""

from __future__ import annotations

__all__ = ["Coords", "DataArray", "Dataset", "sw_version"]

import contextlib
from importlib.metadata import PackageNotFoundError, version

from .lib.coords import Coords
from .lib.da import DataArray
from .lib.ds import Dataset

__version__ = "0.0.0"
with contextlib.suppress(PackageNotFoundError):
    __version__ = version(__name__)


def sw_version(full: bool = False) -> str:
    """Return the software version as obtained from git."""
    if full:
        return __version__

    return __version__.split("+")[0]
