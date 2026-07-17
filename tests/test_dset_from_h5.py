#
# This file is part of Python package: `pyxarr`
#
#     https://github.com/rmvanhees/pyxarr.git
#
# Copyright (c) 2026 - R.M. van Hees (SRON)
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
"""Unit-tests for pyxarr module `dset_from_h5.py`."""

from __future__ import annotations

from importlib.resources import files

import numpy as np
import pytest
from h5yaml.template_h5 import TemplateH5

from pyxarr.dset_from_h5 import dset_from_h5


class TestFromH5:
    """Class with unit-tests for pyxarr.dset_from_h5."""

    _res = TemplateH5(
        [
            files("h5yaml.Data") / "h5_testing.yaml",
            files("h5yaml.Data") / "h5_global_attrs.yaml",
        ]
    )
    _res.set_dims({"number_of_images": 10})
    H5_DEF = _res.asdict
    FID_H5 = _res.diskless()

    def test_full(self: TestFromH5) -> None:
        """Unit-test ..."""
        for key in self.H5_DEF["variables"]:
            _ = dset_from_h5(self.FID_H5[key])
            # try time_units
            for val in ["ns", "us", "ms", "s"]:
                _ = dset_from_h5(self.FID_H5[key], time_units=val)
            if "number_of_images" in self.H5_DEF["variables"][key]["_dims"]:
                _ = dset_from_h5(self.FID_H5[key], data_sel=np.s_[3:7, ...])
            if (
                self.H5_DEF["variables"][key]["_dtype"] == "stats_dtype"
                and "_vlen" not in self.H5_DEF["variables"][key]
            ):
                _ = dset_from_h5(self.FID_H5[key], field="index")

        with pytest.raises(KeyError, match=r"unknown .*") as excinfo:
            _ = dset_from_h5(self.FID_H5["/group_01/stats"], time_units="ps")
        assert "unknown numpy.timedelta64 unit" in str(excinfo.value)

    def test_close(self: TestFromH5) -> None:
        """Close the in-memory HDF5 file."""
        self.FID_H5.close()


def main() -> None:
    """..."""
    obj = TestFromH5()
    obj.test_full()


if __name__ == "__main__":
    main()
