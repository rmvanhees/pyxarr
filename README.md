# pyxarr
[![image](https://img.shields.io/pypi/v/pyxarr.svg?label=release)](https://github.com/rmvanhees/pyxarr/)
[![image](https://img.shields.io/pypi/l/pyxarr.svg)](https://github.com/rmvanhees/pyxarr/LICENSE)
[![image](https://img.shields.io/pypi/dm/pyxarr.svg)](https://pypi.org/project/pyxarr/)
[![image](https://img.shields.io/pypi/status/pyxarr.svg?label=status)](https://pypi.org/project/pyxarr/)

This is a package which provides a minimal and light-weight class to work with
multi-dimensional labeled arrays.

## Description
The software of `pyxarr` is written from scratch. 
I have tried to mimic the classes `DataArray`, `Dataset` and `Coords` from [xarray](https://xarray.dev/). 
My main goal is to keep the code light and fast.


## Installation
Pyxarr is available as `pyxarr` on PyPI. To install it use `pip`:

> $ pip install pyxarr

The module `pyxarr` requires Python3.10+ and Python modules: h5py, numpy. 

## Usage

Working with a pyxarr `DataArray`:
```
import numpy as np
from pyxarr import DataArray

xds = DataArray()
bool(xds) # will return False, thus if xda: ... will work
len(xds)  # will return 0

rng = np.random.default_rng()
xda = DataArray(
   rng.random((120, 11, 17)),
   dims=("time", "y", "x"),
   coords=(
      np.arange("2025-02-24T14:32:00", "2025-02-24T15:32:00", 30, dtype="datetime64[s]"),
      list(range(11)),
      np.arange(17),
   ),
   attrs={
      "long_name": "noisy signal",
      "units": "1",
   },
)
bool(xds)  # returns True
len(xds)   # returns 120
xda.shape  # returns (120, 11, 17)
xda.size   # returns 22440
"x" in xda.coords  # returns True
xda[4, :, :]  # slicing works
xda.coords += ("orbit", list(range(500, 620))  # will add an auxiliary coordinate
xda.swap_dims("orbit", "time")  # will make the auxiliary coordinate the dimension coordinate and visa versa
xda.mean("time")  # will return a new DataArray averaged over the time axis
xda + xda2  # will return a new DataArray with the sum of the data of both arrays, other supported operator are sub, div and mul.
```



## Authors and acknowledgment
The code is developed by R.M. van Hees (SRON)


## License
* Copyright: SRON (https://www.sron.nl)
* License: Apache-2.0
