# pygwmodel: Python wrappers for building geographically weighted models

## Overview

This package includes techniques from a particular branch of spatial statistics,
termed geographically weighted (GW) models.
GW models suit situations when data are not described well by some global model,
but where there are spatial regions where a suitably localized calibration provides a better description.

The goal of **pygwmodel** is to provide conscious and easy-to-use user interface
to high-performance C++ implementations of GW models (see [libgwmodel](https://github.com/GWmodel-Lab/libgwmodel))
based on **GeoPandas**.
We believe with the newly designed interfaces and the underlying C++ core,
users will get fluent experiences.

## Installation

We highly recommend installing this package in a conda environment,
especially on Windows.

```bash
conda install armadillo gsl openblas catch2 numpy geopandas nanobind scikit-build-core
git clone https://github.com/GWmodel-Lab/pygwmodel.git
pip install ./pygwmodel
```

On Windows, it is very important to use the **OpenBLAS** library instead of that bundled with Python to avoid segmentation fault.
To do so, set the environment variable `CMAKE_ARGS`:

```powershell
# powershell
$Env:CMAKE_ARGS="-DBLA_VENDOR=OpenBLAS"
pip install .
```

In command line prompt, it can be done like this:

```bat
set CMAKE_ARGS="-DBLA_VENDOR=OpenBLAS"
pip install .
```

Or using the following line to install this package:

```powershell
pip install . --config-settings=cmake.args=-DBLA_VENDOR=OpenBLAS
```

## Getting started

```py
from pygwmodel import GWRBasic
algorithm = GWRBasic(data, y, x, 36.0).fit()
```

For full usage, please see the unit tests in `test` directory.

## Related work

This package is based on a pure C++ library --- [libgwmodel](https://github.com/GWmodel-Lab/libgwmodel).
This library implements all models,
and **pygwmodel** just calls this package by translating inputs and outputs. 
Besides, it also provides some handy functions for the convenience of python users.

## Getting help

If you encounter a bug, please create an [issue](https://github.com/GWmodel-Lab/pygwmodel/issues) here.
It would be better for us if a minimal reproducible example is also provided.
