Quick Start
===========

.. _quickstart-overview:

Overview
--------

**pygwmodel** is a Python binding for
`libgwmodel <https://github.com/GWmodel-Lab/libgwmodel>`_ that provides
clear, high-performance interfaces for geographically weighted (GW) models
based on **GeoPandas**.

Currently implemented models:

- **Geographically Weighted Regression (GWR)** — :class:`~pygwmodel.gwr_basic.GWRBasic`
- **Multiscale GWR (MGWR)** — :class:`~pygwmodel.gwr_multiscale.GWRMultiscale`
- **Geographically Weighted Summary Statistics (GWSS)** — :class:`~pygwmodel.gwss.GWSS`

All algorithms use a C++17 core, exposed to Python via
`nanobind <https://github.com/wjakob/nanobind>`_, with support for
OpenMP multi-threading and CUDA GPU acceleration.

.. _quickstart-install:

Installation
------------

System Dependencies
~~~~~~~~~~~~~~~~~~~

Install the required native libraries:

.. code-block:: bash

    # Ubuntu/Debian
    sudo apt install libarmadillo-dev libgsl-dev libopenblas-dev

    # or via conda/mamba
    conda install armadillo gsl openblas -c conda-forge

Python Installation
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    git clone https://github.com/GWmodel-Lab/pygwmodel.git
    cd pygwmodel
    git submodule update --init --recursive
    pip install .

On Windows, you **must** use OpenBLAS to avoid segmentation faults:

.. code-block:: powershell

    $Env:CMAKE_ARGS="-DBLA_VENDOR=OpenBLAS"
    pip install .

Or pass the setting directly to pip:

.. code-block:: powershell

    pip install . --config-settings=cmake.args=-DBLA_VENDOR=OpenBLAS

.. _quickstart-dev:

Development Guide
-----------------

Editable Install
~~~~~~~~~~~~~~~~

An editable install lets you edit Python code without reinstalling:

.. code-block:: bash

    pip install nanobind scikit-build-core[pyproject]
    pip install --no-build-isolation -ve .

Running Tests
~~~~~~~~~~~~~

.. code-block:: bash

    # Run directly
    python test/test_gwr_basic.py test/londonhp100.csv
    python test/test_gwr_multiscale.py test/londonhp100.csv

    # Enable OpenMP test cases
    ENABLE_OPENMP=true python test/test_gwr_multiscale.py test/londonhp100.csv

    # Via CTest
    ctest

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    pip install furo sphinx
    sphinx-build -b html -D language=en doc doc/_build/en          # English
    sphinx-build -b html -D language=zh_CN doc doc/_build/zh_CN    # Chinese

Project Structure
~~~~~~~~~~~~~~~~~

.. code-block:: text

    pygwmodel/
    ├── libgwmodel/          # C++ core algorithms (git submodule)
    ├── src/                 # nanobind C++ bindings + Python API
    │   ├── pygwmodel/       # Python wrapper layer
    │   └── *.cpp            # C++ binding source files
    ├── test/                # Integration tests
    ├── doc/                 # Sphinx documentation
    └── pyproject.toml       # Build configuration
