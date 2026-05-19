pygwmodel Documentation
=========================

**pygwmodel** is a Python package providing conscious and easy-to-use interfaces to
high-performance C++ implementations of geographically weighted (GW) models,
based on `libgwmodel <https://github.com/GWmodel-Lab/libgwmodel>`_ and **GeoPandas**.

GW models are a branch of spatial statistics suited to situations where data are
not well described by some global model, but where spatial regions exist where a
suitably localized calibration provides a better description.

Implemented Models
------------------

* **GWRBasic** — Basic Geographically Weighted Regression with a single bandwidth.
* **GWRMultiscale** — Multiscale GWR (MGWR) with parameter-specific bandwidths and
  backfitting algorithm.
* **GWSS** — Geographically Weighted Summary Statistics (averages and correlations).

Installation
------------

We highly recommend installing in a conda environment:

.. code-block:: bash

   conda install armadillo gsl openblas numpy geopandas nanobind scikit-build-core
   git clone https://github.com/GWmodel-Lab/pygwmodel.git
   cd pygwmodel
   git submodule update --init --recursive
   pip install .

On Windows, set the environment variable to use OpenBLAS:

.. code-block:: shell

   set CMAKE_ARGS="-DBLA_VENDOR=OpenBLAS"
   pip install .

Quick Start
-----------

Basic GWR
~~~~~~~~~

.. code-block:: python

   from pygwmodel import GWRBasic, BandwidthWeight, CRSDistance

   algorithm = GWRBasic(
       data,              # GeoDataFrame
       depen_var="PURCHASE",
       indep_vars=["FLOORSZ", "UNEMPLOY", "PROF"],
       weight=BandwidthWeight(36.0, adaptive=True),
       distance=CRSDistance()
   ).fit()

   print(algorithm.diagnostic)       # {'AIC': ..., 'RSquare': ...}
   print(algorithm.result_layer)     # GeoDataFrame with betas, SE, fitted

Multiscale GWR (MGWR)
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pygwmodel import GWRMultiscale, BandwidthWeight

   n_var = 4  # intercept + 3 predictors
   weights = [BandwidthWeight(36.0, adaptive=True) for _ in range(n_var)]

   algorithm = GWRMultiscale(
       data,
       depen_var="PURCHASE",
       indep_vars=["FLOORSZ", "UNEMPLOY", "PROF"],
       weights=weights,
   ).fit()

   print(algorithm.diagnostic)
   # Bandwidths are auto-optimized per variable
   for w in algorithm.weights:
       print(w.bandwidth)

Algorithm Parameters
--------------------

GWRBasic
~~~~~~~~

* ``weight`` — A single :class:`~pygwmodel.spatial_weight.BandwidthWeight` shared
  by all variables.
* ``fit(optimize_bw=...)`` — Optionally auto-select bandwidth via CV or AIC.
* ``fit(optimize_var=...)`` — Optionally auto-select variables via forward
  selection.

GWRMultiscale
~~~~~~~~~~~~~

* ``weights`` — A list of :class:`~pygwmodel.spatial_weight.BandwidthWeight`, one
  per variable (including intercept). Each variable gets its own bandwidth.
* ``bandwidth_initilize`` — Per-variable initialization strategy:
  :class:`~pygwmodel.gwr_multiscale.GWRMultiscale.BandwidthInitilizeType.Null` (auto-select),
  ``Specified`` (fixed), or ``Initial``.
* ``bandwidth_selection_approach`` — Criterion per variable:
  :class:`~pygwmodel.gwr_multiscale.GWRMultiscale.BandwidthSelectionCriterionType.CV`
  or ``AIC``.
* ``criterion_type`` — Backfitting convergence criterion:
  :class:`~pygwmodel.gwr_multiscale.GWRMultiscale.BackFittingCriterionType.CVR`
  or ``dCVR``.
* ``max_iteration`` — Maximum backfitting iterations (default 500).
* ``preditor_centered`` — Whether to center each predictor before fitting.

Spatial Weights
---------------

.. code-block:: python

   from pygwmodel import BandwidthWeight, CRSDistance

   # Adaptive bandwidth: number of nearest neighbors
   bw = BandwidthWeight(bandwidth=36, adaptive=True)

   # Fixed bandwidth: distance in coordinate units
   bw = BandwidthWeight(bandwidth=5000.0, adaptive=False)

   # Kernel function (default: Gaussian)
   bw = BandwidthWeight(bandwidth=36, adaptive=True,
                        kernel=BandwidthWeight.Kernel.Bisquare)

Available kernels: ``Gaussian``, ``Exponential``, ``Bisquare``, ``Tricube``, ``Boxcar``.

Parallel Computing
------------------

Enable multi-threading via OpenMP:

.. code-block:: python

   from pygwmodel import ParallelType

   algorithm = GWRBasic(data, y, x, w).enable_parallel(
       ParallelType.OpenMP, threads=4
   ).fit()

API Reference
-------------

.. toctree::
   :maxdepth: 3
   :caption: Module Reference:

   modules.rst

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
