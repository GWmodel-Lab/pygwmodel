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
* **GWRMultiscale** — Multiscale GWR (MGWR) with per-variable bandwidths and
  backfitting algorithm.
* **GTWR** — Geographically and Temporally Weighted Regression.
* **GWAverage** — Geographically Weighted Summary Statistics (mean, std dev, etc.).
* **GWCorrelation** — Geographically Weighted Correlation coefficients.

Quick Start
-----------

.. code-block:: python

   from pygwmodel import GWRBasic, GWRMultiscale, BandwidthWeight, CRSDistance

   # Basic GWR
   algorithm = GWRBasic(data, y, x,
                        weight=BandwidthWeight(36.0, adaptive=True),
                        distance=CRSDistance()).fit()
   print(algorithm.diagnostic['RSquare'])

   # Multiscale GWR
   mgwr = GWRMultiscale(data, y, x,
                        weights=[BandwidthWeight(36.0, adaptive=True) for _ in range(4)]
                        ).fit()
   print(mgwr.diagnostic)

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   spatial_weight
   models
   models/gwss
   performance
   modules.rst

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
