High-Performance Computing
===========================

.. _perf-overview:

Supported Parallel Algorithms
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 20 15

   * - Algorithm
     - Serial
     - OpenMP
     - CUDA
     - MPI
   * - GWRBasic
     - ✅
     - ✅
     - ✅
     - ✅
   * - GWRMultiscale
     - ✅
     - ✅
     - ✅
     - Serial × :math:`n_{var}`
   * - GWSS (Average)
     - ✅
     - ✅
     - —
     - —
   * - GWSS (Correlation)
     - ✅
     - ✅
     - —
     - —

Notes:

- **Serial** (SerialOnly): single-threaded; suitable for small datasets or debugging.
- **OpenMP**: shared-memory multi-threading; suitable for single-node multi-core
  machines. Requires ``ENABLE_OPENMP`` at build time.
- **CUDA**: NVIDIA GPU acceleration; suitable for large datasets. Requires
  ``ENABLE_CUDA`` at build time.
- **MPI**: distributed computing; suitable for multi-node clusters. Requires
  ``ENABLE_MPI`` at build time.

.. _perf-openmp:

Multi-Threading (OpenMP)
------------------------

Enables shared-memory parallel computation via OpenMP. The core per-sample model
fitting loop is parallelised, with each thread independently computing coefficient
estimates for a subset of samples.

Setting the number of threads:

.. code-block:: python

    from pygwmodel import GWRBasic, ParallelType, BandwidthWeight, CRSDistance

    algorithm = GWRBasic(data, y, x,
                         weight=BandwidthWeight(36.0, adaptive=True),
                         distance=CRSDistance()).enable_parallel(
        ParallelType.OpenMP, threads=4
    ).fit()

GWRMultiscale also supports OpenMP:

.. code-block:: python

    from pygwmodel import GWRMultiscale

    algorithm = GWRMultiscale(data, y, x, weights).enable_parallel(
        ParallelType.OpenMP, threads=8
    ).fit()

Recommended thread count: set to the number of physical CPU cores.

.. _perf-cuda:

GPU Acceleration (CUDA)
-----------------------

Offloads the locally weighted regression matrix operations to a NVIDIA GPU,
suitable for larger datasets.

Group Size (``group_size``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``group_size`` controls how many samples' coefficient estimates are computed
together on the GPU in one batch. Larger groups make better use of GPU
parallelism but are constrained by GPU memory. The internal constraint is:

.. math::

    k \times n \times g \times 8 < \text{GPU Memory}

where :math:`k` is the number of independent variables, :math:`n` is the number
of samples, and :math:`g` is the group size.

Usage:

.. code-block:: python

    algorithm = GWRBasic(data, y, x,
                         weight=BandwidthWeight(36.0, adaptive=True),
                         distance=CRSDistance()).enable_parallel(
        ParallelType.CUDA, gpu_id=0, group_size=64
    ).fit()

For GWRMultiscale:

.. code-block:: python

    algorithm = GWRMultiscale(data, y, x, weights).enable_parallel(
        ParallelType.CUDA, gpu_id=0, group_size=128
    ).fit()

.. _perf-tips:

Performance Tips
----------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Scenario
     - Recommendation
   * - Small datasets (:math:`n < 1000`)
     - Serial is sufficient; overhead is negligible
   * - Medium datasets (:math:`n < 10^4`)
     - Use OpenMP with thread count equal to CPU cores
   * - Large datasets (:math:`n > 10^4`)
     - Use CUDA if available; otherwise OpenMP
   * - Speeding up GWRMultiscale
     - Set ``has_hat_matrix=False`` to skip hat matrix computation, significantly
       reducing memory and computational cost
   * - GWRMultiscale convergence
     - Increase ``criterion_threshold`` to reduce iterations (trades slight accuracy)
