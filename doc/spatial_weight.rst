Spatial Weights
===============

Spatial weights are the core concept of GW models. For each target point :math:`i`,
the distance to every other data point :math:`j` is computed via a distance metric,
and a kernel function converts the distance into a weight :math:`w_{ij}`.
Larger distance means smaller weight.

.. _distance-metrics:

Distance Metrics
----------------

pygwmodel supports the following distance metrics:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Distance Type
     - Description
     - Python Class
   * - CRS Distance
     - Automatic selection based on coordinate reference system: Euclidean for
       projected coordinates, great-circle for geographic coordinates.
     - :class:`~pygwmodel.spatial_weight.CRSDistance`
   * - Minkowski Distance
     - Generalised distance parameterised by :math:`p`: :math:`p=1` is Manhattan,
       :math:`p=2` is Euclidean, :math:`p \to \infty` is Chebyshev.
     - (C++ implemented; Python binding pending)
   * - One-Dimensional Distance
     - Absolute difference along a single spatial or temporal dimension, used as
       the temporal component in spatiotemporal models.
     - (C++ implemented; Python binding pending)
   * - Distance Matrix File
     - Reads distances from a precomputed ``.dmat`` binary distance matrix file.
     - (C++ implemented; Python binding pending)
   * - Spatiotemporal Distance
     - Weighted combination of spatial and temporal distances, supporting
       orthogonal and oblique modes.
     - (C++ implemented; Python binding pending)

CRS Distance
~~~~~~~~~~~~

The coordinate-reference-system distance is the most commonly used metric,
automatically selecting the calculation method based on the CRS.

- **Projected coordinates** (``is_geographic=False``) — Euclidean distance:

  .. math::

      d_{ij} = \sqrt{(u_i - u_j)^2 + (v_i - v_j)^2}

- **Geographic coordinates** (``is_geographic=True``) — great-circle distance
  (geodesic distance).

Usage:

.. code-block:: python

    from pygwmodel import CRSDistance

    # Projected coordinate system (default)
    dist = CRSDistance(is_geographic=False)

    # Geographic coordinate system
    dist = CRSDistance(is_geographic=True)

.. _kernel-functions:

Kernel Functions and Weights
-----------------------------

Kernel functions convert distances :math:`d_{ij}` into weights :math:`w_{ij}`.
pygwmodel supports five kernel functions, configured via
:class:`~pygwmodel.spatial_weight.BandwidthWeight`.

.. list-table::
   :header-rows: 1
   :widths: 15 35 50

   * - Kernel
     - Formula
     - Enum Value
   * - Gaussian
     - :math:`w_{ij} = \exp\left(-\dfrac{d_{ij}^2}{2b^2}\right)`
     - ``BandwidthWeight.Kernel.Gaussian``
   * - Exponential
     - :math:`w_{ij} = \exp\left(-\dfrac{|d_{ij}|}{b}\right)`
     - ``BandwidthWeight.Kernel.Exponential``
   * - Bisquare
     - :math:`w_{ij} = \begin{cases} \left(1 - \left(\frac{d_{ij}}{b}\right)^2\right)^2, & d_{ij} < b \\ 0, & \text{otherwise} \end{cases}`
     - ``BandwidthWeight.Kernel.Bisquare``
   * - Tricube
     - :math:`w_{ij} = \begin{cases} \left(1 - \left(\frac{d_{ij}}{b}\right)^3\right)^3, & d_{ij} < b \\ 0, & \text{otherwise} \end{cases}`
     - ``BandwidthWeight.Kernel.Tricube``
   * - Boxcar
     - :math:`w_{ij} = \begin{cases} 1, & d_{ij} < b \\ 0, & \text{otherwise} \end{cases}`
     - ``BandwidthWeight.Kernel.Boxcar``

Here :math:`b` is the bandwidth and :math:`d_{ij}` is the distance from sample
:math:`i` to sample :math:`j`.

Gaussian and Exponential are continuous kernels — all samples receive non-zero
weights. Bisquare, Tricube, and Boxcar are truncated kernels — samples beyond
the bandwidth receive zero weight, which is useful for emphasising local effects.

Usage:

.. code-block:: python

    from pygwmodel import BandwidthWeight

    # Gaussian kernel (default)
    bw = BandwidthWeight(bandwidth=36.0, adaptive=True,
                         kernel=BandwidthWeight.Kernel.Gaussian)

    # Bisquare kernel
    bw = BandwidthWeight(bandwidth=5000.0, adaptive=False,
                         kernel=BandwidthWeight.Kernel.Bisquare)

.. _bandwidth:

Bandwidth
---------

The bandwidth :math:`b` controls the rate at which spatial weights decay
and is the most important parameter in GW models.

Bandwidth Types
~~~~~~~~~~~~~~~

- **Fixed bandwidth**: The bandwidth value is a distance. Weights are computed
  directly as :math:`w = k(d; b)`.
- **Adaptive bandwidth**: The bandwidth value is a neighbour count. For focus
  point :math:`i`, the effective distance bandwidth is the distance to the
  :math:`b`-th nearest neighbour. Different locations can use different effective
  distance bandwidths, making this suitable for datasets with non-uniform sample
  density.

Usage:

.. code-block:: python

    # Adaptive bandwidth: b=36 means use distance to the 36th nearest neighbour
    bw = BandwidthWeight(bandwidth=36.0, adaptive=True)

    # Fixed bandwidth: b=5000.0 means distance threshold of 5000 coordinate units
    bw = BandwidthWeight(bandwidth=5000.0, adaptive=False)

Bandwidth Selection
~~~~~~~~~~~~~~~~~~~

If you are unsure about the right bandwidth value, the algorithm can select it
automatically.

**GWRBasic**:

.. code-block:: python

    from pygwmodel import GWRBasic

    algorithm = GWRBasic(data, y, x, weight, distance).fit(
        optimize_bw=GWRBasic.BandwidthSelectionCriterionType.CV
    )
    # The optimised bandwidth
    print(algorithm.weight.bandwidth)

**GWRMultiscale** (each variable's bandwidth is optimised independently
during backfitting):

.. code-block:: python

    from pygwmodel import GWRMultiscale

    # BandwidthInitilizeType.Null (default) lets the algorithm auto-select
    algorithm = GWRMultiscale(
        data, y, x, weights,
        bandwidth_initilize=None,   # all auto-select
        bandwidth_selection_approach=None  # all use CV
    ).fit()

    # Check the optimised bandwidths
    for w in algorithm.weights:
        print(w.bandwidth)

Bandwidth selection criteria:

- **CV** (Cross-Validation): Minimizes the cross-validation residual sum of squares.
- **AIC** (Akaike Information Criterion): Minimizes the AIC value.

Spatial Weight Configuration
----------------------------

A :class:`~pygwmodel.spatial_weight.SpatialWeight` combines a distance metric
and a bandwidth weight, created via the factory method:

.. code-block:: python

    from pygwmodel import SpatialWeight, BandwidthWeight, CRSDistance

    sw = SpatialWeight.create(
        distance=CRSDistance(is_geographic=False),
        weight=BandwidthWeight(bandwidth=36.0, adaptive=True)
    )
