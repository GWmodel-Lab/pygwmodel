Geographically and Temporally Weighted Regression (GTWR)
=========================================================

.. _gtwr-overview:

Model Overview
--------------

Geographically and Temporally Weighted Regression (GTWR) extends GWR by
incorporating temporal information via a spatio-temporal distance metric.
GTWR assigns a spatio-temporal ratio parameter :math:`\lambda \in [0, 1]`
that balances the influence of spatial proximity versus temporal proximity
on the weighting scheme.

Key features:

- Supports automatic bandwidth selection via CV or AIC criteria.
- Supports OpenMP parallel computation.
- Provides standard errors and diagnostic statistics.

.. _gtwr-constructor:

Constructor
-----------

.. code-block:: python

    from pygwmodel import GTWR, BandwidthWeight
    from pygwmodel.spatial_weight import CRSSTDistance

    algorithm = GTWR(
        data,                   # GeoDataFrame
        'PURCHASE',             # dependent variable
        ['FLOORSZ', 'UNEMPLOY', 'PROF'],  # independent variables
        times='TIME',           # temporal column
        weight=BandwidthWeight(36.0, adaptive=True),
        distance=CRSSTDistance(lambda_=0.5)
    )

Parameters:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Parameter
     - Type
     - Description
   * - ``sdf``
     - ``GeoDataFrame``
     - Input data with geometry
   * - ``depen_var``
     - ``str``
     - Dependent variable column name
   * - ``indep_vars``
     - ``List[str]``
     - Independent variable column names
   * - ``times``
     - ``str``
     - Column for temporal stamps
   * - ``weight``
     - ``BandwidthWeight``
     - Bandwidth configuration
   * - ``distance``
     - ``CRSSTDistance``
     - Spatio-temporal distance (default CRSSTDistance)
   * - ``has_intercept``
     - ``bool``
     - Include intercept term (default True)

.. _gtwr-examples:

Code Examples
-------------

Basic Fit
~~~~~~~~~

.. code-block:: python

    from pygwmodel import GTWR, BandwidthWeight
    from pygwmodel.spatial_weight import CRSSTDistance

    algorithm = GTWR(
        data, 'PURCHASE', ['FLOORSZ', 'UNEMPLOY', 'PROF'],
        times='TIME',
        weight=BandwidthWeight(36.0, adaptive=True),
        distance=CRSSTDistance(lambda_=0.5)
    ).fit()

    print(algorithm.diagnostic)
    # {'RSS': ..., 'AIC': ..., 'AICc': ..., 'RSquare': ...}

    print(algorithm.result_layer.columns)
    # Index(['Intercept', 'FLOORSZ', 'UNEMPLOY', 'PROF',
    #        'Intercept_SE', 'FLOORSZ_SE', 'UNEMPLOY_SE', 'PROF_SE',
    #        'fitted'])

Bandwidth Auto-Selection
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    algorithm = GTWR(
        data, 'PURCHASE', ['FLOORSZ', 'UNEMPLOY', 'PROF'],
        times='TIME',
        weight=BandwidthWeight(36.0, adaptive=True),
        distance=CRSSTDistance(lambda_=0.5)
    ).fit(optimize_bandwidth=GTWR.BandwidthSelectionCriterionType.CV)

    print(f"Optimal bandwidth: {algorithm.weight.bandwidth}")

Prediction
~~~~~~~~~~

.. code-block:: python

    prediction = algorithm.predict(data)
    print(prediction.columns)
    # Index(['Intercept', 'FLOORSZ', 'UNEMPLOY', 'PROF',
    #        'y_hat', 'residual'])

Parallel Execution
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pygwmodel import GTWR, ParallelType

    algorithm = GTWR(..., weight=...)
    algorithm.enable_parallel(ParallelType.OpenMP, threads=4).fit()
