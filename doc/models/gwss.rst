Geographically Weighted Summary Statistics (GWAverage / GWCorrelation)
======================================================================

.. _gwss-overview:

Model Overview
--------------

Geographically Weighted Summary Statistics performs locally weighted
descriptive statistics on multivariate data, revealing spatial heterogeneity
in the statistical characteristics of variables.

Two classes are provided:

- :class:`~pygwmodel.gwss.GWAverage` — computes local mean, standard deviation,
  variance, skewness, coefficient of variation, and optionally local median,
  interquartile range, and quantile imbalance.
- :class:`~pygwmodel.gwss.GWCorrelation` — computes local Pearson correlation
  coefficients and Spearman rank correlation coefficients for every pair of
  variables.

.. _gwss-average:

GWAverage
---------

For each variable, the following local statistics are computed:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Statistic
     - Attribute
     - Column Name
   * - Local Mean
     - ``local_mean``
     - ``{variable}_Mean``
   * - Local Std Dev
     - ``local_sdev``
     - ``{variable}_SDev``
   * - Local Skewness
     - ``local_skewness``
     - ``{variable}_Skew``
   * - Local CV
     - ``local_cv``
     - ``{variable}_CV``

When ``quantile=True``:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Statistic
     - Attribute
     - Column Name
   * - Local Median
     - ``local_median``
     - ``{variable}_Median``
   * - IQR
     - ``iqr``
     - ``{variable}_IQR``
   * - Quantile Imbalance
     - ``qi``
     - ``{variable}_QI``

.. _gwss-correlation:

GWCorrelation
-------------

For each pair of variables :math:`(X_i, X_j)`, the following are computed:

- Local Pearson correlation coefficient — via locally weighted covariance
- Local Spearman rank correlation coefficient — via locally weighted
  correlation on ranked data

Column name format: ``{var1}.{var2}_Corr`` and ``{var1}.{var2}_SCorr``.

.. _gwss-examples:

Code Examples
-------------

GWAverage
~~~~~~~~~

.. code-block:: python

    from pygwmodel import GWAverage, BandwidthWeight

    vars = ["PURCHASE", "FLOORSZ", "UNEMPLOY", "PROF"]

    gwa = GWAverage(
        data, vars,
        weight=BandwidthWeight(36.0, adaptive=True),
        quantile=False
    ).run()

    result = gwa.result_layer
    print(result.columns)
    # PURCHASE_Mean, PURCHASE_SDev, PURCHASE_Skew, PURCHASE_CV,
    # FLOORSZ_Mean, ...

GWCorrelation
~~~~~~~~~~~~~

.. code-block:: python

    from pygwmodel import GWCorrelation, BandwidthWeight

    gwc = GWCorrelation(data, vars, weight=BandwidthWeight(36.0, adaptive=True))
    result = gwc.run().result_layer

    print(result.columns)
    # PURCHASE.FLOORSZ_Corr, PURCHASE.FLOORSZ_SCorr,
    # PURCHASE.UNEMPLOY_Corr, ...

Parallel Execution
~~~~~~~~~~~~~~~~~~

Both classes support OpenMP multi-threading:

.. code-block:: python

    from pygwmodel import GWAverage, GWCorrelation, ParallelType

    gwa = GWAverage(data, vars, weight=...)
    gwa.enable_parallel(ParallelType.OpenMP, threads=4).run()

    gwc = GWCorrelation(data, vars, weight=...)
    gwc.enable_parallel(ParallelType.OpenMP, threads=4).run()
