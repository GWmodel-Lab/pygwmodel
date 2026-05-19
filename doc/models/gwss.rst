Geographically Weighted Summary Statistics (GWSS)
=================================================

.. _gwss-overview:

Model Overview
--------------

Geographically Weighted Summary Statistics (GWSS) performs locally weighted
descriptive statistics on multivariate data, revealing spatial heterogeneity
in the statistical characteristics of variables.

GWSS supports two modes:

- Average mode — computes local mean, standard deviation, variance, skewness,
  coefficient of variation, and optionally local median, interquartile range,
  and quantile imbalance.
- Correlation mode — computes local Pearson correlation coefficients and
  Spearman rank correlation coefficients.

.. _gwss-modes:

Two Modes
---------

Average Mode
~~~~~~~~~~~~

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

Correlation Mode
~~~~~~~~~~~~~~~~

For each pair of variables :math:`(X_i, X_j)`, the following are computed:

- Local Pearson correlation coefficient — via locally weighted covariance
- Local Spearman rank correlation coefficient — via locally weighted
  correlation on ranked data

Column name format: ``{var1}.{var2}_Corr`` and ``{var1}.{var2}_SCorr``.

.. _gwss-examples:

Code Examples
-------------

Average Mode
~~~~~~~~~~~~

.. code-block:: python

    from pygwmodel import GWSS, BandwidthWeight

    vars = ["PURCHASE", "FLOORSZ", "UNEMPLOY", "PROF"]

    gwss = GWSS(
        data, vars,
        weight=BandwidthWeight(36.0, adaptive=True),
        mode=GWSS.Mode.Average,
        quantile=False
    ).run()

    result = gwss.result_layer
    print(result.columns)
    # PURCHASE_Mean, PURCHASE_SDev, PURCHASE_Skew, PURCHASE_CV,
    # FLOORSZ_Mean, ...

Correlation Mode
~~~~~~~~~~~~~~~~

.. code-block:: python

    gwss = GWSS(data, vars, weight=BandwidthWeight(36.0, adaptive=True))
    result = gwss.run(mode=GWSS.Mode.Correlation).result_layer

    print(result.columns)
    # PURCHASE.FLOORSZ_Corr, PURCHASE.FLOORSZ_SCorr,
    # PURCHASE.UNEMPLOY_Corr, ...
