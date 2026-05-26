Basic Geographically Weighted Regression (GWR)
================================================

.. _gwr-math:

Mathematical Foundation
-----------------------

For a dataset of :math:`n` samples and :math:`p` independent variables,
the basic GWR model at sample :math:`i` is defined as:

.. math::

    y_i = \beta_{i0} + \sum_{k=1}^{p} \beta_{ik} x_{ik} + \varepsilon_i

where:

- :math:`y_i` is the dependent variable,
- :math:`x_{ik}` is the :math:`k`-th independent variable,
- :math:`\beta_{ik}` is the :math:`k`-th coefficient,
- :math:`\beta_{i0}` is the intercept,
- :math:`\varepsilon_i \sim \mathcal{N}(0, \sigma^2)` is the random error.

The locally weighted least-squares estimator of the coefficients is:

.. math::

    \hat{\boldsymbol{\beta}}_i = \left( \mathbf{X}^\top \mathbf{W}_i \mathbf{X} \right)^{-1} \mathbf{X}^\top \mathbf{W}_i \mathbf{y}

where :math:`\mathbf{W}_i = \operatorname{diag}(w_{i1}, w_{i2}, \dots, w_{in})`
is the spatial weighting matrix. Each :math:`w_{ij}` is computed by a kernel
function :math:`k(d_{ij}; b)` based on the distance from sample :math:`i` to
sample :math:`j`.

Diagnostic Information
~~~~~~~~~~~~~~~~~~~~~~

After fitting, the algorithm computes the following diagnostics:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Metric
     - Meaning
     - Key
   * - RSS
     - Residual sum of squares :math:`\sum (y_i - \hat{y}_i)^2`
     - ``diagnostic['RSS']``
   * - AICc
     - Corrected Akaike information criterion (smaller is better)
     - ``diagnostic['AICc']``
   * - ENP
     - Effective number of parameters
     - ``diagnostic['ENP']``
   * - EDF
     - Effective degrees of freedom
     - ``diagnostic['EDF']``
   * - R²
     - Coefficient of determination
     - ``diagnostic['RSquare']``
   * - Adjusted R²
     - Adjusted R-squared
     - ``diagnostic['RSquareAdjust']``

.. _gwr-params:

Key Parameters
--------------

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - Parameter
     - Description
     - Default
   * - ``weight``
     - A single bandwidth weight shared by all variables
     - Required
   * - ``distance``
     - The distance metric to use
     - ``CRSDistance()``
   * - ``has_intercept``
     - Whether to include an intercept term
     - ``True``
   * - ``fit(optimize_bw=...)``
     - Auto-select bandwidth: ``CV`` or ``AIC``
     - ``None``
   * - ``fit(optimize_var=...)``
     - Auto-select variables via forward selection: AIC change threshold
     - ``None``

.. _gwr-examples:

Code Examples
-------------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    from pygwmodel import GWRBasic, BandwidthWeight, CRSDistance

    algorithm = GWRBasic(
        data,
        depen_var="PURCHASE",
        indep_vars=["FLOORSZ", "UNEMPLOY", "PROF"],
        weight=BandwidthWeight(36.0, adaptive=True),
        distance=CRSDistance()
    ).fit()

    # View diagnostic information
    print(algorithm.diagnostic['RSquare'])      # 0.708
    print(algorithm.diagnostic['AICc'])          # 2448.27

    # Get the result layer (GeoDataFrame)
    result = algorithm.result_layer
    print(result.columns)  # Intercept, FLOORSZ, ..., Intercept_SE, ..., fitted

Bandwidth Selection
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    algorithm = GWRBasic(data, y, x, BandwidthWeight(adaptive=True),
                         distance=CRSDistance()).fit(
        optimize_bw=GWRBasic.BandwidthSelectionCriterionType.CV
    )
    print(algorithm.weight.bandwidth)  # Optimised bandwidth: 67

Independent Variable Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    algorithm = GWRBasic(data, y, x, BandwidthWeight(36.0, adaptive=True),
                         distance=CRSDistance()).fit(optimize_var=3.0)

    # Variable combinations and their AICc values
    for vars, aicc in algorithm.indep_var_select_criterions:
        print(f"{vars}: {aicc:.2f}")

    print(algorithm.indep_vars)  # ['FLOORSZ', 'PROF'] — optimal subset

Prediction
~~~~~~~~~~

.. code-block:: python

    prediction = algorithm.predict(new_data)
    print(prediction.columns)  # Intercept, FLOORSZ, ..., y_hat, residual

.. _gwr-refs:

References
----------

* Brunsdon, C., Fotheringham, A. S., & Charlton, M. E. (1996).
  *Geographically weighted regression: a method for exploring spatial nonstationarity*.
  Geographical Analysis, 28(4), 281-298.

* Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
  *Geographically weighted regression: the analysis of spatially varying relationships*.
  John Wiley & Sons.
