Multiscale Geographically Weighted Regression (GWRMultiscale)
=============================================================

.. _mgwr-math:

Mathematical Foundation
-----------------------

Multiscale GWR (MGWR) extends basic GWR by allowing each independent variable
to have its own bandwidth. Different spatial processes may operate at different
spatial scales — the influence of some variables may be highly local (small
bandwidth), while others may have regional or global scale (large bandwidth).

The MGWR model form is the same as basic GWR:

.. math::

    y_i = \beta_{i0}(b_0) + \sum_{k=1}^{p} \beta_{ik}(b_k) x_{ik} + \varepsilon_i

However, each coefficient :math:`\beta_{ik}` is estimated using its own bandwidth
:math:`b_k`. Bandwidth calibration is carried out via the **backfitting**
algorithm [#mgwr]_:

1. **Initialisation**: Compute initial coefficient estimates
   :math:`\boldsymbol{\beta}^{(0)}` using standard GWR with an initial spatial
   weighting configuration.
2. **Backfitting**: For each iteration :math:`t = 1, 2, \dots`:

   a. For each variable :math:`k`:

      - Fix the coefficients of all other variables and compute the residual of the dependent variable for the current variable.
      - Select the optimal bandwidth :math:`b_k` for variable :math:`k` (golden section search with CV/AIC criterion).
      - Fit new coefficients for variable :math:`k` using :math:`b_k`.
      - Update the residuals.

   b. Check the convergence criterion (CVR or dCVR); stop if satisfied.

3. **Diagnostics**: Compute RSS, AICc, ENP, EDF, R², and other diagnostic metrics.

Convergence Criteria
~~~~~~~~~~~~~~~~~~~~

- **CVR** (Change in RSS):

  .. math::

      \text{CVR} = |RSS_t - RSS_{t-1}|

- **dCVR** (relative Change in RSS, default):

  .. math::

      \text{dCVR} = \sqrt{\frac{|RSS_t - RSS_{t-1}|}{RSS_t}}

.. _mgwr-params:

Key Parameters
--------------

.. list-table::
   :header-rows: 1
   :widths: 30 55 15

   * - Parameter
     - Description
     - Default
   * - ``weights``
     - A list of bandwidth weights, one per variable (including intercept)
     - Required
   * - ``distance``
     - The distance metric shared by all variables
     - ``CRSDistance()``
   * - ``has_intercept``
     - Whether to include an intercept term
     - ``True``
   * - ``bandwidth_initilize``
     - Bandwidth initialisation type per variable
     - All ``Null`` (auto-select)
   * - ``bandwidth_selection_approach``
     - Bandwidth selection criterion per variable
     - All ``CV``
   * - ``preditor_centered``
     - Whether to centre each predictor before fitting
     - All ``False``
   * - ``criterion_type``
     - Backfitting convergence criterion type
     - ``dCVR``
   * - ``max_iteration``
     - Maximum number of iterations
     - ``500``
   * - ``criterion_threshold``
     - Convergence threshold
     - ``1e-6``
   * - ``has_hat_matrix``
     - Whether to compute the hat matrix (for diagnostics)
     - ``True``
   * - ``adaptive_lower``
     - Lower bound on neighbour count for adaptive bandwidth selection
     - ``10``

Bandwidth Initialisation Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Type
     - Meaning
   * - ``BandwidthInitilizeType.Null``
     - Not specified; auto-select via golden section search during backfitting
   * - ``BandwidthInitilizeType.Specified``
     - User-specified; skip bandwidth selection and use the fixed value from ``weights``
   * - ``BandwidthInitilizeType.Initial``
     - Initially optimised; may still be adjusted in later backfitting iterations

.. _mgwr-examples:

Code Examples
-------------

Basic Usage (Auto-Select Bandwidths)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pygwmodel import GWRMultiscale, BandwidthWeight, CRSDistance

    n_var = 4  # intercept + 3 independent variables
    weights = [BandwidthWeight(36.0, adaptive=True) for _ in range(n_var)]

    algorithm = GWRMultiscale(
        data,
        depen_var="PURCHASE",
        indep_vars=["FLOORSZ", "UNEMPLOY", "PROF"],
        weights=weights,
        distance=CRSDistance()
    ).fit()

    # Diagnostic information
    print(algorithm.diagnostic)

    # Optimised bandwidths per variable
    for w in algorithm.weights:
        print(f"bandwidth={w.bandwidth}, adaptive={w.adaptive}")

    # Result layer
    result = algorithm.result_layer
    print(result.columns)
    # Intercept, FLOORSZ, UNEMPLOY, PROF,
    # Intercept_SE, FLOORSZ_SE, ..., Intercept_TV, ..., fitted

Specified Bandwidths
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pygwmodel import GWRMultiscale

    n_var = 4
    spec = GWRMultiscale.BandwidthInitilizeType.Specified
    cv = GWRMultiscale.BandwidthSelectionCriterionType.CV

    algorithm = GWRMultiscale(
        data, y, x,
        weights=[BandwidthWeight(36.0, adaptive=True) for _ in range(n_var)],
        bandwidth_initilize=[spec] * n_var,
        bandwidth_selection_approach=[cv] * n_var
    ).fit()

Note: when ``bandwidth_initilize`` is set to ``Specified``, the bandwidths
remain fixed; otherwise they are iteratively optimised during backfitting.

Adjusting Convergence
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    algorithm = GWRMultiscale(data, y, x, weights)
    algorithm.criterion_type = GWRMultiscale.BackFittingCriterionType.CVR
    algorithm.criterion_threshold = 1e-4
    algorithm.max_iteration = 100
    algorithm.fit()

.. _mgwr-refs:

References
----------

.. [#mgwr] Fotheringham, A. S., Yang, W., & Kang, W. (2017).
   *Multiscale geographically weighted regression (MGWR)*.
   Annals of the American Association of Geographers, 107(6), 1247-1265.

* Yu, H., Fotheringham, A. S., Li, Z., Oshan, T., Kang, W., & Wolf, L. J. (2020).
  *Inference in multiscale geographically weighted regression*.
  Geographical Analysis, 52(1), 87-106.
