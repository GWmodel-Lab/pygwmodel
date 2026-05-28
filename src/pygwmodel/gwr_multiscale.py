from typing import List, Optional
import numpy as np
import geopandas as gp
from .spatial_weight import SpatialWeight, Distance, BandwidthWeight, CRSDistance
from .parallel import ParallelType
from ._regression import _GWRMultiscale


class GWRMultiscale:
    """Multiscale Geographically Weighted Regression (MGWR).

    MGWR allows each explanatory variable (including the intercept)
    to operate at its own spatial scale by assigning an independent
    bandwidth to each variable.  The model is calibrated via a
    backfitting algorithm.

    Supports automatic bandwidth selection (CV or AIC) per variable,
    and OpenMP/CUDA parallel computation.

    Parameters
    ----------
    sdf : GeoDataFrame
        Input spatial data.
    depen_var : str
        Dependent variable column name.
    indep_vars : list of str
        Independent variable column names.
    weights : list of BandwidthWeight
        One weight per variable (intercept included if
        ``has_intercept=True``).
    distance : Distance, optional
        Distance metric, default CRSDistance.
    has_intercept : bool, optional
        Whether to include an intercept term (default True).
    bandwidth_initilize : list of BandwidthInitilizeType, optional
        Per-variable bandwidth initialization type; defaults to
        ``Null`` (auto-select) for all.
    bandwidth_selection_approach : list of BandwidthSelectionCriterionType, optional
        Per-variable selection criterion; defaults to CV for all.
    preditor_centered : list of bool, optional
        Whether to center each predictor; defaults to False.
    has_hat_matrix : bool, optional
        Store the hat matrix S (default True).

    Attributes
    ----------
    result_layer : GeoDataFrame or None
        Fitted results after calling :meth:`fit`.
    BandwidthInitilizeType : enum
        Null, Initial, Specified
    BandwidthSelectionCriterionType : enum
        CV, AIC
    BackFittingCriterionType : enum
        CVR, dCVR
    """

    BandwidthInitilizeType = _GWRMultiscale.BandwidthInitilizeType
    BandwidthSelectionCriterionType = _GWRMultiscale.BandwidthSelectionCriterionType
    BackFittingCriterionType = _GWRMultiscale.BackFittingCriterionType

    def __init__(self, sdf: gp.GeoDataFrame, depen_var: str,
                 indep_vars: List[str],
                 weights: List[BandwidthWeight],
                 distance: Distance = CRSDistance(),
                 has_intercept: bool = True,
                 bandwidth_initilize: Optional[List[BandwidthInitilizeType]] = None,
                 bandwidth_selection_approach: Optional[List[BandwidthSelectionCriterionType]] = None,
                 preditor_centered: Optional[List[bool]] = None,
                 has_hat_matrix: bool = True):
        if not isinstance(sdf, gp.GeoDataFrame):
            raise ValueError("sdf must be a GeoDataFrame")
        self.geometry = sdf.geometry
        self.depen_var: str = depen_var
        self.indep_vars: List[str] = indep_vars
        self.has_intercept: bool = has_intercept
        self.distance = distance
        self.result_layer: Optional[gp.GeoDataFrame] = None

        n_var = len(indep_vars) + int(has_intercept)

        if len(weights) != n_var:
            raise ValueError(f"weights must have length {n_var} (one per variable including intercept), got {len(weights)}")

        if bandwidth_initilize is None:
            bandwidth_initilize = [GWRMultiscale.BandwidthInitilizeType.Null] * n_var
        if len(bandwidth_initilize) != n_var:
            raise ValueError(f"bandwidth_initilize must have length {n_var}")

        if bandwidth_selection_approach is None:
            bandwidth_selection_approach = [GWRMultiscale.BandwidthSelectionCriterionType.CV] * n_var
        if len(bandwidth_selection_approach) != n_var:
            raise ValueError(f"bandwidth_selection_approach must have length {n_var}")

        if preditor_centered is None:
            preditor_centered = [False] * n_var
        if len(preditor_centered) != n_var:
            raise ValueError(f"preditor_centered must have length {n_var}")

        self.weights = weights
        self.bandwidth_initilize: List[GWRMultiscale.BandwidthInitilizeType] = bandwidth_initilize
        self.bandwidth_selection_approach: List[GWRMultiscale.BandwidthSelectionCriterionType] = bandwidth_selection_approach

        indep_vars_data = np.asfortranarray(sdf[indep_vars], dtype=np.float64)
        if has_intercept:
            indep_vars_data = np.hstack([np.ones((indep_vars_data.shape[0], 1)), indep_vars_data])

        spatial_weights = [SpatialWeight.create(distance, w) for w in weights]

        self.algorithm = _GWRMultiscale()
        self.algorithm.coords = np.asfortranarray(sdf.geometry.centroid.get_coordinates(), dtype=np.float64)
        self.algorithm.independent = indep_vars_data
        self.algorithm.dependent = np.asfortranarray(sdf[depen_var], dtype=np.float64)
        self.algorithm.spatial_weights = spatial_weights
        self.algorithm.has_intercept = has_intercept
        self.algorithm.has_hat_matrix = has_hat_matrix

        self.algorithm.set_bandwidth_initilize(
            [_GWRMultiscale.BandwidthInitilizeType(t.value) for t in bandwidth_initilize]
        )
        self.algorithm.set_bandwidth_selection_approach(
            [_GWRMultiscale.BandwidthSelectionCriterionType(t.value) for t in bandwidth_selection_approach]
        )
        self.algorithm.preditor_centered = preditor_centered

        self.algorithm.bandwidth_select_threshold = [1.0e-6] * n_var

    def enable_parallel_omp(self, threads: int = 8):
        """Enable OpenMP parallel computation.

        Parameters
        ----------
        threads : int
            Number of threads (default 8).
        """
        if self.algorithm is None:
            raise ValueError("Not initialized")
        if isinstance(threads, int) and threads > 0:
            self.algorithm.parallel_omp(threads)
        else:
            raise ValueError("threads must be a positive integer")
        return self

    def enable_parallel_cuda(self, gpu_id: int = 0, group_size: int = 64):
        """Enable CUDA GPU acceleration.

        Parameters
        ----------
        gpu_id : int
            GPU device ID (default 0).
        group_size : int
            CUDA thread group size (default 64).
        """
        if self.algorithm is None:
            raise ValueError("Not initialized")
        if all([(isinstance(x, int) and x > 0) for x in [gpu_id, group_size]]):
            self.algorithm.parallel_cuda(gpu_id, group_size)
        else:
            raise ValueError("gpu_id and group_size must be positive integers")
        return self

    def enable_parallel(self, type: ParallelType, **kvargs):
        """Enable parallel computation.

        Parameters
        ----------
        type : ParallelType
            Parallel backend (OpenMP or CUDA).
        **kvargs
            Keyword arguments forwarded to ``enable_parallel_omp``
            or ``enable_parallel_cuda``.
        """
        if type == ParallelType.OpenMP:
            self.enable_parallel_omp(**kvargs)
        elif type == ParallelType.CUDA:
            self.enable_parallel_cuda(**kvargs)
        return self

    def fit(self):
        """Run the multiscale GWR backfitting algorithm.

        Returns
        -------
        self
            The fitted instance with ``result_layer`` populated.
        """
        self.algorithm.fit()

        for i, sw in enumerate(self.algorithm.spatial_weights):
            bw_info = sw.weight()
            self.weights[i].bandwidth = bw_info[1]
            self.weights[i].adaptive = bw_info[2]
            self.weights[i].kernel = BandwidthWeight.Kernel(bw_info[3])

        indep_var_names = (['Intercept'] if self.has_intercept else []) + self.indep_vars
        result_data = {
            **{f: self.algorithm.betas[:, i] for i, f in enumerate(indep_var_names)},
            **{f'{f}_SE': self.algorithm.betasSE[:, i] for i, f in enumerate(indep_var_names)},
        }
        if self.algorithm.has_hat_matrix:
            result_data.update(
                {f'{f}_TV': self.algorithm.betasTV[:, i] for i, f in enumerate(indep_var_names)}
            )
        result_data['fitted'] = np.sum(
            self.algorithm.independent * self.algorithm.betas, axis=1
        )
        self.result_layer = gp.GeoDataFrame(result_data, geometry=self.geometry)
        return self

    @property
    def diagnostic(self):
        """dict: Regression diagnostics (RSS, AICc, ENP, EDF,
        RSquare, RSquareAdjust)."""
        return self.algorithm.diagnostic if self.algorithm else None

    @property
    def betas(self):
        """ndarray: Coefficient estimates (n_samples × n_vars)."""
        return self.algorithm.betas if self.algorithm else None

    @property
    def betasSE(self):
        """ndarray: Standard errors of coefficients."""
        return self.algorithm.betasSE if self.algorithm else None

    @property
    def betasTV(self):
        """ndarray: t-values of coefficients (only when
        ``has_hat_matrix=True``)."""
        return self.algorithm.betasTV if self.algorithm else None
