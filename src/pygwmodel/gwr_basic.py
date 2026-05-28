from typing import List, Union, Optional
import numpy as np
import geopandas as gp
from .spatial_weight import SpatialWeight, Distance, BandwidthWeight
from .parallel import ParallelType
from ._regression import _GWRBasic


class GWRBasic:
    """Basic Geographically Weighted Regression (GWR).

    GWR is a local form of linear regression used to model spatially
    varying relationships.  A separate regression is calibrated at each
    observation location, with observations weighted by their distance
    from the calibration point via a kernel function.

    The algorithm can optionally auto-select the optimal bandwidth
    (via CV or AIC) and/or perform forward variable selection.

    Supports OpenMP multi-threading and CUDA GPU acceleration.

    Parameters
    ----------
    sdf : GeoDataFrame
        Input spatial data with a geometry column.
    depen_var : str
        Name of the dependent variable column.
    indep_vars : list of str
        Names of the independent variable columns.
    weight : BandwidthWeight
        Bandwidth and kernel configuration.
    distance : Distance
        Distance metric (e.g. CRSDistance).
    has_intercept : bool
        Whether to include an intercept term (default True).

    Attributes
    ----------
    result_layer : GeoDataFrame or None
        The fitted result, populated after calling :meth:`fit`.
    """

    BandwidthSelectionCriterionType = _GWRBasic.BandwidthSelectionCriterionType

    def __init__(self, sdf: gp.GeoDataFrame, depen_var: str,
                 indep_vars: List[str], weight: BandwidthWeight,
                 distance: Distance, has_intercept=True):
        if not isinstance(sdf, gp.GeoDataFrame):
            raise ValueError("sdf must be a GeoDataFrame")
        self.geometry = sdf.geometry
        self.depen_var: str = depen_var
        self.indep_vars: List[str] = indep_vars
        self.has_intercept: bool = has_intercept
        self.weight = weight
        self.distance = distance
        self.result_layer: Optional[gp.GeoDataFrame] = None
        indep_vars_data = np.asfortranarray(sdf[self.indep_vars], dtype=np.float64)
        if (self.has_intercept):
            indep_vars_data = np.hstack([np.ones((indep_vars_data.shape[0], 1)), indep_vars_data])
        self.algorithm = _GWRBasic()
        self.algorithm.coords = np.asfortranarray(sdf.geometry.centroid.get_coordinates(), dtype=np.float64)
        self.algorithm.independent = indep_vars_data
        self.algorithm.dependent = np.asfortranarray(sdf[self.depen_var], dtype=np.float64)
        self.algorithm.spatial_weight = SpatialWeight.create(distance, weight)

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

    def fit(self,
            optimize_bw: Optional[BandwidthSelectionCriterionType] = None,
            optimize_var: Optional[float] = None):
        """Fit the GWR model.

        Parameters
        ----------
        optimize_bw : BandwidthSelectionCriterionType, optional
            Criterion for automatic bandwidth selection
            (AIC or CV).  If *None* and the weight's bandwidth
            is already set, no selection is performed.
        optimize_var : float, optional
            Positive threshold for forward variable selection.
            Variables are selected until the criterion improvement
            is below this threshold.

        Returns
        -------
        self
            The fitted instance with ``result_layer`` populated.
        """
        if self.weight.bandwidth is None and optimize_bw is None:
            optimize_bw = GWRBasic.BandwidthSelectionCriterionType.CV
        if optimize_bw is not None:
            if optimize_bw == GWRBasic.BandwidthSelectionCriterionType.AIC or optimize_bw == GWRBasic.BandwidthSelectionCriterionType.CV:
                self.algorithm.enable_select_bandwidth(optimize_bw.value)
            else:
                raise ValueError("optimize_bw must be BandwidthSelectionCriterionType.AIC(0) or BandwidthSelectionCriterionType.CV(1)")
        if optimize_var is not None:
            if isinstance(optimize_var, float) and optimize_var > 0:
                self.algorithm.enable_select_variables(optimize_var)
            else:
                raise ValueError("optimize_var must be a positive real number")
        self.algorithm.fit()
        if self.weight.bandwidth is None or optimize_bw is not None:
            self.weight.bandwidth = self.algorithm.spatial_weight.weight()[1]
        if optimize_var is not None:
            self._indep_vars_old = self.indep_vars
            self.indep_vars = [self.indep_vars[v - int(self.has_intercept)] for v in self.algorithm.selected_variables]
            pass
        indep_var_names = (['Intercept'] if self.has_intercept else []) + self.indep_vars
        result_data = {
            **{f: self.algorithm.betas[:, i] for i, f in enumerate(indep_var_names)},
            **{f'{f}_SE': self.algorithm.betasSE[:, i] for i, f in enumerate(indep_var_names)},
            'fitted': self.algorithm.fitted
        }
        self.result_layer = gp.GeoDataFrame(result_data, geometry=self.geometry)
        return self

    @property
    def diagnostic(self):
        """dict: Regression diagnostics (RSS, AIC, AICc, ENP, EDF,
        RSquare, RSquareAdjust)."""
        return self.algorithm.diagnostic if self.algorithm else None

    @property
    def bandwidth_select_criterions(self):
        """list: Bandwidth selection criterion values across the
        golden-section search."""
        return self.algorithm.bandwidth_criterions if self.algorithm else None

    @property
    def indep_var_select_criterions(self):
        """list of tuple: Forward variable selection criterion values,
        each as ``(variable_list, criterion)``."""
        return [([self._indep_vars_old[v - int(self.has_intercept)] for v in varlist], criterion) for varlist, criterion in self.algorithm.variables_criterions] if self.algorithm else None

    def predict(self, targets: gp.GeoDataFrame):
        """Predict coefficients at new locations.

        Parameters
        ----------
        targets : GeoDataFrame
            Locations at which to predict.

        Returns
        -------
        GeoDataFrame
            Predicted coefficients for each target location.
            If the target GeoDataFrame contains the independent
            variables, ``y_hat`` and ``residual`` columns are
            also computed.
        """
        if self.weight.bandwidth is None:
            raise ValueError("Bandwidth cannot be None when predicting")
        predict_locations = np.asfortranarray(targets.centroid.get_coordinates())
        coef_predict = self.algorithm.predict(predict_locations)
        indep_var_names = (['Intercept'] if self.has_intercept else []) + self.indep_vars
        result_data = {
            **{f: coef_predict[:, i] for i, f in enumerate(indep_var_names)}
        }
        if all([x in targets.columns for x in self.indep_vars]):
            px = np.asfortranarray(targets[self.indep_vars])
            if self.has_intercept:
                px = np.hstack([np.ones((px.shape[0], 1)), px])
            result_data['y_hat'] = np.sum(px * coef_predict, axis=1)
            if self.depen_var in targets.columns:
                py = targets[self.depen_var]
                result_data["residual"] = py - result_data["y_hat"]
        return gp.GeoDataFrame(result_data, geometry=targets.geometry)
