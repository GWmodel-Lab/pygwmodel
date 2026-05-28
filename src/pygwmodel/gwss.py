from typing import List, Optional
import numpy as np
import geopandas as gp
from .spatial_weight import BandwidthWeight, CRSDistance, Distance, SpatialWeight
from .parallel import ParallelType
from ._analysis import _GWAverage, _GWCorrelation


class GWAverage:
    """Geographically Weighted Average — local summary statistics.

    Computes locally weighted descriptive statistics for each
    observation, including local mean, standard deviation,
    variance, skewness, coefficient of variation, and optionally
    local median, interquartile range (IQR) and quantile
    imbalance (QI).

    Supports OpenMP parallel computation.

    Parameters
    ----------
    sdf : GeoDataFrame
        Input spatial data.
    vars : list of str
        Variable column names to analyse.
    weight : BandwidthWeight
        Bandwidth and kernel configuration.
    distance : Distance, optional
        Distance metric (default CRSDistance).
    quantile : bool, optional
        Also compute median, IQR, and QI (default False).

    Attributes
    ----------
    result_layer : GeoDataFrame or None
        Results after calling :meth:`run`.
    """

    def __init__(self, sdf: gp.GeoDataFrame, vars: List[str],
                 weight: BandwidthWeight, distance: Distance = CRSDistance(),
                 quantile: bool = False) -> None:
        self.geometry = sdf.geometry
        self.vars = vars
        self.weight = weight
        self.distance = distance
        self.quantile = quantile
        self.result_layer: Optional[gp.GeoDataFrame] = None
        self.algorithm = _GWAverage()
        self.algorithm.coords = np.asfortranarray(
            sdf.geometry.centroid.get_coordinates(), dtype=np.float64)
        self.algorithm.variables = np.asfortranarray(
            sdf[self.vars], dtype=np.float64)
        self.algorithm.spatial_weight = SpatialWeight.create(distance, weight)
        self.algorithm.quantile = self.quantile

    def enable_parallel(self, type: ParallelType, **kwargs):
        """Enable OpenMP parallel computation.

        Parameters
        ----------
        type : ParallelType
            Parallel backend (only OpenMP supported).
        **kwargs
            ``threads`` (int) — number of OpenMP threads.
        """
        if type == ParallelType.OpenMP:
            threads = kwargs.get('threads', 8)
            if isinstance(threads, int) and threads > 0:
                self.algorithm.parallel_omp(threads)
            else:
                raise ValueError("threads must be a positive integer")
        return self

    def run(self, quantile: bool = False):
        """Execute the GW average computation.

        Parameters
        ----------
        quantile : bool
            Whether to compute quantile-based statistics.

        Returns
        -------
        self
            Instance with ``result_layer`` populated.
        """
        self.quantile = quantile
        self.algorithm.quantile = self.quantile
        self.algorithm.run()
        result_data = {
            **{f'{f}_Mean': self.local_mean[:, i]
               for i, f in enumerate(self.vars)},
            **{f'{f}_SDev': self.local_sdev[:, i]
               for i, f in enumerate(self.vars)},
            **{f'{f}_Skew': self.local_skewness[:, i]
               for i, f in enumerate(self.vars)},
            **{f'{f}_CV': self.local_cv[:, i]
               for i, f in enumerate(self.vars)}
        }
        if self.quantile:
            result_data = {
                **result_data,
                **{f'{f}_Median': self.local_median[:, i]
                   for i, f in enumerate(self.vars)},
                **{f'{f}_IQR': self.iqr[:, i]
                   for i, f in enumerate(self.vars)},
                **{f'{f}_QI': self.qi[:, i]
                   for i, f in enumerate(self.vars)}
            }
        self.result_layer = gp.GeoDataFrame(result_data, geometry=self.geometry)
        return self

    @property
    def local_mean(self):
        """ndarray: Local weighted mean for each variable."""
        return self.algorithm.local_mean

    @property
    def local_sdev(self):
        """ndarray: Local weighted standard deviation."""
        return self.algorithm.local_sdev

    @property
    def local_skewness(self):
        """ndarray: Local weighted skewness."""
        return self.algorithm.local_skewness

    @property
    def local_cv(self):
        """ndarray: Local coefficient of variation."""
        return self.algorithm.local_cv

    @property
    def local_var(self):
        """ndarray: Local weighted variance."""
        return self.algorithm.local_var

    @property
    def local_median(self):
        """ndarray: Local weighted median (requires ``quantile=True``)."""
        return self.algorithm.local_median

    @property
    def iqr(self):
        """ndarray: Local interquartile range (requires ``quantile=True``)."""
        return self.algorithm.iqr

    @property
    def qi(self):
        """ndarray: Local quantile imbalance (requires ``quantile=True``)."""
        return self.algorithm.qi


class GWCorrelation:
    """Geographically Weighted Correlation — local correlation statistics.

    Computes local Pearson and Spearman rank correlation coefficients
    for every pair of variables using locally weighted covariance.

    Supports OpenMP parallel computation.

    Parameters
    ----------
    sdf : GeoDataFrame
        Input spatial data.
    vars : list of str
        Variable column names to correlate.
    weight : BandwidthWeight
        Bandwidth and kernel configuration (one weight shared
        across all variable pairs).
    distance : Distance, optional
        Distance metric (default CRSDistance).

    Attributes
    ----------
    result_layer : GeoDataFrame or None
        Results after calling :meth:`run`.
    """

    def __init__(self, sdf: gp.GeoDataFrame, vars: List[str],
                 weight: BandwidthWeight, distance: Distance = CRSDistance()) -> None:
        self.geometry = sdf.geometry
        self.vars = vars
        self.weight = weight
        self.distance = distance
        self.result_layer: Optional[gp.GeoDataFrame] = None
        self.algorithm = _GWCorrelation()
        self.algorithm.coords = np.asfortranarray(
            sdf.geometry.centroid.get_coordinates(), dtype=np.float64)
        vars_mat = np.asfortranarray(sdf[self.vars], dtype=np.float64)
        self.algorithm.variables1 = vars_mat
        self.algorithm.variables2 = vars_mat
        n_var = len(self.vars)
        n_col = n_var * n_var
        sw = SpatialWeight.create(distance, weight)
        self.algorithm.spatial_weights = [sw] * n_col
        self.algorithm.bandwidth_initilize = \
            [_GWCorrelation.BandwidthInitilizeType.Specified] * n_col
        self.algorithm.bandwidth_selection_approach = \
            [_GWCorrelation.BandwidthSelectionCriterionType.CV] * n_col

    def enable_parallel(self, type: ParallelType, **kwargs):
        """Enable OpenMP parallel computation.

        Parameters
        ----------
        type : ParallelType
            Parallel backend (only OpenMP supported).
        **kwargs
            ``threads`` (int) — number of OpenMP threads.
        """
        if type == ParallelType.OpenMP:
            threads = kwargs.get('threads', 8)
            if isinstance(threads, int) and threads > 0:
                self.algorithm.parallel_omp(threads)
            else:
                raise ValueError("threads must be a positive integer")
        return self

    def run(self):
        """Execute the GW correlation computation.

        Returns
        -------
        self
            Instance with ``result_layer`` populated.
        """
        self.algorithm.run()
        n_var = len(self.vars)
        pairs = [(i, j) for i in range(n_var) for j in range(i + 1, n_var)]
        col_indices = [i * n_var + j for i, j in pairs]
        result_data = {
            **{f'{self.vars[ci]}.{self.vars[cj]}_Corr':
               self.local_corr[:, col_idx]
               for (ci, cj), col_idx in zip(pairs, col_indices)},
            **{f'{self.vars[ci]}.{self.vars[cj]}_SCorr':
               self.local_s_corr[:, col_idx]
               for (ci, cj), col_idx in zip(pairs, col_indices)}
        }
        self.result_layer = gp.GeoDataFrame(result_data, geometry=self.geometry)
        return self

    @property
    def local_mean(self):
        """ndarray: Local weighted mean for each variable."""
        return self.algorithm.local_mean

    @property
    def local_sdev(self):
        """ndarray: Local weighted standard deviation."""
        return self.algorithm.local_sdev

    @property
    def local_skewness(self):
        """ndarray: Local weighted skewness."""
        return self.algorithm.local_skewness

    @property
    def local_cv(self):
        """ndarray: Local coefficient of variation."""
        return self.algorithm.local_cv

    @property
    def local_var(self):
        """ndarray: Local weighted variance."""
        return self.algorithm.local_var

    @property
    def local_cov(self):
        """ndarray: Local weighted covariances for each variable
        pair (Pearson)."""
        return self.algorithm.local_cov

    @property
    def local_corr(self):
        """ndarray: Local Pearson correlation coefficients for each
        variable pair."""
        return self.algorithm.local_corr

    @property
    def local_s_corr(self):
        """ndarray: Local Spearman rank correlation coefficients for
        each variable pair."""
        return self.algorithm.local_s_corr
