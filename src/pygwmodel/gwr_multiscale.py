from typing import List, Optional
from enum import IntEnum
import numpy as np
import geopandas as gp
from .spatial_weight import SpatialWeight, Distance, BandwidthWeight, CRSDistance
from .parallel import ParallelType
from ._regression import _GWRMultiscale


class GWRMultiscale:
    """
    Multiscale GWR python high api class.
    """

    class BandwidthInitilizeType(IntEnum):
        Null = _GWRMultiscale.BandwidthInitilizeType.Null
        Initial = _GWRMultiscale.BandwidthInitilizeType.Initial
        Specified = _GWRMultiscale.BandwidthInitilizeType.Specified

    class BandwidthSelectionCriterionType(IntEnum):
        CV = _GWRMultiscale.BandwidthSelectionCriterionType.CV
        AIC = _GWRMultiscale.BandwidthSelectionCriterionType.AIC

    class BackFittingCriterionType(IntEnum):
        CVR = _GWRMultiscale.BackFittingCriterionType.CVR
        dCVR = _GWRMultiscale.BackFittingCriterionType.dCVR

    def __init__(self, sdf: gp.GeoDataFrame, depen_var: str, indep_vars: List[str],
                 weights: List[BandwidthWeight], distance: Distance = CRSDistance(),
                 has_intercept: bool = True,
                 bandwidth_initilize: Optional[List[BandwidthInitilizeType]] = None,
                 bandwidth_selection_approach: Optional[List[BandwidthSelectionCriterionType]] = None,
                 preditor_centered: Optional[List[bool]] = None,
                 has_hat_matrix: bool = True):
        """
        Initialize Multiscale GWR.

        Parameters
        ----------
        sdf : GeoDataFrame
            Input spatial data.
        depen_var : str
            Name of the dependent variable column.
        indep_vars : List[str]
            Names of independent variable columns.
        weights : List[BandwidthWeight]
            Bandwidth weights, one per independent variable (plus intercept if has_intercept=True).
        distance : Distance
            Distance metric. Defaults to CRSDistance.
        has_intercept : bool
            Whether to include an intercept term.
        bandwidth_initilize : Optional[List[BandwidthInitilizeType]]
            Bandwidth initialization types, one per variable.
            Defaults to Null (auto-select) for all.
        bandwidth_selection_approach : Optional[List[BandwidthSelectionCriterionType]]
            Bandwidth selection criterion, one per variable.
            Defaults to CV for all.
        preditor_centered : Optional[List[bool]]
            Whether to center each predictor. Defaults to False for all.
        has_hat_matrix : bool
            Whether to store the hat matrix S for diagnostic computation.
        """
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

        # Default bandwidth initilize: Null (auto-select) for all
        if bandwidth_initilize is None:
            bandwidth_initilize = [GWRMultiscale.BandwidthInitilizeType.Null] * n_var
        if len(bandwidth_initilize) != n_var:
            raise ValueError(f"bandwidth_initilize must have length {n_var}")

        # Default bandwidth selection approach: CV for all
        if bandwidth_selection_approach is None:
            bandwidth_selection_approach = [GWRMultiscale.BandwidthSelectionCriterionType.CV] * n_var
        if len(bandwidth_selection_approach) != n_var:
            raise ValueError(f"bandwidth_selection_approach must have length {n_var}")

        # Default preditor_centered: False for all
        if preditor_centered is None:
            preditor_centered = [False] * n_var
        if len(preditor_centered) != n_var:
            raise ValueError(f"preditor_centered must have length {n_var}")

        self.weights = weights
        self.bandwidth_initilize: List[GWRMultiscale.BandwidthInitilizeType] = bandwidth_initilize
        self.bandwidth_selection_approach: List[GWRMultiscale.BandwidthSelectionCriterionType] = bandwidth_selection_approach

        # Build independent variable matrix
        indep_vars_data = np.asfortranarray(sdf[indep_vars], dtype=np.float64)
        if has_intercept:
            indep_vars_data = np.hstack([np.ones((indep_vars_data.shape[0], 1)), indep_vars_data])

        # Create spatial weights (one per variable)
        spatial_weights = [SpatialWeight.create(distance, w) for w in weights]

        self.algorithm = _GWRMultiscale()
        self.algorithm.coords = np.asfortranarray(sdf.geometry.centroid.get_coordinates(), dtype=np.float64)
        self.algorithm.independent = indep_vars_data
        self.algorithm.dependent = np.asfortranarray(sdf[depen_var], dtype=np.float64)
        self.algorithm.spatial_weights = spatial_weights
        self.algorithm.has_intercept = has_intercept
        self.algorithm.has_hat_matrix = has_hat_matrix

        # Set bandwidth configuration
        self.algorithm.set_bandwidth_initilize(
            [_GWRMultiscale.BandwidthInitilizeType(t.value) for t in bandwidth_initilize]
        )
        self.algorithm.set_bandwidth_selection_approach(
            [_GWRMultiscale.BandwidthSelectionCriterionType(t.value) for t in bandwidth_selection_approach]
        )
        self.algorithm.preditor_centered = preditor_centered

        # Default threshold values (one per variable)
        self.algorithm.bandwidth_select_threshold = [1.0e-6] * n_var

    def enable_parallel_omp(self, threads: int = 8):
        if self.algorithm is None:
            raise ValueError("Not initialized")
        if isinstance(threads, int) and threads > 0:
            self.algorithm.parallel_omp(threads)
        else:
            raise ValueError("threads must be a positive integer")
        return self

    def enable_parallel_cuda(self, gpu_id: int = 0, group_size: int = 64):
        if self.algorithm is None:
            raise ValueError("Not initialized")
        if all([(isinstance(x, int) and x > 0) for x in [gpu_id, group_size]]):
            self.algorithm.parallel_cuda(gpu_id, group_size)
        else:
            raise ValueError("gpu_id and group_size must be positive integers")
        return self

    def enable_parallel(self, type: ParallelType, **kvargs):
        if type == ParallelType.OpenMP:
            self.enable_parallel_omp(**kvargs)
        elif type == ParallelType.CUDA:
            self.enable_parallel_cuda(**kvargs)
        return self

    def fit(self):
        """
        Run the multiscale GWR algorithm and return self.
        """
        self.algorithm.fit()

        # Update bandwidths from fitted results
        for i, sw in enumerate(self.algorithm.spatial_weights):
            bw_info = sw.weight()
            self.weights[i].bandwidth = bw_info[1]
            self.weights[i].adaptive = bw_info[2]
            self.weights[i].kernel = BandwidthWeight.Kernel(bw_info[3])

        # Build result GeoDataFrame
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
        return self.algorithm.diagnostic if self.algorithm else None

    @property
    def betas(self):
        return self.algorithm.betas if self.algorithm else None

    @property
    def betasSE(self):
        return self.algorithm.betasSE if self.algorithm else None

    @property
    def betasTV(self):
        return self.algorithm.betasTV if self.algorithm else None
