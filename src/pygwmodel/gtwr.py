from typing import List, Optional
import numpy as np
import geopandas as gp
from .parallel import ParallelType
from ._regression import _GTWR


class GTWR:
    """Geographically and Temporally Weighted Regression.

    GTWR extends GWR by incorporating temporal information via a
    spatio-temporal distance metric with a spatio-temporal ratio
    parameter (lambda).

    Parameters
    ----------
    sdf : GeoDataFrame
        Input data with geometry column.
    depen_var : str
        Dependent variable column name.
    indep_vars : List[str]
        Independent variable column names.
    times : str
        Column name for temporal stamps.
    weight : BandwidthWeight
        Bandwidth weight configuration.
    distance : CRSSTDistance, optional
        Spatio-temporal distance (default CRSSTDistance).
    has_intercept : bool, optional
        Include intercept term (default True).
    """

    BandwidthSelectionCriterionType = _GTWR.BandwidthSelectionCriterionType

    def __init__(self, sdf: gp.GeoDataFrame, depen_var: str,
                 indep_vars: List[str], times: str,
                 weight, distance=None,
                 has_intercept: bool = True):
        if not isinstance(sdf, gp.GeoDataFrame):
            raise ValueError("sdf must be a GeoDataFrame")
        self.geometry = sdf.geometry
        self.depen_var: str = depen_var
        self.indep_vars: List[str] = indep_vars
        self.times_col: str = times
        self.has_intercept: bool = has_intercept
        self.weight = weight
        if distance is None:
            from .spatial_weight import CRSSTDistance
            distance = CRSSTDistance()
        self.distance = distance
        self.result_layer: Optional[gp.GeoDataFrame] = None
        self.algorithm = _GTWR()
        coords = np.asfortranarray(
            sdf.geometry.centroid.get_coordinates(), dtype=np.float64)
        self.algorithm.coords = coords
        times_vec = np.asfortranarray(sdf[self.times_col], dtype=np.float64)
        self.algorithm.set_times_vec(times_vec)
        indep_vars_data = np.asfortranarray(
            sdf[self.indep_vars], dtype=np.float64)
        if self.has_intercept:
            indep_vars_data = np.hstack(
                [np.ones((indep_vars_data.shape[0], 1)), indep_vars_data])
        self.algorithm.independent = indep_vars_data
        self.algorithm.dependent = np.asfortranarray(
            sdf[self.depen_var], dtype=np.float64)
        from .spatial_weight import SpatialWeight
        self.algorithm.spatial_weight = \
            SpatialWeight.create(distance, weight)
        self.algorithm.prepare_st_distance()

    def enable_parallel(self, type: ParallelType, **kwargs):
        if type == ParallelType.OpenMP:
            threads = kwargs.get('threads', 8)
            if isinstance(threads, int) and threads > 0:
                self.algorithm.parallel_omp(threads)
            else:
                raise ValueError("threads must be a positive integer")
        return self

    def fit(self,
            optimize_bandwidth:
            Optional[BandwidthSelectionCriterionType] = None):
        if (self.weight.bandwidth is None
                and optimize_bandwidth is None):
            optimize_bandwidth = GTWR.BandwidthSelectionCriterionType.CV
        if optimize_bandwidth is not None:
            v = optimize_bandwidth
            self.algorithm.set_select_bandwidth(True, v.value)
        self.algorithm.fit()
        if optimize_bandwidth is not None:
            self.weight.bandwidth = \
                self.algorithm.spatial_weight.weight()[1]
        indep_var_names = (['Intercept'] if self.has_intercept else []) \
            + self.indep_vars
        result_data = {
            **{f: self.algorithm.betas[:, i]
               for i, f in enumerate(indep_var_names)},
            **{f'{f}_SE': self.algorithm.betasSE[:, i]
               for i, f in enumerate(indep_var_names)},
            'fitted': np.sum(
                self.algorithm.independent * self.algorithm.betas, axis=1)
        }
        self.result_layer = gp.GeoDataFrame(
            result_data, geometry=self.geometry)
        return self

    def predict(self, targets: gp.GeoDataFrame):
        if self.weight.bandwidth is None:
            raise ValueError("Bandwidth cannot be None when predicting")
        predict_locations = np.asfortranarray(
            targets.geometry.centroid.get_coordinates())
        coef_predict = self.algorithm.predict(predict_locations)
        indep_var_names = (['Intercept'] if self.has_intercept else []) \
            + self.indep_vars
        result_data = {
            **{f: coef_predict[:, i]
               for i, f in enumerate(indep_var_names)}
        }
        if all(x in targets.columns for x in self.indep_vars):
            px = np.asfortranarray(targets[self.indep_vars])
            if self.has_intercept:
                px = np.hstack([np.ones((px.shape[0], 1)), px])
            result_data['y_hat'] = np.sum(px * coef_predict, axis=1)
            if self.depen_var in targets.columns:
                py = targets[self.depen_var]
                result_data["residual"] = py - result_data["y_hat"]
        return gp.GeoDataFrame(result_data, geometry=targets.geometry)

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
    def bandwidth_criterions(self):
        return self.algorithm.bandwidth_criterions \
            if self.algorithm else None

    @property
    def lambda_(self):
        return self.algorithm.lambda_ if self.algorithm else None

    @property
    def angle(self):
        return self.algorithm.angle if self.algorithm else None
