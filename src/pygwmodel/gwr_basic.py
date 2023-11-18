from typing import List, Union, Optional
import numpy as np
import geopandas as gp
from enum import IntEnum
from .spatial_weight import SpatialWeight, Distance, BandwidthWeight
from ._gwr_basic import _GWRBasic


class ParallelType(IntEnum):
    Serial = 1
    OpenMP = 1 << 1
    CUDA = 1 << 2


class GWRBasic:
    """
    Basic GWR python high api class.
    """

    class BandwidthSelectionCriterionType(IntEnum):
        AIC = _GWRBasic.AIC
        CV = _GWRBasic.CV

    def __init__(self, sdf: gp.GeoDataFrame, depen_var: str, indep_vars: List[str], weight: BandwidthWeight, distance: Distance, has_intercept=True):
        """
        docstring
        """
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
    
    def enable_parallel_omp(self, threads: int=8):
        if self.algorithm is None:
            raise ValueError("Not initialized")
        if isinstance(threads, int) and threads > 0:
            self.algorithm.parallel_omp(threads)
        else:
            raise ValueError("threads must be a positive integer")
        return self
    
    def enable_parallel_cuda(self, gpu_id: int=0, group_size: int=64):
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
    
    def fit(self, optimize_bw: Optional[BandwidthSelectionCriterionType]=None, optimize_var: Optional[float]=None):
        """
        Run algorithm and return result
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
        ''' Get result layer
        '''
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
        return self.algorithm.diagnostic if self.algorithm else None
    
    @property
    def bandwidth_select_criterions(self):
        return self.algorithm.bandwidth_criterions if self.algorithm else None
    
    @property
    def indep_var_select_criterions(self):
        return [([self._indep_vars_old[v - int(self.has_intercept)] for v in varlist], criterion) for varlist, criterion in self.algorithm.variables_criterions] if self.algorithm else None


    def predict(self, targets: gp.GeoDataFrame):
        """
        Predict
        """
        if self.weight.bandwidth is None:
            raise ValueError("Bandwidth cannot be None when predicting")
        predict_locations = np.asfortranarray(targets.centroid.get_coordinates())
        coef_predict = self.algorithm.predict(predict_locations)
        ''' Get result layer
        '''
        indep_var_names = (['Intercept'] if self.has_intercept else []) + self.indep_vars
        result_data = {
            **{f: coef_predict[:, i] for i, f in enumerate(indep_var_names)}
        }
        if all([x in targets.columns for x in self.indep_vars]):
            ''' If all variables are in predicting targets, calculate estimated y
            '''
            px = np.asfortranarray(targets[self.indep_vars])
            if self.has_intercept:
                px = np.hstack([np.ones((px.shape[0], 1)), px])
            result_data['y_hat'] = np.sum(px * coef_predict, axis=1)
            if self.depen_var in targets.columns:
                py = targets[self.depen_var]
                result_data["residual"] = py - result_data["y_hat"]
        return gp.GeoDataFrame(result_data, geometry=targets.geometry)
