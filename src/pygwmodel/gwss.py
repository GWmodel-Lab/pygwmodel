from typing import List, Optional
from enum import Enum
import numpy as np
import geopandas as gp
from .spatial_weight import *
from .parallel import ParallelType
from ._analysis import _GWSS


class GWSS:

    class Mode(Enum):
        Average = _GWSS.Mode.Average
        Correlation = _GWSS.Mode.Correlation
    
    def __init__(self, sdf: gp.GeoDataFrame, vars: List[str], weight: BandwidthWeight, distance: Distance=CRSDistance(), mode: Mode=Mode.Average, quantile: bool=False) -> None:
        self.geometry = sdf.geometry
        self.vars = vars
        self.mode = mode
        self.weight = weight
        self.distance = distance
        self.quantile = quantile
        self.result_layer: Optional[gp.GeoDataFrame] = None
        self.algorithm = _GWSS()
        self.algorithm.coords = np.asfortranarray(sdf.geometry.centroid.get_coordinates(), dtype=np.float64)
        self.algorithm.variables = np.asfortranarray(sdf[self.vars], dtype=np.float64)
        self.algorithm.spatial_weight = SpatialWeight.create(distance, weight)
        self.algorithm.quantile = self.quantile
        self.algorithm.set_mode(self.mode.value)
    
    def enable_parallel_omp(self, threads: int=8):
        if self.algorithm is None:
            raise ValueError("Not initialized")
        if isinstance(threads, int) and threads > 0:
            self.algorithm.parallel_omp(threads)
        else:
            raise ValueError("threads must be a positive integer")
        return self

    def enable_parallel(self, type: ParallelType, **kvargs):
        if type == ParallelType.OpenMP:
            self.enable_parallel_omp(**kvargs)
        return self
    
    def run(self, mode: Mode=Mode.Average, quantile: bool=False):
        self.mode = mode
        self.algorithm.set_mode(self.mode.value)
        self.algorithm.quantile = quantile
        self.algorithm.run()
        if self.mode == GWSS.Mode.Average:
            result_data = {
                **{f'{f}_Mean': self.local_mean[:, i] for i, f in enumerate(self.vars)},
                **{f'{f}_SDev': self.local_sdev[:, i] for i, f in enumerate(self.vars)},
                **{f'{f}_Skew': self.local_skewness[:, i] for i, f in enumerate(self.vars)},
                **{f'{f}_CV': self.local_cv[:, i] for i, f in enumerate(self.vars)}
            }
            if self.quantile:
                result_data = {
                    **result_data,
                    **{f'{f}_Median': self.local_median[:, i] for i, f in enumerate(self.vars)},
                    **{f'{f}_IQR': self.iqr[:, i] for i, f in enumerate(self.vars)},
                    **{f'{f}_QI': self.qi[:, i] for i, f in enumerate(self.vars)}
                }
        elif self.mode == GWSS.Mode.Correlation:
            columns = [(fi, fj) for i, fi in enumerate(self.vars) for _, fj in enumerate(self.vars[(i+1):])]
            result_data = {
                **{f'{fi}.{fj}_Corr': self.local_corr[:, i] for i, (fi, fj) in enumerate(columns)},
                **{f'{fi}.{fj}_SCorr': self.local_s_corr[:, i] for i, (fi, fj) in enumerate(columns)}
            }
        else:
            raise ValueError("Not such mode.")
        self.result_layer = gp.GeoDataFrame(result_data, geometry=self.geometry)
        return self

    @property
    def local_mean(self):
        return self.algorithm.local_mean
        
    @property
    def local_sdev(self):
        return self.algorithm.local_sdev
        
    @property
    def local_skewness(self):
        return self.algorithm.local_skewness
        
    @property
    def local_cv(self):
        return self.algorithm.local_cv
        
    @property
    def local_var(self):
        return self.algorithm.local_var
        
    @property
    def local_median(self):
        return self.algorithm.local_median
        
    @property
    def iqr(self):
        return self.algorithm.iqr
        
    @property
    def qi(self):
        return self.algorithm.qi
        
    @property
    def local_cov(self):
        return self.algorithm.local_cov
        
    @property
    def local_corr(self):
        return self.algorithm.local_corr
        
    @property
    def local_s_corr(self):
        return self.algorithm.local_s_corr
        
