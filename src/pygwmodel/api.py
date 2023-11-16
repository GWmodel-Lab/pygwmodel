from typing import List, Union, Optional
import numpy as np
import geopandas as gp
from enum import IntEnum
from .py_spatial_weight import SpatialWeight as SpatialWeightBind
from .py_gwr_basic import GWRBasic as GWRBasicBind

class KernelType(IntEnum):
    GAUSSIAN = 0


class GWSSMode(IntEnum):
    Average = 0
    Correlation = 1


class GWRBasic:
    """
    Basic GWR python high api class.
    """

    class BandwidthSelectionCriterionType(IntEnum):
        AIC = GWRBasicBind.AIC
        CV = GWRBasicBind.CV

    def __init__(self, sdf: gp.GeoDataFrame, depen_var: str, indep_vars: List[str], bw: Union[float, None]=None, adaptive: bool=True, kernel: KernelType=KernelType.GAUSSIAN, longlat: bool=True, has_intercept=True):
        """
        docstring
        """
        if not isinstance(sdf, gp.GeoDataFrame):
            raise ValueError("sdf must be a GeoDataFrame")
        self.sdf = sdf
        self.depen_var = depen_var
        self.indep_vars = indep_vars
        self.has_intercept = has_intercept
        self.bw = bw
        self.kernel = kernel
        self.adaptive = adaptive
        self.longlat = longlat
        self.result_layer = None
        self.diagnostic = None
        self.bandwidth_select_criterions: Optional[list[tuple[float, float]]] = None
        self.indep_var_select_criterions: Optional[list[tuple[list[str], float]]] = None
    
    def fit(self, hatmatrix: bool=True, optimize_bw: Optional[BandwidthSelectionCriterionType]=None, optimize_var: Optional[float]=None, multithreads: Optional[int]=None):
        """
        Run algorithm and return result
        """
        ''' Extract data
        '''
        # cyg_distance = CyCRSDistance(self.longlat)
        # cyg_weight = CyBandwidthWeight(self.bw, self.adaptive, self.kernel.value)
        sw = SpatialWeightBind()
        sw.set_distance_crs(self.longlat)
        sw.set_weight_bandwidth(self.bw, self.adaptive, self.kernel.value)
        ''' Create cython GWR
        '''
        depen_var = np.asfortranarray(self.sdf[self.depen_var], dtype=np.float64)
        indep_vars = np.asfortranarray(self.sdf[self.indep_vars], dtype=np.float64)
        if (self.has_intercept):
            indep_vars = np.hstack([np.ones((indep_vars.shape[0], 1)), indep_vars])
        coords = np.asfortranarray(self.sdf.geometry.centroid.get_coordinates(), dtype=np.float64)
        # algorithm = CyGWRBasic(coords, depen_var, indep_vars, cyg_weight, cyg_distance, self.has_intercept)
        algorithm = GWRBasicBind()
        algorithm.coords = coords
        algorithm.independent = indep_vars
        algorithm.dependent = depen_var
        algorithm.spatial_weight = sw
        if self.bw is None and optimize_bw is None:
            optimize_bw = GWRBasic.BandwidthSelectionCriterionType.CV
        if optimize_bw is not None:
            if optimize_bw == GWRBasic.BandwidthSelectionCriterionType.AIC or optimize_bw == GWRBasic.BandwidthSelectionCriterionType.CV:
                algorithm.select_bandwidth = True
                algorithm.select_bandwidth_criterion = optimize_bw.value
            else:
                raise ValueError("optimize_bw must be BandwidthSelectionCriterionType.AIC(0) or BandwidthSelectionCriterionType.CV(1)")
        if optimize_var is not None:
            if isinstance(optimize_var, float) and optimize_var > 0:
                algorithm.select_variables = True
                algorithm.select_variables_threshold = optimize_var
            else:
                raise ValueError("optimize_var must be a positive real number")
        # if multithreads is not None:
        #     if isinstance(multithreads, int) and multithreads > 0:
        #         algorithm.enable_openmp(multithreads)
        #     else:
        #         raise ValueError("multithreads must be a positive integer")
        algorithm.fit()
        if self.bw is None or optimize_bw is not None:
            self.bw = algorithm.spatial_weight.weight()[1]
            self.bandwidth_select_criterions = algorithm.bandwidth_criterions
        if optimize_var is not None:
            self.indep_var_select_criterions = [([self.indep_vars[v - int(self.has_intercept)] for v in varlist], criterion) for varlist, criterion in algorithm.variables_criterions]
            self.indep_vars = [self.indep_vars[v - int(self.has_intercept)] for v in algorithm.selected_variables]
            pass
        ''' Get result layer
        '''
        indep_var_names = (['Intercept'] if self.has_intercept else []) + self.indep_vars
        result_data = {
            **{f: algorithm.betas[:, i] for i, f in enumerate(indep_var_names)},
            **{f'{f}_SE': algorithm.betasSE[:, i] for i, f in enumerate(indep_var_names)},
            'fitted': algorithm.fitted
        }
        self.result_layer = gp.GeoDataFrame(result_data, geometry=self.sdf.geometry)
        ''' Get diagnostic
        '''
        if hatmatrix:
            self.diagnostic = algorithm.diagnostic
        return self

    # def predict(self, targets: gp.GeoDataFrame, multithreads: int=None):
    #     """
    #     Predict
    #     """
    #     if self.bw is None:
    #         raise ValueError("Bandwidth cannot be None when predicting")
    #     ''' Extract data
    #     '''
    #     cyg_distance = CyCRSDistance(self.longlat)
    #     cyg_weight = CyBandwidthWeight(self.bw, self.adaptive, self.kernel.value)
    #     cyg_depen_var = np.asfortranarray(self.sdf[self.depen_var])
    #     cyg_indep_vars = np.asfortranarray(self.sdf[self.indep_vars])
    #     if (self.has_intercept):
    #         cyg_indep_vars = np.hstack([np.ones((cyg_indep_vars.shape[0], 1)), cyg_indep_vars])
    #     cyg_coords = np.asfortranarray(self.sdf.geometry.centroid.get_coordinates())
    #     ''' Create cython GWR
    #     '''
    #     cyg_gwr_basic = CyGWRBasic(cyg_coords, cyg_depen_var, cyg_indep_vars, cyg_weight, cyg_distance, self.has_intercept)
    #     cyg_predict_locations = np.asfortranarray(targets.centroid.get_coordinates())
    #     if multithreads is not None:
    #         if isinstance(multithreads, int) and multithreads > 0:
    #             cyg_gwr_basic.enable_openmp(multithreads)
    #         else:
    #             raise ValueError("multithreads must be a positive integer")
    #     cyg_gwr_predict = cyg_gwr_basic.predict(cyg_predict_locations)
    #     ''' Get result layer
    #     '''
    #     indep_var_names = (['Intercept'] if self.has_intercept else []) + self.indep_vars
    #     result_data = {
    #         **{f: cyg_gwr_predict[:, i] for i, f in enumerate(indep_var_names)}
    #     }
    #     if all([x in targets.columns for x in self.indep_vars]):
    #         ''' If all variables are in predicting targets, calculate estimated y
    #         '''
    #         px = np.asfortranarray(targets[self.indep_vars])
    #         if self.has_intercept:
    #             px = np.hstack([np.ones((cyg_coords.shape[0], 1)), px])
    #         result_data['y_hat'] = np.sum(px * cyg_gwr_predict, axis=1)
    #         if self.depen_var in targets.columns:
    #             py = targets[self.depen_var]
    #             result_data["residual"] = py - result_data["y_hat"]
    #     return gp.GeoDataFrame(result_data, geometry=targets.geometry)
