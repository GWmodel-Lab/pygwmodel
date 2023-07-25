import numpy as np
import pandas as pd
import geopandas as gp
from shapely.geometry import Point
from typing import List, Union
from enum import Enum
from .pygwmodel import CyCRSDistance
from .pygwmodel import CyBandwidthWeight
from .pygwmodel import CyGWRBasic


class KernelType(Enum):
    GAUSSIAN = 0

class BandwidthSelectionCriterionType(Enum):
    AIC = 0
    CV = 1


class GWRBasic:
    """
    Basic GWR python high api class.
    """

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
        self.bandwidth_select_criterions = None
        self.indep_var_select_criterions = None
    
    def fit(self, hatmatrix: bool=True, optimize_bw: BandwidthSelectionCriterionType=None, optimize_var: float=None, multithreads: int=None):
        """
        Run algorithm and return result
        """
        ''' Extract data
        '''
        cyg_distance = CyCRSDistance(self.longlat)
        cyg_weight = CyBandwidthWeight(self.bw, self.adaptive, self.kernel.value)
        ''' Create cython GWR
        '''
        cyg_depen_var = np.asfortranarray(self.sdf[self.depen_var])
        indep_var_names = (['Intercept'] if self.has_intercept else []) + self.indep_vars
        cyg_indep_vars = np.asfortranarray(self.sdf[self.indep_vars])
        if (self.has_intercept):
            cyg_indep_vars = np.hstack([np.ones((cyg_indep_vars.shape[0], 1)), cyg_indep_vars])
        cyg_coords = np.asfortranarray(self.sdf.geometry.centroid.get_coordinates())
        cyg_gwr_basic = CyGWRBasic(cyg_coords, cyg_depen_var, cyg_indep_vars, cyg_weight, cyg_distance, hatmatrix)
        if self.bw is None and optimize_bw is None:
            optimize_bw = BandwidthSelectionCriterionType.CV
        if optimize_bw is not None:
            if optimize_bw == BandwidthSelectionCriterionType.AIC or optimize_bw == BandwidthSelectionCriterionType.CV:
                cyg_gwr_basic.enable_bandwidth_autoselection(optimize_bw.value)
            else:
                raise ValueError("optimize_bw must be BandwidthSelectionCriterionType.AIC(0) or BandwidthSelectionCriterionType.CV(1)")
        if optimize_var is not None:
            if isinstance(optimize_var, float) and optimize_var > 0:
                cyg_gwr_basic.enable_indep_var_autoselection(optimize_var)
            else:
                raise ValueError("optimize_var must be a positive real number")
        if multithreads is not None:
            if isinstance(multithreads, int) and multithreads > 0:
                cyg_gwr_basic.enable_openmp(multithreads)
            else:
                raise ValueError("multithreads must be a positive integer")
        cyg_gwr_basic.fit()
        if self.bw is None or optimize_bw is not None:
            self.bw = cyg_gwr_basic.bandwidth()
        #     self.bandwidth_select_criterions = cyg_gwr_basic.bandwidth_select_criterions()
        # if optimize_var is not None:
        #     self.indep_vars = [v.decode() for v in cyg_gwr_basic.indep_vars()]
        #     self.indep_var_select_criterions = cyg_gwr_basic.indep_var_select_criterions()
        ''' Get result layer
        '''
        result_data = {
            **{f: cyg_gwr_basic.betas[:, i] for i, f in enumerate(indep_var_names)},
            **{f'{f}_SE': cyg_gwr_basic.betas[:, i] for i, f in enumerate(indep_var_names)},
        }
        self.result_layer = gp.GeoDataFrame({
            **result_data,
            "geometry": self.sdf.geometry
        })
        ''' Get diagnostic
        '''
        if hatmatrix:
            self.diagnostic = cyg_gwr_basic.diagnostic()
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
    #     ''' Create cython GWR
    #     '''
    #     cyg_depen_var = self.sdf.loc[self.depen_var]
    #     cyg_indep_vars = self.sdf.loc[self.indep_vars]
    #     cyg_gwr_basic = CyGWRBasic(cyg_depen_var, cyg_indep_vars, cyg_weight, cyg_distance, False)
    #     cyg_gwr_basic.set_predict_layer(cyg_predict_layer)
    #     if multithreads is not None:
    #         if isinstance(multithreads, int) and multithreads > 0:
    #             cyg_gwr_basic.enable_openmp(multithreads)
    #         else:
    #             raise ValueError("multithreads must be a positive integer")
    #     ''' Get result layer
    #     '''
    #     cyg_gwr_basic.run()
    #     return layer_to_sdf(cyg_gwr_basic.result_layer, targets.geometry)


# class GWSS:
#     """
#     GWSS python high api class.
#     """

#     def __init__(self, sdf: gp.GeoDataFrame, variables: List[str], bw: float, adaptive: bool=True, kernel: KernelType=KernelType.GAUSSIAN, longlat: bool=True, quantile: bool=False, first_only: bool=False):
#         """
#         docstring
#         """
#         if not isinstance(sdf, gp.GeoDataFrame):
#             raise ValueError("sdf must be a GeoDataFrame")
#         self.sdf = sdf
#         self.variables = variables
#         self.bw = bw
#         self.kernel = kernel
#         self.adaptive = adaptive
#         self.longlat = longlat
#         self.result_layer = None
#         self.quantile = quantile
#         self.first_only = first_only

#     def fit(self, multithreads: int=None):
#         """
#         Run algorithm and return result
#         """
#         ''' Extract data
#         '''
#         cyg_data_layer = sdf_to_layer(self.sdf, self.variables)
#         cyg_distance = CyCRSDistance(self.longlat)
#         cyg_weight = CyBandwidthWeight(self.bw, self.adaptive, self.kernel.value)
#         cyg_in_vars = CyVariableList([CyVariable(i, True, n.encode("utf-8")) for i, n in enumerate(self.variables)])
#         ''' Create cython GWSS
#         '''
#         cyg_gwss = CyGWSS(cyg_data_layer, cyg_in_vars, cyg_weight, cyg_distance, self.quantile, self.first_only)
#         if multithreads is not None:
#             if isinstance(multithreads, int) and multithreads > 0:
#                 cyg_gwss.enable_openmp(multithreads)
#             else:
#                 raise ValueError("multithreads must be a positive integer")
#         cyg_gwss.run()
#         self.result_layer = layer_to_sdf(cyg_gwss.result_layer, self.sdf.geometry)
#         return self


# class GWPCA:
#     """
#     GWPCA python high api class.
#     """
#     result_layer = None
#     loadings = None
#     local_pv = None

#     def __init__(self, sdf: gp.GeoDataFrame, variables: List[str], bw: float, adaptive: bool=True, kernel: KernelType=KernelType.GAUSSIAN, longlat: bool=True, keepComponents: int=2):
#         """
#         docstring
#         """
#         if not isinstance(sdf, gp.GeoDataFrame):
#             raise ValueError("sdf must be a GeoDataFrame")
#         self.sdf = sdf
#         self.variables = variables
#         self.bw = bw
#         self.kernel = kernel
#         self.adaptive = adaptive
#         self.longlat = longlat
#         self.keepComponents = keepComponents

#     def fit(self):
#         """
#         Run algorithm and return result
#         """
#         ''' Extract data
#         '''
#         cyg_data_layer = sdf_to_layer(self.sdf, self.variables)
#         cyg_distance = CyCRSDistance(self.longlat)
#         cyg_weight = CyBandwidthWeight(self.bw, self.adaptive, self.kernel.value)
#         # cyg_spatial_weight = cyg_sw.SpatialWeight(cyg_distance, cyg_weight)
#         cyg_in_vars = CyVariableList([CyVariable(i, True, n.encode("utf-8")) for i, n in enumerate(self.variables)])
#         ''' Create cython GWPCA
#         '''
#         cyg_gwpca = CyGWPCA(cyg_data_layer, cyg_in_vars, cyg_weight, cyg_distance, self.keepComponents)
#         cyg_gwpca.run()
#         self.result_layer = layer_to_sdf(cyg_gwpca.result_layer, self.sdf.geometry)
#         ''' Get loadings
#         '''
#         self.local_pv = cyg_gwpca.local_pv()
#         self.loadings = cyg_gwpca.loadings()
#         return self
