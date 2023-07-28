# distutils: language = c++
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.pair cimport pair
import numpy as np
cimport numpy as np
import geopandas as gp
from .gwmodel cimport mat, vec, cube
from .gwmodel cimport Distance, CRSDistance
from .gwmodel cimport Weight, BandwidthWeight
from .gwmodel cimport SpatialWeight
from .gwmodel cimport GWRBasic, RegressionDiagnostic
from .gwmodel cimport BandwidthCriterionList, VariablesCriterionList
from .gwmodel cimport ParallelType


cdef mat numpy2mat(double[::1, :] array):
    cdef unsigned long long rows = array.shape[0]
    cdef unsigned long long cols = array.shape[1]
    return mat(&array[0, 0], rows, cols)

cdef vec numpy2vec(double[:] array):
    cdef unsigned long long rows = array.shape[0]
    return vec(&array[0], rows)


cdef mat2numpy(const mat& arma_mat):
    cdef const double* src = arma_mat.memptr()
    cdef unsigned long long rows = arma_mat.n_rows
    cdef unsigned long long cols = arma_mat.n_cols
    result = np.zeros((rows, cols), dtype=np.float64, order="F")
    cdef double[::1, :] dst = result
    cdef unsigned long long i
    for i in range(cols):
        for j in range(rows):
            dst[j, i] = src[i * rows + j]
    return result

cdef vec2numpy(const vec& arma_vec):
    cdef const double* src = arma_vec.memptr()
    cdef unsigned long long rows = arma_vec.n_rows
    result = np.zeros((rows,), dtype=np.float64, order="F")
    cdef double[:] dst = result
    cdef unsigned long long i
    for i in range(rows):
        dst[i] = src[i]
    return result

cdef cube2numpy(const cube& arma_cube):
    cdef const double* src = arma_cube.memptr()
    cdef unsigned long long rows = arma_cube.n_rows
    cdef unsigned long long cols = arma_cube.n_cols
    cdef unsigned long long slices = arma_cube.n_slices
    result = np.zeros((slices, rows, cols), dtype=np.float64, order="C")
    cdef double[:, :, ::1] dst = result
    cdef unsigned long long i, j, k
    for k in range(slices):
        for j in range(cols):
            for i in range(rows):
                dst[k, i, j] = src[k * cols * rows + j * rows + i]
    return result


cdef class CyDistance:
    __type = None
    cdef Distance* _c_instance

    def __cinit__(self):
        self._c_instance = NULL

    def distance_type(self):
        return self.__type


cdef class CyCRSDistance(CyDistance):
    __type = "CRS"

    def __cinit__(self, bint geographic):
        self._c_instance = new CRSDistance(geographic)


cdef class CyWeight:
    __type = None
    cdef Weight* _c_instance

    def __cinit__(self):
        self._c_instance = NULL

    def weight_type(self):
        return self.__type


cdef class CyBandwidthWeight(CyWeight):
    __type = "Bandwidth"
    
    def __cinit__(self, double size, bint adaptive, int kernel):
        self._c_instance = new BandwidthWeight(size, adaptive, <BandwidthWeight.KernelFunctionType>kernel)


cdef class CyGWRBasic:
    cdef GWRBasic* _c_instance

    def __cinit__(self, double[::1, :] coords, double[:] depen_var, double[::1, :] indep_vars, CyWeight weight, CyDistance distance, bint intercept):
        self._c_instance = new GWRBasic()
        self._c_instance.setCoords(numpy2mat(coords))
        self._c_instance.setDependentVariable(numpy2vec(depen_var))
        self._c_instance.setIndependentVariables(numpy2mat(indep_vars))
        cdef SpatialWeight spatial = SpatialWeight(weight._c_instance, distance._c_instance)
        self._c_instance.setSpatialWeight(spatial)
        self._c_instance.setHasIntercept(intercept)
    
    def enable_bandwidth_autoselection(self, int criterion):
        self._c_instance.setIsAutoselectBandwidth(True)
        self._c_instance.setBandwidthSelectionCriterion(<GWRBasic.BandwidthSelectionCriterionType>criterion)
    
    def enable_indep_var_autoselection(self, double threshold):
        self._c_instance.setIsAutoselectIndepVars(True)
        self._c_instance.setIndepVarSelectionThreshold(threshold)
    
    def enable_openmp(self, int threads):
        self._c_instance.setParallelType(ParallelType.OpenMP)
        self._c_instance.setOmpThreadNum(threads)

    def fit(self, bint hatmatrix=True):
        self._c_instance.setHasHatMatrix(hatmatrix)
        self._c_instance.fit()
    
    @property
    def dependent_variable(self):
        return vec2numpy(self._c_instance.dependentVariable())
    
    @property
    def independent_variables(self):
        return mat2numpy(self._c_instance.independentVariables())
    
    @property
    def betas(self):
        return mat2numpy(self._c_instance.betas())
    
    @property
    def betasSE(self):
        return mat2numpy(self._c_instance.betasSE())
    
    @property
    def diagnostic(self):
        cdef RegressionDiagnostic diag = self._c_instance.diagnostic()
        return {
            "RSS": diag.RSS,
            "AIC": diag.AIC,
            "AICc": diag.AICc,
            "ENP": diag.ENP,
            "EDF": diag.EDF,
            "RSquare": diag.RSquare,
            "RSquareAdjust": diag.RSquareAdjust
        }
    
    @property
    def bandwidth_select_criterions(self):
        cdef BandwidthCriterionList criterion_c = self._c_instance.bandwidthSelectionCriterionList()
        criterion_py = []
        cdef unsigned long long size = criterion_c.size()
        cdef pair[double, double] item
        for i in range(size):
            item = criterion_c.at(i)
            criterion_py.append((item.first, item.second))
        return criterion_py

    @property
    def indep_var_select_criterions(self):
        cdef VariablesCriterionList criterion_c = self._c_instance.indepVarsSelectionCriterionList()
        criterion_py = []
        cdef unsigned long long criterion_size = criterion_c.size()
        cdef pair[vector[size_t], double] item
        cdef vector[size_t] var_list
        cdef unsigned long long var_size
        cdef size_t var
        for i in range(criterion_size):
            item = criterion_c.at(i)
            var_list = item.first
            var_size = var_list.size()
            var_list_py = []
            for j in range(var_size):
                var = var_list.at(j)
                var_list_py.append(var)
            criterion_py.append((var_list_py, item.second))
        return criterion_py

    @property
    def bandwidth(self):
        cdef SpatialWeight spatial_weight = self._c_instance.spatialWeight()
        return (<BandwidthWeight*>(spatial_weight.weight())).bandwidth()
    
    @property
    def selected_indep_vars(self):
        return [i for i in self._c_instance.selectedVariables()]


# cdef class CyGWSS:
#     cdef GWSS* _c_instance

#     def __cinit__(self, CySimpleLayer layer, CyVariableList variable_list, CyWeight weight, CyDistance distance, bint quantile, bint first_only):
#         self._c_instance = new GWSS()
#         self._c_instance.setSourceLayer(layer._c_instance)
#         self._c_instance.setVariables(variable_list._c_instance)
#         cdef SpatialWeight spatial = SpatialWeight(weight._c_instance, distance._c_instance)
#         self._c_instance.setSpatialWeight(spatial)
#         self._c_instance.setQuantile(quantile)
#         self._c_instance.setIsCorrWithFirstOnly(first_only)
    
#     def enable_openmp(self, int threads):
#         self._c_instance.setParallelType(ParallelType.OpenMP)
#         self._c_instance.setOmpThreadNum(threads)
    
#     def valid(self):
#         return self._c_instance.isValid()
    
#     def run(self):
#         self._c_instance.run()

#     def local_mean(self):
#         return mat2numpy(self._c_instance.localMean())

#     def local_sdev(self):
#         return mat2numpy(self._c_instance.localSDev())

#     def local_skewness(self):
#         return mat2numpy(self._c_instance.localSkewness())

#     def local_cv(self):
#         return mat2numpy(self._c_instance.localCV())

#     def local_var(self):
#         return mat2numpy(self._c_instance.localVar())

#     def local_median(self):
#         return mat2numpy(self._c_instance.localMedian())

#     def iqr(self):
#         return mat2numpy(self._c_instance.iqr())

#     def qi(self):
#         return mat2numpy(self._c_instance.qi())

#     def local_cov(self):
#         return mat2numpy(self._c_instance.localCov())

#     def local_corr(self):
#         return mat2numpy(self._c_instance.localCorr())

#     def local_scorr(self):
#         return mat2numpy(self._c_instance.localSCorr())
    
#     @property
#     def result_layer(self):
#         cdef SimpleLayer* layer = self._c_instance.resultLayer()
#         return CySimpleLayer(mat2numpy(layer.points()), 
#                              mat2numpy(layer.data()),
#                              name_vector2list(layer.fields()))


# cdef class CyGWPCA:
#     cdef GWPCA* _c_instance
    
#     def __cinit__(self, CySimpleLayer layer, CyVariableList variable_list, CyWeight weight, CyDistance distance, int keepComponents):
#         self._c_instance = new GWPCA()
#         self._c_instance.setSourceLayer(layer._c_instance)
#         self._c_instance.setVariables(variable_list._c_instance)
#         cdef SpatialWeight spatial = SpatialWeight(weight._c_instance, distance._c_instance)
#         self._c_instance.setSpatialWeight(spatial)
#         self._c_instance.setKeepComponents(keepComponents)
    
#     def valid(self):
#         return self._c_instance.isValid()
    
#     def run(self):
#         self._c_instance.run()
    
#     def local_pv(self):
#         return mat2numpy(self._c_instance.localPV())

#     def sdev(self):
#         return mat2numpy(self._c_instance.sdev())
    
#     def loadings(self):
#         return cube2numpy(self._c_instance.loadings())

#     def scores(self):
#         return cube2numpy(self._c_instance.scores())
    
#     @property
#     def result_layer(self):
#         cdef SimpleLayer* layer = self._c_instance.resultLayer()
#         return CySimpleLayer(mat2numpy(layer.points()), 
#                              mat2numpy(layer.data()),
#                              name_vector2list(layer.fields()))
        

