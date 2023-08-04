from libcpp.vector cimport vector
from libcpp.pair cimport pair

cdef extern from "armadillo" namespace "arma":
    cdef cppclass mat:
        mat()
        mat(int n_rows, int n_cols) except +
        mat(double * aux_mem, int n_rows, int n_cols) except +
        int n_rows
        int n_cols
        int n_elem
        double* memptr()
    
    cdef cppclass vec:
        vec()
        vec(int n_rows) except +
        vec(double* aux_mem, int n_rows) except +
        int n_rows
        int n_elem
        double* memptr()
    
    cdef cppclass cube:
        cube()
        cube(int n_row, int n_cols, int n_slices) except +
        cube(double* aux_mem, int n_row, int n_cols, int n_slices) except +
        int n_rows
        int n_cols
        int n_elem
        int n_slices
        double* memptr()
        mat slice(unsigned long long slice_number)

cdef extern from "spatialweight/Distance.h" namespace "gwm":
    cdef cppclass Distance:
        distance()
        maxDistance()
        minDistance()

cdef extern from "spatialweight/CRSDistance.h" namespace "gwm":
    cdef cppclass CRSDistance(Distance):
        CRSDistance()
        CRSDistance(bint isGeographic)
        bint geographic() const
        void setGeographic(bint geographic)

cdef extern from "spatialweight/Weight.h" namespace "gwm":
    cdef cppclass Weight:
        weight()

cdef extern from "spatialweight/BandwidthWeight.h" namespace "gwm":
    cdef cppclass BandwidthWeight(Weight):
        enum KernelFunctionType:
            Gaussian = 0
            Exponential = 1
            Bisquare = 2
            Tricube = 3
            Boxcar = 4
        BandwidthWeight()
        BandwidthWeight(double size, bint adaptive, KernelFunctionType kernel)
        double bandwidth() const
        void setBandwidth(double bandwidth)
        bint adaptive() const
        void setAdaptive(bint adaptive)
        KernelFunctionType kernel() const
        void setKernel(const KernelFunctionType &kernel)

cdef extern from "spatialweight/SpatialWeight.h" namespace "gwm":
    cdef cppclass SpatialWeight:
        SpatialWeight()
        SpatialWeight(Weight* weight, Distance* distance)
        Weight *weight() const
        void setWeight(Weight *weight)
        Distance *distance() const
        void setDistance(Distance *distance)


cdef extern from "Status.h" namespace "gwm":
    cdef enum class Status:
        Success
        Terminated


cdef extern from "Algorithm.h" namespace "gwm":
    cdef cppclass Algorithm:
        Algorithm()
        bint isValid()


cdef extern from "SpatialAlgorithm.h" namespace "gwm":
    cdef cppclass SpatialAlgorithm(Algorithm):
        SpatialAlgorithm()
        mat coords()
        void setCoords(const mat& coords)


cdef extern from "SpatialMonoscaleAlgorithm.h" namespace "gwm":
    cdef cppclass SpatialMonoscaleAlgorithm(SpatialAlgorithm):
        SpatialMonoscaleAlgorithm()
        SpatialWeight spatialWeight() const
        void setSpatialWeight(const SpatialWeight &spatialWeight)


cdef extern from "IMultivariableAnalysis.h" namespace "gwm":
    cdef cppclass IMultivariableAnalysis:
        mat variables() const
        void setVariables(const mat& variables)
        void run()


cdef extern from "IParallelizable.h" namespace "gwm":
    enum ParallelType:
        SerialOnly = 1
        OpenMP = 2
        CUDA = 4

    cdef cppclass IParallelizable:
        int parallelAbility() const
        ParallelType parallelType() const
        void setParallelType(const ParallelType& type)
    
    cdef cppclass IParallelOpenmpEnabled(IParallelizable):
        void setOmpThreadNum(const int threadNum)


cdef extern from "RegressionDiagnostic.h" namespace "gwm":
    cdef cppclass RegressionDiagnostic:
        double RSS
        double AIC
        double AICc
        double ENP
        double EDF
        double RSquare
        double RSquareAdjust


cdef extern from "IRegressionAnalysis.h" namespace "gwm":
    cdef cppclass IRegressionAnalysis:
        vec dependentVariable() const
        void setDependentVariable(const vec& variable)
        mat independentVariables() const
        void setIndependentVariables(const mat& variables)
        RegressionDiagnostic diagnostic() const
        bint hasIntercept() const
        void setHasIntercept(const bint has)
        mat predict(const mat& locations)
        mat fit()


cdef extern from "IBandwidthSelectable.h" namespace "gwm":
    cdef cppclass IBandwidthSelectable:
        pass
    ctypedef vector[pair[double, double] ] BandwidthCriterionList


cdef extern from "IVarialbeSelectable.h" namespace "gwm":
    cdef cppclass IVarialbeSelectable:
        vector[size_t] selectedVariables
    ctypedef vector[pair[vector[size_t], double] ] VariablesCriterionList


cdef extern from "GWRBase.h" namespace "gwm":
    cdef cppclass GWRBase(SpatialMonoscaleAlgorithm, IRegressionAnalysis):
        GWRBase()
        mat betas() const


cdef extern from "GWRBasic.h" namespace "gwm":
    cdef cppclass GWRBasic(GWRBase, IBandwidthSelectable, IVarialbeSelectable, IParallelOpenmpEnabled):
        enum BandwidthSelectionCriterionType:
            AIC = 0
            CV = 1
        GWRBasic()
        GWRBasic(const mat& x, const vec& y, const mat& coords, const SpatialWeight& spatialWeight, bint hasHatMatrix, bint hasIntercept)
        bint isAutoselectBandwidth() const
        void setIsAutoselectBandwidth(bint isAutoSelect)
        BandwidthSelectionCriterionType bandwidthSelectionCriterion() const
        void setBandwidthSelectionCriterion(const BandwidthSelectionCriterionType& criterion)
        bint isAutoselectIndepVars() const
        void setIsAutoselectIndepVars(bint isAutoSelect)
        double indepVarSelectionThreshold() const
        void setIndepVarSelectionThreshold(double threshold)
        VariablesCriterionList indepVarsSelectionCriterionList() const
        BandwidthCriterionList bandwidthSelectionCriterionList() const
        bint hasHatMatrix() const
        void setHasHatMatrix(const bint has)
        mat betasSE()
        vec sHat()
        vec qDiag()
        mat s()
        vector[size_t] selectedVariables()


cdef extern from "GWSS.h" namespace "gwm::GWSS":
    cdef enum class GWSSMode:
        Average
        Correlation


cdef extern from "GWSS.h" namespace "gwm":
    cdef cppclass GWSS(SpatialMonoscaleAlgorithm, IMultivariableAnalysis, IParallelOpenmpEnabled):
        GWSS() except +
        GWSS(const mat x, const mat coords, const SpatialWeight& spatialWeight) except +
        bint quantile() const
        void setQuantile(bint quantile)
        bint isCorrWithFirstOnly() const
        void setIsCorrWithFirstOnly(bint corrWithFirstOnly)
        mat localMean() const
        mat localSDev() const
        mat localSkewness() const
        mat localCV() const
        mat localVar() const
        mat localMedian() const
        mat iqr() const
        mat qi() const
        mat localCov() const
        mat localCorr() const
        mat localSCorr() const
        void setGWSSMode(GWSSMode mode)


cdef extern from "GWPCA.h" namespace "gwm":
    cdef cppclass GWPCA(SpatialMonoscaleAlgorithm, IMultivariableAnalysis):
        GWPCA() except +
        GWPCA(const mat x, const mat coords, const SpatialWeight& spatialWeight) except +
        int keepComponents()
        void setKeepComponents(int k)
        mat localPV()
        mat sdev()
        cube loadings()
        cube scores()
