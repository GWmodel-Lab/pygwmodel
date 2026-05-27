#include <nanobind/nanobind.h>
#include <GWAverage.h>
#include <GWCorrelation.h>
#include "common.hpp"
#include "parallel.hpp"

namespace nb = nanobind;

void init_gwss(nb::module_& m)
{
    nb::class_<gwm::GWAverage, gwm::SpatialMonoscaleAlgorithm> _GWAverage(m, "_GWAverage");
    _GWAverage
        .def(nb::init<>())
        .def_prop_rw("variables", &gwm::GWAverage::variables, &gwm::GWAverage::setVariables, nb::rv_policy::move)
        .def_prop_rw("quantile", &gwm::GWAverage::quantile, &gwm::GWAverage::setQuantile)
        .def("run", &gwm::GWAverage::run)
        .def_prop_ro("local_mean", &gwm::GWAverage::localMean, nb::rv_policy::move)
        .def_prop_ro("local_sdev", &gwm::GWAverage::localSDev, nb::rv_policy::move)
        .def_prop_ro("local_skewness", &gwm::GWAverage::localSkewness, nb::rv_policy::move)
        .def_prop_ro("local_cv", &gwm::GWAverage::localCV, nb::rv_policy::move)
        .def_prop_ro("local_var", &gwm::GWAverage::localVar, nb::rv_policy::move)
        .def_prop_ro("local_median", &gwm::GWAverage::localMedian, nb::rv_policy::move)
        .def_prop_ro("iqr", &gwm::GWAverage::iqr, nb::rv_policy::move)
        .def_prop_ro("qi", &gwm::GWAverage::qi, nb::rv_policy::move)
        ;
    def_parallel_info(_GWAverage);
    def_parallel_openmp(_GWAverage);

    nb::class_<gwm::GWCorrelation, gwm::SpatialMultiscaleAlgorithm> _GWCorrelation(m, "_GWCorrelation");
    _GWCorrelation
        .def(nb::init<>())
        .def_prop_rw("variables", &gwm::GWCorrelation::variables2, &gwm::GWCorrelation::setVariables2, nb::rv_policy::move)
        .def("run", &gwm::GWCorrelation::run)
        .def_prop_ro("local_mean", &gwm::GWCorrelation::localMean, nb::rv_policy::move)
        .def_prop_ro("local_sdev", &gwm::GWCorrelation::localSDev, nb::rv_policy::move)
        .def_prop_ro("local_skewness", &gwm::GWCorrelation::localSkewness, nb::rv_policy::move)
        .def_prop_ro("local_cv", &gwm::GWCorrelation::localCV, nb::rv_policy::move)
        .def_prop_ro("local_var", &gwm::GWCorrelation::localVar, nb::rv_policy::move)
        .def_prop_ro("local_cov", &gwm::GWCorrelation::localCov, nb::rv_policy::move)
        .def_prop_ro("local_corr", &gwm::GWCorrelation::localCorr, nb::rv_policy::move)
        .def_prop_ro("local_s_corr", &gwm::GWCorrelation::localSCorr, nb::rv_policy::move)
        ;
    def_parallel_info(_GWCorrelation);
    def_parallel_openmp(_GWCorrelation);
}
