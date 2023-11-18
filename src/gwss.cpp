#include <nanobind/nanobind.h>
#include <GWSS.h>
#include "common.hpp"
#include "parallel.hpp"

namespace nb = nanobind;

void init_gwss(nb::module_& m)
{
    nb::class_<gwm::GWSS, gwm::SpatialMonoscaleAlgorithm> _GWSS(m, "_GWSS");

    nb::enum_<gwm::GWSS::GWSSMode>(_GWSS, "Mode")
        .value("Average", gwm::GWSS::GWSSMode::Average)
        .value("Correlation", gwm::GWSS::GWSSMode::Correlation)
        ;

    _GWSS
        .def(nb::init<>())
        .def("set_mode", &gwm::GWSS::setGWSSMode)
        .def_prop_rw("variables", &gwm::GWSS::variables, &gwm::GWSS::setVariables)
        .def_prop_rw("quantile", &gwm::GWSS::quantile, &gwm::GWSS::setQuantile)
        .def_prop_rw("corr_with_first", &gwm::GWSS::isCorrWithFirstOnly, &gwm::GWSS::setIsCorrWithFirstOnly)
        .def_prop_ro("local_mean", &gwm::GWSS::localMean)
        .def_prop_ro("local_sdev", &gwm::GWSS::localSDev)
        .def_prop_ro("local_skewness", &gwm::GWSS::localSkewness)
        .def_prop_ro("local_cv", &gwm::GWSS::localCV)
        .def_prop_ro("local_var", &gwm::GWSS::localVar)
        .def_prop_ro("local_median", &gwm::GWSS::localMedian)
        .def_prop_ro("iqr", &gwm::GWSS::iqr)
        .def_prop_ro("qi", &gwm::GWSS::qi)
        .def_prop_ro("local_cov", &gwm::GWSS::localCov)
        .def_prop_ro("local_corr", &gwm::GWSS::localCorr)
        .def_prop_ro("local_s_corr", &gwm::GWSS::localSCorr)
        ;
    
    def_parallel_info(_GWSS);
    def_parallel_openmp(_GWSS);
}
