#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <GWRBasic.h>
#include "common.hpp"
#include "parallel.hpp"

namespace nb = nanobind;

void init_gwr_basic(nb::module_& m)
{
    nb::class_<gwm::GWRBasic, gwm::GWRBase> _GWRBasic(m, "_GWRBasic");

    nb::enum_<gwm::GWRBasic::BandwidthSelectionCriterionType>(_GWRBasic, "BandwidthSelectionCriterionType")
        .value("AIC", gwm::GWRBasic::BandwidthSelectionCriterionType::AIC)
        .value("CV", gwm::GWRBasic::BandwidthSelectionCriterionType::CV)
        .export_values();

    _GWRBasic
        .def(nb::init<>())
        .def_prop_ro(
            "select_bandwidth_enabled",
            &gwm::GWRBasic::isAutoselectBandwidth
        )
        .def(
            "enable_select_bandwidth",
            [](gwm::GWRBasic &instance, int criterion)
            {
                instance.setIsAutoselectBandwidth(true);
                instance.setBandwidthSelectionCriterion((gwm::GWRBasic::BandwidthSelectionCriterionType)criterion); 
            }
        )
        .def_prop_ro(
            "select_variables_enabled",
            &gwm::GWRBasic::isAutoselectIndepVars
        )
        .def(
            "enable_select_variables",
            [](gwm::GWRBasic &instance, double threshold)
            { 
                instance.setIsAutoselectIndepVars(true);
                instance.setIndepVarSelectionThreshold(threshold); 
            }
        )
        .def(
            "fit",
            [](gwm::GWRBasic &instance){ instance.fit(); }
        )
        .def(
            "predict",
            [](gwm::GWRBasic &instance, arma::mat locs){ return instance.predict(locs); },
            nb::rv_policy::move
        )
        .def_prop_ro(
            "betasSE",
            &gwm::GWRBasic::betasSE,
            nb::rv_policy::move
        )
        .def_prop_ro(
            "fitted",
            [](gwm::GWRBasic &instance){ return instance.Fitted(instance.independentVariables(), instance.betas()); },
            nb::rv_policy::move
        )
        .def_prop_ro(
            "bandwidth_criterions",
            &gwm::GWRBasic::bandwidthSelectionCriterionList
        )
        .def_prop_ro(
            "variables_criterions",
            &gwm::GWRBasic::indepVarsSelectionCriterionList
        )
        .def_prop_ro(
            "selected_variables",
            &gwm::GWRBasic::selectedVariables
        )
        ;
    
    def_parallel_info(_GWRBasic);
    def_parallel_openmp(_GWRBasic);
    def_parallel_cuda(_GWRBasic);
}
