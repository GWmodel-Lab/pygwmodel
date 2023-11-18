#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <GWRBasic.h>
#include "common.h"

namespace nb = nanobind;

void init_base(nb::module_& m);

NB_MODULE(_gwr_basic, m)
{
    init_base(m);

    nb::class_<gwm::GWRBase, gwm::SpatialMonoscaleAlgorithm>(m, "_GWRBase")
        .def_prop_ro(
            "betas",
            [](gwm::GWRBase &instance){ return instance.betas(); },
            nb::rv_policy::move
        )
        .def_prop_ro(
            "diagnostic",
            [](gwm::GWRBase &instance){ return wrap(instance.diagnostic()); }
        )
        .def_prop_rw(
            "dependent",
            [](gwm::GWRBase &instance){ return instance.dependentVariable(); },
            [](gwm::GWRBase &instance, arma::vec y){ instance.setDependentVariable(y); },
            nb::rv_policy::move
        )
        .def_prop_rw(
            "independent",
            [](gwm::GWRBase &instance){ return instance.independentVariables(); },
            [](gwm::GWRBase &instance, arma::mat x){ instance.setIndependentVariables(x); },
            nb::rv_policy::move
        )
        ;

    nb::class_<gwm::GWRBasic, gwm::GWRBase> gwr_basic(m, "_GWRBasic");

    nb::enum_<gwm::GWRBasic::BandwidthSelectionCriterionType>(gwr_basic, "BandwidthSelectionCriterionType")
        .value("AIC", gwm::GWRBasic::BandwidthSelectionCriterionType::AIC)
        .value("CV", gwm::GWRBasic::BandwidthSelectionCriterionType::CV)
        .export_values();

    gwr_basic
        .def(nb::init<>())
        .def_prop_ro(
            "select_bandwidth_enabled",
            [](gwm::GWRBasic &instance){ return instance.isAutoselectBandwidth(); }
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
            [](gwm::GWRBasic &instance){ return instance.isAutoselectIndepVars(); }
        )
        .def(
            "enable_select_variables",
            [](gwm::GWRBasic &instance, double threshold)
            { 
                instance.setIsAutoselectIndepVars(true);
                instance.setIndepVarSelectionThreshold(threshold); 
            }
        )
        .def_prop_ro(
            "parallel_type",
            [](gwm::GWRBasic &instance){ return int(instance.parallelType()); }
        )
        .def(
            "parallel_omp",
            [](gwm::GWRBasic &instance, int threadNum)
            {
                instance.setParallelType(gwm::ParallelType::OpenMP);
                instance.setOmpThreadNum(threadNum);
            }
        )
        .def(
            "parallel_cuda",
            [](gwm::GWRBasic &instance, int gpuId, int groupSize)
            {
                instance.setParallelType(gwm::ParallelType::CUDA);
                instance.setGPUId(gpuId);
                instance.setGroupSize(groupSize);
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
            [](gwm::GWRBasic &instance){ return instance.betasSE(); },
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
}
