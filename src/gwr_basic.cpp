#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <GWRBasic.h>
#include "common.h"

namespace nb = nanobind;

NB_MODULE(py_gwr_basic, m)
{
    nb::class_<gwm::GWRBasic> gwr_basic(m, "GWRBasic");

    nb::enum_<gwm::GWRBasic::BandwidthSelectionCriterionType>(gwr_basic, "BandwidthSelectionCriterionType")
        .value("AIC", gwm::GWRBasic::BandwidthSelectionCriterionType::AIC)
        .value("CV", gwm::GWRBasic::BandwidthSelectionCriterionType::CV)
        .export_values();

    gwr_basic
        .def(nb::init<>())
        .def_prop_rw(
            "dependent",
            [](gwm::GWRBasic &instance){ return instance.dependentVariable(); },
            [](gwm::GWRBasic &instance, arma::vec y){ instance.setDependentVariable(y); },
            nb::rv_policy::move
        )
        .def_prop_rw(
            "independent",
            [](gwm::GWRBasic &instance){ return instance.independentVariables(); },
            [](gwm::GWRBasic &instance, arma::mat x){ instance.setIndependentVariables(x); },
            nb::rv_policy::move
        )
        .def_prop_rw(
            "coords",
            [](gwm::GWRBasic &instance){ return instance.coords(); },
            [](gwm::GWRBasic &instance, arma::mat coords){ instance.setCoords(coords); },
            nb::rv_policy::move
        )
        .def_prop_rw(
            "spatial_weight",
            [](gwm::GWRBasic &instance)
            {
                return nb::cast(instance.spatialWeight());
            },
            [](gwm::GWRBasic &instance, nb::handle_t<gwm::SpatialWeight> sw)
            {
                instance.setSpatialWeight(nb::cast<gwm::SpatialWeight &>(sw));
            }
        )
        .def_prop_rw(
            "select_bandwidth",
            [](gwm::GWRBasic &instance){ return instance.isAutoselectBandwidth(); },
            [](gwm::GWRBasic &instance, bool flag){ instance.setIsAutoselectBandwidth(flag); }
        )
        .def_prop_rw(
            "select_bandwidth_criterion",
            [](gwm::GWRBasic &instance){ return instance.bandwidthSelectionCriterion(); },
            [](gwm::GWRBasic &instance, int criterion){ instance.setBandwidthSelectionCriterion((gwm::GWRBasic::BandwidthSelectionCriterionType)criterion); }
        )
        .def_prop_rw(
            "select_variables",
            [](gwm::GWRBasic &instance){ return instance.isAutoselectIndepVars(); },
            [](gwm::GWRBasic &instance, bool flag){ instance.setIsAutoselectIndepVars(flag); }
        )
        .def_prop_rw(
            "select_variables_threshold",
            [](gwm::GWRBasic &instance){ return instance.indepVarSelectionThreshold(); },
            [](gwm::GWRBasic &instance, double threshold){ instance.setIndepVarSelectionThreshold(threshold); }
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
            "diagnostic",
            [](gwm::GWRBasic &instance){ return wrap(instance.diagnostic()); }
        )
        .def_prop_ro(
            "betas",
            [](gwm::GWRBasic &instance){ return instance.betas(); },
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
