#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <GWRMultiscale.h>
#include "common.hpp"
#include "parallel.hpp"

namespace nb = nanobind;

void init_gwr_multiscale(nb::module_& m)
{
    nb::class_<gwm::GWRMultiscale, gwm::SpatialMultiscaleAlgorithm> _GWRMultiscale(m, "_GWRMultiscale");

    nb::enum_<gwm::GWRMultiscale::BandwidthInitilizeType>(_GWRMultiscale, "BandwidthInitilizeType")
        .value("Null", gwm::GWRMultiscale::BandwidthInitilizeType::Null)
        .value("Initial", gwm::GWRMultiscale::BandwidthInitilizeType::Initial)
        .value("Specified", gwm::GWRMultiscale::BandwidthInitilizeType::Specified)
        .export_values();

    nb::enum_<gwm::GWRMultiscale::BandwidthSelectionCriterionType>(_GWRMultiscale, "BandwidthSelectionCriterionType")
        .value("CV", gwm::GWRMultiscale::BandwidthSelectionCriterionType::CV)
        .value("AIC", gwm::GWRMultiscale::BandwidthSelectionCriterionType::AIC)
        .export_values();

    nb::enum_<gwm::GWRMultiscale::BackFittingCriterionType>(_GWRMultiscale, "BackFittingCriterionType")
        .value("CVR", gwm::GWRMultiscale::BackFittingCriterionType::CVR)
        .value("dCVR", gwm::GWRMultiscale::BackFittingCriterionType::dCVR)
        .export_values();

    _GWRMultiscale
        .def(nb::init<>())
        // IRegressionAnalysis interface
        .def_prop_rw("dependent", &gwm::GWRMultiscale::dependentVariable, &gwm::GWRMultiscale::setDependentVariable, nb::rv_policy::move)
        .def_prop_rw("independent", &gwm::GWRMultiscale::independentVariables, &gwm::GWRMultiscale::setIndependentVariables, nb::rv_policy::move)
        .def_prop_rw("has_intercept", &gwm::GWRMultiscale::hasIntercept, &gwm::GWRMultiscale::setHasIntercept)
        // Configuration
        .def_prop_ro("bandwidth_initilize", &gwm::GWRMultiscale::bandwidthInitilize)
        .def("set_bandwidth_initilize", &gwm::GWRMultiscale::setBandwidthInitilize)
        .def_prop_ro("bandwidth_selection_approach", &gwm::GWRMultiscale::bandwidthSelectionApproach)
        .def("set_bandwidth_selection_approach", &gwm::GWRMultiscale::setBandwidthSelectionApproach)
        .def_prop_rw(
            "preditor_centered",
            [](gwm::GWRMultiscale &instance)
            {
                nb::list result;
                for (bool v : instance.preditorCentered())
                    result.append(v);
                return result;
            },
            [](gwm::GWRMultiscale &instance, nb::list lst)
            {
                std::vector<bool> vec;
                for (nb::handle item : lst)
                    vec.push_back(nb::cast<bool>(item));
                instance.setPreditorCentered(vec);
            }
        )
        .def_prop_rw("bandwidth_select_threshold", &gwm::GWRMultiscale::bandwidthSelectThreshold, &gwm::GWRMultiscale::setBandwidthSelectThreshold)
        .def_prop_rw("bandwidth_select_retry_times", &gwm::GWRMultiscale::bandwidthSelectRetryTimes, &gwm::GWRMultiscale::setBandwidthSelectRetryTimes)
        .def_prop_rw("max_iteration", &gwm::GWRMultiscale::maxIteration, &gwm::GWRMultiscale::setMaxIteration)
        .def_prop_rw("criterion_type", &gwm::GWRMultiscale::criterionType, &gwm::GWRMultiscale::setCriterionType)
        .def_prop_rw("criterion_threshold", &gwm::GWRMultiscale::criterionThreshold, &gwm::GWRMultiscale::setCriterionThreshold)
        .def_prop_rw("has_hat_matrix", &gwm::GWRMultiscale::hasHatMatrix, &gwm::GWRMultiscale::setHasHatMatrix)
        .def_prop_rw("adaptive_lower", &gwm::GWRMultiscale::adaptiveLower, &gwm::GWRMultiscale::setAdaptiveLower)
        .def("set_golden_lower_bounds", &gwm::GWRMultiscale::setGoldenLowerBounds)
        .def("set_golden_upper_bounds", &gwm::GWRMultiscale::setGoldenUpperBounds)
        // Fit
        .def("fit", [](gwm::GWRMultiscale &instance){ instance.fit(); })
        // Results
        .def_prop_ro("betas", &gwm::GWRMultiscale::betas, nb::rv_policy::move)
        .def_prop_ro("betasSE", &gwm::GWRMultiscale::betasSE, nb::rv_policy::move)
        .def_prop_ro("betasTV", &gwm::GWRMultiscale::betasTV, nb::rv_policy::move)
        .def_prop_ro(
            "diagnostic",
            [](gwm::GWRMultiscale &instance){ return wrap(instance.diagnostic()); }
        )
        ;

    def_parallel_info(_GWRMultiscale);
    def_parallel_openmp(_GWRMultiscale);
    def_parallel_cuda(_GWRMultiscale);
}
