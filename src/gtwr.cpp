#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/ndarray.h>
#include <GTWR.h>
#include <spatialweight/CRSSTDistance.h>
#include "common.hpp"
#include "parallel.hpp"

namespace nb = nanobind;

struct GTWRHelper : gwm::GTWR
{
    void prepare_st_distance()
    {
        mStdistance = mSpatialWeight.distance<gwm::CRSSTDistance>();
    }

    void set_times_vec(const arma::vec &times)
    {
        vTimes = times;
    }
};

void init_gtwr(nb::module_& m)
{
    nb::class_<GTWRHelper, gwm::GWRBase> _GTWR(m, "_GTWR");

    nb::enum_<gwm::GTWR::BandwidthSelectionCriterionType>(_GTWR, "BandwidthSelectionCriterionType")
        .value("AIC", gwm::GTWR::BandwidthSelectionCriterionType::AIC)
        .value("CV", gwm::GTWR::BandwidthSelectionCriterionType::CV)
        .export_values();

    _GTWR
        .def(nb::init<>())
        .def(
            "prepare_st_distance",
            &GTWRHelper::prepare_st_distance
        )
        .def_prop_ro(
            "select_bandwidth_enabled",
            &gwm::GTWR::isAutoselectBandwidth
        )
        .def(
            "set_select_bandwidth",
            [](GTWRHelper &instance, bool enable, int criterion)
            {
                instance.setIsAutoselectBandwidth(enable);
                instance.setBandwidthSelectionCriterion(
                    (gwm::GTWR::BandwidthSelectionCriterionType)criterion);
            }
        )
        .def(
            "set_select_lambda",
            [](GTWRHelper &instance, bool enable)
            { instance.setIsAutoselectLambda(enable); }
        )
        .def(
            "set_select_lambda_bw",
            [](GTWRHelper &instance, bool enable)
            { instance.setIsAutoselectLambdaBw(enable); }
        )
        .def_prop_rw(
            "has_hat_matrix",
            &gwm::GTWR::hasHatMatrix,
            &gwm::GTWR::setHasHatMatrix
        )
        .def_prop_ro(
            "bandwidth_criterions",
            &gwm::GTWR::bandwidthSelectionCriterionList
        )
        .def(
            "set_times_vec",
            [](GTWRHelper &instance, const arma::vec &times)
            {
                instance.set_times_vec(times);
            }
        )
        .def(
            "fit",
            [](GTWRHelper &instance){ instance.fit(); }
        )
        .def(
            "predict",
            [](GTWRHelper &instance, arma::mat locs){ return instance.predict(locs); },
            nb::rv_policy::move
        )
        .def_prop_ro(
            "fitted",
            [](GTWRHelper &instance){
                return instance.Fitted(instance.independentVariables(), instance.betas());
            },
            nb::rv_policy::move
        )
        .def(
            "predict",
            [](gwm::GTWR &instance, arma::mat locs){ return instance.predict(locs); },
            nb::rv_policy::move
        )
        .def_prop_ro(
            "betasSE",
            &gwm::GTWR::betasSE,
            nb::rv_policy::move
        )
        .def_prop_ro(
            "fitted",
            [](gwm::GTWR &instance){
                return instance.Fitted(instance.independentVariables(), instance.betas());
            },
            nb::rv_policy::move
        )
        .def_prop_ro(
            "s_hat",
            &gwm::GTWR::sHat,
            nb::rv_policy::move
        )
        .def_prop_ro(
            "q_diag",
            &gwm::GTWR::qDiag,
            nb::rv_policy::move
        )
        .def_prop_ro(
            "s",
            &gwm::GTWR::s,
            nb::rv_policy::move
        )
        .def_prop_ro(
            "lambda_",
            &gwm::GTWR::getLambda
        )
        .def_prop_ro(
            "angle",
            &gwm::GTWR::getAngle
        )
        ;

    def_parallel_info(_GTWR);
    def_parallel_openmp(_GTWR);
}
