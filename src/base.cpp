#include <nanobind/nanobind.h>
#include <nanobind/trampoline.h>
#include <nanobind/stl/vector.h>
#include <SpatialMonoscaleAlgorithm.h>
#include <SpatialMultiscaleAlgorithm.h>
#include <GWRBase.h>
#include <IRegressionAnalysis.h>
#include <IParallelizable.h>
#include "common.hpp"
#include "parallel.hpp"

namespace nb = nanobind;

void init_base(nb::module_& m)
{
    nb::class_<gwm::SpatialAlgorithm>(m, "_SpatialAlgorithm")
        .def_prop_rw(
            "coords",
            &gwm::SpatialAlgorithm::coords,
            &gwm::SpatialAlgorithm::setCoords,
            nb::rv_policy::move
        )
        ;
    
    nb::class_<gwm::SpatialMonoscaleAlgorithm, gwm::SpatialAlgorithm>(m, "SpatialMonoscaleAlgorithm")
        .def_prop_rw(
            "spatial_weight",
            [](gwm::SpatialMonoscaleAlgorithm &instance)
            {
                return nb::cast(instance.spatialWeight());
            },
            [](gwm::SpatialMonoscaleAlgorithm &instance, nb::handle_t<gwm::SpatialWeight> sw)
            {
                instance.setSpatialWeight(nb::cast<gwm::SpatialWeight &>(sw));
            }
        )
        ;

    nb::class_<gwm::SpatialMultiscaleAlgorithm, gwm::SpatialAlgorithm>(m, "_SpatialMultiscaleAlgorithm")
        .def_prop_rw(
            "spatial_weights",
            [](gwm::SpatialMultiscaleAlgorithm &instance)
            {
                nb::list result;
                for (const auto& sw : instance.spatialWeights())
                    result.append(nb::cast(sw));
                return result;
            },
            [](gwm::SpatialMultiscaleAlgorithm &instance, nb::list sw_list)
            {
                std::vector<gwm::SpatialWeight> weights;
                for (nb::handle item : sw_list)
                    weights.push_back(nb::cast<gwm::SpatialWeight &>(item));
                instance.setSpatialWeights(weights);
            }
        )
        ;
}