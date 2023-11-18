#include <nanobind/nanobind.h>
#include <nanobind/trampoline.h>
#include <SpatialMonoscaleAlgorithm.h>
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
}