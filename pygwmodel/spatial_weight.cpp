#include <nanobind/nanobind.h>
#include <GWRBasic.h>
#include "common.h"

namespace nb = nanobind;

NB_MODULE(py_spatial_weight, m)
{
    nb::class_<gwm::SpatialWeight>(m, "SpatialWeight")
        .def(nb::init<>())
        .def(
            "weight",
            [](gwm::SpatialWeight &sw)
            {
                auto bw = sw.weight<gwm::BandwidthWeight>();
                return nb::make_tuple("BandwidthWeight", bw->bandwidth(), bw->adaptive(), bw->kernel());
            }
        )
        .def(
            "distance",
            [](gwm::SpatialWeight &sw)
            {
                switch (sw.distance()->type())
                {
                case gwm::Distance::DistanceType::CRSDistance:
                {
                    auto dist = sw.distance<gwm::CRSDistance>();
                    return nb::make_tuple("CRSDistance", dist->geographic());
                }
                default:
                    throw nb::type_error("Unsupported distance type.");
                }
            }
        )
        .def(
            "set_weight_bandwidth",
            [](gwm::SpatialWeight &sw, double bandwidth, bool adaptive, int kernel)
            {
                sw.setWeight(gwm::BandwidthWeight(bandwidth, adaptive, (gwm::BandwidthWeight::KernelFunctionType)kernel));
            }
        )
        .def(
            "set_distance_crs",
            [](gwm::SpatialWeight &sw, bool geographic)
            {
                sw.setDistance(gwm::CRSDistance(geographic));
            }
        )
        ;
};
