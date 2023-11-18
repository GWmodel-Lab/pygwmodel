#include <nanobind/nanobind.h>
#include <GWRBase.h>
#include "common.hpp"

namespace nb = nanobind;

void init_base(nb::module_& m);
void init_gwr_basic(nb::module_& m);

NB_MODULE(_regression, m)
{
    init_base(m);

    nb::class_<gwm::GWRBase, gwm::SpatialMonoscaleAlgorithm>(m, "_GWRBase")
        .def_prop_ro(
            "betas",
            &gwm::GWRBase::betas,
            nb::rv_policy::move
        )
        .def_prop_ro(
            "diagnostic",
            [](gwm::GWRBase &instance){ return wrap(instance.diagnostic()); }
        )
        .def_prop_rw(
            "dependent",
            &gwm::GWRBase::dependentVariable,
            &gwm::GWRBase::setDependentVariable,
            nb::rv_policy::move
        )
        .def_prop_rw(
            "independent",
            &gwm::GWRBase::independentVariables,
            &gwm::GWRBase::setIndependentVariables,
            nb::rv_policy::move
        )
        ;
    
    init_gwr_basic(m);
}
