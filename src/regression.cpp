#include <nanobind/nanobind.h>
#include <GWRBase.h>
#include "common.h"

namespace nb = nanobind;

void init_base(nb::module_& m);
void init_gwr_basic(nb::module_& m);

NB_MODULE(_regression, m)
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
    
    init_gwr_basic(m);
}
