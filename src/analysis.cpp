#include <nanobind/nanobind.h>

namespace nb = nanobind;

void init_base(nb::module_& m);

NB_MODULE(_analysis, m)
{
    init_base(m);
}
