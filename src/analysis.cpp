#include <nanobind/nanobind.h>

namespace nb = nanobind;

void init_base(nb::module_& m);
void init_gwss(nb::module_& m);

NB_MODULE(_analysis, m)
{
    init_base(m);
    init_gwss(m);
}
