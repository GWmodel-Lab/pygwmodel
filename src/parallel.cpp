#include <nanobind/nanobind.h>
#include <IParallelizable.h>

namespace nb = nanobind;

NB_MODULE(_parallel, m)
{
    nb::enum_<gwm::ParallelType>(m, "_ParallelType")
        .value("SerialOnly", gwm::ParallelType::SerialOnly)
        .value("OpenMP", gwm::ParallelType::OpenMP)
        .value("CUDA", gwm::ParallelType::CUDA)
        ;
}
