#include "parallel.hpp"

void init_parallel(nb::module_& m)
{
    nb::enum_<gwm::ParallelType>(m, "_ParallelType")
        .value("SerialOnly", gwm::ParallelType::SerialOnly)
        .value("OpenMP", gwm::ParallelType::OpenMP)
        .value("CUDA", gwm::ParallelType::CUDA)
        .export_values();
}
