#include <nanobind/nanobind.h>
#include <IParallelizable.h>

namespace nb = nanobind;

// void init_parallel(nb::module_& m);

template <typename T, typename... Ts>
void def_parallel_info(nb::class_<T, Ts...>& c)
{
    c.def_prop_ro("parallel_ability", &T::parallelAbility);
    c.def_prop_ro(
        "parallel_type",
        [](T &instance){ return int(instance.parallelType()); }
    );
}

template <typename T, typename... Ts>
void def_parallel_openmp(nb::class_<T, Ts...>& c)
{
    c.def(
        "parallel_omp",
        [](T &instance, int threadNum)
        {
            instance.setParallelType(gwm::ParallelType::OpenMP);
            instance.setOmpThreadNum(threadNum);
        }
    );
}

template <typename T, typename... Ts>
void def_parallel_cuda(nb::class_<T, Ts...>& c)
{
    c.def(
        "parallel_cuda",
        [](T &instance, int gpuId, int groupSize)
        {
            instance.setParallelType(gwm::ParallelType::CUDA);
            instance.setGPUId(gpuId);
            instance.setGroupSize(groupSize);
        }
    );
}
