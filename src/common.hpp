#ifndef UTILS_H
#define UTILS_H

#include <armadillo>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <RegressionDiagnostic.h>

namespace nb = nanobind;


template<typename T>
constexpr int is_vector_v = bool(T::is_col);

template<typename T>
constexpr int ndim_v = is_vector_v<T> ? 1 : 2;

template <typename T, typename Scalar = typename T::elem_type>
using array_for_arma_t = nb::ndarray<
    nb::numpy,
    Scalar,
    std::conditional_t<
        is_vector_v<T> == 1,
        nb::shape<nb::any>,
        nb::shape<nb::any, nb::any>
    >,
    std::conditional_t<
        is_vector_v<T> == 1,
        nb::any_contig,
        nb::f_contig
    >
>;

template<typename T>
struct nb::detail::type_caster<T, nb::detail::enable_if_t<nb::detail::is_ndarray_scalar_v<typename T::elem_type>>>
{
    using Scalar = typename T::elem_type;
    using NDArray = array_for_arma_t<T>;
    using NDArrayCaster = nb::detail::make_caster<NDArray>;

    NB_TYPE_CASTER(T, NDArrayCaster::Name)

    bool from_python(handle src, uint8_t flags, nb::detail::cleanup_list *cleanup) noexcept
    {
        using NDArrayConst = array_for_arma_t<T, const Scalar>;
        nb::detail::make_caster<NDArrayConst> caster;
        if (!caster.from_python(src, flags, cleanup))
            return false;
        
        const NDArrayConst &array = caster.value;
        if constexpr (ndim_v<T> == 1)
            value.resize(array.shape(0));
        else
            value.resize(array.shape(0), array.shape(1));
        memcpy(value.memptr(), array.data(), array.size() * sizeof(Scalar));

        return true;
    }

    static handle from_cpp(T &&v, rv_policy policy, nb::detail::cleanup_list *cleanup) noexcept
    {
        if (policy == rv_policy::automatic ||
            policy == rv_policy::automatic_reference)
            policy = rv_policy::move;

        return from_cpp((const T &) v, policy, cleanup);
    }

    static handle from_cpp(const T &v, rv_policy policy, nb::detail::cleanup_list *cleanup) noexcept
    {
        size_t shape[ndim_v<T>];
        int64_t strides[ndim_v<T>];

        if constexpr (is_vector_v<T> == 1)
        {
            shape[0] = v.n_elem;
            strides[0] = 1;
        }
        else
        {
            shape[0] = v.n_rows;
            shape[1] = v.n_cols;
            strides[0] = 1;
            strides[1] = v.n_rows;
        }

        void *ptr = (void *)v.memptr();
        switch (policy)
        {
        case rv_policy::automatic:
            policy = rv_policy::move;
            break;
        case rv_policy::automatic_reference:
            policy = rv_policy::reference;
            break;
        default:
            break;
        }

        object owner;
        if (policy == rv_policy::move)
        {
            T *temp = new T(std::move(v));
            owner = capsule(temp, [](void *p) noexcept { delete (T *)p; });
            ptr = temp->memptr();
            policy = rv_policy::reference;
        }
        else if (policy == rv_policy::reference_internal)
        {
            owner = borrow(cleanup->self());
            policy = rv_policy::reference;
        }

        object o = steal(NDArrayCaster::from_cpp(
            NDArray(ptr, ndim_v<T>, shape, owner, strides),
            policy, cleanup
        ));

        return o.release();
    }
};

inline auto wrap(const gwm::RegressionDiagnostic& diagnostic)
{
    nb::dict result;
    result["RSS"] = diagnostic.RSS;
    result["AIC"] = diagnostic.AIC;
    result["AICc"] = diagnostic.AICc;
    result["ENP"] = diagnostic.ENP;
    result["EDF"] = diagnostic.EDF;
    result["RSquare"] = diagnostic.RSquare;
    result["RSquareAdjust"] = diagnostic.RSquareAdjust;
    return result;
}

#endif  // UTILS_H
