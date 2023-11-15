#ifndef UTILS_H
#define UTILS_H

#include <armadillo>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

using nvec = nb::ndarray<double_t, nb::shape<nb::any>, nb::device::cpu, nb::f_contig>;
using nmat = nb::ndarray<double_t, nb::shape<nb::any, nb::any>, nb::device::cpu, nb::f_contig>;
using pnvec = nb::ndarray<nb::numpy, const double, nb::shape<nb::any>, nb::f_contig>;
using pnmat = nb::ndarray<nb::numpy, const double, nb::shape<nb::any, nb::any>, nb::f_contig>;

inline arma::mat as(nmat src) {
    size_t rows = src.shape(0);
    size_t cols = src.shape(1);
    return arma::mat(src.data(), rows, cols);
}

inline arma::vec as(nvec src) {
    size_t rows = src.shape(0);
    return arma::vec(src.data(), rows);
}

inline auto wrap(const arma::mat& src) {
    size_t shape[2] = { src.n_rows, src.n_cols };
    return pnmat(src.mem, 2, shape);
}

inline auto wrap(const arma::vec& src) {
    size_t shape[1] = { src.n_elem };
    return pnvec(src.mem, 1, shape);
}

#endif  // UTILS_H
