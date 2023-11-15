#ifndef UTILS_H
#define UTILS_H

#include <armadillo>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <RegressionDiagnostic.h>

namespace nb = nanobind;

using nvec = nb::ndarray<nb::numpy, const double, nb::shape<nb::any>>;
using nmat = nb::ndarray<nb::numpy, const double, nb::shape<nb::any, nb::any>, nb::f_contig>;

inline arma::mat as(nmat src)
{
    return arma::mat(src.data(), src.shape(0), src.shape(1));
}

inline arma::vec as(nvec src)
{
    return arma::vec(src.data(), src.shape(0));
}

inline auto wrap(const arma::mat& src)
{
    arma::mat *temp = new arma::mat(src.mem, src.n_rows, src.n_cols);
    auto owner = nb::capsule(temp, [](void *p) noexcept { delete (arma::mat*)p; });
    return nmat(temp->memptr(), { temp->n_rows, temp->n_cols }, nb::handle(), { 1, int64_t(temp->n_rows) });
}

inline auto wrap(const arma::vec& src)
{
    arma::vec *temp = new arma::vec(src.mem, src.n_elem);
    auto owner = nb::capsule(temp, [](void *p) noexcept { delete (arma::vec*)p; });
    return nvec(temp->memptr(), { temp->n_elem }, owner, { 1 });
}

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
