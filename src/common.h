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
    return nmat(src.memptr(), { src.n_rows, src.n_cols }, nb::handle(), { 1, int64_t(src.n_rows) });
}

inline auto wrap(const arma::vec& src)
{
    return nvec(src.memptr(), { src.n_elem }, nb::handle(), { 1 });
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
