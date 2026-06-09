#pragma once

#ifndef VNLB_LINALG_USE_ACCELERATE
#define VNLB_LINALG_USE_ACCELERATE 0
#endif

#ifndef VNLB_LINALG_USE_OPENBLAS
#define VNLB_LINALG_USE_OPENBLAS 0
#endif

#include "linalg.hpp"

#include <span>

namespace vnlb::linalg::backend {

enum class SymmetricProduct {
    SampleRows,
    SampleColumns,
};

inline constexpr bool has_centered_syrk =
    VNLB_LINALG_USE_ACCELERATE || VNLB_LINALG_USE_OPENBLAS;
inline constexpr bool has_topk_symmetric_eigen =
    VNLB_LINALG_USE_ACCELERATE || VNLB_LINALG_USE_OPENBLAS;

[[nodiscard]] bool compute_centered_syrk(ConstMatrixView<float> samples,
                                         MatrixView<float> output, float scale,
                                         SymmetricProduct product);

[[nodiscard]] bool compute_centered_syrk(ConstMatrixView<double> samples,
                                         MatrixView<double> output,
                                         double scale,
                                         SymmetricProduct product);

[[nodiscard]] bool topk_symmetric_eigen(
    ConstMatrixView<float> symmetric, std::span<float> eigenvalues,
    MatrixView<float> eigenvectors, SymmetricEigenWorkspace<float>& workspace,
    SymmetricEigenResult& result);

[[nodiscard]] bool topk_symmetric_eigen(
    ConstMatrixView<double> symmetric, std::span<double> eigenvalues,
    MatrixView<double> eigenvectors, SymmetricEigenWorkspace<double>& workspace,
    SymmetricEigenResult& result);

} // namespace vnlb::linalg::backend
