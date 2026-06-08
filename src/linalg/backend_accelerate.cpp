#include "backend.hpp"

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace vnlb::linalg::backend {
namespace {

static_assert(sizeof(__CLPK_integer) == sizeof(int),
              "Accelerate LAPACK integer size must match int");

[[nodiscard]] int checked_blas_int(std::size_t value, const char* name) {
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::length_error(std::string(name) + " exceeds BLAS int range");
    }
    return static_cast<int>(value);
}

template <Real T> T abs_value(T value) {
    using std::abs;
    return abs(value);
}

template <Real T> T sqrt_value(T value) {
    using std::sqrt;
    return sqrt(value);
}

template <Real T>
void mirror_upper_to_lower(MatrixView<T> matrix) {
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t col = row + 1; col < matrix.cols(); ++col) {
            matrix(col, row) = matrix(row, col);
        }
    }
}

template <Real T>
void copy_matrix(ConstMatrixView<T> source, MatrixView<T> destination) {
    for (std::size_t row = 0; row < source.rows(); ++row) {
        std::copy_n(source.row_data(row), source.cols(),
                    destination.row_data(row));
    }
}

template <Real T>
void canonicalize_vector_row(MatrixView<T> vectors, std::size_t row) {
    T norm_squared{};
    T max_abs{};
    std::size_t sign_pivot = 0;
    T* row_data = vectors.row_data(row);

    for (std::size_t col = 0; col < vectors.cols(); ++col) {
        const T value = row_data[col];
        norm_squared += value * value;
        const T value_abs = abs_value(value);
        if (value_abs > max_abs) {
            max_abs = value_abs;
            sign_pivot = col;
        }
    }

    if (norm_squared <= std::numeric_limits<T>::min()) {
        return;
    }

    T scale = T{1} / sqrt_value(norm_squared);
    if (row_data[sign_pivot] < T{}) {
        scale = -scale;
    }

    for (std::size_t col = 0; col < vectors.cols(); ++col) {
        row_data[col] *= scale;
    }
}

template <Real T>
void call_syrk(CBLAS_TRANSPOSE transpose, int dimension, int reduction, T alpha,
               const T* input, int input_stride, T* output, int output_stride);

template <>
void call_syrk<float>(CBLAS_TRANSPOSE transpose, int dimension, int reduction,
                      float alpha, const float* input, int input_stride,
                      float* output, int output_stride) {
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
    cblas_ssyrk(CblasRowMajor, CblasUpper, transpose, dimension, reduction,
                alpha, input, input_stride, 0.0F, output, output_stride);
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
}

template <>
void call_syrk<double>(CBLAS_TRANSPOSE transpose, int dimension, int reduction,
                       double alpha, const double* input, int input_stride,
                       double* output, int output_stride) {
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
    cblas_dsyrk(CblasRowMajor, CblasUpper, transpose, dimension, reduction,
                alpha, input, input_stride, 0.0, output, output_stride);
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
}

template <Real T>
bool compute_centered_syrk_accelerate(ConstMatrixView<T> samples,
                                      MatrixView<T> output, T scale,
                                      SymmetricProduct product) {
    if (output.rows() == 0 || output.cols() == 0) {
        return true;
    }

    const CBLAS_TRANSPOSE transpose =
        product == SymmetricProduct::SampleRows ? CblasNoTrans : CblasTrans;
    const int dimension =
        checked_blas_int(output.rows(), "symmetric output dimension");
    const int reduction = checked_blas_int(
        transpose == CblasNoTrans ? samples.cols() : samples.rows(),
        "syrk reduction dimension");
    const int input_stride = checked_blas_int(samples.stride(), "input stride");
    const int output_stride =
        checked_blas_int(output.stride(), "output stride");
    call_syrk<T>(transpose, dimension, reduction, scale, samples.data(),
                 input_stride, output.data(), output_stride);
    mirror_upper_to_lower(output);
    return true;
}

template <Real T>
int call_syevr(__CLPK_integer dimension, __CLPK_integer first_index,
               __CLPK_integer last_index, T* matrix, T* eigenvalues,
               T* eigenvectors, __CLPK_integer* found, __CLPK_integer* support,
               T* work, __CLPK_integer work_size, __CLPK_integer* iwork,
               __CLPK_integer iwork_size);

template <>
int call_syevr<float>(__CLPK_integer dimension, __CLPK_integer first_index,
                      __CLPK_integer last_index, float* matrix,
                      float* eigenvalues, float* eigenvectors,
                      __CLPK_integer* found, __CLPK_integer* support,
                      float* work, __CLPK_integer work_size,
                      __CLPK_integer* iwork, __CLPK_integer iwork_size) {
    char jobz = 'V';
    char range = 'I';
    char uplo = 'U';
    __CLPK_integer lda = dimension;
    float vl = 0.0F;
    float vu = 0.0F;
    float abstol = 0.0F;
    __CLPK_integer ldz = dimension;
    __CLPK_integer info = 0;
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
    ssyevr_(&jobz, &range, &uplo, &dimension, matrix, &lda, &vl, &vu,
            &first_index, &last_index, &abstol, found, eigenvalues,
            eigenvectors, &ldz, support, work, &work_size, iwork, &iwork_size,
            &info);
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
    return static_cast<int>(info);
}

template <>
int call_syevr<double>(__CLPK_integer dimension, __CLPK_integer first_index,
                       __CLPK_integer last_index, double* matrix,
                       double* eigenvalues, double* eigenvectors,
                       __CLPK_integer* found, __CLPK_integer* support,
                       double* work, __CLPK_integer work_size,
                       __CLPK_integer* iwork, __CLPK_integer iwork_size) {
    char jobz = 'V';
    char range = 'I';
    char uplo = 'U';
    __CLPK_integer lda = dimension;
    double vl = 0.0;
    double vu = 0.0;
    double abstol = 0.0;
    __CLPK_integer ldz = dimension;
    __CLPK_integer info = 0;
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
    dsyevr_(&jobz, &range, &uplo, &dimension, matrix, &lda, &vl, &vu,
            &first_index, &last_index, &abstol, found, eigenvalues,
            eigenvectors, &ldz, support, work, &work_size, iwork, &iwork_size,
            &info);
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
    return static_cast<int>(info);
}

template <Real T>
bool topk_symmetric_eigen_accelerate(
    ConstMatrixView<T> symmetric, std::span<T> eigenvalues,
    MatrixView<T> eigenvectors, SymmetricEigenWorkspace<T>& workspace,
    SymmetricEigenResult& result) {
    const std::size_t dimension = symmetric.rows();
    const std::size_t rank = eigenvalues.size();
    if (dimension == 0 || rank == 0) {
        result = {.computed = 0, .sweeps = 0, .converged = true};
        return true;
    }

    workspace.resize(dimension);
    MatrixView<T> matrix = workspace.matrix_view();
    MatrixView<T> vectors = workspace.vectors_view();
    copy_matrix(symmetric, matrix);

    const auto n = static_cast<__CLPK_integer>(dimension);
    const auto first_index = static_cast<__CLPK_integer>(dimension - rank + 1);
    const auto last_index = static_cast<__CLPK_integer>(dimension);
    __CLPK_integer found = 0;
    const auto work_size =
        static_cast<__CLPK_integer>(workspace.lapack_work().size());
    const auto iwork_size =
        static_cast<__CLPK_integer>(workspace.lapack_iwork().size());
    auto* iwork =
        reinterpret_cast<__CLPK_integer*>(workspace.lapack_iwork().data());
    auto* support =
        reinterpret_cast<__CLPK_integer*>(workspace.lapack_support().data());

    const int info = call_syevr<T>(
        n, first_index, last_index, matrix.data(),
        workspace.lapack_values().data(), vectors.data(), &found, support,
        workspace.lapack_work().data(), work_size, iwork, iwork_size);
    if (info != 0 || found != static_cast<__CLPK_integer>(rank)) {
        result = {.computed = rank, .sweeps = 0, .converged = false};
        return false;
    }

    const T* values = workspace.lapack_values().data();
    for (std::size_t out = 0; out < rank; ++out) {
        const std::size_t source_col = rank - 1 - out;
        eigenvalues[out] = values[source_col];
        for (std::size_t row = 0; row < dimension; ++row) {
            eigenvectors(out, row) =
                vectors.data()[row + source_col * dimension];
        }
        canonicalize_vector_row(eigenvectors, out);
    }

    result = {.computed = rank, .sweeps = 0, .converged = true};
    return true;
}

} // namespace

bool compute_centered_syrk(ConstMatrixView<float> samples,
                           MatrixView<float> output, float scale,
                           SymmetricProduct product) {
    return compute_centered_syrk_accelerate(samples, output, scale, product);
}

bool compute_centered_syrk(ConstMatrixView<double> samples,
                           MatrixView<double> output, double scale,
                           SymmetricProduct product) {
    return compute_centered_syrk_accelerate(samples, output, scale, product);
}

bool topk_symmetric_eigen(ConstMatrixView<float> symmetric,
                          std::span<float> eigenvalues,
                          MatrixView<float> eigenvectors,
                          SymmetricEigenWorkspace<float>& workspace,
                          SymmetricEigenResult& result) {
    return topk_symmetric_eigen_accelerate(symmetric, eigenvalues, eigenvectors,
                                           workspace, result);
}

bool topk_symmetric_eigen(ConstMatrixView<double> symmetric,
                          std::span<double> eigenvalues,
                          MatrixView<double> eigenvectors,
                          SymmetricEigenWorkspace<double>& workspace,
                          SymmetricEigenResult& result) {
    return topk_symmetric_eigen_accelerate(symmetric, eigenvalues, eigenvectors,
                                           workspace, result);
}

} // namespace vnlb::linalg::backend
