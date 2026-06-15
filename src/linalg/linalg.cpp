#include "linalg.hpp"

#include "backend.hpp"
#include "common/compiler.hpp"
#include "common/validation.hpp"
#include "kernels_highway.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace vnlb::linalg {
namespace {

#if VNLB_INTERNAL_VALIDATION_ENABLED
[[noreturn]] void fail_shape(const char* message) {
    throw std::invalid_argument(message);
}

void require_shape(bool condition, const char* message) {
    if (!condition) {
        fail_shape(message);
    }
}
#define VNLB_LINALG_REQUIRE_SHAPE(condition, message)                          \
    require_shape((condition), (message))
#else
#define VNLB_LINALG_REQUIRE_SHAPE(condition, message) ((void)0)
#endif

std::size_t checked_element_count(std::size_t rows, std::size_t cols) {
    if (rows != 0 && cols > std::numeric_limits<std::size_t>::max() / rows) {
        throw std::length_error("matrix dimensions overflow size_t");
    }
    return rows * cols;
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
T dot_contiguous(const T* VNLB_RESTRICT left, const T* VNLB_RESTRICT right,
                 std::size_t count) noexcept {
    return kernels::dot_contiguous_highway(left, right, count);
}

template <Real T>
T dot_centered_rows(const T* VNLB_RESTRICT left, const T* VNLB_RESTRICT right,
                    const T* VNLB_RESTRICT mean, std::size_t count) noexcept {
    return kernels::dot_centered_rows_highway(left, right, mean, count);
}

template <Real T> void fill_matrix(MatrixView<T> matrix, T value) {
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        std::fill_n(matrix.row_data(row), matrix.cols(), value);
    }
}

template <Real T> void mirror_upper_to_lower(MatrixView<T> matrix) {
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t col = row + 1; col < matrix.cols(); ++col) {
            matrix(col, row) = matrix(row, col);
        }
    }
}

template <Real T>
void require_mean_shape([[maybe_unused]] ConstMatrixView<T> samples,
                        [[maybe_unused]] std::span<const T> mean)
    VNLB_INTERNAL_VALIDATION_NOEXCEPT {
    VNLB_LINALG_REQUIRE_SHAPE(mean.size() == samples.cols(),
                              "mean length must match matrix columns");
}

template <Real T>
void require_same_shape([[maybe_unused]] ConstMatrixView<T> input,
                        [[maybe_unused]] MatrixView<T> output,
                        [[maybe_unused]] const char* message)
    VNLB_INTERNAL_VALIDATION_NOEXCEPT {
    VNLB_LINALG_REQUIRE_SHAPE(output.rows() == input.rows() &&
                                  output.cols() == input.cols(),
                              message);
}

template <Real T>
void require_square([[maybe_unused]] ConstMatrixView<T> matrix,
                    [[maybe_unused]] const char* message)
    VNLB_INTERNAL_VALIDATION_NOEXCEPT {
    VNLB_LINALG_REQUIRE_SHAPE(matrix.rows() == matrix.cols(), message);
}

template <Real T>
void require_square([[maybe_unused]] MatrixView<T> matrix,
                    [[maybe_unused]] const char* message)
    VNLB_INTERNAL_VALIDATION_NOEXCEPT {
    VNLB_LINALG_REQUIRE_SHAPE(matrix.rows() == matrix.cols(), message);
}

template <Real T> T max_abs_matrix(ConstMatrixView<T> matrix) {
    T max_value{};
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        const T* VNLB_RESTRICT row_data = matrix.row_data(row);
        for (std::size_t col = 0; col < matrix.cols(); ++col) {
            max_value = std::max(max_value, abs_value(row_data[col]));
        }
    }
    return max_value;
}

template <Real T> T max_abs_offdiag(ConstMatrixView<T> matrix) {
    T max_value{};
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        const T* VNLB_RESTRICT row_data = matrix.row_data(row);
        for (std::size_t col = row + 1; col < matrix.cols(); ++col) {
            max_value = std::max(max_value, abs_value(row_data[col]));
        }
    }
    return max_value;
}

template <Real T> void set_identity(MatrixView<T> matrix) {
    fill_matrix(matrix, T{});
    for (std::size_t index = 0; index < matrix.rows(); ++index) {
        matrix(index, index) = T{1};
    }
}

template <Real T>
void copy_matrix(ConstMatrixView<T> source, MatrixView<T> destination) {
    require_same_shape(source, destination,
                       "destination matrix shape must match source");
    if (source.cols() == 0) {
        return;
    }
    for (std::size_t row = 0; row < source.rows(); ++row) {
        const T* VNLB_RESTRICT source_row = source.row_data(row);
        T* VNLB_RESTRICT destination_row = destination.row_data(row);
        std::memcpy(destination_row, source_row, source.cols() * sizeof(T));
    }
}

template <Real T>
void jacobi_rotate(MatrixView<T> matrix, MatrixView<T> vectors,
                   std::size_t pivot, std::size_t target) {
    const T app = matrix(pivot, pivot);
    const T aqq = matrix(target, target);
    const T apq = matrix(pivot, target);

    if (apq == T{}) {
        return;
    }

    const T tau = (aqq - app) / (static_cast<T>(2) * apq);
    const T tau_abs = abs_value(tau);
    const T t_abs = T{1} / (tau_abs + sqrt_value(T{1} + (tau * tau)));
    const T t = tau < T{} ? -t_abs : t_abs;
    const T c = T{1} / sqrt_value(T{1} + (t * t));
    const T s = t * c;
    const T cc = c * c;
    const T ss = s * s;
    const T cs = c * s;

    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        if (row == pivot || row == target) {
            continue;
        }

        const T arp = matrix(row, pivot);
        const T arq = matrix(row, target);
        const T new_rp = (c * arp) - (s * arq);
        const T new_rq = (s * arp) + (c * arq);
        matrix(row, pivot) = new_rp;
        matrix(pivot, row) = new_rp;
        matrix(row, target) = new_rq;
        matrix(target, row) = new_rq;
    }

    matrix(pivot, pivot) =
        (cc * app) - (static_cast<T>(2) * cs * apq) + (ss * aqq);
    matrix(target, target) =
        (ss * app) + (static_cast<T>(2) * cs * apq) + (cc * aqq);
    matrix(pivot, target) = T{};
    matrix(target, pivot) = T{};

    for (std::size_t row = 0; row < vectors.rows(); ++row) {
        const T vrp = vectors(row, pivot);
        const T vrq = vectors(row, target);
        vectors(row, pivot) = (c * vrp) - (s * vrq);
        vectors(row, target) = (s * vrp) + (c * vrq);
    }
}

template <Real T>
void canonicalize_vector_row(MatrixView<T> vectors, std::size_t row) {
    T norm_squared{};
    T max_abs{};
    std::size_t sign_pivot = 0;
    T* VNLB_RESTRICT row_data = vectors.row_data(row);

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
void normalize_or_zero(T* VNLB_RESTRICT row, std::size_t count, T floor) {
    const T norm_squared = dot_contiguous(row, row, count);

    if (norm_squared <= floor) {
        std::fill_n(row, count, T{});
        return;
    }

    const T scale = T{1} / sqrt_value(norm_squared);
    kernels::scale_contiguous_highway(row, scale, count);
}

[[nodiscard]] bool worth_calling_syrk_backend(std::size_t dimension) noexcept {
    constexpr std::size_t backend_min_dimension = 32;
    return dimension >= backend_min_dimension;
}

} // namespace

template <Real T> Matrix<T>::Matrix(std::size_t rows, std::size_t cols) {
    resize(rows, cols);
}

template <Real T> void Matrix<T>::resize(std::size_t rows, std::size_t cols) {
    storage_.resize(checked_element_count(rows, cols));
    rows_ = rows;
    cols_ = cols;
}

template <Real T> void Matrix<T>::fill(T value) {
    std::fill(storage_.begin(), storage_.end(), value);
}

template <Real T>
void SymmetricEigenWorkspace<T>::resize(std::size_t dimension) {
    matrix_.resize(dimension, dimension);
    vectors_.resize(dimension, dimension);
    order_.resize(dimension);
    lapack_values_.resize(dimension);

    const std::size_t work_size =
        dimension == 0 ? 1
                       : std::max<std::size_t>(1 + (6 * dimension) +
                                                   (2 * dimension * dimension),
                                               26 * dimension);
    const std::size_t iwork_size =
        dimension == 0
            ? 1
            : std::max<std::size_t>(3 + (5 * dimension), 10 * dimension);
    lapack_work_.resize(work_size);
    lapack_iwork_.resize(iwork_size);
    lapack_support_.resize(2 * std::max<std::size_t>(dimension, 1));
}

template <Real T>
void compute_mean(ConstMatrixView<T> samples,
                  std::span<T> mean) VNLB_INTERNAL_VALIDATION_NOEXCEPT {
    VNLB_LINALG_REQUIRE_SHAPE(samples.rows() > 0,
                              "cannot compute mean of an empty sample set");
    VNLB_LINALG_REQUIRE_SHAPE(mean.size() == samples.cols(),
                              "mean length must match matrix columns");

    std::fill(mean.begin(), mean.end(), T{});
    for (std::size_t row = 0; row < samples.rows(); ++row) {
        const T* VNLB_RESTRICT row_data = samples.row_data(row);
        for (std::size_t col = 0; col < samples.cols(); ++col) {
            mean[col] += row_data[col];
        }
    }

    const T inv_count = T{1} / static_cast<T>(samples.rows());
    for (T& value : mean) {
        value *= inv_count;
    }
}

template <Real T>
void center_rows(ConstMatrixView<T> samples, std::span<const T> mean,
                 MatrixView<T> centered) VNLB_INTERNAL_VALIDATION_NOEXCEPT {
    require_mean_shape(samples, mean);
    require_same_shape(samples, centered,
                       "centered matrix shape must match samples");

    for (std::size_t row = 0; row < samples.rows(); ++row) {
        const T* VNLB_RESTRICT input = samples.row_data(row);
        T* VNLB_RESTRICT output = centered.row_data(row);
        kernels::center_row_highway(input, mean.data(), output, samples.cols());
    }
}

template <Real T>
void compute_gram_centered(ConstMatrixView<T> centered_samples,
                           MatrixView<T> gram, T scale) {
    VNLB_LINALG_REQUIRE_SHAPE(
        gram.rows() == centered_samples.rows() &&
            gram.cols() == centered_samples.rows(),
        "gram matrix must be samples.rows x samples.rows");

    if constexpr (backend::has_centered_syrk &&
                  (std::same_as<T, float> || std::same_as<T, double>)) {
        if (worth_calling_syrk_backend(gram.rows()) &&
            backend::compute_centered_syrk(
                centered_samples, gram, scale,
                backend::SymmetricProduct::SampleRows)) {
            return;
        }
    }

    for (std::size_t row = 0; row < centered_samples.rows(); ++row) {
        const T* VNLB_RESTRICT left = centered_samples.row_data(row);
        for (std::size_t col = 0; col <= row; ++col) {
            const T value =
                scale * dot_contiguous(left, centered_samples.row_data(col),
                                       centered_samples.cols());
            gram(row, col) = value;
            gram(col, row) = value;
        }
    }
}

template <Real T>
void compute_gram(ConstMatrixView<T> samples, std::span<const T> mean,
                  MatrixView<T> gram,
                  T scale) VNLB_INTERNAL_VALIDATION_NOEXCEPT {
    require_mean_shape(samples, mean);
    VNLB_LINALG_REQUIRE_SHAPE(
        gram.rows() == samples.rows() && gram.cols() == samples.rows(),
        "gram matrix must be samples.rows x samples.rows");

    for (std::size_t row = 0; row < samples.rows(); ++row) {
        const T* VNLB_RESTRICT left = samples.row_data(row);
        for (std::size_t col = 0; col <= row; ++col) {
            const T value =
                scale * dot_centered_rows(left, samples.row_data(col),
                                          mean.data(), samples.cols());
            gram(row, col) = value;
            gram(col, row) = value;
        }
    }
}

template <Real T>
void compute_covariance_centered(ConstMatrixView<T> centered_samples,
                                 MatrixView<T> covariance, T scale) {
    require_square(covariance, "covariance matrix must be square");
    VNLB_LINALG_REQUIRE_SHAPE(
        covariance.rows() == centered_samples.cols(),
        "covariance matrix must be samples.cols x samples.cols");

    if constexpr (backend::has_centered_syrk &&
                  (std::same_as<T, float> || std::same_as<T, double>)) {
        if (worth_calling_syrk_backend(covariance.rows()) &&
            backend::compute_centered_syrk(
                centered_samples, covariance, scale,
                backend::SymmetricProduct::SampleColumns)) {
            return;
        }
    }

    fill_matrix(covariance, T{});
    for (std::size_t sample = 0; sample < centered_samples.rows(); ++sample) {
        const T* VNLB_RESTRICT row = centered_samples.row_data(sample);
        for (std::size_t col0 = 0; col0 < centered_samples.cols(); ++col0) {
            const T left = row[col0];
            T* VNLB_RESTRICT cov_row = covariance.row_data(col0);
            kernels::add_scaled_contiguous_highway(
                cov_row + col0, row + col0, left,
                centered_samples.cols() - col0);
        }
    }

    for (std::size_t row = 0; row < covariance.rows(); ++row) {
        for (std::size_t col = row; col < covariance.cols(); ++col) {
            const T value = covariance(row, col) * scale;
            covariance(row, col) = value;
            covariance(col, row) = value;
        }
    }
}

template <Real T>
void compute_covariance(ConstMatrixView<T> samples, std::span<const T> mean,
                        MatrixView<T> covariance,
                        T scale) VNLB_INTERNAL_VALIDATION_NOEXCEPT {
    require_mean_shape(samples, mean);
    require_square(covariance, "covariance matrix must be square");
    VNLB_LINALG_REQUIRE_SHAPE(
        covariance.rows() == samples.cols(),
        "covariance matrix must be samples.cols x samples.cols");

    fill_matrix(covariance, T{});
    for (std::size_t sample = 0; sample < samples.rows(); ++sample) {
        const T* VNLB_RESTRICT row = samples.row_data(sample);
        for (std::size_t col0 = 0; col0 < samples.cols(); ++col0) {
            const T left = row[col0] - mean[col0];
            T* VNLB_RESTRICT cov_row = covariance.row_data(col0);
            for (std::size_t col1 = col0; col1 < samples.cols(); ++col1) {
                cov_row[col1] += left * (row[col1] - mean[col1]);
            }
        }
    }

    for (std::size_t row = 0; row < covariance.rows(); ++row) {
        for (std::size_t col = row; col < covariance.cols(); ++col) {
            const T value = covariance(row, col) * scale;
            covariance(row, col) = value;
            covariance(col, row) = value;
        }
    }
}

template <Real T>
SymmetricEigenResult topk_symmetric_eigen_jacobi(
    ConstMatrixView<T> symmetric, std::span<T> eigenvalues,
    MatrixView<T> eigenvectors, SymmetricEigenWorkspace<T>& workspace,
    JacobiEigenOptions<T> options) {
    require_square(symmetric, "symmetric eigen input must be square");
    VNLB_LINALG_REQUIRE_SHAPE(
        eigenvectors.rows() == eigenvalues.size(),
        "eigenvectors must have one row per requested eigenvalue");
    VNLB_LINALG_REQUIRE_SHAPE(eigenvectors.cols() == symmetric.rows(),
                              "eigenvector width must match matrix dimension");

    const std::size_t dimension = symmetric.rows();
    const std::size_t rank = eigenvalues.size();
    VNLB_LINALG_REQUIRE_SHAPE(rank <= dimension,
                              "requested eigen rank exceeds matrix dimension");

    workspace.resize(dimension);
    MatrixView<T> matrix = workspace.matrix_view();
    MatrixView<T> vectors = workspace.vectors_view();
    copy_matrix(symmetric, matrix);
    set_identity(vectors);

    const T matrix_scale = std::max(T{1}, max_abs_matrix(matrix.as_const()));
    const T threshold = std::max(options.absolute_tolerance,
                                 options.relative_tolerance * matrix_scale);

    SymmetricEigenResult result{
        .computed = rank, .sweeps = 0, .converged = true};

    if (dimension > 1) {
        result.converged = false;
        for (std::size_t sweep = 0; sweep < options.max_sweeps; ++sweep) {
            result.sweeps = sweep + 1;
            if (max_abs_offdiag(matrix.as_const()) <= threshold) {
                result.converged = true;
                break;
            }

            for (std::size_t pivot = 0; pivot + 1 < dimension; ++pivot) {
                for (std::size_t target = pivot + 1; target < dimension;
                     ++target) {
                    if (abs_value(matrix(pivot, target)) > threshold) {
                        jacobi_rotate(matrix, vectors, pivot, target);
                    }
                }
            }
        }

        if (!result.converged && max_abs_offdiag(matrix.as_const()) <=
                                     static_cast<T>(8) * threshold) {
            result.converged = true;
        }
    }

    std::span<std::size_t> order = workspace.order();
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t left, std::size_t right) {
                         return matrix(left, left) > matrix(right, right);
                     });

    for (std::size_t out = 0; out < rank; ++out) {
        const std::size_t source_col = order[out];
        eigenvalues[out] = matrix(source_col, source_col);
        for (std::size_t row = 0; row < dimension; ++row) {
            eigenvectors(out, row) = vectors(row, source_col);
        }
        canonicalize_vector_row(eigenvectors, out);
    }

    return result;
}

template <Real T>
SymmetricEigenResult topk_symmetric_eigen(ConstMatrixView<T> symmetric,
                                          std::span<T> eigenvalues,
                                          MatrixView<T> eigenvectors,
                                          SymmetricEigenWorkspace<T>& workspace,
                                          JacobiEigenOptions<T> options) {
    require_square(symmetric, "symmetric eigen input must be square");
    VNLB_LINALG_REQUIRE_SHAPE(
        eigenvectors.rows() == eigenvalues.size(),
        "eigenvectors must have one row per requested eigenvalue");
    VNLB_LINALG_REQUIRE_SHAPE(eigenvectors.cols() == symmetric.rows(),
                              "eigenvector width must match matrix dimension");
    VNLB_LINALG_REQUIRE_SHAPE(eigenvalues.size() <= symmetric.rows(),
                              "requested eigen rank exceeds matrix dimension");

    constexpr std::size_t backend_min_dimension = 64;
    if (symmetric.rows() >= backend_min_dimension) {
        if constexpr (backend::has_topk_symmetric_eigen &&
                      (std::same_as<T, float> || std::same_as<T, double>)) {
            SymmetricEigenResult result{};
            if (backend::topk_symmetric_eigen(
                    symmetric, eigenvalues, eigenvectors, workspace, result)) {
                return result;
            }
        }
    }

    return topk_symmetric_eigen_jacobi(symmetric, eigenvalues, eigenvectors,
                                       workspace, options);
}

template <Real T>
void map_dual_eigenvectors_to_basis(
    ConstMatrixView<T> centered_samples, std::span<const T> gram_eigenvalues,
    ConstMatrixView<T> gram_eigenvectors, MatrixView<T> basis_vectors,
    T eigenvalue_floor) VNLB_INTERNAL_VALIDATION_NOEXCEPT {
    VNLB_LINALG_REQUIRE_SHAPE(
        gram_eigenvectors.rows() <= gram_eigenvalues.size(),
        "not enough Gram eigenvalues for Gram eigenvectors");
    VNLB_LINALG_REQUIRE_SHAPE(gram_eigenvectors.cols() ==
                                  centered_samples.rows(),
                              "Gram eigenvector width must match sample count");
    VNLB_LINALG_REQUIRE_SHAPE(
        basis_vectors.rows() == gram_eigenvectors.rows(),
        "basis vector count must match Gram eigenvector count");
    VNLB_LINALG_REQUIRE_SHAPE(basis_vectors.cols() == centered_samples.cols(),
                              "basis vector width must match sample dimension");

    for (std::size_t basis = 0; basis < basis_vectors.rows(); ++basis) {
        T* VNLB_RESTRICT basis_row = basis_vectors.row_data(basis);
        std::fill_n(basis_row, basis_vectors.cols(), T{});

        const T eigenvalue = gram_eigenvalues[basis];
        if (eigenvalue <= eigenvalue_floor) {
            continue;
        }

        const T* VNLB_RESTRICT sample_vector =
            gram_eigenvectors.row_data(basis);
        for (std::size_t sample = 0; sample < centered_samples.rows();
             ++sample) {
            const T coefficient = sample_vector[sample];
            const T* VNLB_RESTRICT sample_row =
                centered_samples.row_data(sample);
            kernels::add_scaled_contiguous_highway(
                basis_row, sample_row, coefficient, centered_samples.cols());
        }

        const T inv_sqrt_eigenvalue = T{1} / sqrt_value(eigenvalue);
        kernels::scale_contiguous_highway(basis_row, inv_sqrt_eigenvalue,
                                          basis_vectors.cols());

        normalize_or_zero(basis_row, basis_vectors.cols(), eigenvalue_floor);
    }
}

template <Real T>
void project_low_rank_centered(
    ConstMatrixView<T> centered_samples, ConstMatrixView<T> basis_vectors,
    MatrixView<T> output, std::size_t rank) VNLB_INTERNAL_VALIDATION_NOEXCEPT {
    require_same_shape(centered_samples, output,
                       "output matrix shape must match samples");
    VNLB_LINALG_REQUIRE_SHAPE(rank <= basis_vectors.rows(),
                              "projection rank exceeds basis vector count");
    VNLB_LINALG_REQUIRE_SHAPE(basis_vectors.cols() == centered_samples.cols(),
                              "basis vector width must match sample dimension");

    for (std::size_t sample = 0; sample < centered_samples.rows(); ++sample) {
        const T* VNLB_RESTRICT sample_row = centered_samples.row_data(sample);
        T* VNLB_RESTRICT output_row = output.row_data(sample);
        std::fill_n(output_row, output.cols(), T{});

        for (std::size_t basis = 0; basis < rank; ++basis) {
            const T* VNLB_RESTRICT basis_row = basis_vectors.row_data(basis);
            const T coefficient =
                dot_contiguous(sample_row, basis_row, centered_samples.cols());
            kernels::add_scaled_contiguous_highway(
                output_row, basis_row, coefficient, centered_samples.cols());
        }
    }
}

template <Real T>
void project_low_rank(ConstMatrixView<T> samples, std::span<const T> mean,
                      ConstMatrixView<T> basis_vectors, MatrixView<T> output,
                      std::size_t rank) VNLB_INTERNAL_VALIDATION_NOEXCEPT {
    require_mean_shape(samples, mean);
    require_same_shape(samples, output,
                       "output matrix shape must match samples");
    VNLB_LINALG_REQUIRE_SHAPE(rank <= basis_vectors.rows(),
                              "projection rank exceeds basis vector count");
    VNLB_LINALG_REQUIRE_SHAPE(basis_vectors.cols() == samples.cols(),
                              "basis vector width must match sample dimension");

    for (std::size_t sample = 0; sample < samples.rows(); ++sample) {
        const T* VNLB_RESTRICT sample_row = samples.row_data(sample);
        T* VNLB_RESTRICT output_row = output.row_data(sample);
        if (!mean.empty()) {
            std::memcpy(output_row, mean.data(), mean.size() * sizeof(T));
        }

        for (std::size_t basis = 0; basis < rank; ++basis) {
            const T* VNLB_RESTRICT basis_row = basis_vectors.row_data(basis);
            T coefficient{};
            for (std::size_t col = 0; col < samples.cols(); ++col) {
                coefficient += (sample_row[col] - mean[col]) * basis_row[col];
            }
            kernels::add_scaled_contiguous_highway(output_row, basis_row,
                                                   coefficient, samples.cols());
        }
    }
}

template class Matrix<float>;
template class Matrix<double>;
template class SymmetricEigenWorkspace<float>;
template class SymmetricEigenWorkspace<double>;

template void compute_mean<float>(ConstMatrixView<float>, std::span<float>);
template void compute_mean<double>(ConstMatrixView<double>, std::span<double>);

template void center_rows<float>(ConstMatrixView<float>, std::span<const float>,
                                 MatrixView<float>);
template void center_rows<double>(ConstMatrixView<double>,
                                  std::span<const double>, MatrixView<double>);

template void compute_gram_centered<float>(ConstMatrixView<float>,
                                           MatrixView<float>, float);
template void compute_gram_centered<double>(ConstMatrixView<double>,
                                            MatrixView<double>, double);

template void compute_gram<float>(ConstMatrixView<float>,
                                  std::span<const float>, MatrixView<float>,
                                  float);
template void compute_gram<double>(ConstMatrixView<double>,
                                   std::span<const double>, MatrixView<double>,
                                   double);

template void compute_covariance_centered<float>(ConstMatrixView<float>,
                                                 MatrixView<float>, float);
template void compute_covariance_centered<double>(ConstMatrixView<double>,
                                                  MatrixView<double>, double);

template void compute_covariance<float>(ConstMatrixView<float>,
                                        std::span<const float>,
                                        MatrixView<float>, float);
template void compute_covariance<double>(ConstMatrixView<double>,
                                         std::span<const double>,
                                         MatrixView<double>, double);

template SymmetricEigenResult
topk_symmetric_eigen<float>(ConstMatrixView<float>, std::span<float>,
                            MatrixView<float>, SymmetricEigenWorkspace<float>&,
                            JacobiEigenOptions<float>);
template SymmetricEigenResult topk_symmetric_eigen<double>(
    ConstMatrixView<double>, std::span<double>, MatrixView<double>,
    SymmetricEigenWorkspace<double>&, JacobiEigenOptions<double>);

template void map_dual_eigenvectors_to_basis<float>(ConstMatrixView<float>,
                                                    std::span<const float>,
                                                    ConstMatrixView<float>,
                                                    MatrixView<float>, float);
template void map_dual_eigenvectors_to_basis<double>(ConstMatrixView<double>,
                                                     std::span<const double>,
                                                     ConstMatrixView<double>,
                                                     MatrixView<double>,
                                                     double);

template void project_low_rank_centered<float>(ConstMatrixView<float>,
                                               ConstMatrixView<float>,
                                               MatrixView<float>, std::size_t);
template void project_low_rank_centered<double>(ConstMatrixView<double>,
                                                ConstMatrixView<double>,
                                                MatrixView<double>,
                                                std::size_t);

template void project_low_rank<float>(ConstMatrixView<float>,
                                      std::span<const float>,
                                      ConstMatrixView<float>, MatrixView<float>,
                                      std::size_t);
template void project_low_rank<double>(ConstMatrixView<double>,
                                       std::span<const double>,
                                       ConstMatrixView<double>,
                                       MatrixView<double>, std::size_t);

#undef VNLB_LINALG_REQUIRE_SHAPE

} // namespace vnlb::linalg
