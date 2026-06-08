#include "linalg/linalg.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string_view>
#include <vector>

namespace {

using vnlb::linalg::ConstMatrixView;
using vnlb::linalg::Matrix;
using vnlb::linalg::MatrixView;
using vnlb::linalg::SymmetricEigenWorkspace;

template <typename T>
void require_close(T actual, T expected, T tolerance, std::string_view label) {
    using std::abs;
    if (abs(actual - expected) > tolerance) {
        std::cerr << label << ": expected " << expected << ", got " << actual
                  << '\n';
        std::exit(EXIT_FAILURE);
    }
}

template <typename T>
void require_vector_close(ConstMatrixView<T> actual,
                          ConstMatrixView<T> expected, T tolerance,
                          std::string_view label) {
    if (actual.rows() != expected.rows() || actual.cols() != expected.cols()) {
        std::cerr << label << ": shape mismatch\n";
        std::exit(EXIT_FAILURE);
    }

    for (std::size_t row = 0; row < actual.rows(); ++row) {
        for (std::size_t col = 0; col < actual.cols(); ++col) {
            require_close(actual(row, col), expected(row, col), tolerance,
                          label);
        }
    }
}

template <typename T>
T dot_row(ConstMatrixView<T> matrix, std::size_t row_a, std::size_t row_b) {
    T sum{};
    for (std::size_t col = 0; col < matrix.cols(); ++col) {
        sum += matrix(row_a, col) * matrix(row_b, col);
    }
    return sum;
}

template <typename T>
void require_eigen_residual(ConstMatrixView<T> matrix,
                            ConstMatrixView<T> vectors,
                            std::span<const T> values, T tolerance) {
    for (std::size_t vector = 0; vector < vectors.rows(); ++vector) {
        for (std::size_t row = 0; row < matrix.rows(); ++row) {
            T projected{};
            for (std::size_t col = 0; col < matrix.cols(); ++col) {
                projected += matrix(row, col) * vectors(vector, col);
            }
            require_close(projected, values[vector] * vectors(vector, row),
                          tolerance, "eigen residual");
        }

        require_close(dot_row(vectors, vector, vector), T{1}, tolerance,
                      "eigen norm");
    }
}

void test_mean_center_gram_covariance() {
    const std::vector<double> sample_data{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const ConstMatrixView<double> samples(sample_data.data(), 3, 2);

    std::vector<double> mean(2);
    vnlb::linalg::compute_mean(samples, std::span<double>(mean));
    require_close(mean[0], 3.0, 1e-12, "mean[0]");
    require_close(mean[1], 4.0, 1e-12, "mean[1]");

    Matrix<double> centered(3, 2);
    vnlb::linalg::center_rows(samples, std::span<const double>(mean),
                              centered.view());

    const std::vector<double> expected_centered{-2.0, -2.0, 0.0, 0.0, 2.0, 2.0};
    require_vector_close(
        centered.cview(),
        ConstMatrixView<double>(expected_centered.data(), 3, 2), 1e-12,
        "centered samples");

    Matrix<double> gram(3, 3);
    vnlb::linalg::compute_gram_centered(centered.cview(), gram.view());
    const std::vector<double> expected_gram{8.0, 0.0,  -8.0, 0.0, 0.0,
                                            0.0, -8.0, 0.0,  8.0};
    require_vector_close(gram.cview(),
                         ConstMatrixView<double>(expected_gram.data(), 3, 3),
                         1e-12, "centered Gram");

    Matrix<double> gram_direct(3, 3);
    vnlb::linalg::compute_gram(samples, std::span<const double>(mean),
                               gram_direct.view());
    require_vector_close(gram_direct.cview(), gram.cview(), 1e-12,
                         "direct Gram");

    Matrix<double> covariance(2, 2);
    vnlb::linalg::compute_covariance_centered(centered.cview(),
                                              covariance.view());
    const std::vector<double> expected_covariance{8.0, 8.0, 8.0, 8.0};
    require_vector_close(
        covariance.cview(),
        ConstMatrixView<double>(expected_covariance.data(), 2, 2), 1e-12,
        "centered covariance");

    Matrix<double> covariance_direct(2, 2);
    vnlb::linalg::compute_covariance(samples, std::span<const double>(mean),
                                     covariance_direct.view());
    require_vector_close(covariance_direct.cview(), covariance.cview(), 1e-12,
                         "direct covariance");
}

void test_symmetric_eigen_diagonal() {
    const std::vector<double> matrix_data{4.0, 0.0, 0.0, 0.0, 1.0,
                                          0.0, 0.0, 0.0, 9.0};
    const ConstMatrixView<double> matrix(matrix_data.data(), 3, 3);

    std::vector<double> values(2);
    Matrix<double> vectors(2, 3);
    SymmetricEigenWorkspace<double> workspace;
    const auto result = vnlb::linalg::topk_symmetric_eigen(
        matrix, std::span<double>(values), vectors.view(), workspace);

    require_close(values[0], 9.0, 1e-12, "largest diagonal eigenvalue");
    require_close(values[1], 4.0, 1e-12, "second diagonal eigenvalue");
    require_eigen_residual(matrix, vectors.cview(),
                           std::span<const double>(values), 1e-12);
    if (!result.converged) {
        std::cerr << "diagonal eigen solve did not converge\n";
        std::exit(EXIT_FAILURE);
    }
}

void test_symmetric_eigen_dense() {
    const std::vector<double> matrix_data{2.0, 1.0, 1.0, 2.0};
    const ConstMatrixView<double> matrix(matrix_data.data(), 2, 2);

    std::vector<double> values(2);
    Matrix<double> vectors(2, 2);
    SymmetricEigenWorkspace<double> workspace;
    const auto result = vnlb::linalg::topk_symmetric_eigen(
        matrix, std::span<double>(values), vectors.view(), workspace);

    require_close(values[0], 3.0, 1e-12, "largest dense eigenvalue");
    require_close(values[1], 1.0, 1e-12, "second dense eigenvalue");
    require_eigen_residual(matrix, vectors.cview(),
                           std::span<const double>(values), 1e-12);
    require_close(dot_row(vectors.cview(), 0, 1), 0.0, 1e-12,
                  "eigen orthogonality");
    if (!result.converged) {
        std::cerr << "dense eigen solve did not converge\n";
        std::exit(EXIT_FAILURE);
    }
}

void test_dual_basis_and_projection() {
    const std::vector<double> centered_data{1.0, 0.0, -1.0, 0.0};
    const ConstMatrixView<double> centered(centered_data.data(), 2, 2);

    const std::vector<double> gram_values{2.0};
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    const std::vector<double> gram_vectors{inv_sqrt2, -inv_sqrt2};
    const ConstMatrixView<double> gram_eigenvectors(gram_vectors.data(), 1, 2);

    Matrix<double> basis(1, 2);
    vnlb::linalg::map_dual_eigenvectors_to_basis(
        centered, std::span<const double>(gram_values), gram_eigenvectors,
        basis.view());
    require_close(basis(0, 0), 1.0, 1e-12, "dual basis x");
    require_close(basis(0, 1), 0.0, 1e-12, "dual basis y");

    Matrix<double> projected_centered(2, 2);
    vnlb::linalg::project_low_rank_centered(centered, basis.cview(),
                                            projected_centered.view(), 1);
    require_vector_close(projected_centered.cview(), centered, 1e-12,
                         "centered low-rank projection");

    const std::vector<double> sample_data{2.0, 2.0, 0.0, 2.0};
    const std::vector<double> mean{1.0, 2.0};
    Matrix<double> projected(2, 2);
    vnlb::linalg::project_low_rank(
        ConstMatrixView<double>(sample_data.data(), 2, 2),
        std::span<const double>(mean), basis.cview(), projected.view(), 1);
    require_vector_close(projected.cview(),
                         ConstMatrixView<double>(sample_data.data(), 2, 2),
                         1e-12, "uncentered low-rank projection");
}

} // namespace

int main() {
    test_mean_center_gram_covariance();
    test_symmetric_eigen_diagonal();
    test_symmetric_eigen_dense();
    test_dual_basis_and_projection();
    return EXIT_SUCCESS;
}
