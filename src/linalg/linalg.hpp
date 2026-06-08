#pragma once

#include "common/validation.hpp"

#include <concepts>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

namespace vnlb::linalg {

template <typename T>
concept Real = std::floating_point<T>;

template <Real T> class ConstMatrixView {
  public:
    constexpr ConstMatrixView() noexcept = default;

    constexpr ConstMatrixView(const T* data, std::size_t rows,
                              std::size_t cols) noexcept
        : data_(data), rows_(rows), cols_(cols), stride_(cols) {}

    constexpr ConstMatrixView(const T* data, std::size_t rows, std::size_t cols,
                              std::size_t stride) noexcept
        : data_(data), rows_(rows), cols_(cols), stride_(stride) {}

    [[nodiscard]] constexpr const T* data() const noexcept { return data_; }
    [[nodiscard]] constexpr std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] constexpr std::size_t cols() const noexcept { return cols_; }
    [[nodiscard]] constexpr std::size_t stride() const noexcept {
        return stride_;
    }
    [[nodiscard]] constexpr bool empty() const noexcept {
        return rows_ == 0 || cols_ == 0;
    }

    [[nodiscard]] constexpr const T* row_data(std::size_t row) const noexcept {
        return data_ + (row * stride_);
    }

    [[nodiscard]] constexpr const T&
    operator()(std::size_t row, std::size_t col) const noexcept {
        return row_data(row)[col];
    }

  private:
    const T* data_ = nullptr;
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::size_t stride_ = 0;
};

template <Real T> class MatrixView {
  public:
    constexpr MatrixView() noexcept = default;

    constexpr MatrixView(T* data, std::size_t rows, std::size_t cols) noexcept
        : data_(data), rows_(rows), cols_(cols), stride_(cols) {}

    constexpr MatrixView(T* data, std::size_t rows, std::size_t cols,
                         std::size_t stride) noexcept
        : data_(data), rows_(rows), cols_(cols), stride_(stride) {}

    [[nodiscard]] constexpr T* data() const noexcept { return data_; }
    [[nodiscard]] constexpr std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] constexpr std::size_t cols() const noexcept { return cols_; }
    [[nodiscard]] constexpr std::size_t stride() const noexcept {
        return stride_;
    }
    [[nodiscard]] constexpr bool empty() const noexcept {
        return rows_ == 0 || cols_ == 0;
    }

    [[nodiscard]] constexpr T* row_data(std::size_t row) const noexcept {
        return data_ + (row * stride_);
    }

    [[nodiscard]] constexpr T& operator()(std::size_t row,
                                          std::size_t col) const noexcept {
        return row_data(row)[col];
    }

    [[nodiscard]] constexpr ConstMatrixView<T> as_const() const noexcept {
        return ConstMatrixView<T>(data_, rows_, cols_, stride_);
    }

  private:
    T* data_ = nullptr;
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::size_t stride_ = 0;
};

template <Real T> class Matrix {
  public:
    Matrix() = default;
    Matrix(std::size_t rows, std::size_t cols);

    void resize(std::size_t rows, std::size_t cols);
    void fill(T value);

    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
    [[nodiscard]] bool empty() const noexcept {
        return rows_ == 0 || cols_ == 0;
    }
    [[nodiscard]] std::size_t size() const noexcept { return storage_.size(); }

    [[nodiscard]] T* data() noexcept { return storage_.data(); }
    [[nodiscard]] const T* data() const noexcept { return storage_.data(); }

    [[nodiscard]] T* row_data(std::size_t row) noexcept {
        return storage_.data() + (row * cols_);
    }
    [[nodiscard]] const T* row_data(std::size_t row) const noexcept {
        return storage_.data() + (row * cols_);
    }

    [[nodiscard]] T& operator()(std::size_t row, std::size_t col) noexcept {
        return row_data(row)[col];
    }

    [[nodiscard]] const T& operator()(std::size_t row,
                                      std::size_t col) const noexcept {
        return row_data(row)[col];
    }

    [[nodiscard]] MatrixView<T> view() noexcept {
        return MatrixView<T>(storage_.data(), rows_, cols_);
    }

    [[nodiscard]] ConstMatrixView<T> view() const noexcept {
        return ConstMatrixView<T>(storage_.data(), rows_, cols_);
    }

    [[nodiscard]] ConstMatrixView<T> cview() const noexcept {
        return ConstMatrixView<T>(storage_.data(), rows_, cols_);
    }

    [[nodiscard]] std::span<T> values() noexcept {
        return std::span<T>(storage_);
    }
    [[nodiscard]] std::span<const T> values() const noexcept {
        return std::span<const T>(storage_);
    }

  private:
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<T> storage_;
};

template <Real T> struct JacobiEigenOptions {
    std::size_t max_sweeps = 64;
    T relative_tolerance =
        static_cast<T>(64) * std::numeric_limits<T>::epsilon();
    T absolute_tolerance = T{};
};

struct SymmetricEigenResult {
    std::size_t computed = 0;
    std::size_t sweeps = 0;
    bool converged = false;
};

template <Real T> class SymmetricEigenWorkspace {
  public:
    void resize(std::size_t dimension);
    [[nodiscard]] std::size_t dimension() const noexcept {
        return matrix_.rows();
    }
    [[nodiscard]] MatrixView<T> matrix_view() noexcept {
        return matrix_.view();
    }
    [[nodiscard]] MatrixView<T> vectors_view() noexcept {
        return vectors_.view();
    }
    [[nodiscard]] std::span<std::size_t> order() noexcept {
        return std::span<std::size_t>(order_);
    }
    [[nodiscard]] std::span<T> lapack_values() noexcept {
        return std::span<T>(lapack_values_);
    }
    [[nodiscard]] std::span<T> lapack_work() noexcept {
        return std::span<T>(lapack_work_);
    }
    [[nodiscard]] std::span<int> lapack_iwork() noexcept {
        return std::span<int>(lapack_iwork_);
    }
    [[nodiscard]] std::span<int> lapack_support() noexcept {
        return std::span<int>(lapack_support_);
    }

  private:
    template <Real U>
    friend SymmetricEigenResult
    topk_symmetric_eigen(ConstMatrixView<U>, std::span<U>, MatrixView<U>,
                         SymmetricEigenWorkspace<U>&, JacobiEigenOptions<U>);

    Matrix<T> matrix_;
    Matrix<T> vectors_;
    std::vector<std::size_t> order_;
    std::vector<T> lapack_values_;
    std::vector<T> lapack_work_;
    std::vector<int> lapack_iwork_;
    std::vector<int> lapack_support_;
};

template <Real T>
void compute_mean(ConstMatrixView<T> samples,
                  std::span<T> mean) VNLB_INTERNAL_VALIDATION_NOEXCEPT;

template <Real T>
void center_rows(ConstMatrixView<T> samples, std::span<const T> mean,
                 MatrixView<T> centered) VNLB_INTERNAL_VALIDATION_NOEXCEPT;

template <Real T>
void compute_gram_centered(ConstMatrixView<T> centered_samples,
                           MatrixView<T> gram, T scale = T{1});

template <Real T>
void compute_gram(ConstMatrixView<T> samples, std::span<const T> mean,
                  MatrixView<T> gram,
                  T scale = T{1}) VNLB_INTERNAL_VALIDATION_NOEXCEPT;

template <Real T>
void compute_covariance_centered(ConstMatrixView<T> centered_samples,
                                 MatrixView<T> covariance, T scale = T{1});

template <Real T>
void compute_covariance(ConstMatrixView<T> samples, std::span<const T> mean,
                        MatrixView<T> covariance,
                        T scale = T{1}) VNLB_INTERNAL_VALIDATION_NOEXCEPT;

template <Real T>
SymmetricEigenResult topk_symmetric_eigen(ConstMatrixView<T> symmetric,
                                          std::span<T> eigenvalues,
                                          MatrixView<T> eigenvectors,
                                          SymmetricEigenWorkspace<T>& workspace,
                                          JacobiEigenOptions<T> options = {});

template <Real T>
void map_dual_eigenvectors_to_basis(
    ConstMatrixView<T> centered_samples, std::span<const T> gram_eigenvalues,
    ConstMatrixView<T> gram_eigenvectors, MatrixView<T> basis_vectors,
    T eigenvalue_floor = static_cast<T>(128) *
                         std::numeric_limits<T>::epsilon())
    VNLB_INTERNAL_VALIDATION_NOEXCEPT;

template <Real T>
void project_low_rank_centered(
    ConstMatrixView<T> centered_samples, ConstMatrixView<T> basis_vectors,
    MatrixView<T> output, std::size_t rank) VNLB_INTERNAL_VALIDATION_NOEXCEPT;

template <Real T>
void project_low_rank(ConstMatrixView<T> samples, std::span<const T> mean,
                      ConstMatrixView<T> basis_vectors, MatrixView<T> output,
                      std::size_t rank) VNLB_INTERNAL_VALIDATION_NOEXCEPT;

} // namespace vnlb::linalg
