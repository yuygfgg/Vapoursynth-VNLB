#pragma once

#include <cstddef>

namespace vnlb::linalg::kernels {

template <typename T>
[[nodiscard]] inline T dot_contiguous_generic(const T* left, const T* right,
                                              std::size_t count) noexcept {
    T sum0{};
    T sum1{};
    T sum2{};
    T sum3{};

    std::size_t index = 0;
    for (; index + 4 <= count; index += 4) {
        sum0 += left[index] * right[index];
        sum1 += left[index + 1] * right[index + 1];
        sum2 += left[index + 2] * right[index + 2];
        sum3 += left[index + 3] * right[index + 3];
    }

    T sum = (sum0 + sum1) + (sum2 + sum3);
    for (; index < count; ++index) {
        sum += left[index] * right[index];
    }
    return sum;
}

template <typename T>
[[nodiscard]] inline T dot_centered_rows_generic(const T* left, const T* right,
                                                 const T* mean,
                                                 std::size_t count) noexcept {
    T sum0{};
    T sum1{};
    T sum2{};
    T sum3{};

    std::size_t index = 0;
    for (; index + 4 <= count; index += 4) {
        sum0 += (left[index] - mean[index]) * (right[index] - mean[index]);
        sum1 += (left[index + 1] - mean[index + 1]) *
                (right[index + 1] - mean[index + 1]);
        sum2 += (left[index + 2] - mean[index + 2]) *
                (right[index + 2] - mean[index + 2]);
        sum3 += (left[index + 3] - mean[index + 3]) *
                (right[index + 3] - mean[index + 3]);
    }

    T sum = (sum0 + sum1) + (sum2 + sum3);
    for (; index < count; ++index) {
        sum += (left[index] - mean[index]) * (right[index] - mean[index]);
    }
    return sum;
}

template <typename T>
inline void center_row_generic(const T* input, const T* mean, T* output,
                               std::size_t count) noexcept {
    std::size_t index = 0;
    for (; index + 4 <= count; index += 4) {
        output[index] = input[index] - mean[index];
        output[index + 1] = input[index + 1] - mean[index + 1];
        output[index + 2] = input[index + 2] - mean[index + 2];
        output[index + 3] = input[index + 3] - mean[index + 3];
    }
    for (; index < count; ++index) {
        output[index] = input[index] - mean[index];
    }
}

template <typename T>
inline void add_contiguous_generic(T* output, const T* input,
                                   std::size_t count) noexcept {
    std::size_t index = 0;
    for (; index + 4 <= count; index += 4) {
        output[index] += input[index];
        output[index + 1] += input[index + 1];
        output[index + 2] += input[index + 2];
        output[index + 3] += input[index + 3];
    }
    for (; index < count; ++index) {
        output[index] += input[index];
    }
}

template <typename T>
inline void add_scaled_contiguous_generic(T* output, const T* input, T scale,
                                          std::size_t count) noexcept {
    std::size_t index = 0;
    for (; index + 4 <= count; index += 4) {
        output[index] += scale * input[index];
        output[index + 1] += scale * input[index + 1];
        output[index + 2] += scale * input[index + 2];
        output[index + 3] += scale * input[index + 3];
    }
    for (; index < count; ++index) {
        output[index] += scale * input[index];
    }
}

template <typename T>
inline void scale_contiguous_generic(T* row, T scale,
                                     std::size_t count) noexcept {
    std::size_t index = 0;
    for (; index + 4 <= count; index += 4) {
        row[index] *= scale;
        row[index + 1] *= scale;
        row[index + 2] *= scale;
        row[index + 3] *= scale;
    }
    for (; index < count; ++index) {
        row[index] *= scale;
    }
}

} // namespace vnlb::linalg::kernels
