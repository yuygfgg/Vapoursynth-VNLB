#pragma once

#include "kernels_highway.hpp"

#include <cstddef>

namespace vnlb::linalg::kernels {

template <typename T>
[[nodiscard]] inline T dot_contiguous(const T* left, const T* right,
                                      std::size_t count) noexcept {
    return dot_contiguous_highway(left, right, count);
}

template <typename T>
[[nodiscard]] inline T dot_centered_rows(const T* left, const T* right,
                                         const T* mean,
                                         std::size_t count) noexcept {
    return dot_centered_rows_highway(left, right, mean, count);
}

template <typename T>
inline void center_row(const T* input, const T* mean, T* output,
                       std::size_t count) noexcept {
    center_row_highway(input, mean, output, count);
}

template <typename T>
inline void add_contiguous(T* output, const T* input,
                           std::size_t count) noexcept {
    add_contiguous_highway(output, input, count);
}

template <typename T>
inline void add_scaled_contiguous(T* output, const T* input, T scale,
                                  std::size_t count) noexcept {
    add_scaled_contiguous_highway(output, input, scale, count);
}

template <typename T>
inline void scale_contiguous(T* row, T scale, std::size_t count) noexcept {
    scale_contiguous_highway(row, scale, count);
}

} // namespace vnlb::linalg::kernels
