#pragma once

#include "common/compiler.hpp"
#include "kernels_highway.hpp"

#include <cstddef>

namespace vnlb::linalg::kernels {

template <typename T>
[[nodiscard]] inline T dot_contiguous(const T* VNLB_RESTRICT left,
                                      const T* VNLB_RESTRICT right,
                                      std::size_t count) noexcept {
    return dot_contiguous_highway(left, right, count);
}

template <typename T>
[[nodiscard]] inline T
dot_centered_rows(const T* VNLB_RESTRICT left, const T* VNLB_RESTRICT right,
                  const T* VNLB_RESTRICT mean, std::size_t count) noexcept {
    return dot_centered_rows_highway(left, right, mean, count);
}

template <typename T>
inline void center_row(const T* VNLB_RESTRICT input,
                       const T* VNLB_RESTRICT mean, T* VNLB_RESTRICT output,
                       std::size_t count) noexcept {
    center_row_highway(input, mean, output, count);
}

template <typename T>
inline void copy_contiguous(const T* VNLB_RESTRICT input,
                            T* VNLB_RESTRICT output,
                            std::size_t count) noexcept {
    copy_contiguous_highway(input, output, count);
}

template <typename T>
inline void add_contiguous(T* VNLB_RESTRICT output,
                           const T* VNLB_RESTRICT input,
                           std::size_t count) noexcept {
    add_contiguous_highway(output, input, count);
}

template <typename T>
inline void add_scaled_contiguous(T* VNLB_RESTRICT output,
                                  const T* VNLB_RESTRICT input, T scale,
                                  std::size_t count) noexcept {
    add_scaled_contiguous_highway(output, input, scale, count);
}

template <typename T>
inline void add_scalar_contiguous(T* VNLB_RESTRICT row, T value,
                                  std::size_t count) noexcept {
    add_scalar_contiguous_highway(row, value, count);
}

template <typename T>
inline void add_contiguous_and_scalar_contiguous(T* VNLB_RESTRICT output,
                                                 T* VNLB_RESTRICT scalar_output,
                                                 const T* VNLB_RESTRICT input,
                                                 T scalar,
                                                 std::size_t count) noexcept {
    add_contiguous_and_scalar_contiguous_highway(output, scalar_output, input,
                                                 scalar, count);
}

template <typename T>
inline void scale_contiguous(T* VNLB_RESTRICT row, T scale,
                             std::size_t count) noexcept {
    scale_contiguous_highway(row, scale, count);
}

} // namespace vnlb::linalg::kernels
