#pragma once

#ifndef VNLB_LINALG_USE_AARCH64_NEON
#define VNLB_LINALG_USE_AARCH64_NEON 0
#endif

#include "kernels_generic.hpp"

#if VNLB_LINALG_USE_AARCH64_NEON
#include "kernels_aarch64_neon.hpp"
#endif

#include <concepts>
#include <cstddef>

namespace vnlb::linalg::kernels {

template <typename T>
[[nodiscard]] inline T dot_contiguous(const T* left, const T* right,
                                      std::size_t count) noexcept {
#if VNLB_LINALG_USE_AARCH64_NEON
    if constexpr (std::same_as<T, float>) {
        return dot_contiguous_aarch64_neon(left, right, count);
    } else
#endif
    {
        return dot_contiguous_generic(left, right, count);
    }
}

template <typename T>
[[nodiscard]] inline T dot_centered_rows(const T* left, const T* right,
                                         const T* mean,
                                         std::size_t count) noexcept {
#if VNLB_LINALG_USE_AARCH64_NEON
    if constexpr (std::same_as<T, float>) {
        return dot_centered_rows_aarch64_neon(left, right, mean, count);
    } else
#endif
    {
        return dot_centered_rows_generic(left, right, mean, count);
    }
}

template <typename T>
inline void center_row(const T* input, const T* mean, T* output,
                       std::size_t count) noexcept {
#if VNLB_LINALG_USE_AARCH64_NEON
    if constexpr (std::same_as<T, float>) {
        center_row_aarch64_neon(input, mean, output, count);
    } else
#endif
    {
        center_row_generic(input, mean, output, count);
    }
}

template <typename T>
inline void add_contiguous(T* output, const T* input,
                           std::size_t count) noexcept {
#if VNLB_LINALG_USE_AARCH64_NEON
    if constexpr (std::same_as<T, float>) {
        add_contiguous_aarch64_neon(output, input, count);
    } else
#endif
    {
        add_contiguous_generic(output, input, count);
    }
}

template <typename T>
inline void add_scaled_contiguous(T* output, const T* input, T scale,
                                  std::size_t count) noexcept {
#if VNLB_LINALG_USE_AARCH64_NEON
    if constexpr (std::same_as<T, float>) {
        add_scaled_contiguous_aarch64_neon(output, input, scale, count);
    } else
#endif
    {
        add_scaled_contiguous_generic(output, input, scale, count);
    }
}

template <typename T>
inline void scale_contiguous(T* row, T scale, std::size_t count) noexcept {
#if VNLB_LINALG_USE_AARCH64_NEON
    if constexpr (std::same_as<T, float>) {
        scale_contiguous_aarch64_neon(row, scale, count);
    } else
#endif
    {
        scale_contiguous_generic(row, scale, count);
    }
}

} // namespace vnlb::linalg::kernels
