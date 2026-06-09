#pragma once

#include <hwy/highway.h>

#include <concepts>
#include <cstddef>

namespace vnlb::linalg::kernels {

template <std::floating_point T>
[[nodiscard]] inline T dot_contiguous_highway(const T* left, const T* right,
                                              std::size_t count) noexcept {
    namespace hn = hwy::HWY_NAMESPACE;
    const hn::ScalableTag<T> d;
    const std::size_t lanes = hn::Lanes(d);

    auto sum = hn::Zero(d);
    std::size_t index = 0;
    for (; index + lanes <= count; index += lanes) {
        sum = hn::MulAdd(hn::LoadU(d, left + index),
                         hn::LoadU(d, right + index), sum);
    }

    T total = hn::ReduceSum(d, sum);
    for (; index < count; ++index) {
        total += left[index] * right[index];
    }
    return total;
}

template <std::floating_point T>
[[nodiscard]] inline T dot_centered_rows_highway(const T* left, const T* right,
                                                 const T* mean,
                                                 std::size_t count) noexcept {
    namespace hn = hwy::HWY_NAMESPACE;
    const hn::ScalableTag<T> d;
    const std::size_t lanes = hn::Lanes(d);

    auto sum = hn::Zero(d);
    std::size_t index = 0;
    for (; index + lanes <= count; index += lanes) {
        const auto mean_values = hn::LoadU(d, mean + index);
        const auto left_values =
            hn::Sub(hn::LoadU(d, left + index), mean_values);
        const auto right_values =
            hn::Sub(hn::LoadU(d, right + index), mean_values);
        sum = hn::MulAdd(left_values, right_values, sum);
    }

    T total = hn::ReduceSum(d, sum);
    for (; index < count; ++index) {
        total += (left[index] - mean[index]) * (right[index] - mean[index]);
    }
    return total;
}

template <std::floating_point T>
inline void center_row_highway(const T* input, const T* mean, T* output,
                               std::size_t count) noexcept {
    namespace hn = hwy::HWY_NAMESPACE;
    const hn::ScalableTag<T> d;
    const std::size_t lanes = hn::Lanes(d);

    std::size_t index = 0;
    for (; index + lanes <= count; index += lanes) {
        hn::StoreU(
            hn::Sub(hn::LoadU(d, input + index), hn::LoadU(d, mean + index)), d,
            output + index);
    }
    for (; index < count; ++index) {
        output[index] = input[index] - mean[index];
    }
}

template <std::floating_point T>
inline void add_contiguous_highway(T* output, const T* input,
                                   std::size_t count) noexcept {
    namespace hn = hwy::HWY_NAMESPACE;
    const hn::ScalableTag<T> d;
    const std::size_t lanes = hn::Lanes(d);

    std::size_t index = 0;
    for (; index + lanes <= count; index += lanes) {
        hn::StoreU(
            hn::Add(hn::LoadU(d, output + index), hn::LoadU(d, input + index)),
            d, output + index);
    }
    for (; index < count; ++index) {
        output[index] += input[index];
    }
}

template <std::floating_point T>
inline void add_scaled_contiguous_highway(T* output, const T* input, T scale,
                                          std::size_t count) noexcept {
    namespace hn = hwy::HWY_NAMESPACE;
    const hn::ScalableTag<T> d;
    const std::size_t lanes = hn::Lanes(d);
    const auto scale_values = hn::Set(d, scale);

    std::size_t index = 0;
    for (; index + lanes <= count; index += lanes) {
        hn::StoreU(hn::MulAdd(scale_values, hn::LoadU(d, input + index),
                              hn::LoadU(d, output + index)),
                   d, output + index);
    }
    for (; index < count; ++index) {
        output[index] += scale * input[index];
    }
}

template <std::floating_point T>
inline void scale_contiguous_highway(T* row, T scale,
                                     std::size_t count) noexcept {
    namespace hn = hwy::HWY_NAMESPACE;
    const hn::ScalableTag<T> d;
    const std::size_t lanes = hn::Lanes(d);
    const auto scale_values = hn::Set(d, scale);

    std::size_t index = 0;
    for (; index + lanes <= count; index += lanes) {
        hn::StoreU(hn::Mul(hn::LoadU(d, row + index), scale_values), d,
                   row + index);
    }
    for (; index < count; ++index) {
        row[index] *= scale;
    }
}

} // namespace vnlb::linalg::kernels
