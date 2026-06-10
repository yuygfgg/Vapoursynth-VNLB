#pragma once

#include "common/compiler.hpp"

#include <hwy/highway.h>

#include <concepts>
#include <cstddef>

namespace vnlb::linalg::kernels {

template <std::floating_point T>
[[nodiscard]] inline T dot_contiguous_highway(const T* VNLB_RESTRICT left,
                                              const T* VNLB_RESTRICT right,
                                              std::size_t count) noexcept {
    namespace hn = hwy::HWY_NAMESPACE;
    const hn::ScalableTag<T> d;
    const std::size_t lanes = hn::Lanes(d);

    if (count == lanes) {
        return hn::ReduceSum(d,
                             hn::Mul(hn::LoadU(d, left), hn::LoadU(d, right)));
    }

    if (count == (2 * lanes)) {
        auto sum = hn::Mul(hn::LoadU(d, left), hn::LoadU(d, right));
        sum = hn::MulAdd(hn::LoadU(d, left + lanes),
                         hn::LoadU(d, right + lanes), sum);
        return hn::ReduceSum(d, sum);
    }

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
[[nodiscard]] inline T dot_centered_rows_highway(const T* VNLB_RESTRICT left,
                                                 const T* VNLB_RESTRICT right,
                                                 const T* VNLB_RESTRICT mean,
                                                 std::size_t count) noexcept {
    namespace hn = hwy::HWY_NAMESPACE;
    const hn::ScalableTag<T> d;
    const std::size_t lanes = hn::Lanes(d);

    if (count == lanes) {
        const auto mean_values = hn::LoadU(d, mean);
        const auto left_values = hn::Sub(hn::LoadU(d, left), mean_values);
        const auto right_values = hn::Sub(hn::LoadU(d, right), mean_values);
        return hn::ReduceSum(d, hn::Mul(left_values, right_values));
    }

    if (count == (2 * lanes)) {
        auto mean_values = hn::LoadU(d, mean);
        auto left_values = hn::Sub(hn::LoadU(d, left), mean_values);
        auto right_values = hn::Sub(hn::LoadU(d, right), mean_values);
        auto sum = hn::Mul(left_values, right_values);

        mean_values = hn::LoadU(d, mean + lanes);
        left_values = hn::Sub(hn::LoadU(d, left + lanes), mean_values);
        right_values = hn::Sub(hn::LoadU(d, right + lanes), mean_values);
        sum = hn::MulAdd(left_values, right_values, sum);
        return hn::ReduceSum(d, sum);
    }

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
inline void
center_row_highway(const T* VNLB_RESTRICT input, const T* VNLB_RESTRICT mean,
                   T* VNLB_RESTRICT output, std::size_t count) noexcept {
    namespace hn = hwy::HWY_NAMESPACE;
    const hn::ScalableTag<T> d;
    const std::size_t lanes = hn::Lanes(d);

    std::size_t index = 0;
    if (count == lanes) {
        hn::StoreU(hn::Sub(hn::LoadU(d, input), hn::LoadU(d, mean)), d, output);
        return;
    }
    if (count == (2 * lanes)) {
        hn::StoreU(hn::Sub(hn::LoadU(d, input), hn::LoadU(d, mean)), d, output);
        hn::StoreU(
            hn::Sub(hn::LoadU(d, input + lanes), hn::LoadU(d, mean + lanes)), d,
            output + lanes);
        return;
    }
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
inline void copy_contiguous_highway(const T* VNLB_RESTRICT input,
                                    T* VNLB_RESTRICT output,
                                    std::size_t count) noexcept {
    namespace hn = hwy::HWY_NAMESPACE;
    const hn::ScalableTag<T> d;
    const std::size_t lanes = hn::Lanes(d);

    std::size_t index = 0;
    if (count == lanes) {
        hn::StoreU(hn::LoadU(d, input), d, output);
        return;
    }
    if (count == (2 * lanes)) {
        hn::StoreU(hn::LoadU(d, input), d, output);
        hn::StoreU(hn::LoadU(d, input + lanes), d, output + lanes);
        return;
    }
    for (; index + lanes <= count; index += lanes) {
        hn::StoreU(hn::LoadU(d, input + index), d, output + index);
    }
    for (; index < count; ++index) {
        output[index] = input[index];
    }
}

template <std::floating_point T>
inline void add_contiguous_highway(T* VNLB_RESTRICT output,
                                   const T* VNLB_RESTRICT input,
                                   std::size_t count) noexcept {
    namespace hn = hwy::HWY_NAMESPACE;
    const hn::ScalableTag<T> d;
    const std::size_t lanes = hn::Lanes(d);

    std::size_t index = 0;
    if (count == lanes) {
        hn::StoreU(hn::Add(hn::LoadU(d, output), hn::LoadU(d, input)), d,
                   output);
        return;
    }
    if (count == (2 * lanes)) {
        hn::StoreU(hn::Add(hn::LoadU(d, output), hn::LoadU(d, input)), d,
                   output);
        hn::StoreU(
            hn::Add(hn::LoadU(d, output + lanes), hn::LoadU(d, input + lanes)),
            d, output + lanes);
        return;
    }
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
inline void add_scaled_contiguous_highway(T* VNLB_RESTRICT output,
                                          const T* VNLB_RESTRICT input, T scale,
                                          std::size_t count) noexcept {
    namespace hn = hwy::HWY_NAMESPACE;
    const hn::ScalableTag<T> d;
    const std::size_t lanes = hn::Lanes(d);
    const auto scale_values = hn::Set(d, scale);

    std::size_t index = 0;
    if (count == lanes) {
        hn::StoreU(
            hn::MulAdd(scale_values, hn::LoadU(d, input), hn::LoadU(d, output)),
            d, output);
        return;
    }
    if (count == (2 * lanes)) {
        hn::StoreU(
            hn::MulAdd(scale_values, hn::LoadU(d, input), hn::LoadU(d, output)),
            d, output);
        hn::StoreU(hn::MulAdd(scale_values, hn::LoadU(d, input + lanes),
                              hn::LoadU(d, output + lanes)),
                   d, output + lanes);
        return;
    }
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
inline void add_scalar_contiguous_highway(T* VNLB_RESTRICT row, T value,
                                          std::size_t count) noexcept {
    namespace hn = hwy::HWY_NAMESPACE;
    const hn::ScalableTag<T> d;
    const std::size_t lanes = hn::Lanes(d);
    const auto value_values = hn::Set(d, value);

    std::size_t index = 0;
    if (count == lanes) {
        hn::StoreU(hn::Add(hn::LoadU(d, row), value_values), d, row);
        return;
    }
    if (count == (2 * lanes)) {
        hn::StoreU(hn::Add(hn::LoadU(d, row), value_values), d, row);
        hn::StoreU(hn::Add(hn::LoadU(d, row + lanes), value_values), d,
                   row + lanes);
        return;
    }
    for (; index + lanes <= count; index += lanes) {
        hn::StoreU(hn::Add(hn::LoadU(d, row + index), value_values), d,
                   row + index);
    }
    for (; index < count; ++index) {
        row[index] += value;
    }
}

template <std::floating_point T>
inline void add_contiguous_and_scalar_contiguous_highway(
    T* VNLB_RESTRICT output, T* VNLB_RESTRICT scalar_output,
    const T* VNLB_RESTRICT input, T scalar, std::size_t count) noexcept {
    namespace hn = hwy::HWY_NAMESPACE;
    const hn::ScalableTag<T> d;
    const std::size_t lanes = hn::Lanes(d);
    const auto scalar_values = hn::Set(d, scalar);

    std::size_t index = 0;
    if (count == lanes) {
        hn::StoreU(hn::Add(hn::LoadU(d, output), hn::LoadU(d, input)), d,
                   output);
        hn::StoreU(hn::Add(hn::LoadU(d, scalar_output), scalar_values), d,
                   scalar_output);
        return;
    }
    if (count == (2 * lanes)) {
        hn::StoreU(hn::Add(hn::LoadU(d, output), hn::LoadU(d, input)), d,
                   output);
        hn::StoreU(hn::Add(hn::LoadU(d, scalar_output), scalar_values), d,
                   scalar_output);
        hn::StoreU(
            hn::Add(hn::LoadU(d, output + lanes), hn::LoadU(d, input + lanes)),
            d, output + lanes);
        hn::StoreU(hn::Add(hn::LoadU(d, scalar_output + lanes), scalar_values),
                   d, scalar_output + lanes);
        return;
    }
    for (; index + lanes <= count; index += lanes) {
        hn::StoreU(
            hn::Add(hn::LoadU(d, output + index), hn::LoadU(d, input + index)),
            d, output + index);
        hn::StoreU(hn::Add(hn::LoadU(d, scalar_output + index), scalar_values),
                   d, scalar_output + index);
    }
    for (; index < count; ++index) {
        output[index] += input[index];
        scalar_output[index] += scalar;
    }
}

template <std::floating_point T>
inline void scale_contiguous_highway(T* VNLB_RESTRICT row, T scale,
                                     std::size_t count) noexcept {
    namespace hn = hwy::HWY_NAMESPACE;
    const hn::ScalableTag<T> d;
    const std::size_t lanes = hn::Lanes(d);
    const auto scale_values = hn::Set(d, scale);

    std::size_t index = 0;
    if (count == lanes) {
        hn::StoreU(hn::Mul(hn::LoadU(d, row), scale_values), d, row);
        return;
    }
    if (count == (2 * lanes)) {
        hn::StoreU(hn::Mul(hn::LoadU(d, row), scale_values), d, row);
        hn::StoreU(hn::Mul(hn::LoadU(d, row + lanes), scale_values), d,
                   row + lanes);
        return;
    }
    for (; index + lanes <= count; index += lanes) {
        hn::StoreU(hn::Mul(hn::LoadU(d, row + index), scale_values), d,
                   row + index);
    }
    for (; index < count; ++index) {
        row[index] *= scale;
    }
}

} // namespace vnlb::linalg::kernels
