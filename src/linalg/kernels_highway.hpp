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
    for (; index + (4 * lanes) <= count; index += 4 * lanes) {
        auto sum0 =
            hn::Mul(hn::LoadU(d, left + index), hn::LoadU(d, right + index));
        auto sum1 = hn::Mul(hn::LoadU(d, left + index + lanes),
                            hn::LoadU(d, right + index + lanes));
        auto sum2 = hn::Mul(hn::LoadU(d, left + index + (2 * lanes)),
                            hn::LoadU(d, right + index + (2 * lanes)));
        auto sum3 = hn::Mul(hn::LoadU(d, left + index + (3 * lanes)),
                            hn::LoadU(d, right + index + (3 * lanes)));
        sum = hn::Add(sum, hn::Add(hn::Add(sum0, sum1), hn::Add(sum2, sum3)));
    }
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
    for (; index + (4 * lanes) <= count; index += 4 * lanes) {
        const auto mean0 = hn::LoadU(d, mean + index);
        const auto left0 = hn::Sub(hn::LoadU(d, left + index), mean0);
        const auto right0 = hn::Sub(hn::LoadU(d, right + index), mean0);

        const auto mean1 = hn::LoadU(d, mean + index + lanes);
        const auto left1 = hn::Sub(hn::LoadU(d, left + index + lanes), mean1);
        const auto right1 = hn::Sub(hn::LoadU(d, right + index + lanes), mean1);

        const auto mean2 = hn::LoadU(d, mean + index + (2 * lanes));
        const auto left2 =
            hn::Sub(hn::LoadU(d, left + index + (2 * lanes)), mean2);
        const auto right2 =
            hn::Sub(hn::LoadU(d, right + index + (2 * lanes)), mean2);

        const auto mean3 = hn::LoadU(d, mean + index + (3 * lanes));
        const auto left3 =
            hn::Sub(hn::LoadU(d, left + index + (3 * lanes)), mean3);
        const auto right3 =
            hn::Sub(hn::LoadU(d, right + index + (3 * lanes)), mean3);

        sum = hn::Add(
            sum,
            hn::Add(hn::Add(hn::Mul(left0, right0), hn::Mul(left1, right1)),
                    hn::Add(hn::Mul(left2, right2), hn::Mul(left3, right3))));
    }
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
    for (; index + (4 * lanes) <= count; index += 4 * lanes) {
        hn::StoreU(
            hn::Sub(hn::LoadU(d, input + index), hn::LoadU(d, mean + index)), d,
            output + index);
        hn::StoreU(hn::Sub(hn::LoadU(d, input + index + lanes),
                           hn::LoadU(d, mean + index + lanes)),
                   d, output + index + lanes);
        hn::StoreU(hn::Sub(hn::LoadU(d, input + index + (2 * lanes)),
                           hn::LoadU(d, mean + index + (2 * lanes))),
                   d, output + index + (2 * lanes));
        hn::StoreU(hn::Sub(hn::LoadU(d, input + index + (3 * lanes)),
                           hn::LoadU(d, mean + index + (3 * lanes))),
                   d, output + index + (3 * lanes));
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
    for (; index + (4 * lanes) <= count; index += 4 * lanes) {
        hn::StoreU(hn::LoadU(d, input + index), d, output + index);
        hn::StoreU(hn::LoadU(d, input + index + lanes), d,
                   output + index + lanes);
        hn::StoreU(hn::LoadU(d, input + index + (2 * lanes)), d,
                   output + index + (2 * lanes));
        hn::StoreU(hn::LoadU(d, input + index + (3 * lanes)), d,
                   output + index + (3 * lanes));
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
    for (; index + (4 * lanes) <= count; index += 4 * lanes) {
        hn::StoreU(
            hn::Add(hn::LoadU(d, output + index), hn::LoadU(d, input + index)),
            d, output + index);
        hn::StoreU(hn::Add(hn::LoadU(d, output + index + lanes),
                           hn::LoadU(d, input + index + lanes)),
                   d, output + index + lanes);
        hn::StoreU(hn::Add(hn::LoadU(d, output + index + (2 * lanes)),
                           hn::LoadU(d, input + index + (2 * lanes))),
                   d, output + index + (2 * lanes));
        hn::StoreU(hn::Add(hn::LoadU(d, output + index + (3 * lanes)),
                           hn::LoadU(d, input + index + (3 * lanes))),
                   d, output + index + (3 * lanes));
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
    for (; index + (4 * lanes) <= count; index += 4 * lanes) {
        hn::StoreU(hn::MulAdd(scale_values, hn::LoadU(d, input + index),
                              hn::LoadU(d, output + index)),
                   d, output + index);
        hn::StoreU(hn::MulAdd(scale_values, hn::LoadU(d, input + index + lanes),
                              hn::LoadU(d, output + index + lanes)),
                   d, output + index + lanes);
        hn::StoreU(hn::MulAdd(scale_values,
                              hn::LoadU(d, input + index + (2 * lanes)),
                              hn::LoadU(d, output + index + (2 * lanes))),
                   d, output + index + (2 * lanes));
        hn::StoreU(hn::MulAdd(scale_values,
                              hn::LoadU(d, input + index + (3 * lanes)),
                              hn::LoadU(d, output + index + (3 * lanes))),
                   d, output + index + (3 * lanes));
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
    for (; index + (4 * lanes) <= count; index += 4 * lanes) {
        hn::StoreU(hn::Add(hn::LoadU(d, row + index), value_values), d,
                   row + index);
        hn::StoreU(hn::Add(hn::LoadU(d, row + index + lanes), value_values), d,
                   row + index + lanes);
        hn::StoreU(
            hn::Add(hn::LoadU(d, row + index + (2 * lanes)), value_values), d,
            row + index + (2 * lanes));
        hn::StoreU(
            hn::Add(hn::LoadU(d, row + index + (3 * lanes)), value_values), d,
            row + index + (3 * lanes));
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
    for (; index + (4 * lanes) <= count; index += 4 * lanes) {
        hn::StoreU(
            hn::Add(hn::LoadU(d, output + index), hn::LoadU(d, input + index)),
            d, output + index);
        hn::StoreU(hn::Add(hn::LoadU(d, scalar_output + index), scalar_values),
                   d, scalar_output + index);
        hn::StoreU(hn::Add(hn::LoadU(d, output + index + lanes),
                           hn::LoadU(d, input + index + lanes)),
                   d, output + index + lanes);
        hn::StoreU(
            hn::Add(hn::LoadU(d, scalar_output + index + lanes), scalar_values),
            d, scalar_output + index + lanes);
        hn::StoreU(hn::Add(hn::LoadU(d, output + index + (2 * lanes)),
                           hn::LoadU(d, input + index + (2 * lanes))),
                   d, output + index + (2 * lanes));
        hn::StoreU(hn::Add(hn::LoadU(d, scalar_output + index + (2 * lanes)),
                           scalar_values),
                   d, scalar_output + index + (2 * lanes));
        hn::StoreU(hn::Add(hn::LoadU(d, output + index + (3 * lanes)),
                           hn::LoadU(d, input + index + (3 * lanes))),
                   d, output + index + (3 * lanes));
        hn::StoreU(hn::Add(hn::LoadU(d, scalar_output + index + (3 * lanes)),
                           scalar_values),
                   d, scalar_output + index + (3 * lanes));
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
    for (; index + (4 * lanes) <= count; index += 4 * lanes) {
        hn::StoreU(hn::Mul(hn::LoadU(d, row + index), scale_values), d,
                   row + index);
        hn::StoreU(hn::Mul(hn::LoadU(d, row + index + lanes), scale_values), d,
                   row + index + lanes);
        hn::StoreU(
            hn::Mul(hn::LoadU(d, row + index + (2 * lanes)), scale_values), d,
            row + index + (2 * lanes));
        hn::StoreU(
            hn::Mul(hn::LoadU(d, row + index + (3 * lanes)), scale_values), d,
            row + index + (3 * lanes));
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
