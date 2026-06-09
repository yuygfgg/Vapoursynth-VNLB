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
    const std::size_t step = lanes * 4;

    auto sum0 = hn::Zero(d);
    auto sum1 = hn::Zero(d);
    auto sum2 = hn::Zero(d);
    auto sum3 = hn::Zero(d);

    std::size_t index = 0;
    for (; index + step <= count; index += step) {
        sum0 = hn::MulAdd(hn::LoadU(d, left + index),
                          hn::LoadU(d, right + index), sum0);
        sum1 = hn::MulAdd(hn::LoadU(d, left + index + lanes),
                          hn::LoadU(d, right + index + lanes), sum1);
        sum2 = hn::MulAdd(hn::LoadU(d, left + index + (lanes * 2)),
                          hn::LoadU(d, right + index + (lanes * 2)), sum2);
        sum3 = hn::MulAdd(hn::LoadU(d, left + index + (lanes * 3)),
                          hn::LoadU(d, right + index + (lanes * 3)), sum3);
    }

    sum0 = hn::Add(hn::Add(sum0, sum1), hn::Add(sum2, sum3));
    for (; index + lanes <= count; index += lanes) {
        sum0 = hn::MulAdd(hn::LoadU(d, left + index),
                          hn::LoadU(d, right + index), sum0);
    }

    T sum = hn::ReduceSum(d, sum0);
    for (; index < count; ++index) {
        sum += left[index] * right[index];
    }
    return sum;
}

template <std::floating_point T>
[[nodiscard]] inline T dot_centered_rows_highway(const T* left, const T* right,
                                                 const T* mean,
                                                 std::size_t count) noexcept {
    namespace hn = hwy::HWY_NAMESPACE;
    const hn::ScalableTag<T> d;
    const std::size_t lanes = hn::Lanes(d);
    const std::size_t step = lanes * 4;

    auto sum0 = hn::Zero(d);
    auto sum1 = hn::Zero(d);
    auto sum2 = hn::Zero(d);
    auto sum3 = hn::Zero(d);

    std::size_t index = 0;
    for (; index + step <= count; index += step) {
        const auto mean0 = hn::LoadU(d, mean + index);
        const auto left0 = hn::Sub(hn::LoadU(d, left + index), mean0);
        const auto right0 = hn::Sub(hn::LoadU(d, right + index), mean0);
        sum0 = hn::MulAdd(left0, right0, sum0);

        const auto mean1 = hn::LoadU(d, mean + index + lanes);
        const auto left1 = hn::Sub(hn::LoadU(d, left + index + lanes), mean1);
        const auto right1 = hn::Sub(hn::LoadU(d, right + index + lanes), mean1);
        sum1 = hn::MulAdd(left1, right1, sum1);

        const auto mean2 = hn::LoadU(d, mean + index + (lanes * 2));
        const auto left2 =
            hn::Sub(hn::LoadU(d, left + index + (lanes * 2)), mean2);
        const auto right2 =
            hn::Sub(hn::LoadU(d, right + index + (lanes * 2)), mean2);
        sum2 = hn::MulAdd(left2, right2, sum2);

        const auto mean3 = hn::LoadU(d, mean + index + (lanes * 3));
        const auto left3 =
            hn::Sub(hn::LoadU(d, left + index + (lanes * 3)), mean3);
        const auto right3 =
            hn::Sub(hn::LoadU(d, right + index + (lanes * 3)), mean3);
        sum3 = hn::MulAdd(left3, right3, sum3);
    }

    sum0 = hn::Add(hn::Add(sum0, sum1), hn::Add(sum2, sum3));
    for (; index + lanes <= count; index += lanes) {
        const auto mean_values = hn::LoadU(d, mean + index);
        const auto left_values =
            hn::Sub(hn::LoadU(d, left + index), mean_values);
        const auto right_values =
            hn::Sub(hn::LoadU(d, right + index), mean_values);
        sum0 = hn::MulAdd(left_values, right_values, sum0);
    }

    T sum = hn::ReduceSum(d, sum0);
    for (; index < count; ++index) {
        sum += (left[index] - mean[index]) * (right[index] - mean[index]);
    }
    return sum;
}

template <std::floating_point T>
inline void center_row_highway(const T* input, const T* mean, T* output,
                               std::size_t count) noexcept {
    namespace hn = hwy::HWY_NAMESPACE;
    const hn::ScalableTag<T> d;
    const std::size_t lanes = hn::Lanes(d);
    const std::size_t step = lanes * 4;

    std::size_t index = 0;
    for (; index + step <= count; index += step) {
        hn::StoreU(
            hn::Sub(hn::LoadU(d, input + index), hn::LoadU(d, mean + index)), d,
            output + index);
        hn::StoreU(hn::Sub(hn::LoadU(d, input + index + lanes),
                           hn::LoadU(d, mean + index + lanes)),
                   d, output + index + lanes);
        hn::StoreU(hn::Sub(hn::LoadU(d, input + index + (lanes * 2)),
                           hn::LoadU(d, mean + index + (lanes * 2))),
                   d, output + index + (lanes * 2));
        hn::StoreU(hn::Sub(hn::LoadU(d, input + index + (lanes * 3)),
                           hn::LoadU(d, mean + index + (lanes * 3))),
                   d, output + index + (lanes * 3));
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
inline void add_contiguous_highway(T* output, const T* input,
                                   std::size_t count) noexcept {
    namespace hn = hwy::HWY_NAMESPACE;
    const hn::ScalableTag<T> d;
    const std::size_t lanes = hn::Lanes(d);
    const std::size_t step = lanes * 4;

    std::size_t index = 0;
    for (; index + step <= count; index += step) {
        hn::StoreU(
            hn::Add(hn::LoadU(d, output + index), hn::LoadU(d, input + index)),
            d, output + index);
        hn::StoreU(hn::Add(hn::LoadU(d, output + index + lanes),
                           hn::LoadU(d, input + index + lanes)),
                   d, output + index + lanes);
        hn::StoreU(hn::Add(hn::LoadU(d, output + index + (lanes * 2)),
                           hn::LoadU(d, input + index + (lanes * 2))),
                   d, output + index + (lanes * 2));
        hn::StoreU(hn::Add(hn::LoadU(d, output + index + (lanes * 3)),
                           hn::LoadU(d, input + index + (lanes * 3))),
                   d, output + index + (lanes * 3));
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
inline void add_scaled_contiguous_highway(T* output, const T* input, T scale,
                                          std::size_t count) noexcept {
    namespace hn = hwy::HWY_NAMESPACE;
    const hn::ScalableTag<T> d;
    const std::size_t lanes = hn::Lanes(d);
    const std::size_t step = lanes * 4;
    const auto scale_values = hn::Set(d, scale);

    std::size_t index = 0;
    for (; index + step <= count; index += step) {
        hn::StoreU(hn::MulAdd(scale_values, hn::LoadU(d, input + index),
                              hn::LoadU(d, output + index)),
                   d, output + index);
        hn::StoreU(hn::MulAdd(scale_values, hn::LoadU(d, input + index + lanes),
                              hn::LoadU(d, output + index + lanes)),
                   d, output + index + lanes);
        hn::StoreU(hn::MulAdd(scale_values,
                              hn::LoadU(d, input + index + (lanes * 2)),
                              hn::LoadU(d, output + index + (lanes * 2))),
                   d, output + index + (lanes * 2));
        hn::StoreU(hn::MulAdd(scale_values,
                              hn::LoadU(d, input + index + (lanes * 3)),
                              hn::LoadU(d, output + index + (lanes * 3))),
                   d, output + index + (lanes * 3));
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
inline void scale_contiguous_highway(T* row, T scale,
                                     std::size_t count) noexcept {
    namespace hn = hwy::HWY_NAMESPACE;
    const hn::ScalableTag<T> d;
    const std::size_t lanes = hn::Lanes(d);
    const std::size_t step = lanes * 4;
    const auto scale_values = hn::Set(d, scale);

    std::size_t index = 0;
    for (; index + step <= count; index += step) {
        hn::StoreU(hn::Mul(hn::LoadU(d, row + index), scale_values), d,
                   row + index);
        hn::StoreU(hn::Mul(hn::LoadU(d, row + index + lanes), scale_values), d,
                   row + index + lanes);
        hn::StoreU(
            hn::Mul(hn::LoadU(d, row + index + (lanes * 2)), scale_values), d,
            row + index + (lanes * 2));
        hn::StoreU(
            hn::Mul(hn::LoadU(d, row + index + (lanes * 3)), scale_values), d,
            row + index + (lanes * 3));
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
