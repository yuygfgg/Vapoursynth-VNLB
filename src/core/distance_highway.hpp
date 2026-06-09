#pragma once

#include <hwy/highway.h>

#include <cstddef>

namespace vnlb::core::distance {

[[nodiscard]] inline float add_bounded_squared_row_distance_highway(
    float distance, const float* left, const float* right, int count) noexcept {
    if (count <= 0) {
        return distance;
    }

    namespace hn = hwy::HWY_NAMESPACE;
    const hn::CappedTag<float, 4> d;
    const std::size_t lanes = hn::Lanes(d);
    const std::size_t step = lanes * 4;
    const auto total = static_cast<std::size_t>(count);

    auto sum0 = hn::Zero(d);
    auto sum1 = hn::Zero(d);
    auto sum2 = hn::Zero(d);
    auto sum3 = hn::Zero(d);

    std::size_t index = 0;
    for (; index + step <= total; index += step) {
        const auto left0 = hn::LoadU(d, left + index);
        const auto right0 = hn::LoadU(d, right + index);
        const auto diff0 = hn::Sub(left0, right0);
        sum0 = hn::MulAdd(diff0, diff0, sum0);

        const auto left1 = hn::LoadU(d, left + index + lanes);
        const auto right1 = hn::LoadU(d, right + index + lanes);
        const auto diff1 = hn::Sub(left1, right1);
        sum1 = hn::MulAdd(diff1, diff1, sum1);

        const auto left2 = hn::LoadU(d, left + index + (lanes * 2));
        const auto right2 = hn::LoadU(d, right + index + (lanes * 2));
        const auto diff2 = hn::Sub(left2, right2);
        sum2 = hn::MulAdd(diff2, diff2, sum2);

        const auto left3 = hn::LoadU(d, left + index + (lanes * 3));
        const auto right3 = hn::LoadU(d, right + index + (lanes * 3));
        const auto diff3 = hn::Sub(left3, right3);
        sum3 = hn::MulAdd(diff3, diff3, sum3);
    }

    sum0 = hn::Add(hn::Add(sum0, sum1), hn::Add(sum2, sum3));
    for (; index + lanes <= total; index += lanes) {
        const auto left_values = hn::LoadU(d, left + index);
        const auto right_values = hn::LoadU(d, right + index);
        const auto diff = hn::Sub(left_values, right_values);
        sum0 = hn::MulAdd(diff, diff, sum0);
    }

    distance += hn::ReduceSum(d, sum0);
    for (; index < total; ++index) {
        const float diff = left[index] - right[index];
        distance += diff * diff;
    }
    return distance;
}

} // namespace vnlb::core::distance
