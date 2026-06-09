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
    const auto total = static_cast<std::size_t>(count);

    auto sum = hn::Zero(d);
    std::size_t index = 0;
    for (; index + lanes <= total; index += lanes) {
        const auto left_values = hn::LoadU(d, left + index);
        const auto right_values = hn::LoadU(d, right + index);
        const auto diff = hn::Sub(left_values, right_values);
        sum = hn::MulAdd(diff, diff, sum);
    }

    distance += hn::ReduceSum(d, sum);
    for (; index < total; ++index) {
        const float diff = left[index] - right[index];
        distance += diff * diff;
    }
    return distance;
}

} // namespace vnlb::core::distance
