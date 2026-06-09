#pragma once

#include "distance_highway.hpp"

namespace vnlb::core::distance {

[[nodiscard]] inline float
add_squared_row_distance(float distance, const float* left, const float* right,
                         int count) noexcept {
    return add_squared_row_distance_highway(distance, left, right, count);
}

} // namespace vnlb::core::distance
