#pragma once

#include "distance_highway.hpp"

namespace vnlb::core::distance {

[[nodiscard]] inline float
add_bounded_squared_row_distance(float distance, const float* left,
                                 const float* right, int count,
                                 [[maybe_unused]] float stop_after) noexcept {
    return add_bounded_squared_row_distance_highway(distance, left, right,
                                                    count);
}

} // namespace vnlb::core::distance
