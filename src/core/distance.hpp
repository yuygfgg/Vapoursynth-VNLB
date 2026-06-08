#pragma once

#ifndef VNLB_CORE_USE_AARCH64_NEON
#define VNLB_CORE_USE_AARCH64_NEON 0
#endif

#include "distance_generic.hpp"

#if VNLB_CORE_USE_AARCH64_NEON
#include "distance_aarch64_neon.hpp"
#endif

namespace vnlb::core::distance {

[[nodiscard]] inline float
add_bounded_squared_row_distance(float distance, const float* left,
                                 const float* right, int count,
                                 float stop_after) noexcept {
#if VNLB_CORE_USE_AARCH64_NEON
    if (count >= 4) {
        return add_bounded_squared_row_distance_aarch64_neon(distance, left,
                                                             right, count);
    }
#endif
    return add_bounded_squared_row_distance_generic(distance, left, right,
                                                    count, stop_after);
}

} // namespace vnlb::core::distance
