#pragma once

namespace vnlb::core::distance {

[[nodiscard]] inline float
add_bounded_squared_row_distance_generic(float distance, const float* left,
                                         const float* right, int count,
                                         float stop_after) noexcept {
    int index = 0;
    for (; index + 4 <= count; index += 4) {
        const float diff0 = left[index] - right[index];
        const float diff1 = left[index + 1] - right[index + 1];
        const float diff2 = left[index + 2] - right[index + 2];
        const float diff3 = left[index + 3] - right[index + 3];
        distance +=
            diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
        if (distance > stop_after) {
            return distance;
        }
    }

    for (; index < count; ++index) {
        const float diff = left[index] - right[index];
        distance += diff * diff;
        if (distance > stop_after) {
            return distance;
        }
    }
    return distance;
}

} // namespace vnlb::core::distance
