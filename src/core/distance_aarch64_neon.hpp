#pragma once

#include <arm_neon.h>

namespace vnlb::core::distance {

[[nodiscard]] inline float add_bounded_squared_row_distance_aarch64_neon(
    float distance, const float* left, const float* right, int count) noexcept {
    float32x4_t sum0 = vdupq_n_f32(0.0F);
    float32x4_t sum1 = vdupq_n_f32(0.0F);
    float32x4_t sum2 = vdupq_n_f32(0.0F);
    float32x4_t sum3 = vdupq_n_f32(0.0F);

    int index = 0;
    for (; index + 16 <= count; index += 16) {
        const float32x4_t left0 = vld1q_f32(left + index);
        const float32x4_t right0 = vld1q_f32(right + index);
        const float32x4_t diff0 = vsubq_f32(left0, right0);
        sum0 = vfmaq_f32(sum0, diff0, diff0);

        const float32x4_t left1 = vld1q_f32(left + index + 4);
        const float32x4_t right1 = vld1q_f32(right + index + 4);
        const float32x4_t diff1 = vsubq_f32(left1, right1);
        sum1 = vfmaq_f32(sum1, diff1, diff1);

        const float32x4_t left2 = vld1q_f32(left + index + 8);
        const float32x4_t right2 = vld1q_f32(right + index + 8);
        const float32x4_t diff2 = vsubq_f32(left2, right2);
        sum2 = vfmaq_f32(sum2, diff2, diff2);

        const float32x4_t left3 = vld1q_f32(left + index + 12);
        const float32x4_t right3 = vld1q_f32(right + index + 12);
        const float32x4_t diff3 = vsubq_f32(left3, right3);
        sum3 = vfmaq_f32(sum3, diff3, diff3);
    }

    sum0 = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));
    for (; index + 4 <= count; index += 4) {
        const float32x4_t left_values = vld1q_f32(left + index);
        const float32x4_t right_values = vld1q_f32(right + index);
        const float32x4_t diff = vsubq_f32(left_values, right_values);
        sum0 = vfmaq_f32(sum0, diff, diff);
    }

    distance += vaddvq_f32(sum0);
    for (; index < count; ++index) {
        const float diff = left[index] - right[index];
        distance += diff * diff;
    }
    return distance;
}

} // namespace vnlb::core::distance
