#pragma once

#include <arm_neon.h>

#include <cstddef>

namespace vnlb::linalg::kernels {

[[nodiscard]] inline float
dot_contiguous_aarch64_neon(const float* left, const float* right,
                            std::size_t count) noexcept {
    float32x4_t sum0 = vdupq_n_f32(0.0F);
    float32x4_t sum1 = vdupq_n_f32(0.0F);
    float32x4_t sum2 = vdupq_n_f32(0.0F);
    float32x4_t sum3 = vdupq_n_f32(0.0F);

    std::size_t index = 0;
    for (; index + 16 <= count; index += 16) {
        sum0 =
            vfmaq_f32(sum0, vld1q_f32(left + index), vld1q_f32(right + index));
        sum1 = vfmaq_f32(sum1, vld1q_f32(left + index + 4),
                         vld1q_f32(right + index + 4));
        sum2 = vfmaq_f32(sum2, vld1q_f32(left + index + 8),
                         vld1q_f32(right + index + 8));
        sum3 = vfmaq_f32(sum3, vld1q_f32(left + index + 12),
                         vld1q_f32(right + index + 12));
    }

    sum0 = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));
    for (; index + 4 <= count; index += 4) {
        sum0 =
            vfmaq_f32(sum0, vld1q_f32(left + index), vld1q_f32(right + index));
    }

    float sum = vaddvq_f32(sum0);
    for (; index < count; ++index) {
        sum += left[index] * right[index];
    }
    return sum;
}

[[nodiscard]] inline float
dot_centered_rows_aarch64_neon(const float* left, const float* right,
                               const float* mean, std::size_t count) noexcept {
    float32x4_t sum0 = vdupq_n_f32(0.0F);
    float32x4_t sum1 = vdupq_n_f32(0.0F);
    float32x4_t sum2 = vdupq_n_f32(0.0F);
    float32x4_t sum3 = vdupq_n_f32(0.0F);

    std::size_t index = 0;
    for (; index + 16 <= count; index += 16) {
        const float32x4_t mean0 = vld1q_f32(mean + index);
        const float32x4_t left0 = vsubq_f32(vld1q_f32(left + index), mean0);
        const float32x4_t right0 = vsubq_f32(vld1q_f32(right + index), mean0);
        sum0 = vfmaq_f32(sum0, left0, right0);

        const float32x4_t mean1 = vld1q_f32(mean + index + 4);
        const float32x4_t left1 = vsubq_f32(vld1q_f32(left + index + 4), mean1);
        const float32x4_t right1 =
            vsubq_f32(vld1q_f32(right + index + 4), mean1);
        sum1 = vfmaq_f32(sum1, left1, right1);

        const float32x4_t mean2 = vld1q_f32(mean + index + 8);
        const float32x4_t left2 = vsubq_f32(vld1q_f32(left + index + 8), mean2);
        const float32x4_t right2 =
            vsubq_f32(vld1q_f32(right + index + 8), mean2);
        sum2 = vfmaq_f32(sum2, left2, right2);

        const float32x4_t mean3 = vld1q_f32(mean + index + 12);
        const float32x4_t left3 =
            vsubq_f32(vld1q_f32(left + index + 12), mean3);
        const float32x4_t right3 =
            vsubq_f32(vld1q_f32(right + index + 12), mean3);
        sum3 = vfmaq_f32(sum3, left3, right3);
    }

    sum0 = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));
    for (; index + 4 <= count; index += 4) {
        const float32x4_t mean_values = vld1q_f32(mean + index);
        const float32x4_t left_values =
            vsubq_f32(vld1q_f32(left + index), mean_values);
        const float32x4_t right_values =
            vsubq_f32(vld1q_f32(right + index), mean_values);
        sum0 = vfmaq_f32(sum0, left_values, right_values);
    }

    float sum = vaddvq_f32(sum0);
    for (; index < count; ++index) {
        sum += (left[index] - mean[index]) * (right[index] - mean[index]);
    }
    return sum;
}

inline void center_row_aarch64_neon(const float* input, const float* mean,
                                    float* output, std::size_t count) noexcept {
    std::size_t index = 0;
    for (; index + 16 <= count; index += 16) {
        vst1q_f32(output + index,
                  vsubq_f32(vld1q_f32(input + index), vld1q_f32(mean + index)));
        vst1q_f32(output + index + 4, vsubq_f32(vld1q_f32(input + index + 4),
                                                vld1q_f32(mean + index + 4)));
        vst1q_f32(output + index + 8, vsubq_f32(vld1q_f32(input + index + 8),
                                                vld1q_f32(mean + index + 8)));
        vst1q_f32(output + index + 12, vsubq_f32(vld1q_f32(input + index + 12),
                                                 vld1q_f32(mean + index + 12)));
    }
    for (; index + 4 <= count; index += 4) {
        vst1q_f32(output + index,
                  vsubq_f32(vld1q_f32(input + index), vld1q_f32(mean + index)));
    }
    for (; index < count; ++index) {
        output[index] = input[index] - mean[index];
    }
}

inline void add_contiguous_aarch64_neon(float* output, const float* input,
                                        std::size_t count) noexcept {
    std::size_t index = 0;
    for (; index + 16 <= count; index += 16) {
        vst1q_f32(output + index, vaddq_f32(vld1q_f32(output + index),
                                            vld1q_f32(input + index)));
        vst1q_f32(output + index + 4, vaddq_f32(vld1q_f32(output + index + 4),
                                                vld1q_f32(input + index + 4)));
        vst1q_f32(output + index + 8, vaddq_f32(vld1q_f32(output + index + 8),
                                                vld1q_f32(input + index + 8)));
        vst1q_f32(output + index + 12,
                  vaddq_f32(vld1q_f32(output + index + 12),
                            vld1q_f32(input + index + 12)));
    }
    for (; index + 4 <= count; index += 4) {
        vst1q_f32(output + index, vaddq_f32(vld1q_f32(output + index),
                                            vld1q_f32(input + index)));
    }
    for (; index < count; ++index) {
        output[index] += input[index];
    }
}

inline void add_scaled_contiguous_aarch64_neon(float* output,
                                               const float* input, float scale,
                                               std::size_t count) noexcept {
    std::size_t index = 0;
    for (; index + 16 <= count; index += 16) {
        vst1q_f32(output + index, vfmaq_n_f32(vld1q_f32(output + index),
                                              vld1q_f32(input + index), scale));
        vst1q_f32(output + index + 4,
                  vfmaq_n_f32(vld1q_f32(output + index + 4),
                              vld1q_f32(input + index + 4), scale));
        vst1q_f32(output + index + 8,
                  vfmaq_n_f32(vld1q_f32(output + index + 8),
                              vld1q_f32(input + index + 8), scale));
        vst1q_f32(output + index + 12,
                  vfmaq_n_f32(vld1q_f32(output + index + 12),
                              vld1q_f32(input + index + 12), scale));
    }
    for (; index + 4 <= count; index += 4) {
        vst1q_f32(output + index, vfmaq_n_f32(vld1q_f32(output + index),
                                              vld1q_f32(input + index), scale));
    }
    for (; index < count; ++index) {
        output[index] += scale * input[index];
    }
}

inline void scale_contiguous_aarch64_neon(float* row, float scale,
                                          std::size_t count) noexcept {
    const float32x4_t scale_values = vdupq_n_f32(scale);
    std::size_t index = 0;
    for (; index + 16 <= count; index += 16) {
        vst1q_f32(row + index, vmulq_f32(vld1q_f32(row + index), scale_values));
        vst1q_f32(row + index + 4,
                  vmulq_f32(vld1q_f32(row + index + 4), scale_values));
        vst1q_f32(row + index + 8,
                  vmulq_f32(vld1q_f32(row + index + 8), scale_values));
        vst1q_f32(row + index + 12,
                  vmulq_f32(vld1q_f32(row + index + 12), scale_values));
    }
    for (; index + 4 <= count; index += 4) {
        vst1q_f32(row + index, vmulq_f32(vld1q_f32(row + index), scale_values));
    }
    for (; index < count; ++index) {
        row[index] *= scale;
    }
}

} // namespace vnlb::linalg::kernels
