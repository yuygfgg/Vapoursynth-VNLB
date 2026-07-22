#include "cuda/aggregation.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr int skipped = 77;

void check_cuda(cudaError_t status, std::string_view operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

template <typename T> class DeviceArray {
  public:
    explicit DeviceArray(std::size_t count) : count_(count) {
        if (count_ != 0) {
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_),
                                  count_ * sizeof(T)),
                       "cudaMalloc");
        }
    }

    ~DeviceArray() {
        if (data_ != nullptr) {
            (void)cudaFree(data_);
        }
    }

    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    [[nodiscard]] T* data() noexcept { return data_; }
    [[nodiscard]] const T* data() const noexcept { return data_; }

    void upload(std::span<const T> source) {
        if (source.size() != count_) {
            throw std::invalid_argument("device upload size mismatch");
        }
        check_cuda(cudaMemcpy(data_, source.data(), count_ * sizeof(T),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy host to device");
    }

    void download(std::span<T> destination) const {
        if (destination.size() != count_) {
            throw std::invalid_argument("device download size mismatch");
        }
        check_cuda(cudaMemcpy(destination.data(), data_, count_ * sizeof(T),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy device to host");
    }

  private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

void require_close(float actual, float expected, float tolerance,
                   std::string_view label) {
    const float error = std::abs(actual - expected);
    const float limit = tolerance * (1.0F + std::abs(expected));
    if (!(error <= limit)) {
        throw std::runtime_error(std::string(label) + " mismatch");
    }
}

[[nodiscard]] std::size_t plane_values(const vnlbcu::AggregationShape& shape) {
    return static_cast<std::size_t>(shape.width) * shape.height;
}

[[nodiscard]] int sample_dimension(const vnlbcu::AggregationShape& shape) {
    return shape.channels * shape.patch_time * shape.patch_size *
           shape.patch_size;
}

std::vector<float> make_window(int patch_size, float gamma) {
    std::vector<float> window(static_cast<std::size_t>(patch_size * patch_size),
                              1.0F);
    if (gamma == 0.0F || patch_size <= 1) {
        return window;
    }
    const float center = 0.5F * static_cast<float>(patch_size - 1);
    const float radius = std::max(center, 0.5F);
    float maximum = 0.0F;
    for (int y = 0; y < patch_size; ++y) {
        const float dy = (static_cast<float>(y) - center) / radius;
        for (int x = 0; x < patch_size; ++x) {
            const float dx = (static_cast<float>(x) - center) / radius;
            const float raw = std::exp(-0.5F * ((dx * dx) + (dy * dy)));
            window[static_cast<std::size_t>(y * patch_size + x)] = raw;
            maximum = std::max(maximum, raw);
        }
    }
    const float inverse_maximum = maximum > 0.0F ? 1.0F / maximum : 1.0F;
    for (float& value : window) {
        value = std::pow(value * inverse_maximum, gamma);
    }
    return window;
}

struct ReferenceContribution {
    std::vector<float> numerators;
    std::vector<float> weights;
};

ReferenceContribution
reference_scatter(const vnlbcu::AggregationShape& shape,
                  const vnlbcu::AggregationParameters& parameters,
                  std::span<const float> filtered, std::span<const float> logs,
                  std::span<const int> retained_counts,
                  std::span<const vnlbcu::PatchMatch> matches,
                  float window_gamma) {
    const std::size_t pixels = plane_values(shape);
    ReferenceContribution result{
        .numerators = std::vector<float>(
            static_cast<std::size_t>(shape.channels) * shape.slots * pixels),
        .weights =
            std::vector<float>(static_cast<std::size_t>(shape.slots) * pixels),
    };
    const int patch_area = shape.patch_size * shape.patch_size;
    const int channel_patch_dim = shape.patch_time * patch_area;
    const int sample_dim = sample_dimension(shape);
    const std::vector<float> window =
        make_window(shape.patch_size, window_gamma);

    for (int group = 0; group < shape.max_groups; ++group) {
        const int similar = retained_counts[static_cast<std::size_t>(group)];
        const std::size_t descriptor_base =
            static_cast<std::size_t>(group) * shape.retained_stride;
        const std::size_t sample_group_base =
            descriptor_base * static_cast<std::size_t>(sample_dim);
        for (int sample = 0; sample < similar; ++sample) {
            float log_sum = 0.0F;
            for (int model = 0; model < parameters.log_weight_model_count;
                 ++model) {
                log_sum += logs[static_cast<std::size_t>(model) *
                                    parameters.log_weight_model_stride +
                                descriptor_base + sample];
            }
            float average =
                log_sum / static_cast<float>(parameters.log_weight_model_count);
            average = std::clamp(average, -80.0F, 80.0F);
            const float patch_weight = std::exp(average);
            const vnlbcu::PatchMatch match = matches[descriptor_base + sample];
            const std::size_t sample_base =
                sample_group_base +
                static_cast<std::size_t>(sample) * sample_dim;
            for (int dt = 0; dt < shape.patch_time; ++dt) {
                const int slot = match.frame + dt - parameters.anchor_frame +
                                 shape.search_bwd;
                if (slot < 0 || slot >= shape.slots) {
                    continue;
                }
                for (int y = 0; y < shape.patch_size; ++y) {
                    for (int x = 0; x < shape.patch_size; ++x) {
                        const int output_x = match.x + x;
                        const int output_y = match.y + y;
                        const int position = y * shape.patch_size + x;
                        const std::size_t pixel =
                            static_cast<std::size_t>(output_y) * shape.width +
                            output_x;
                        const float contribution_weight =
                            patch_weight *
                            window[static_cast<std::size_t>(position)];
                        result.weights[static_cast<std::size_t>(slot) * pixels +
                                       pixel] += contribution_weight;
                        for (int channel = 0; channel < shape.channels;
                             ++channel) {
                            const std::size_t input =
                                sample_base +
                                static_cast<std::size_t>(channel) *
                                    channel_patch_dim +
                                static_cast<std::size_t>(dt) * patch_area +
                                position;
                            const std::size_t output =
                                (static_cast<std::size_t>(channel) *
                                     shape.slots +
                                 slot) *
                                    pixels +
                                pixel;
                            result.numerators[output] +=
                                contribution_weight * filtered[input];
                        }
                    }
                }
            }
        }
    }
    return result;
}

void run_scatter_reference_case() {
    const vnlbcu::AggregationShape shape{
        .max_groups = 3,
        .width = 17,
        .height = 15,
        .channels = 2,
        .slots = 4,
        .retained_stride = 5,
        .patch_size = 3,
        .patch_time = 2,
        .search_window = 5,
        .search_bwd = 1,
    };
    constexpr float window_gamma = 1.35F;
    const std::size_t descriptors =
        static_cast<std::size_t>(shape.max_groups) * shape.retained_stride;
    const int sample_dim = sample_dimension(shape);
    const std::size_t samples = descriptors * sample_dim;
    const std::vector<int> counts{5, 3, 4};
    std::vector<vnlbcu::PatchMatch> matches(descriptors);
    const std::vector<vnlbcu::PatchMatch> active_matches = {
        {0.0F, 1, 1, 5}, {0.1F, 3, 2, 6}, {0.2F, 5, 3, 7}, {0.3F, 2, 4, 5},
        {0.4F, 7, 1, 6}, {0.0F, 4, 3, 6}, {0.1F, 6, 5, 5}, {0.2F, 8, 6, 7},
        {0.0F, 2, 8, 5}, {0.1F, 5, 7, 6}, {0.2F, 9, 2, 6}, {0.3F, 11, 9, 7},
    };
    std::size_t active = 0;
    for (int group = 0; group < shape.max_groups; ++group) {
        const std::size_t base =
            static_cast<std::size_t>(group) * shape.retained_stride;
        for (int sample = 0; sample < counts[static_cast<std::size_t>(group)];
             ++sample) {
            matches[base + sample] = active_matches[active++];
        }
    }

    std::vector<float> filtered(samples);
    for (int group = 0; group < shape.max_groups; ++group) {
        const std::size_t group_base = static_cast<std::size_t>(group) *
                                       shape.retained_stride * sample_dim;
        for (int sample = 0; sample < shape.retained_stride; ++sample) {
            for (int dimension = 0; dimension < sample_dim; ++dimension) {
                filtered[group_base +
                         static_cast<std::size_t>(sample) * sample_dim +
                         dimension] = 0.25F * static_cast<float>(group + 1) +
                                      0.03125F * static_cast<float>(sample) +
                                      0.001F * static_cast<float>(dimension);
            }
        }
    }

    constexpr int log_models = 2;
    const std::ptrdiff_t log_stride = static_cast<std::ptrdiff_t>(descriptors);
    std::vector<float> logs(static_cast<std::size_t>(log_models) * descriptors);
    for (int model = 0; model < log_models; ++model) {
        for (int group = 0; group < shape.max_groups; ++group) {
            for (int sample = 0; sample < shape.retained_stride; ++sample) {
                logs[static_cast<std::size_t>(model) * descriptors +
                     static_cast<std::size_t>(group) * shape.retained_stride +
                     sample] = -0.17F * static_cast<float>(model + 1) -
                               0.09F * static_cast<float>(group) -
                               0.025F * static_cast<float>(sample);
            }
        }
    }

    const vnlbcu::AggregationParameters parameters{
        .anchor_frame = 6,
        .log_weight_model_count = log_models,
        .log_weight_model_stride = log_stride,
    };
    const ReferenceContribution expected = reference_scatter(
        shape, parameters, filtered, logs, counts, matches, window_gamma);

    DeviceArray<float> device_filtered(filtered.size());
    DeviceArray<float> device_logs(logs.size());
    DeviceArray<int> device_counts(counts.size());
    DeviceArray<vnlbcu::PatchMatch> device_matches(matches.size());
    DeviceArray<float> device_numerators(expected.numerators.size());
    DeviceArray<float> device_weights(expected.weights.size());
    device_filtered.upload(filtered);
    device_logs.upload(logs);
    device_counts.upload(counts);
    device_matches.upload(matches);

    const vnlbcu::DeviceContributionView contributions =
        vnlbcu::make_contiguous_contribution_view(
            shape, device_numerators.data(), device_weights.data());
    vnlbcu::Aggregator aggregator;
    aggregator.reserve(shape, window_gamma);
    aggregator.enqueue_clear(contributions);
    aggregator.enqueue_scatter(shape, parameters,
                               vnlbcu::DeviceAggregationBatch{
                                   .filtered_samples = device_filtered.data(),
                                   .log_patch_weights = device_logs.data(),
                                   .retained_counts = device_counts.data(),
                                   .matches = device_matches.data(),
                                   .groups = shape.max_groups,
                                   .contributions = contributions,
                               });
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize scatter");

    std::vector<float> actual_numerators(expected.numerators.size());
    std::vector<float> actual_weights(expected.weights.size());
    device_numerators.download(actual_numerators);
    device_weights.download(actual_weights);
    for (std::size_t index = 0; index < actual_numerators.size(); ++index) {
        require_close(actual_numerators[index], expected.numerators[index],
                      3.0e-5F, "scatter numerator");
    }
    for (std::size_t index = 0; index < actual_weights.size(); ++index) {
        require_close(actual_weights[index], expected.weights[index], 3.0e-5F,
                      "scatter weight");
    }
}

std::size_t strided_storage_size(int outer_count, std::ptrdiff_t outer_stride,
                                 int inner_count, std::ptrdiff_t inner_stride,
                                 int height, std::ptrdiff_t row_stride,
                                 int width) {
    return static_cast<std::size_t>(outer_count - 1) * outer_stride +
           static_cast<std::size_t>(inner_count - 1) * inner_stride +
           static_cast<std::size_t>(height - 1) * row_stride + width;
}

void run_strided_clear_pack_normalize_case() {
    const vnlbcu::AggregationShape shape{
        .max_groups = 2,
        .width = 9,
        .height = 7,
        .channels = 3,
        .slots = 4,
        .retained_stride = 3,
        .patch_size = 3,
        .patch_time = 2,
        .search_window = 5,
        .search_bwd = 1,
    };
    const std::ptrdiff_t numerator_row = shape.width + 3;
    const std::ptrdiff_t numerator_slot = numerator_row * shape.height + 5;
    const std::ptrdiff_t numerator_channel = numerator_slot * shape.slots + 7;
    const std::ptrdiff_t weight_row = shape.width + 2;
    const std::ptrdiff_t weight_slot = weight_row * shape.height + 4;
    const std::size_t numerator_storage = strided_storage_size(
        shape.channels, numerator_channel, shape.slots, numerator_slot,
        shape.height, numerator_row, shape.width);
    const std::size_t weight_storage = strided_storage_size(
        1, 0, shape.slots, weight_slot, shape.height, weight_row, shape.width);
    constexpr float sentinel = 12345.0F;
    std::vector<float> host_numerators(numerator_storage, sentinel);
    std::vector<float> host_weights(weight_storage, sentinel);
    DeviceArray<float> device_numerators(numerator_storage);
    DeviceArray<float> device_weights(weight_storage);
    device_numerators.upload(host_numerators);
    device_weights.upload(host_weights);

    const vnlbcu::DeviceContributionView view{
        .numerators = device_numerators.data(),
        .weights = device_weights.data(),
        .width = shape.width,
        .height = shape.height,
        .channels = shape.channels,
        .slots = shape.slots,
        .numerator_row_stride = numerator_row,
        .numerator_slot_stride = numerator_slot,
        .numerator_channel_stride = numerator_channel,
        .weight_row_stride = weight_row,
        .weight_slot_stride = weight_slot,
    };
    vnlbcu::Aggregator aggregator;
    aggregator.reserve(shape, 0.75F);
    aggregator.enqueue_clear(view);
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize clear");
    device_numerators.download(host_numerators);
    device_weights.download(host_weights);

    std::vector<float> expected_numerators(numerator_storage, sentinel);
    std::vector<float> expected_weights(weight_storage, sentinel);
    for (int channel = 0; channel < shape.channels; ++channel) {
        for (int slot = 0; slot < shape.slots; ++slot) {
            for (int y = 0; y < shape.height; ++y) {
                for (int x = 0; x < shape.width; ++x) {
                    expected_numerators
                        [static_cast<std::size_t>(channel) * numerator_channel +
                         static_cast<std::size_t>(slot) * numerator_slot +
                         static_cast<std::size_t>(y) * numerator_row + x] =
                            0.0F;
                    expected_weights[static_cast<std::size_t>(slot) *
                                         weight_slot +
                                     static_cast<std::size_t>(y) * weight_row +
                                     x] = 0.0F;
                }
            }
        }
    }
    if (host_numerators != expected_numerators ||
        host_weights != expected_weights) {
        throw std::runtime_error(
            "strided clear touched padding or missed data");
    }

    std::fill(host_numerators.begin(), host_numerators.end(), sentinel);
    std::fill(host_weights.begin(), host_weights.end(), sentinel);
    const std::size_t pixels = plane_values(shape);
    std::vector<float> expected_packed(aggregator.packed_values());
    for (int channel = 0; channel < shape.channels; ++channel) {
        for (int slot = 0; slot < shape.slots; ++slot) {
            for (int y = 0; y < shape.height; ++y) {
                for (int x = 0; x < shape.width; ++x) {
                    const float channel_value = static_cast<float>(channel);
                    const float slot_value = static_cast<float>(slot);
                    const float y_value = static_cast<float>(y);
                    const float x_value = static_cast<float>(x);
                    const float numerator = 1000.0F * channel_value +
                                            100.0F * slot_value +
                                            10.0F * y_value + x_value + 0.25F;
                    const float weight =
                        100.0F * slot_value + 10.0F * y_value + x_value + 0.5F;
                    host_numerators
                        [static_cast<std::size_t>(channel) * numerator_channel +
                         static_cast<std::size_t>(slot) * numerator_slot +
                         static_cast<std::size_t>(y) * numerator_row + x] =
                            numerator;
                    host_weights[static_cast<std::size_t>(slot) * weight_slot +
                                 static_cast<std::size_t>(y) * weight_row + x] =
                        weight;
                    const std::size_t pixel =
                        static_cast<std::size_t>(y) * shape.width + x;
                    const std::size_t plane =
                        static_cast<std::size_t>(channel * shape.slots + slot);
                    expected_packed[plane * 2 * pixels + pixel] = numerator;
                    expected_packed[plane * 2 * pixels + pixels + pixel] =
                        weight;
                }
            }
        }
    }
    device_numerators.upload(host_numerators);
    device_weights.upload(host_weights);
    DeviceArray<float> device_packed(expected_packed.size());
    aggregator.enqueue_pack(view, device_packed.data());
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize pack");
    std::vector<float> actual_packed(expected_packed.size());
    device_packed.download(actual_packed);
    if (actual_packed != expected_packed) {
        throw std::runtime_error("packed contribution layout mismatch");
    }

    constexpr int normalize_slot = 2;
    for (int y = 0; y < shape.height; ++y) {
        for (int x = 0; x < shape.width; ++x) {
            const bool covered = ((x + y) % 3) != 0;
            const float x_value = static_cast<float>(x);
            const float y_value = static_cast<float>(y);
            const float weight =
                covered ? 1.25F + 0.01F * (x_value + y_value) : 0.0F;
            host_weights[static_cast<std::size_t>(normalize_slot) *
                             weight_slot +
                         static_cast<std::size_t>(y) * weight_row + x] = weight;
            for (int channel = 0; channel < shape.channels; ++channel) {
                const float normalized = 20.0F * static_cast<float>(channel) +
                                         0.5F * y_value + 0.125F * x_value;
                host_numerators
                    [static_cast<std::size_t>(channel) * numerator_channel +
                     static_cast<std::size_t>(normalize_slot) * numerator_slot +
                     static_cast<std::size_t>(y) * numerator_row + x] =
                        normalized * weight;
            }
        }
    }
    device_numerators.upload(host_numerators);
    device_weights.upload(host_weights);

    const std::ptrdiff_t source_row = shape.width + 2;
    const std::ptrdiff_t source_channel = source_row * shape.height + 3;
    const std::ptrdiff_t source_frame_stride =
        source_channel * shape.channels + 5;
    constexpr int source_frames = 2;
    const std::size_t source_storage =
        static_cast<std::size_t>(source_frames) * source_frame_stride;
    std::vector<float> host_source(source_storage, sentinel);
    for (int frame = 0; frame < source_frames; ++frame) {
        for (int channel = 0; channel < shape.channels; ++channel) {
            for (int y = 0; y < shape.height; ++y) {
                for (int x = 0; x < shape.width; ++x) {
                    const float frame_value = static_cast<float>(frame);
                    const float channel_value = static_cast<float>(channel);
                    const float y_value = static_cast<float>(y);
                    const float x_value = static_cast<float>(x);
                    host_source[static_cast<std::size_t>(frame) *
                                    source_frame_stride +
                                static_cast<std::size_t>(channel) *
                                    source_channel +
                                static_cast<std::size_t>(y) * source_row + x] =
                        200.0F * frame_value + 50.0F * channel_value +
                        5.0F * y_value + x_value;
                }
            }
        }
    }
    DeviceArray<float> device_source(source_storage);
    device_source.upload(host_source);
    const vnlbcu::DeviceVideoView source{
        .data = device_source.data(),
        .width = shape.width,
        .height = shape.height,
        .channels = shape.channels,
        .frames = source_frames,
        .first_frame = 10,
        .source_frames = 12,
        .row_stride = source_row,
        .channel_stride = source_channel,
        .frame_stride = source_frame_stride,
    };
    constexpr int source_frame = 11;
    const std::ptrdiff_t output_row = shape.width + 4;
    const std::ptrdiff_t output_channel = output_row * shape.height + 2;
    const std::size_t output_storage =
        static_cast<std::size_t>(shape.channels - 1) * output_channel +
        static_cast<std::size_t>(shape.height - 1) * output_row + shape.width;
    std::vector<float> host_output(output_storage, sentinel);
    DeviceArray<float> device_output(output_storage);
    device_output.upload(host_output);
    aggregator.enqueue_normalize(view, normalize_slot, source, source_frame,
                                 vnlbcu::DeviceMutableFrameView{
                                     .data = device_output.data(),
                                     .width = shape.width,
                                     .height = shape.height,
                                     .channels = shape.channels,
                                     .row_stride = output_row,
                                     .channel_stride = output_channel,
                                 });
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize normalize");
    device_output.download(host_output);
    for (int channel = 0; channel < shape.channels; ++channel) {
        for (int y = 0; y < shape.height; ++y) {
            for (int x = 0; x < shape.width; ++x) {
                const bool covered = ((x + y) % 3) != 0;
                const float channel_value = static_cast<float>(channel);
                const float y_value = static_cast<float>(y);
                const float x_value = static_cast<float>(x);
                const float expected =
                    covered
                        ? 20.0F * channel_value + 0.5F * y_value +
                              0.125F * x_value
                        : host_source[source_frame_stride +
                                      static_cast<std::size_t>(channel) *
                                          source_channel +
                                      static_cast<std::size_t>(y) * source_row +
                                      x];
                const float actual =
                    host_output[static_cast<std::size_t>(channel) *
                                    output_channel +
                                static_cast<std::size_t>(y) * output_row + x];
                require_close(actual, expected, 1.0e-6F, "normalized output");
            }
        }
    }
}

void run_multi_source_normalize_case() {
    const vnlbcu::AggregationShape shape{
        .max_groups = 1,
        .width = 8,
        .height = 6,
        .channels = 3,
        .slots = 4,
        .retained_stride = 1,
        .patch_size = 3,
        .patch_time = 2,
        .search_window = 3,
        .search_bwd = 1,
    };
    constexpr float sentinel = 9876.0F;
    const std::ptrdiff_t numerator_row_a = shape.width + 2;
    const std::ptrdiff_t numerator_slot_a = numerator_row_a * shape.height + 3;
    const std::ptrdiff_t numerator_channel_a =
        numerator_slot_a * shape.slots + 5;
    const std::ptrdiff_t weight_row_a = shape.width + 1;
    const std::ptrdiff_t weight_slot_a = weight_row_a * shape.height + 2;
    const std::size_t numerator_storage_a = strided_storage_size(
        shape.channels, numerator_channel_a, shape.slots, numerator_slot_a,
        shape.height, numerator_row_a, shape.width);
    const std::size_t weight_storage_a =
        strided_storage_size(1, 0, shape.slots, weight_slot_a, shape.height,
                             weight_row_a, shape.width);
    std::vector<float> host_numerators_a(numerator_storage_a, sentinel);
    std::vector<float> host_weights_a(weight_storage_a, sentinel);

    const std::size_t pixels = plane_values(shape);
    std::vector<float> host_numerators_b(
        static_cast<std::size_t>(shape.channels) * shape.slots * pixels,
        sentinel);
    std::vector<float> host_weights_b(
        static_cast<std::size_t>(shape.slots) * pixels, sentinel);
    constexpr int slot_a0 = 0;
    constexpr int slot_a1 = 2;
    constexpr int slot_b = 1;

    std::vector<float> expected(static_cast<std::size_t>(shape.channels) *
                                pixels);
    std::vector<float> expected_fallback(expected.size());
    for (int y = 0; y < shape.height; ++y) {
        for (int x = 0; x < shape.width; ++x) {
            const int mode = (x + 2 * y) % 5;
            const float weights[3] = {
                mode == 0 || mode == 2 ? 0.0F
                                       : 0.75F + 0.01F * static_cast<float>(x),
                mode == 0 || mode == 1 ? 0.0F
                                       : 1.25F + 0.02F * static_cast<float>(y),
                mode == 0 || mode == 3
                    ? 0.0F
                    : 0.5F + 0.01F * static_cast<float>(x + y),
            };
            host_weights_a[static_cast<std::size_t>(slot_a0) * weight_slot_a +
                           static_cast<std::size_t>(y) * weight_row_a + x] =
                weights[0];
            host_weights_a[static_cast<std::size_t>(slot_a1) * weight_slot_a +
                           static_cast<std::size_t>(y) * weight_row_a + x] =
                weights[1];
            host_weights_b[static_cast<std::size_t>(slot_b) * pixels +
                           static_cast<std::size_t>(y) * shape.width + x] =
                weights[2];

            const float weight_sum = weights[0] + weights[1] + weights[2];
            for (int channel = 0; channel < shape.channels; ++channel) {
                const float values[3] = {
                    10.0F * static_cast<float>(channel) +
                        0.5F * static_cast<float>(y) +
                        0.125F * static_cast<float>(x),
                    30.0F + 7.0F * static_cast<float>(channel) +
                        0.25F * static_cast<float>(y) +
                        0.0625F * static_cast<float>(x),
                    -5.0F + 3.0F * static_cast<float>(channel) +
                        0.75F * static_cast<float>(y) -
                        0.03125F * static_cast<float>(x),
                };
                const float numerator_a0 = weights[0] == 0.0F
                                               ? 100000.0F + values[0]
                                               : weights[0] * values[0];
                const float numerator_a1 = weights[1] == 0.0F
                                               ? 200000.0F + values[1]
                                               : weights[1] * values[1];
                const float numerator_b = weights[2] == 0.0F
                                              ? 300000.0F + values[2]
                                              : weights[2] * values[2];
                host_numerators_a
                    [static_cast<std::size_t>(channel) * numerator_channel_a +
                     static_cast<std::size_t>(slot_a0) * numerator_slot_a +
                     static_cast<std::size_t>(y) * numerator_row_a + x] =
                        numerator_a0;
                host_numerators_a
                    [static_cast<std::size_t>(channel) * numerator_channel_a +
                     static_cast<std::size_t>(slot_a1) * numerator_slot_a +
                     static_cast<std::size_t>(y) * numerator_row_a + x] =
                        numerator_a1;
                host_numerators_b[(static_cast<std::size_t>(channel) *
                                       shape.slots +
                                   slot_b) *
                                      pixels +
                                  static_cast<std::size_t>(y) * shape.width +
                                  x] = numerator_b;
                const std::size_t output_index =
                    static_cast<std::size_t>(channel) * pixels +
                    static_cast<std::size_t>(y) * shape.width + x;
                if (weight_sum != 0.0F) {
                    expected[output_index] =
                        (weights[0] * values[0] + weights[1] * values[1] +
                         weights[2] * values[2]) /
                        weight_sum;
                }
            }
        }
    }

    DeviceArray<float> device_numerators_a(host_numerators_a.size());
    DeviceArray<float> device_weights_a(host_weights_a.size());
    DeviceArray<float> device_numerators_b(host_numerators_b.size());
    DeviceArray<float> device_weights_b(host_weights_b.size());
    device_numerators_a.upload(host_numerators_a);
    device_weights_a.upload(host_weights_a);
    device_numerators_b.upload(host_numerators_b);
    device_weights_b.upload(host_weights_b);
    const vnlbcu::DeviceContributionView view_a{
        .numerators = device_numerators_a.data(),
        .weights = device_weights_a.data(),
        .width = shape.width,
        .height = shape.height,
        .channels = shape.channels,
        .slots = shape.slots,
        .numerator_row_stride = numerator_row_a,
        .numerator_slot_stride = numerator_slot_a,
        .numerator_channel_stride = numerator_channel_a,
        .weight_row_stride = weight_row_a,
        .weight_slot_stride = weight_slot_a,
    };
    const vnlbcu::DeviceContributionView view_b =
        vnlbcu::make_contiguous_contribution_view(
            shape, device_numerators_b.data(), device_weights_b.data());
    const std::vector<vnlbcu::DeviceContributionSource> sources{
        vnlbcu::make_contribution_source(view_a, slot_a0),
        vnlbcu::make_contribution_source(view_a, slot_a1),
        vnlbcu::make_contribution_source(view_b, slot_b),
    };
    DeviceArray<vnlbcu::DeviceContributionSource> device_sources(
        sources.size());
    device_sources.upload(sources);

    const std::ptrdiff_t source_row = shape.width + 3;
    const std::ptrdiff_t source_channel = source_row * shape.height + 2;
    const std::ptrdiff_t source_frame_stride =
        source_channel * shape.channels + 4;
    constexpr int source_frames = 2;
    constexpr int first_source_frame = 20;
    constexpr int source_frame = 21;
    std::vector<float> host_source(static_cast<std::size_t>(source_frames) *
                                       source_frame_stride,
                                   sentinel);
    for (int frame = 0; frame < source_frames; ++frame) {
        for (int channel = 0; channel < shape.channels; ++channel) {
            for (int y = 0; y < shape.height; ++y) {
                for (int x = 0; x < shape.width; ++x) {
                    const float value = 400.0F * static_cast<float>(frame) +
                                        40.0F * static_cast<float>(channel) +
                                        4.0F * static_cast<float>(y) +
                                        static_cast<float>(x);
                    host_source[static_cast<std::size_t>(frame) *
                                    source_frame_stride +
                                static_cast<std::size_t>(channel) *
                                    source_channel +
                                static_cast<std::size_t>(y) * source_row + x] =
                        value;
                    const std::size_t output_index =
                        static_cast<std::size_t>(channel) * pixels +
                        static_cast<std::size_t>(y) * shape.width + x;
                    expected_fallback[output_index] =
                        frame == 1 ? value : expected_fallback[output_index];
                    if ((x + 2 * y) % 5 == 0 && frame == 1) {
                        expected[output_index] = value;
                    }
                }
            }
        }
    }
    DeviceArray<float> device_source(host_source.size());
    device_source.upload(host_source);
    const std::vector<const float*> source_frame_data{
        device_source.data(), device_source.data() + source_frame_stride};
    DeviceArray<const float*> device_source_frame_data(
        source_frame_data.size());
    device_source_frame_data.upload(source_frame_data);
    const vnlbcu::DeviceVideoView fallback_source{
        .data = nullptr,
        .frame_data = device_source_frame_data.data(),
        .width = shape.width,
        .height = shape.height,
        .channels = shape.channels,
        .frames = source_frames,
        .first_frame = first_source_frame,
        .source_frames = first_source_frame + source_frames,
        .row_stride = source_row,
        .channel_stride = source_channel,
        .frame_stride = 0,
    };

    const std::ptrdiff_t output_row = shape.width + 4;
    const std::ptrdiff_t output_channel = output_row * shape.height + 3;
    const std::size_t output_storage =
        static_cast<std::size_t>(shape.channels - 1) * output_channel +
        static_cast<std::size_t>(shape.height - 1) * output_row + shape.width;
    std::vector<float> host_output(output_storage, sentinel);
    DeviceArray<float> device_output(host_output.size());
    device_output.upload(host_output);
    const vnlbcu::DeviceMutableFrameView output{
        .data = device_output.data(),
        .width = shape.width,
        .height = shape.height,
        .channels = shape.channels,
        .row_stride = output_row,
        .channel_stride = output_channel,
    };
    vnlbcu::Aggregator aggregator;
    aggregator.reserve(shape, 0.0F);
    aggregator.enqueue_normalize_many(device_sources.data(),
                                      static_cast<int>(sources.size()),
                                      fallback_source, source_frame, output);
    check_cuda(cudaDeviceSynchronize(),
               "cudaDeviceSynchronize multi-source normalize");
    device_output.download(host_output);
    for (int channel = 0; channel < shape.channels; ++channel) {
        for (int y = 0; y < shape.height; ++y) {
            for (int x = 0; x < shape.width; ++x) {
                const std::size_t packed_index =
                    static_cast<std::size_t>(channel) * pixels +
                    static_cast<std::size_t>(y) * shape.width + x;
                const std::size_t strided_index =
                    static_cast<std::size_t>(channel) * output_channel +
                    static_cast<std::size_t>(y) * output_row + x;
                require_close(host_output[strided_index],
                              expected[packed_index], 2.0e-6F,
                              "multi-source normalized output");
            }
        }
    }

    std::fill(host_output.begin(), host_output.end(), sentinel);
    device_output.upload(host_output);
    aggregator.enqueue_normalize_many(nullptr, 0, fallback_source, source_frame,
                                      output);
    check_cuda(cudaDeviceSynchronize(),
               "cudaDeviceSynchronize zero-source normalize");
    device_output.download(host_output);
    for (int channel = 0; channel < shape.channels; ++channel) {
        for (int y = 0; y < shape.height; ++y) {
            for (int x = 0; x < shape.width; ++x) {
                const std::size_t packed_index =
                    static_cast<std::size_t>(channel) * pixels +
                    static_cast<std::size_t>(y) * shape.width + x;
                const std::size_t strided_index =
                    static_cast<std::size_t>(channel) * output_channel +
                    static_cast<std::size_t>(y) * output_row + x;
                require_close(host_output[strided_index],
                              expected_fallback[packed_index], 0.0F,
                              "zero-source fallback output");
            }
        }
    }
}

} // namespace

int main() {
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        device_count == 0) {
        std::cout
            << "CUDA device unavailable; skipping vnlbcu aggregation test\n";
        return skipped;
    }
    check_cuda(status, "cudaGetDeviceCount");

    try {
        run_scatter_reference_case();
        run_strided_clear_pack_normalize_case();
        run_multi_source_normalize_case();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
