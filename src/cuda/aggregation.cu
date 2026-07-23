#include "cuda/aggregation.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace vnlbcu {
namespace {

constexpr int warp_width = 32;
constexpr int block_threads = 256;
constexpr int warps_per_block = block_threads / warp_width;
constexpr float minimum_log_weight = -80.0F;
constexpr float maximum_log_weight = 80.0F;

struct PatchWindowEntry {
    float weight;
    int row;
};

[[noreturn]] void throw_cuda(cudaError_t status, const char* operation) {
    std::ostringstream message;
    message << operation << " failed: " << cudaGetErrorString(status);
    throw std::runtime_error(message.str());
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw_cuda(status, operation);
    }
}

int checked_add_int(int left, int right, const char* label) {
    const long long result = static_cast<long long>(left) + right;
    if (result < std::numeric_limits<int>::min() ||
        result > std::numeric_limits<int>::max()) {
        throw std::length_error(label);
    }
    return static_cast<int>(result);
}

int checked_product_int(int left, int right, const char* label) {
    const long long result = static_cast<long long>(left) * right;
    if (result < 0 || result > std::numeric_limits<int>::max()) {
        throw std::length_error(label);
    }
    return static_cast<int>(result);
}

std::size_t checked_product(std::size_t left, std::size_t right,
                            const char* label) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::length_error(label);
    }
    return left * right;
}

std::size_t plane_values(const AggregationShape& shape) {
    return checked_product(static_cast<std::size_t>(shape.width),
                           static_cast<std::size_t>(shape.height),
                           "CUDA aggregation plane size overflows");
}

std::size_t numerator_value_count(const AggregationShape& shape) {
    return checked_product(
        checked_product(static_cast<std::size_t>(shape.channels),
                        static_cast<std::size_t>(shape.slots),
                        "CUDA numerator plane count overflows"),
        plane_values(shape), "CUDA numerator size overflows");
}

std::size_t weight_value_count(const AggregationShape& shape) {
    return checked_product(static_cast<std::size_t>(shape.slots),
                           plane_values(shape),
                           "CUDA aggregation weight size overflows");
}

std::size_t packed_value_count(const AggregationShape& shape) {
    return checked_product(numerator_value_count(shape), std::size_t{2},
                           "CUDA packed contribution size overflows");
}

int sample_dimension(const AggregationShape& shape) {
    const int patch_area = checked_product_int(
        shape.patch_size, shape.patch_size, "CUDA patch area overflows");
    const int temporal = checked_product_int(patch_area, shape.patch_time,
                                             "CUDA patch dimension overflows");
    return checked_product_int(temporal, shape.channels,
                               "CUDA sample dimension overflows");
}

void validate_shape(const AggregationShape& shape) {
    if (shape.max_groups <= 0 || shape.width <= 0 || shape.height <= 0 ||
        shape.channels <= 0 || shape.slots <= 0 || shape.retained_stride <= 0) {
        throw std::invalid_argument(
            "CUDA aggregation geometry must be positive");
    }
    if (shape.patch_size <= 0 || shape.patch_size > shape.width ||
        shape.patch_size > shape.height || shape.patch_time <= 0) {
        throw std::invalid_argument("invalid CUDA aggregation patch geometry");
    }
    if (shape.search_window <= 0 || (shape.search_window & 1) == 0 ||
        shape.search_bwd < 0 || shape.search_bwd >= shape.slots) {
        throw std::invalid_argument(
            "invalid CUDA aggregation search/slot geometry");
    }
    if (shape.slots <
        checked_add_int(shape.search_bwd, shape.patch_time,
                        "CUDA aggregation slot count overflows")) {
        throw std::invalid_argument(
            "CUDA aggregation has too few temporal contribution slots");
    }
    (void)sample_dimension(shape);
    (void)numerator_value_count(shape);
    (void)weight_value_count(shape);
    (void)packed_value_count(shape);
}

bool same_reserved_geometry(const AggregationShape& requested,
                            const AggregationShape& reserved) noexcept {
    return requested.width == reserved.width &&
           requested.height == reserved.height &&
           requested.channels == reserved.channels &&
           requested.slots == reserved.slots &&
           requested.retained_stride == reserved.retained_stride &&
           requested.patch_size == reserved.patch_size &&
           requested.patch_time == reserved.patch_time &&
           requested.search_window == reserved.search_window &&
           requested.search_bwd == reserved.search_bwd &&
           requested.max_groups <= reserved.max_groups;
}

void validate_contributions(const DeviceContributionView& view,
                            const AggregationShape& shape) {
    if (view.numerators == nullptr || view.weights == nullptr) {
        throw std::invalid_argument(
            "CUDA aggregation requires numerator and weight storage");
    }
    if (view.width != shape.width || view.height != shape.height ||
        view.channels != shape.channels || view.slots != shape.slots) {
        throw std::invalid_argument(
            "CUDA contribution view shape does not match aggregation shape");
    }
    if (view.numerator_row_stride < shape.width ||
        view.weight_row_stride < shape.width ||
        view.numerator_slot_stride <= 0 || view.numerator_channel_stride <= 0 ||
        view.weight_slot_stride <= 0) {
        throw std::invalid_argument("invalid CUDA contribution strides");
    }

    const std::size_t numerator_plane =
        checked_product(static_cast<std::size_t>(view.numerator_row_stride),
                        static_cast<std::size_t>(shape.height),
                        "CUDA numerator stride overflows");
    const std::size_t numerator_channel =
        checked_product(static_cast<std::size_t>(view.numerator_slot_stride),
                        static_cast<std::size_t>(shape.slots),
                        "CUDA numerator channel stride overflows");
    const std::size_t weight_plane = checked_product(
        static_cast<std::size_t>(view.weight_row_stride),
        static_cast<std::size_t>(shape.height), "CUDA weight stride overflows");
    if (static_cast<std::size_t>(view.numerator_slot_stride) <
            numerator_plane ||
        static_cast<std::size_t>(view.numerator_channel_stride) <
            numerator_channel ||
        static_cast<std::size_t>(view.weight_slot_stride) < weight_plane) {
        throw std::invalid_argument("CUDA contribution strides overlap");
    }
}

void validate_parameters(const AggregationParameters& parameters,
                         const DeviceAggregationBatch& batch,
                         const AggregationShape& shape) {
    if (parameters.anchor_frame < 0) {
        throw std::invalid_argument(
            "CUDA aggregation anchor frame must be non-negative");
    }
    if (batch.filtered_samples == nullptr || batch.matches == nullptr ||
        batch.groups <= 0 || batch.groups > shape.max_groups) {
        throw std::invalid_argument("invalid CUDA aggregation batch");
    }
    if (batch.log_patch_weights != nullptr) {
        if (parameters.log_weight_model_count <= 0) {
            throw std::invalid_argument(
                "CUDA log-weight model count must be positive");
        }
        if (parameters.log_weight_model_count > 1) {
            const std::size_t minimum_stride =
                checked_product(static_cast<std::size_t>(batch.groups),
                                static_cast<std::size_t>(shape.retained_stride),
                                "CUDA log-weight model stride overflows");
            if (parameters.log_weight_model_stride <= 0 ||
                static_cast<std::size_t>(parameters.log_weight_model_stride) <
                    minimum_stride) {
                throw std::invalid_argument(
                    "CUDA log-weight model planes overlap");
            }
        }
    }
    validate_contributions(batch.contributions, shape);
}

void validate_source(DeviceVideoView source, int source_frame,
                     const AggregationShape& shape) {
    const long long source_end = static_cast<long long>(source.first_frame) +
                                 static_cast<long long>(source.frames);
    if ((source.data == nullptr && source.frame_data == nullptr) ||
        source.width != shape.width || source.height != shape.height ||
        source.channels != shape.channels || source.frames <= 0 ||
        source_frame < source.first_frame ||
        static_cast<long long>(source_frame) >= source_end) {
        throw std::invalid_argument(
            "CUDA normalization fallback source is invalid");
    }
    if (source.row_stride < shape.width || source.channel_stride <= 0 ||
        (source.frame_data == nullptr && source.frame_stride <= 0)) {
        throw std::invalid_argument(
            "CUDA normalization source strides are invalid");
    }
    const std::size_t plane =
        checked_product(static_cast<std::size_t>(source.row_stride),
                        static_cast<std::size_t>(shape.height),
                        "CUDA normalization source plane overflows");
    const std::size_t frame =
        checked_product(static_cast<std::size_t>(source.channel_stride),
                        static_cast<std::size_t>(shape.channels),
                        "CUDA normalization source frame overflows");
    if (static_cast<std::size_t>(source.channel_stride) < plane ||
        (source.frame_data == nullptr &&
         static_cast<std::size_t>(source.frame_stride) < frame)) {
        throw std::invalid_argument(
            "CUDA normalization source strides overlap");
    }
}

void validate_output(DeviceMutableFrameView output,
                     const AggregationShape& shape) {
    if (output.data == nullptr || output.width != shape.width ||
        output.height != shape.height || output.channels != shape.channels ||
        output.row_stride < shape.width || output.channel_stride <= 0) {
        throw std::invalid_argument(
            "CUDA normalization output view is invalid");
    }
    const std::size_t plane =
        checked_product(static_cast<std::size_t>(output.row_stride),
                        static_cast<std::size_t>(shape.height),
                        "CUDA normalization output plane overflows");
    if (static_cast<std::size_t>(output.channel_stride) < plane) {
        throw std::invalid_argument("CUDA normalization output planes overlap");
    }
}

bool contiguous_contributions(const DeviceContributionView& view,
                              const AggregationShape& shape) noexcept {
    const std::ptrdiff_t plane =
        static_cast<std::ptrdiff_t>(shape.width) * shape.height;
    return view.numerator_row_stride == shape.width &&
           view.numerator_slot_stride == plane &&
           view.numerator_channel_stride ==
               plane * static_cast<std::ptrdiff_t>(shape.slots) &&
           view.weight_row_stride == shape.width &&
           view.weight_slot_stride == plane;
}

int launch_blocks(std::size_t values) {
    const std::size_t blocks =
        (values + static_cast<std::size_t>(block_threads) - 1) /
        static_cast<std::size_t>(block_threads);
    return static_cast<int>(std::min<std::size_t>(blocks, 65535));
}

template <typename T> class DeviceBuffer {
  public:
    DeviceBuffer() = default;
    ~DeviceBuffer() { reset_noexcept(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          capacity_(std::exchange(other.capacity_, 0)) {}

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            reset_noexcept();
            data_ = std::exchange(other.data_, nullptr);
            capacity_ = std::exchange(other.capacity_, 0);
        }
        return *this;
    }

    void reserve(std::size_t count) {
        if (count <= capacity_) {
            return;
        }
        T* replacement = nullptr;
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&replacement),
                              checked_product(count, sizeof(T),
                                              "CUDA allocation overflows")),
                   "cudaMalloc");
        if (data_ != nullptr) {
            const cudaError_t status = cudaFree(data_);
            if (status != cudaSuccess) {
                (void)cudaFree(replacement);
                throw_cuda(status, "cudaFree");
            }
        }
        data_ = replacement;
        capacity_ = count;
    }

    [[nodiscard]] T* data() noexcept { return data_; }
    [[nodiscard]] const T* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t bytes() const noexcept {
        return capacity_ * sizeof(T);
    }

  private:
    void reset_noexcept() noexcept {
        if (data_ != nullptr) {
            (void)cudaFree(data_);
        }
        data_ = nullptr;
        capacity_ = 0;
    }

    T* data_ = nullptr;
    std::size_t capacity_ = 0;
};

__device__ __forceinline__ int retained_count(const int* counts, int group,
                                              int retained_stride) {
    if (counts == nullptr) {
        return retained_stride;
    }
    return max(0, min(counts[group], retained_stride));
}

__device__ __forceinline__ float patch_weight(const float* log_weights,
                                              int group, int sample,
                                              int retained_stride,
                                              int model_count,
                                              std::ptrdiff_t model_stride) {
    if (log_weights == nullptr) {
        return 1.0F;
    }

    const std::size_t sample_index =
        static_cast<std::size_t>(group) * retained_stride + sample;
    float log_sum = 0.0F;
    for (int model = 0; model < model_count; ++model) {
        const std::ptrdiff_t model_offset =
            model_count == 1
                ? 0
                : static_cast<std::ptrdiff_t>(model) * model_stride;
        log_sum += log_weights[model_offset +
                               static_cast<std::ptrdiff_t>(sample_index)];
    }
    float averaged = log_sum / static_cast<float>(model_count);
    if (averaged < minimum_log_weight) {
        averaged = minimum_log_weight;
    } else if (averaged > maximum_log_weight) {
        averaged = maximum_log_weight;
    }
    return expf(averaged);
}

__device__ __forceinline__ std::ptrdiff_t
numerator_offset(const DeviceContributionView& view, int channel, int slot,
                 int y, int x) {
    return static_cast<std::ptrdiff_t>(channel) *
               view.numerator_channel_stride +
           static_cast<std::ptrdiff_t>(slot) * view.numerator_slot_stride +
           static_cast<std::ptrdiff_t>(y) * view.numerator_row_stride + x;
}

__device__ __forceinline__ std::ptrdiff_t
weight_offset(const DeviceContributionView& view, int slot, int y, int x) {
    return static_cast<std::ptrdiff_t>(slot) * view.weight_slot_stride +
           static_cast<std::ptrdiff_t>(y) * view.weight_row_stride + x;
}

__global__ void
scatter_direct_kernel(AggregationShape shape, AggregationParameters parameters,
                      DeviceAggregationBatch batch,
                      const PatchWindowEntry* __restrict__ window) {
    const int group = static_cast<int>(blockIdx.x);
    const int thread = static_cast<int>(threadIdx.x);
    if (group >= batch.groups) {
        return;
    }
    if (batch.active_groups != nullptr && batch.active_groups[group] == 0U) {
        return;
    }

    const int lane = thread & (warp_width - 1);
    const int warp = thread / warp_width;
    const int patch_area = shape.patch_size * shape.patch_size;
    const int channel_patch_dim = shape.patch_time * patch_area;
    const int sample_dim = shape.channels * channel_patch_dim;
    const int similar =
        retained_count(batch.retained_counts, group, shape.retained_stride);
    const std::size_t descriptor_base =
        static_cast<std::size_t>(group) * shape.retained_stride;
    const std::size_t sample_group_base =
        descriptor_base * static_cast<std::size_t>(sample_dim);

    for (int sample = warp; sample < similar; sample += warps_per_block) {
        int match_x = 0;
        int match_y = 0;
        int match_frame = 0;
        float sample_weight = 0.0F;
        if (lane == 0) {
            const PatchMatch match = batch.matches[descriptor_base + sample];
            match_x = match.x;
            match_y = match.y;
            match_frame = match.frame;
            sample_weight = patch_weight(batch.log_patch_weights, group, sample,
                                         shape.retained_stride,
                                         parameters.log_weight_model_count,
                                         parameters.log_weight_model_stride);
        }
        match_x = __shfl_sync(0xffffffffU, match_x, 0);
        match_y = __shfl_sync(0xffffffffU, match_y, 0);
        match_frame = __shfl_sync(0xffffffffU, match_frame, 0);
        sample_weight = __shfl_sync(0xffffffffU, sample_weight, 0);
        const std::size_t sample_base =
            sample_group_base + static_cast<std::size_t>(sample) * sample_dim;

        for (int frame_delta = 0; frame_delta < shape.patch_time;
             ++frame_delta) {
            const int output_offset =
                match_frame + frame_delta - parameters.anchor_frame;
            const int slot = output_offset + shape.search_bwd;
            if (slot < 0 || slot >= shape.slots) {
                continue;
            }
            for (int position = lane; position < patch_area;
                 position += warp_width) {
                const PatchWindowEntry window_entry = window[position];
                const int patch_y = window_entry.row;
                const int patch_x = position - patch_y * shape.patch_size;
                const int output_x = match_x + patch_x;
                const int output_y = match_y + patch_y;
                if (output_x < 0 || output_x >= shape.width || output_y < 0 ||
                    output_y >= shape.height) {
                    continue;
                }
                const float contribution_weight =
                    sample_weight * window_entry.weight;
                atomicAdd(batch.contributions.weights +
                              weight_offset(batch.contributions, slot, output_y,
                                            output_x),
                          contribution_weight);
                for (int channel = 0; channel < shape.channels; ++channel) {
                    const std::size_t sample_index =
                        sample_base +
                        static_cast<std::size_t>(channel) * channel_patch_dim +
                        static_cast<std::size_t>(frame_delta) * patch_area +
                        position;
                    atomicAdd(batch.contributions.numerators +
                                  numerator_offset(batch.contributions, channel,
                                                   slot, output_y, output_x),
                              contribution_weight *
                                  batch.filtered_samples[sample_index]);
                }
            }
        }
    }
}

__global__ void clear_strided_kernel(DeviceContributionView view,
                                     std::size_t numerator_values,
                                     std::size_t weight_values) {
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    const std::size_t pixels =
        static_cast<std::size_t>(view.width) * view.height;
    for (std::size_t index = thread; index < numerator_values;
         index += stride) {
        const std::size_t pixel = index % pixels;
        const std::size_t plane = index / pixels;
        const int slot = static_cast<int>(plane % view.slots);
        const int channel = static_cast<int>(plane / view.slots);
        const int y = static_cast<int>(pixel / view.width);
        const int x = static_cast<int>(pixel % view.width);
        view.numerators[numerator_offset(view, channel, slot, y, x)] = 0.0F;
    }
    for (std::size_t index = thread; index < weight_values; index += stride) {
        const std::size_t pixel = index % pixels;
        const int slot = static_cast<int>(index / pixels);
        const int y = static_cast<int>(pixel / view.width);
        const int x = static_cast<int>(pixel % view.width);
        view.weights[weight_offset(view, slot, y, x)] = 0.0F;
    }
}

__global__ void pack_kernel(DeviceContributionView view, float* packed) {
    const std::size_t pixels =
        static_cast<std::size_t>(view.width) * view.height;
    const std::size_t values =
        static_cast<std::size_t>(view.channels) * view.slots * pixels;
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = thread; index < values; index += stride) {
        const std::size_t pixel = index % pixels;
        const std::size_t plane = index / pixels;
        const int slot = static_cast<int>(plane % view.slots);
        const int channel = static_cast<int>(plane / view.slots);
        const int y = static_cast<int>(pixel / view.width);
        const int x = static_cast<int>(pixel % view.width);
        const std::size_t output = plane * 2 * pixels + pixel;
        packed[output] =
            view.numerators[numerator_offset(view, channel, slot, y, x)];
        packed[output + pixels] = view.weights[weight_offset(view, slot, y, x)];
    }
}

__device__ __forceinline__ const float*
video_frame_base(const DeviceVideoView& view, int frame) {
    const std::ptrdiff_t local_frame = frame - view.first_frame;
    return view.frame_data != nullptr
               ? view.frame_data[local_frame]
               : view.data + local_frame * view.frame_stride;
}

__global__ void normalize_kernel(DeviceContributionView contributions, int slot,
                                 DeviceVideoView source, int source_frame,
                                 DeviceMutableFrameView output) {
    const std::size_t pixels =
        static_cast<std::size_t>(output.width) * output.height;
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    const float* const source_data = video_frame_base(source, source_frame);
    for (std::size_t pixel = thread; pixel < pixels; pixel += stride) {
        const int y = static_cast<int>(pixel / output.width);
        const int x = static_cast<int>(pixel % output.width);
        const float weight =
            contributions.weights[weight_offset(contributions, slot, y, x)];
        const bool has_weight = weight > 0.0F;
        const float inverse_weight = has_weight ? 1.0F / weight : 0.0F;
        for (int channel = 0; channel < output.channels; ++channel) {
            float value = 0.0F;
            if (has_weight) {
                value = contributions.numerators[numerator_offset(
                            contributions, channel, slot, y, x)] *
                        inverse_weight;
            } else {
                const std::ptrdiff_t source_index =
                    static_cast<std::ptrdiff_t>(channel) *
                        source.channel_stride +
                    static_cast<std::ptrdiff_t>(y) * source.row_stride + x;
                value = source_data[source_index];
            }
            output
                .data[static_cast<std::ptrdiff_t>(channel) *
                          output.channel_stride +
                      static_cast<std::ptrdiff_t>(y) * output.row_stride + x] =
                value;
        }
    }
}

__device__ __forceinline__ std::ptrdiff_t
source_numerator_offset(const DeviceContributionSource& source, int channel,
                        int y, int x) {
    return static_cast<std::ptrdiff_t>(channel) *
               source.numerator_channel_stride +
           static_cast<std::ptrdiff_t>(y) * source.numerator_row_stride + x;
}

__device__ __forceinline__ std::ptrdiff_t
source_weight_offset(const DeviceContributionSource& source, int y, int x) {
    return static_cast<std::ptrdiff_t>(y) * source.weight_row_stride + x;
}

template <int Channels>
__global__ void normalize_many_small_channels_kernel(
    const DeviceContributionSource* __restrict__ sources, int source_count,
    DeviceVideoView fallback_source, int source_frame,
    DeviceMutableFrameView output) {
    const std::size_t pixels =
        static_cast<std::size_t>(output.width) * output.height;
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    const float* const fallback_data =
        video_frame_base(fallback_source, source_frame);
    for (std::size_t pixel = thread; pixel < pixels; pixel += stride) {
        const int y = static_cast<int>(pixel / output.width);
        const int x = static_cast<int>(pixel % output.width);
        float numerator[Channels]{};
        float weight_sum = 0.0F;
        for (int source_index = 0; source_index < source_count;
             ++source_index) {
            const DeviceContributionSource source = sources[source_index];
            const float weight =
                source.weights[source_weight_offset(source, y, x)];
            if (weight == 0.0F) {
                continue;
            }
            weight_sum += weight;
#pragma unroll
            for (int channel = 0; channel < Channels; ++channel) {
                numerator[channel] += source.numerators[source_numerator_offset(
                    source, channel, y, x)];
            }
        }

        const bool has_weight = weight_sum > 0.0F;
        const float inverse_weight = has_weight ? 1.0F / weight_sum : 0.0F;
#pragma unroll
        for (int channel = 0; channel < Channels; ++channel) {
            float value = 0.0F;
            if (has_weight) {
                value = numerator[channel] * inverse_weight;
            } else {
                const std::ptrdiff_t source_index =
                    static_cast<std::ptrdiff_t>(channel) *
                        fallback_source.channel_stride +
                    static_cast<std::ptrdiff_t>(y) *
                        fallback_source.row_stride +
                    x;
                value = fallback_data[source_index];
            }
            output
                .data[static_cast<std::ptrdiff_t>(channel) *
                          output.channel_stride +
                      static_cast<std::ptrdiff_t>(y) * output.row_stride + x] =
                value;
        }
    }
}

__global__ void normalize_many_channels_kernel(
    const DeviceContributionSource* __restrict__ sources, int source_count,
    DeviceVideoView fallback_source, int source_frame,
    DeviceMutableFrameView output) {
    const std::size_t pixels =
        static_cast<std::size_t>(output.width) * output.height;
    const std::size_t values =
        static_cast<std::size_t>(output.channels) * pixels;
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    const float* const fallback_data =
        video_frame_base(fallback_source, source_frame);
    for (std::size_t index = thread; index < values; index += stride) {
        const int channel = static_cast<int>(index / pixels);
        const std::size_t pixel = index % pixels;
        const int y = static_cast<int>(pixel / output.width);
        const int x = static_cast<int>(pixel % output.width);
        float numerator = 0.0F;
        float weight_sum = 0.0F;
        for (int source_index = 0; source_index < source_count;
             ++source_index) {
            const DeviceContributionSource source = sources[source_index];
            const float weight =
                source.weights[source_weight_offset(source, y, x)];
            if (weight == 0.0F) {
                continue;
            }
            weight_sum += weight;
            numerator +=
                source
                    .numerators[source_numerator_offset(source, channel, y, x)];
        }

        float value = 0.0F;
        if (weight_sum > 0.0F) {
            value = numerator / weight_sum;
        } else {
            const std::ptrdiff_t source_index =
                static_cast<std::ptrdiff_t>(channel) *
                    fallback_source.channel_stride +
                static_cast<std::ptrdiff_t>(y) * fallback_source.row_stride + x;
            value = fallback_data[source_index];
        }
        output
            .data[static_cast<std::ptrdiff_t>(channel) * output.channel_stride +
                  static_cast<std::ptrdiff_t>(y) * output.row_stride + x] =
            value;
    }
}

} // namespace

DeviceContributionView
make_contiguous_contribution_view(const AggregationShape& shape,
                                  float* numerators, float* weights) {
    validate_shape(shape);
    const std::ptrdiff_t plane =
        static_cast<std::ptrdiff_t>(shape.width) * shape.height;
    return DeviceContributionView{
        .numerators = numerators,
        .weights = weights,
        .width = shape.width,
        .height = shape.height,
        .channels = shape.channels,
        .slots = shape.slots,
        .numerator_row_stride = shape.width,
        .numerator_slot_stride = plane,
        .numerator_channel_stride =
            plane * static_cast<std::ptrdiff_t>(shape.slots),
        .weight_row_stride = shape.width,
        .weight_slot_stride = plane,
    };
}

DeviceContributionSource
make_contribution_source(const DeviceContributionView& contributions,
                         int slot) {
    if (contributions.numerators == nullptr ||
        contributions.weights == nullptr) {
        throw std::invalid_argument(
            "CUDA contribution source requires numerator and weight storage");
    }
    if (slot < 0 || slot >= contributions.slots) {
        throw std::invalid_argument(
            "CUDA contribution source slot is out of range");
    }
    if (contributions.numerator_row_stride <= 0 ||
        contributions.numerator_slot_stride <= 0 ||
        contributions.numerator_channel_stride <= 0 ||
        contributions.weight_row_stride <= 0 ||
        contributions.weight_slot_stride <= 0) {
        throw std::invalid_argument("invalid CUDA contribution source strides");
    }
    return DeviceContributionSource{
        .numerators =
            contributions.numerators + static_cast<std::ptrdiff_t>(slot) *
                                           contributions.numerator_slot_stride,
        .weights = contributions.weights + static_cast<std::ptrdiff_t>(slot) *
                                               contributions.weight_slot_stride,
        .numerator_row_stride = contributions.numerator_row_stride,
        .numerator_channel_stride = contributions.numerator_channel_stride,
        .weight_row_stride = contributions.weight_row_stride,
    };
}

class Aggregator::Impl {
  public:
    explicit Impl(int requested_device) {
        if (requested_device < 0) {
            check_cuda(cudaGetDevice(&device_), "cudaGetDevice");
        } else {
            device_ = requested_device;
            check_cuda(cudaSetDevice(device_), "cudaSetDevice");
        }
    }

    ~Impl() { (void)cudaSetDevice(device_); }

    void reserve(const AggregationShape& shape, float window_gamma) {
        validate_shape(shape);
        if (!std::isfinite(window_gamma) || window_gamma < 0.0F) {
            throw std::invalid_argument("CUDA aggregation window gamma must be "
                                        "finite and non-negative");
        }
        check_cuda(cudaSetDevice(device_), "cudaSetDevice");

        const int patch_area =
            checked_product_int(shape.patch_size, shape.patch_size,
                                "CUDA aggregation patch area overflows");
        std::vector<PatchWindowEntry> host_window(
            static_cast<std::size_t>(patch_area));
        for (int position = 0; position < patch_area; ++position) {
            host_window[static_cast<std::size_t>(position)] =
                PatchWindowEntry{1.0F, position / shape.patch_size};
        }
        if (window_gamma != 0.0F && shape.patch_size > 1) {
            const float center =
                0.5F * static_cast<float>(shape.patch_size - 1);
            const float radius = std::max(center, 0.5F);
            float maximum = 0.0F;
            for (int y = 0; y < shape.patch_size; ++y) {
                const float dy = (static_cast<float>(y) - center) / radius;
                for (int x = 0; x < shape.patch_size; ++x) {
                    const float dx = (static_cast<float>(x) - center) / radius;
                    const float raw = std::exp(-0.5F * ((dx * dx) + (dy * dy)));
                    host_window[static_cast<std::size_t>(y * shape.patch_size +
                                                         x)]
                        .weight = raw;
                    maximum = std::max(maximum, raw);
                }
            }
            const float inverse_maximum =
                maximum > 0.0F ? 1.0F / maximum : 1.0F;
            for (PatchWindowEntry& entry : host_window) {
                entry.weight =
                    std::pow(entry.weight * inverse_maximum, window_gamma);
            }
        }
        window_.reserve(host_window.size());
        check_cuda(cudaMemcpy(window_.data(), host_window.data(),
                              host_window.size() * sizeof(PatchWindowEntry),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy(aggregation window)");

        shape_ = shape;
        numerator_values_ = numerator_value_count(shape);
        weight_values_ = weight_value_count(shape);
        packed_values_ = packed_value_count(shape);
        reserved_ = true;
    }

    void enqueue_clear(const DeviceContributionView& contributions,
                       cudaStream_t stream) const {
        require_reserved();
        validate_contributions(contributions, shape_);
        check_cuda(cudaSetDevice(device_), "cudaSetDevice");
        if (contiguous_contributions(contributions, shape_)) {
            check_cuda(cudaMemsetAsync(
                           contributions.numerators, 0,
                           checked_product(numerator_values_, sizeof(float),
                                           "CUDA numerator clear overflows"),
                           stream),
                       "cudaMemsetAsync(aggregation numerators)");
            check_cuda(
                cudaMemsetAsync(contributions.weights, 0,
                                checked_product(weight_values_, sizeof(float),
                                                "CUDA weight clear overflows"),
                                stream),
                "cudaMemsetAsync(aggregation weights)");
            return;
        }

        const std::size_t values = std::max(numerator_values_, weight_values_);
        clear_strided_kernel<<<launch_blocks(values), block_threads, 0,
                               stream>>>(contributions, numerator_values_,
                                         weight_values_);
        check_cuda(cudaPeekAtLastError(), "launch aggregation clear kernel");
    }

    void enqueue_scatter(const AggregationShape& shape,
                         const AggregationParameters& parameters,
                         const DeviceAggregationBatch& batch,
                         cudaStream_t stream) const {
        validate_shape(shape);
        require_reserved();
        if (!same_reserved_geometry(shape, shape_)) {
            throw std::logic_error(
                "CUDA Aggregator is not reserved for this shape");
        }
        validate_parameters(parameters, batch, shape);
        check_cuda(cudaSetDevice(device_), "cudaSetDevice");

        scatter_direct_kernel<<<static_cast<unsigned int>(batch.groups),
                                block_threads, 0, stream>>>(
            shape, parameters, batch, window_.data());
        check_cuda(cudaPeekAtLastError(),
                   "launch direct aggregation scatter kernel");
    }

    void enqueue_pack(const DeviceContributionView& contributions,
                      float* packed, cudaStream_t stream) const {
        require_reserved();
        validate_contributions(contributions, shape_);
        if (packed == nullptr) {
            throw std::invalid_argument(
                "CUDA packed contribution output is null");
        }
        check_cuda(cudaSetDevice(device_), "cudaSetDevice");
        const std::size_t source_values = numerator_values_;
        pack_kernel<<<launch_blocks(source_values), block_threads, 0, stream>>>(
            contributions, packed);
        check_cuda(cudaPeekAtLastError(), "launch aggregation pack kernel");
    }

    void enqueue_normalize(const DeviceContributionView& contributions,
                           int slot, DeviceVideoView source, int source_frame,
                           DeviceMutableFrameView output,
                           cudaStream_t stream) const {
        require_reserved();
        validate_contributions(contributions, shape_);
        if (slot < 0 || slot >= shape_.slots) {
            throw std::invalid_argument(
                "CUDA normalization slot is out of range");
        }
        validate_source(source, source_frame, shape_);
        validate_output(output, shape_);
        check_cuda(cudaSetDevice(device_), "cudaSetDevice");
        const std::size_t pixels = plane_values(shape_);
        normalize_kernel<<<launch_blocks(pixels), block_threads, 0, stream>>>(
            contributions, slot, source, source_frame, output);
        check_cuda(cudaPeekAtLastError(),
                   "launch aggregation normalize kernel");
    }

    void enqueue_normalize_many(const DeviceContributionSource* device_sources,
                                int source_count, DeviceVideoView source,
                                int source_frame, DeviceMutableFrameView output,
                                cudaStream_t stream) const {
        require_reserved();
        if (source_count < 0 ||
            (source_count != 0 && device_sources == nullptr)) {
            throw std::invalid_argument(
                "invalid CUDA normalization contribution source array");
        }
        validate_source(source, source_frame, shape_);
        validate_output(output, shape_);
        check_cuda(cudaSetDevice(device_), "cudaSetDevice");
        const std::size_t pixels = plane_values(shape_);
        switch (shape_.channels) {
        case 1:
            normalize_many_small_channels_kernel<1>
                <<<launch_blocks(pixels), block_threads, 0, stream>>>(
                    device_sources, source_count, source, source_frame, output);
            break;
        case 2:
            normalize_many_small_channels_kernel<2>
                <<<launch_blocks(pixels), block_threads, 0, stream>>>(
                    device_sources, source_count, source, source_frame, output);
            break;
        case 3:
            normalize_many_small_channels_kernel<3>
                <<<launch_blocks(pixels), block_threads, 0, stream>>>(
                    device_sources, source_count, source, source_frame, output);
            break;
        case 4:
            normalize_many_small_channels_kernel<4>
                <<<launch_blocks(pixels), block_threads, 0, stream>>>(
                    device_sources, source_count, source, source_frame, output);
            break;
        default:
            normalize_many_channels_kernel<<<
                launch_blocks(checked_product(
                    pixels, static_cast<std::size_t>(shape_.channels),
                    "CUDA normalization value count overflows")),
                block_threads, 0, stream>>>(device_sources, source_count,
                                            source, source_frame, output);
            break;
        }
        check_cuda(cudaPeekAtLastError(),
                   "launch aggregation multi-source normalize kernel");
    }

    [[nodiscard]] std::size_t workspace_bytes() const noexcept {
        return window_.bytes();
    }

    void require_reserved() const {
        if (!reserved_) {
            throw std::logic_error("CUDA Aggregator has not been reserved");
        }
    }

    int device_ = 0;
    AggregationShape shape_{};
    bool reserved_ = false;
    std::size_t numerator_values_ = 0;
    std::size_t weight_values_ = 0;
    std::size_t packed_values_ = 0;
    DeviceBuffer<PatchWindowEntry> window_;
};

Aggregator::Aggregator(int device) : impl_(std::make_unique<Impl>(device)) {}

Aggregator::~Aggregator() = default;

Aggregator::Aggregator(Aggregator&&) noexcept = default;

Aggregator& Aggregator::operator=(Aggregator&&) noexcept = default;

void Aggregator::reserve(const AggregationShape& shape, float window_gamma) {
    if (impl_ == nullptr) {
        throw std::logic_error("CUDA Aggregator was moved from");
    }
    impl_->reserve(shape, window_gamma);
}

void Aggregator::enqueue_clear(const DeviceContributionView& contributions,
                               cudaStream_t stream) const {
    if (impl_ == nullptr) {
        throw std::logic_error("CUDA Aggregator was moved from");
    }
    impl_->enqueue_clear(contributions, stream);
}

void Aggregator::enqueue_scatter(const AggregationShape& shape,
                                 const AggregationParameters& parameters,
                                 const DeviceAggregationBatch& batch,
                                 cudaStream_t stream) const {
    if (impl_ == nullptr) {
        throw std::logic_error("CUDA Aggregator was moved from");
    }
    impl_->enqueue_scatter(shape, parameters, batch, stream);
}

void Aggregator::enqueue_pack(const DeviceContributionView& contributions,
                              float* packed, cudaStream_t stream) const {
    if (impl_ == nullptr) {
        throw std::logic_error("CUDA Aggregator was moved from");
    }
    impl_->enqueue_pack(contributions, packed, stream);
}

void Aggregator::enqueue_normalize(const DeviceContributionView& contributions,
                                   int slot, DeviceVideoView fallback_source,
                                   int source_frame,
                                   DeviceMutableFrameView output,
                                   cudaStream_t stream) const {
    if (impl_ == nullptr) {
        throw std::logic_error("CUDA Aggregator was moved from");
    }
    impl_->enqueue_normalize(contributions, slot, fallback_source, source_frame,
                             output, stream);
}

void Aggregator::enqueue_normalize_many(
    const DeviceContributionSource* device_sources, int source_count,
    DeviceVideoView fallback_source, int source_frame,
    DeviceMutableFrameView output, cudaStream_t stream) const {
    if (impl_ == nullptr) {
        throw std::logic_error("CUDA Aggregator was moved from");
    }
    impl_->enqueue_normalize_many(device_sources, source_count, fallback_source,
                                  source_frame, output, stream);
}

std::size_t Aggregator::numerator_values() const noexcept {
    return impl_ == nullptr ? 0 : impl_->numerator_values_;
}

std::size_t Aggregator::weight_values() const noexcept {
    return impl_ == nullptr ? 0 : impl_->weight_values_;
}

std::size_t Aggregator::packed_values() const noexcept {
    return impl_ == nullptr ? 0 : impl_->packed_values_;
}

std::size_t Aggregator::workspace_bytes() const noexcept {
    return impl_ == nullptr ? 0 : impl_->workspace_bytes();
}

int Aggregator::device() const noexcept {
    return impl_ == nullptr ? -1 : impl_->device_;
}

} // namespace vnlbcu
