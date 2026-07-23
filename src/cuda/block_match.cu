#include "cuda/block_match.hpp"

#include <cub/block/block_radix_sort.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace vnlbcu {
namespace {

constexpr int block_threads = 256;
constexpr int max_items_per_thread = 9;
constexpr int fused_candidate_capacity = block_threads * max_items_per_thread;
constexpr int tile_candidate_capacity = fused_candidate_capacity;
constexpr int auto_fused_candidate_threshold = fused_candidate_capacity;
constexpr int auto_chunk_partial_threshold = tile_candidate_capacity;
constexpr int auto_chunk_min_average_candidates = 900;
constexpr int auto_chunk_many_min_average_candidates = 1024;
constexpr int auto_chunk_basic_small_max_tasks = 10;
constexpr int auto_chunk_coupled_small_max_tasks = 5;
using MatchKey = std::uint64_t;
constexpr MatchKey padding_match_key = ~MatchKey{0};

enum class CacheMode : std::uint8_t {
    Tile,
    Reference,
    Direct,
};

struct SearchWindow {
    int x_low;
    int y_low;
    int frame;
};

struct SelectedOrigin {
    int x;
    int y;
    int frame;
};

struct TileLayout {
    int candidate_width = 0;
    int candidate_height = 0;
    int tiles_x = 0;
    int tiles_y = 0;
    int tiles_per_slot = 0;
    int total_tiles = 0;
};

__device__ __forceinline__ MatchKey make_match_key(float distance,
                                                   std::uint32_t candidate_id) {
    return (static_cast<MatchKey>(__float_as_uint(distance)) << 32U) |
           candidate_id;
}

__device__ __forceinline__ float match_key_distance(MatchKey key) {
    return __uint_as_float(static_cast<std::uint32_t>(key >> 32U));
}

__device__ __forceinline__ std::uint32_t match_key_candidate(MatchKey key) {
    return static_cast<std::uint32_t>(key);
}

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

int checked_add(int left, int right, const char* label) {
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

template <typename T> class DeviceBuffer {
  public:
    DeviceBuffer() = default;
    ~DeviceBuffer() {
        if (data_ != nullptr) {
            (void)cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void reset() {
        if (data_ != nullptr) {
            check_cuda(cudaFree(data_), "cudaFree(CUDA matcher workspace)");
            data_ = nullptr;
            capacity_ = 0;
        }
    }

    void reserve(std::size_t count, const char* label) {
        if (count <= capacity_) {
            return;
        }
        T* replacement = nullptr;
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&replacement),
                              checked_product(count, sizeof(T), label)),
                   "cudaMalloc(CUDA matcher workspace)");
        if (data_ != nullptr) {
            const cudaError_t status = cudaFree(data_);
            if (status != cudaSuccess) {
                (void)cudaFree(replacement);
                throw_cuda(status, "cudaFree(CUDA matcher workspace)");
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
    T* data_ = nullptr;
    std::size_t capacity_ = 0;
};

int valid_origin_count(int axis, int patch) {
    return checked_add(axis, 1 - patch, "valid patch origin count overflows");
}

int temporal_count_for(const MatchBatchShape& shape) {
    const int requested =
        checked_add(checked_add(shape.search_bwd, shape.search_fwd,
                                "temporal search count overflows"),
                    1, "temporal search count overflows");
    const int origins =
        valid_origin_count(shape.source_frames, shape.patch_time);
    return std::min(requested, origins);
}

int window_width_for(const MatchBatchShape& shape) {
    return std::min(shape.search_window,
                    valid_origin_count(shape.width, shape.patch_size));
}

int window_height_for(const MatchBatchShape& shape) {
    return std::min(shape.search_window,
                    valid_origin_count(shape.height, shape.patch_size));
}

int candidate_count_for(const MatchBatchShape& shape) {
    const int area =
        checked_product_int(window_width_for(shape), window_height_for(shape),
                            "spatial candidate count overflows");
    return checked_product_int(area, temporal_count_for(shape),
                               "candidate count overflows");
}

int divide_round_up(int value, int divisor) {
    return 1 + ((value - 1) / divisor);
}

TileLayout tile_layout_for(const MatchBatchShape& shape) {
    const int window_width = window_width_for(shape);
    const int window_height = window_height_for(shape);
    const int candidate_width = std::min(window_width, tile_candidate_capacity);
    const int candidate_height = std::max(
        1, std::min(window_height, tile_candidate_capacity / candidate_width));
    const int tiles_x = divide_round_up(window_width, candidate_width);
    const int tiles_y = divide_round_up(window_height, candidate_height);
    const int tiles_per_slot = checked_product_int(
        tiles_x, tiles_y, "CUDA matcher tile count overflows");
    return TileLayout{
        .candidate_width = candidate_width,
        .candidate_height = candidate_height,
        .tiles_x = tiles_x,
        .tiles_y = tiles_y,
        .tiles_per_slot = tiles_per_slot,
        .total_tiles =
            checked_product_int(tiles_per_slot, temporal_count_for(shape),
                                "CUDA matcher temporal tile count overflows"),
    };
}

int tile_candidate_count(const MatchBatchShape& shape, const TileLayout& layout,
                         int task) {
    const int spatial_task = task % layout.tiles_per_slot;
    const int tile_y = spatial_task / layout.tiles_x;
    const int tile_x = spatial_task - (tile_y * layout.tiles_x);
    const int x_begin = tile_x * layout.candidate_width;
    const int y_begin = tile_y * layout.candidate_height;
    const int width =
        std::min(layout.candidate_width, window_width_for(shape) - x_begin);
    const int height =
        std::min(layout.candidate_height, window_height_for(shape) - y_begin);
    return checked_product_int(width, height,
                               "CUDA matcher tile area overflows");
}

bool auto_chunk_geometry(const MatchBatchShape& shape, const TileLayout& layout,
                         int candidates, int partial_count) {
    if (partial_count > auto_chunk_partial_threshold) {
        return false;
    }
    const int average_candidates = candidates / layout.total_tiles;
    const bool coupled_match =
        shape.stage == Stage::Final && shape.channels > 1;
    const int small_task_limit = coupled_match
                                     ? auto_chunk_coupled_small_max_tasks
                                     : auto_chunk_basic_small_max_tasks;
    const int minimum_average = layout.total_tiles <= small_task_limit
                                    ? auto_chunk_min_average_candidates
                                    : auto_chunk_many_min_average_candidates;
    return average_candidates >= minimum_average;
}

int patch_dimension_for(const MatchBatchShape& shape) {
    const int area = checked_product_int(shape.patch_size, shape.patch_size,
                                         "patch area overflows");
    const int temporal = checked_product_int(area, shape.patch_time,
                                             "patch dimension overflows");
    return checked_product_int(temporal, shape.channels,
                               "sample dimension overflows");
}

int match_dimension_for(const MatchBatchShape& shape) {
    const int area = checked_product_int(shape.patch_size, shape.patch_size,
                                         "patch area overflows");
    const int temporal = checked_product_int(area, shape.patch_time,
                                             "match dimension overflows");
    return shape.stage == Stage::Basic
               ? temporal
               : checked_product_int(temporal, shape.channels,
                                     "match dimension overflows");
}

void validate_stage(Stage stage) {
    if (stage != Stage::Basic && stage != Stage::Final) {
        throw std::invalid_argument("invalid CUDA VNLB stage");
    }
}

void validate_strategy(MatchStrategy strategy) {
    switch (strategy) {
    case MatchStrategy::Auto:
    case MatchStrategy::Fused:
    case MatchStrategy::Chunked:
    case MatchStrategy::FullSort:
        return;
    }
    throw std::invalid_argument("invalid CUDA block-match strategy");
}

void validate_shape(const MatchBatchShape& shape) {
    validate_stage(shape.stage);
    validate_strategy(shape.strategy);
    if (shape.groups <= 0 || shape.width <= 0 || shape.height <= 0 ||
        shape.channels <= 0 || shape.frames <= 0 || shape.source_frames <= 0) {
        throw std::invalid_argument("CUDA match geometry must be positive");
    }
    if (shape.first_frame < 0 || shape.first_frame > shape.source_frames ||
        shape.frames > shape.source_frames - shape.first_frame) {
        throw std::invalid_argument(
            "CUDA frame cache must fit inside the source clip");
    }
    if (shape.patch_size <= 0 || shape.patch_size > shape.width ||
        shape.patch_size > shape.height || shape.patch_time <= 0 ||
        shape.patch_time > shape.source_frames) {
        throw std::invalid_argument("invalid CUDA match patch geometry");
    }
    if (shape.search_window <= 0 || (shape.search_window & 1) == 0 ||
        shape.search_bwd < 0 || shape.search_fwd < 0) {
        throw std::invalid_argument("invalid CUDA match search geometry");
    }
    if (shape.requested_similar <= 0 || shape.retained_stride <= 0) {
        throw std::invalid_argument("CUDA match group sizes must be positive");
    }

    const int candidates = candidate_count_for(shape);
    const int effective_similar = std::min(shape.requested_similar, candidates);
    if (shape.retained_stride < effective_similar ||
        shape.retained_stride > candidates) {
        throw std::invalid_argument(
            "CUDA retained stride must cover min(K, candidate_count)");
    }
    (void)patch_dimension_for(shape);
    (void)match_dimension_for(shape);
}

void validate_parameters(const MatchParameters& parameters) {
    if (!std::isfinite(parameters.tau) || parameters.tau < 0.0F) {
        throw std::invalid_argument(
            "CUDA block-match tau must be finite and non-negative");
    }
}

void validate_video(const DeviceVideoView& view, const MatchBatchShape& shape,
                    const char* label) {
    if ((view.data == nullptr && view.frame_data == nullptr) ||
        view.width != shape.width || view.height != shape.height ||
        view.channels != shape.channels || view.frames != shape.frames ||
        view.first_frame != shape.first_frame ||
        view.source_frames != shape.source_frames) {
        throw std::invalid_argument(std::string(label) +
                                    " device video shape mismatch");
    }
    if (view.row_stride < shape.width || view.channel_stride <= 0 ||
        (view.frame_data == nullptr && view.frame_stride <= 0)) {
        throw std::invalid_argument(std::string(label) +
                                    " device video strides are invalid");
    }
    const std::size_t minimum_channel = checked_product(
        static_cast<std::size_t>(view.row_stride),
        static_cast<std::size_t>(shape.height), "video plane size overflows");
    const std::size_t minimum_frame = checked_product(
        static_cast<std::size_t>(view.channel_stride),
        static_cast<std::size_t>(shape.channels), "video frame size overflows");
    if (static_cast<std::size_t>(view.channel_stride) < minimum_channel ||
        (view.frame_data == nullptr &&
         static_cast<std::size_t>(view.frame_stride) < minimum_frame)) {
        throw std::invalid_argument(std::string(label) +
                                    " device video strides overlap");
    }
}

void validate_batch(const MatchBatchShape& shape,
                    const DeviceMatchBatch& batch) {
    validate_video(batch.noisy, shape, "noisy");
    if (shape.stage == Stage::Final) {
        validate_video(batch.basic, shape, "basic");
    }
    if (batch.anchors == nullptr || batch.retained_counts == nullptr ||
        batch.noisy_samples == nullptr) {
        throw std::invalid_argument(
            "CUDA matcher requires anchors, retained counts and noisy output");
    }
    if (shape.stage == Stage::Final && batch.basic_samples == nullptr) {
        throw std::invalid_argument(
            "CUDA Final matcher requires a basic sample output");
    }
}

bool same_reserved_geometry(const MatchBatchShape& left,
                            const MatchBatchShape& right) noexcept {
    return left.stage == right.stage && left.width == right.width &&
           left.height == right.height && left.channels == right.channels &&
           left.source_frames == right.source_frames &&
           left.patch_size == right.patch_size &&
           left.patch_time == right.patch_time &&
           left.search_window == right.search_window &&
           left.search_bwd == right.search_bwd &&
           left.search_fwd == right.search_fwd &&
           left.requested_similar == right.requested_similar &&
           left.retained_stride == right.retained_stride &&
           left.strategy == right.strategy;
}

__device__ __forceinline__ int shifted_range_low(int center, int half,
                                                 int max_origin) {
    const long long start = static_cast<long long>(center) - half;
    const long long end = static_cast<long long>(center) + half;
    const long long low_overflow = start < 0 ? start : 0;
    const long long high_overflow = end > max_origin ? end - max_origin : 0;
    const long long shift = low_overflow + high_overflow;
    const long long shifted = start - shift;
    const long long clamped =
        shifted < 0 ? 0 : (shifted > max_origin ? max_origin : shifted);
    return static_cast<int>(clamped);
}

__device__ __forceinline__ int scheduled_frame(const MatchBatchShape& shape,
                                               int anchor_frame, int slot) {
    if (slot == 0) {
        return anchor_frame;
    }
    const int max_origin = shape.source_frames - shape.patch_time;
    const long long search_start =
        static_cast<long long>(anchor_frame) - shape.search_bwd;
    const long long search_end =
        static_cast<long long>(anchor_frame) + shape.search_fwd;
    const long long low_overflow = search_start < 0 ? search_start : 0;
    const long long high_overflow =
        search_end > max_origin ? search_end - max_origin : 0;
    const long long shift = low_overflow + high_overflow;
    long long low = search_start - shift;
    long long high = search_end - shift;
    low = low < 0 ? 0 : low;
    high = high > max_origin ? max_origin : high;
    const long long forward = high - anchor_frame;
    const long long frame =
        slot <= forward
            ? static_cast<long long>(anchor_frame) + slot
            : static_cast<long long>(anchor_frame) - (slot - forward);
    return static_cast<int>(frame < low ? low : (frame > high ? high : frame));
}

__device__ __forceinline__ SearchWindow make_window(
    const MatchBatchShape& shape, const DeviceMatchBatch& batch,
    const PatchOrigin& anchor, int group, int slot, int temporal_count) {
    int center_x = anchor.x;
    int center_y = anchor.y;
    int frame = scheduled_frame(shape, anchor.frame, slot);
    if (batch.search_centers != nullptr) {
        const SearchCenter center =
            batch.search_centers[(static_cast<std::size_t>(group) *
                                  temporal_count) +
                                 slot];
        center_x = center.x;
        center_y = center.y;
        frame = center.frame;
    }
    const int half = (shape.search_window - 1) / 2;
    return SearchWindow{
        shifted_range_low(center_x, half, shape.width - shape.patch_size),
        shifted_range_low(center_y, half, shape.height - shape.patch_size),
        frame,
    };
}

__device__ __forceinline__ float video_sample(const DeviceVideoView& view,
                                              int frame, int channel, int y,
                                              int x) {
    const std::ptrdiff_t local_frame = frame - view.first_frame;
    const float* const frame_base =
        view.frame_data != nullptr
            ? view.frame_data[local_frame]
            : view.data + local_frame * view.frame_stride;
    const std::ptrdiff_t index =
        static_cast<std::ptrdiff_t>(channel) * view.channel_stride +
        static_cast<std::ptrdiff_t>(y) * view.row_stride + x;
    return frame_base[index];
}

template <Stage stage>
__device__ __forceinline__ DeviceVideoView
matching_source(const DeviceMatchBatch& batch) {
    if constexpr (stage == Stage::Basic) {
        return batch.noisy;
    }
    return batch.basic;
}

template <int StaticPatch>
__device__ __forceinline__ int patch_size(const MatchBatchShape& shape) {
    if constexpr (StaticPatch != 0) {
        return StaticPatch;
    }
    return shape.patch_size;
}

struct DeviceTile {
    int slot;
    int x_begin;
    int y_begin;
    int width;
    int height;
};

__device__ __forceinline__ DeviceTile make_tile(const TileLayout& layout,
                                                int window_width,
                                                int window_height, int task) {
    const int slot = task / layout.tiles_per_slot;
    const int spatial_task = task - (slot * layout.tiles_per_slot);
    const int tile_y = spatial_task / layout.tiles_x;
    const int tile_x = spatial_task - (tile_y * layout.tiles_x);
    const int x_begin = tile_x * layout.candidate_width;
    const int y_begin = tile_y * layout.candidate_height;
    return DeviceTile{
        slot,
        x_begin,
        y_begin,
        min(layout.candidate_width, window_width - x_begin),
        min(layout.candidate_height, window_height - y_begin),
    };
}

template <Stage stage, CacheMode cache_mode, int StaticPatch>
__device__ __forceinline__ float
candidate_distance(const MatchBatchShape& shape, const DeviceVideoView& search,
                   const PatchOrigin& anchor, const SearchWindow& window,
                   int candidate_x, int candidate_y, const float* reference,
                   const float* tile, int tile_width, int tile_plane_values) {
    const int patch = patch_size<StaticPatch>(shape);
    const int patch_area = patch * patch;
    const int match_channels = stage == Stage::Basic ? 1 : shape.channels;
    const int match_planes = match_channels * shape.patch_time;
    float distance = 0.0F;
    for (int plane = 0; plane < match_planes; ++plane) {
        const int channel =
            stage == Stage::Basic ? 0 : plane / shape.patch_time;
        const int frame_delta = plane % shape.patch_time;
        for (int py = 0; py < patch; ++py) {
#pragma unroll
            for (int px = 0; px < StaticPatch; ++px) {
                const int reference_index =
                    (plane * patch_area) + (py * patch) + px;
                float reference_value = 0.0F;
                if constexpr (cache_mode == CacheMode::Direct) {
                    reference_value =
                        video_sample(search, anchor.frame + frame_delta,
                                     channel, anchor.y + py, anchor.x + px);
                } else {
                    reference_value = reference[reference_index];
                }
                float candidate_value = 0.0F;
                if constexpr (cache_mode == CacheMode::Tile) {
                    candidate_value = tile[(plane * tile_plane_values) +
                                           ((candidate_y + py) * tile_width) +
                                           candidate_x + px];
                } else {
                    candidate_value =
                        video_sample(search, window.frame + frame_delta,
                                     channel, window.y_low + candidate_y + py,
                                     window.x_low + candidate_x + px);
                }
                const float difference = reference_value - candidate_value;
                distance = fmaf(difference, difference, distance);
            }
            if constexpr (StaticPatch == 0) {
                for (int px = 0; px < patch; ++px) {
                    const int reference_index =
                        (plane * patch_area) + (py * patch) + px;
                    float reference_value = 0.0F;
                    if constexpr (cache_mode == CacheMode::Direct) {
                        reference_value =
                            video_sample(search, anchor.frame + frame_delta,
                                         channel, anchor.y + py, anchor.x + px);
                    } else {
                        reference_value = reference[reference_index];
                    }
                    float candidate_value = 0.0F;
                    if constexpr (cache_mode == CacheMode::Tile) {
                        candidate_value =
                            tile[(plane * tile_plane_values) +
                                 ((candidate_y + py) * tile_width) +
                                 candidate_x + px];
                    } else {
                        candidate_value = video_sample(
                            search, window.frame + frame_delta, channel,
                            window.y_low + candidate_y + py,
                            window.x_low + candidate_x + px);
                    }
                    const float difference = reference_value - candidate_value;
                    distance = fmaf(difference, difference, distance);
                }
            }
        }
    }
    if (!isfinite(distance)) {
        distance = CUDART_INF_F;
    }
    return distance;
}

template <Stage stage, int ItemsPerThread, CacheMode cache_mode,
          int StaticPatch>
__global__ void
match_select_gather_kernel(MatchBatchShape shape, MatchParameters parameters,
                           DeviceMatchBatch batch, int temporal_count,
                           int window_width, int window_height,
                           int candidate_count, int sample_dim) {
    using BlockSort = cub::BlockRadixSort<float, block_threads, ItemsPerThread,
                                          std::uint32_t>;
    extern __shared__ __align__(16) unsigned char shared_raw[];
    __shared__ PatchOrigin anchor;
    __shared__ int retained;
    __shared__ float retention_threshold;

    const int group = static_cast<int>(blockIdx.x);
    const int thread = static_cast<int>(threadIdx.x);
    if (group >= shape.groups) {
        return;
    }
    if (thread == 0) {
        anchor = batch.anchors[group];
    }
    __syncthreads();

    const int patch = patch_size<StaticPatch>(shape);
    const int patch_area = patch * patch;
    const int match_channels = stage == Stage::Basic ? 1 : shape.channels;
    const int match_planes = match_channels * shape.patch_time;
    const int match_dim = match_planes * patch_area;
    const int tile_width = window_width + patch - 1;
    const int tile_height = window_height + patch - 1;
    const int tile_plane_values = tile_width * tile_height;
    const int tile_values = match_planes * tile_plane_values;
    const int spatial_candidates = window_width * window_height;
    const DeviceVideoView search = matching_source<stage>(batch);

    float* reference = nullptr;
    float* tile = nullptr;
    float* distances = reinterpret_cast<float*>(shared_raw);
    if constexpr (cache_mode == CacheMode::Tile) {
        reference = reinterpret_cast<float*>(shared_raw);
        tile = reference + match_dim;
        distances = tile + tile_values;
    } else if constexpr (cache_mode == CacheMode::Reference) {
        reference = reinterpret_cast<float*>(shared_raw);
        distances = reference + match_dim;
    }

    if constexpr (cache_mode != CacheMode::Direct) {
        for (int index = thread; index < match_dim; index += block_threads) {
            const int plane = index / patch_area;
            const int position = index - (plane * patch_area);
            const int channel =
                stage == Stage::Basic ? 0 : plane / shape.patch_time;
            const int frame_delta = plane % shape.patch_time;
            const int py = position / patch;
            const int px = position - (py * patch);
            reference[index] =
                video_sample(search, anchor.frame + frame_delta, channel,
                             anchor.y + py, anchor.x + px);
        }
        __syncthreads();
    }

    for (int slot = 0; slot < temporal_count; ++slot) {
        SearchWindow current_window{};
        const int warp_lane = thread & 31;
        if (warp_lane == 0) {
            current_window =
                make_window(shape, batch, anchor, group, slot, temporal_count);
        }
        current_window.x_low =
            __shfl_sync(0xffffffffU, current_window.x_low, 0);
        current_window.y_low =
            __shfl_sync(0xffffffffU, current_window.y_low, 0);
        current_window.frame =
            __shfl_sync(0xffffffffU, current_window.frame, 0);

        if constexpr (cache_mode == CacheMode::Tile) {
            for (int index = thread; index < tile_values;
                 index += block_threads) {
                const int plane = index / tile_plane_values;
                const int position = index - (plane * tile_plane_values);
                const int channel =
                    stage == Stage::Basic ? 0 : plane / shape.patch_time;
                const int frame_delta = plane % shape.patch_time;
                const int ty = position / tile_width;
                const int tx = position - (ty * tile_width);
                tile[index] = video_sample(
                    search, current_window.frame + frame_delta, channel,
                    current_window.y_low + ty, current_window.x_low + tx);
            }
            __syncthreads();
        }

        for (int local_candidate = thread; local_candidate < spatial_candidates;
             local_candidate += block_threads) {
            const int candidate_y = local_candidate / window_width;
            const int candidate_x =
                local_candidate - (candidate_y * window_width);
            float distance = 0.0F;
            for (int plane = 0; plane < match_planes; ++plane) {
                const int channel =
                    stage == Stage::Basic ? 0 : plane / shape.patch_time;
                const int frame_delta = plane % shape.patch_time;
                for (int py = 0; py < patch; ++py) {
#pragma unroll
                    for (int px = 0; px < StaticPatch; ++px) {
                        const int reference_index =
                            (plane * patch_area) + (py * patch) + px;
                        float reference_value = 0.0F;
                        if constexpr (cache_mode == CacheMode::Direct) {
                            reference_value = video_sample(
                                search, anchor.frame + frame_delta, channel,
                                anchor.y + py, anchor.x + px);
                        } else {
                            reference_value = reference[reference_index];
                        }

                        float candidate_value = 0.0F;
                        if constexpr (cache_mode == CacheMode::Tile) {
                            const int tile_index =
                                (plane * tile_plane_values) +
                                ((candidate_y + py) * tile_width) +
                                candidate_x + px;
                            candidate_value = tile[tile_index];
                        } else {
                            candidate_value = video_sample(
                                search, current_window.frame + frame_delta,
                                channel,
                                current_window.y_low + candidate_y + py,
                                current_window.x_low + candidate_x + px);
                        }
                        const float difference =
                            reference_value - candidate_value;
                        distance = fmaf(difference, difference, distance);
                    }
                    if constexpr (StaticPatch == 0) {
                        for (int px = 0; px < patch; ++px) {
                            const int reference_index =
                                (plane * patch_area) + (py * patch) + px;
                            float reference_value = 0.0F;
                            if constexpr (cache_mode == CacheMode::Direct) {
                                reference_value = video_sample(
                                    search, anchor.frame + frame_delta, channel,
                                    anchor.y + py, anchor.x + px);
                            } else {
                                reference_value = reference[reference_index];
                            }
                            float candidate_value = 0.0F;
                            if constexpr (cache_mode == CacheMode::Tile) {
                                const int tile_index =
                                    (plane * tile_plane_values) +
                                    ((candidate_y + py) * tile_width) +
                                    candidate_x + px;
                                candidate_value = tile[tile_index];
                            } else {
                                candidate_value = video_sample(
                                    search, current_window.frame + frame_delta,
                                    channel,
                                    current_window.y_low + candidate_y + py,
                                    current_window.x_low + candidate_x + px);
                            }
                            const float difference =
                                reference_value - candidate_value;
                            distance = fmaf(difference, difference, distance);
                        }
                    }
                }
            }
            if (!isfinite(distance)) {
                distance = CUDART_INF_F;
            }
            distances[(slot * spatial_candidates) + local_candidate] = distance;
        }
        __syncthreads();
    }

    float thread_distances[ItemsPerThread];
    std::uint32_t thread_ids[ItemsPerThread];
#pragma unroll
    for (int item = 0; item < ItemsPerThread; ++item) {
        const int rank = (thread * ItemsPerThread) + item;
        if (rank < candidate_count) {
            thread_distances[item] = distances[rank];
            thread_ids[item] = static_cast<std::uint32_t>(rank);
        } else {
            thread_distances[item] = CUDART_INF_F;
            thread_ids[item] = 0xffffffffU;
        }
    }
    __syncthreads();

    auto& sort_storage =
        *reinterpret_cast<typename BlockSort::TempStorage*>(shared_raw);
    BlockSort(sort_storage).SortBlockedToStriped(thread_distances, thread_ids);
    if (shape.retained_stride <= 64) {
        const int effective_similar =
            min(shape.requested_similar, candidate_count);
        if (thread == 0) {
            retained = 0;
        }
        if (thread == effective_similar - 1) {
            retention_threshold = fmaxf(parameters.tau, thread_distances[0]);
        }
        __syncthreads();

        const int lane = thread & 31;
        const int warp = thread / 32;
        const unsigned int retained_mask = __ballot_sync(
            0xffffffffU, thread < shape.retained_stride &&
                             thread_distances[0] <= retention_threshold);
        if (lane == 0 && warp * 32 < shape.retained_stride) {
            atomicAdd(&retained, __popc(retained_mask));
        }
        __syncthreads();
        if (thread == 0) {
            batch.retained_counts[group] = retained;
        }
    } else {
        __syncthreads();

        float* selected_distances = reinterpret_cast<float*>(shared_raw);
#pragma unroll
        for (int item = 0; item < ItemsPerThread; ++item) {
            const int rank = (item * block_threads) + thread;
            if (rank < shape.retained_stride) {
                selected_distances[rank] = thread_distances[item];
            }
        }
        __syncthreads();

        if (thread == 0) {
            const int effective_similar =
                min(shape.requested_similar, candidate_count);
            const float kth = selected_distances[effective_similar - 1];
            const float threshold = fmaxf(parameters.tau, kth);
            int count = effective_similar;
            while (count < shape.retained_stride &&
                   selected_distances[count] <= threshold) {
                ++count;
            }
            retained = count;
            batch.retained_counts[group] = count;
        }
        __syncthreads();
    }

    const std::size_t descriptor_base =
        static_cast<std::size_t>(group) * shape.retained_stride;
    // CUB's striped output remains live in the per-thread arrays. Once the
    // retained count is known, reuse the sort's shared region for decoded
    // patch origins instead of round-tripping descriptors through global
    // memory or decoding the candidate id for every value.
    auto* selected_origins = reinterpret_cast<SelectedOrigin*>(shared_raw);
#pragma unroll
    for (int item = 0; item < ItemsPerThread; ++item) {
        const int sample = (item * block_threads) + thread;
        if (sample < retained) {
            const std::uint32_t candidate_id = thread_ids[item];
            const int slot =
                static_cast<int>(candidate_id) / spatial_candidates;
            const int local =
                static_cast<int>(candidate_id) - (slot * spatial_candidates);
            const int dy = local / window_width;
            const int dx = local - (dy * window_width);
            const SearchWindow window =
                make_window(shape, batch, anchor, group, slot, temporal_count);
            const SelectedOrigin origin{window.x_low + dx, window.y_low + dy,
                                        window.frame};
            selected_origins[sample] = origin;
            if (batch.candidate_ids != nullptr) {
                batch.candidate_ids[descriptor_base + sample] = candidate_id;
            }
            if (batch.matches != nullptr) {
                batch.matches[descriptor_base + sample] = PatchMatch{
                    thread_distances[item], origin.x, origin.y, origin.frame};
            }
        }
    }
    __syncthreads();

    const std::size_t sample_group_base =
        static_cast<std::size_t>(group) * shape.retained_stride * sample_dim;
    constexpr int warp_threads = 32;
    constexpr int warps_per_block = block_threads / warp_threads;
    const int lane = thread & (warp_threads - 1);
    const int warp = thread / warp_threads;
    const int channel_patch_dim = shape.patch_time * patch_area;
    for (int sample = warp; sample < retained; sample += warps_per_block) {
        const SelectedOrigin origin = selected_origins[sample];
        const std::size_t output_base =
            sample_group_base + (static_cast<std::size_t>(sample) * sample_dim);
        for (int channel = 0; channel < shape.channels; ++channel) {
            const std::size_t channel_output =
                output_base +
                (static_cast<std::size_t>(channel) * channel_patch_dim);
            for (int frame_delta = 0; frame_delta < shape.patch_time;
                 ++frame_delta) {
                const int frame = origin.frame + frame_delta;
                const std::size_t plane_output =
                    channel_output +
                    (static_cast<std::size_t>(frame_delta) * patch_area);
                for (int position = lane; position < patch_area;
                     position += warp_threads) {
                    const int py = position / patch;
                    const int px = position - (py * patch);
                    const int x = origin.x + px;
                    const int y = origin.y + py;
                    const std::size_t output = plane_output + position;
                    batch.noisy_samples[output] =
                        video_sample(batch.noisy, frame, channel, y, x);
                    if constexpr (stage == Stage::Final) {
                        batch.basic_samples[output] =
                            video_sample(batch.basic, frame, channel, y, x);
                    }
                }
            }
        }
    }
}

template <Stage stage, int StaticPatch>
__global__ void single_match_gather_kernel(MatchBatchShape shape,
                                           DeviceMatchBatch batch,
                                           int window_width, int window_height,
                                           int sample_dim) {
    const int group = static_cast<int>(blockIdx.x);
    const int thread = static_cast<int>(threadIdx.x);
    if (group >= shape.groups) {
        return;
    }
    const PatchOrigin anchor = batch.anchors[group];
    const int patch = patch_size<StaticPatch>(shape);
    const int patch_area = patch * patch;
    const int spatial_candidates = window_width * window_height;
    const int half = (shape.search_window - 1) / 2;
    const int x_low =
        shifted_range_low(anchor.x, half, shape.width - shape.patch_size);
    const int y_low =
        shifted_range_low(anchor.y, half, shape.height - shape.patch_size);
    const int local = (anchor.y - y_low) * window_width + (anchor.x - x_low);
    const std::uint32_t candidate_id = local >= 0 && local < spatial_candidates
                                           ? static_cast<std::uint32_t>(local)
                                           : 0xffffffffU;
    const std::size_t descriptor_base =
        static_cast<std::size_t>(group) * shape.retained_stride;
    if (thread == 0) {
        batch.retained_counts[group] = 1;
        if (batch.candidate_ids != nullptr) {
            batch.candidate_ids[descriptor_base] = candidate_id;
        }
        if (batch.matches != nullptr) {
            batch.matches[descriptor_base] =
                PatchMatch{0.0F, anchor.x, anchor.y, anchor.frame};
        }
    }

    const std::size_t output_base =
        static_cast<std::size_t>(group) * shape.retained_stride * sample_dim;
    for (int dimension = thread; dimension < sample_dim;
         dimension += block_threads) {
        const int channel_patch_dim = shape.patch_time * patch_area;
        const int channel = dimension / channel_patch_dim;
        const int channel_position = dimension - (channel * channel_patch_dim);
        const int frame_delta = channel_position / patch_area;
        const int position = channel_position - (frame_delta * patch_area);
        const int py = position / patch;
        const int px = position - (py * patch);
        batch.noisy_samples[output_base + dimension] =
            video_sample(batch.noisy, anchor.frame + frame_delta, channel,
                         anchor.y + py, anchor.x + px);
        if constexpr (stage == Stage::Final) {
            batch.basic_samples[output_base + dimension] =
                video_sample(batch.basic, anchor.frame + frame_delta, channel,
                             anchor.y + py, anchor.x + px);
        }
    }
}

template <Stage stage, int ItemsPerThread, CacheMode cache_mode,
          int StaticPatch>
__global__ void chunk_match_select_kernel(
    MatchBatchShape shape, DeviceMatchBatch batch, TileLayout layout,
    int temporal_count, int window_width, int window_height,
    const int* partial_offsets, int partial_per_group, MatchKey* partial_keys) {
    using BlockSort =
        cub::BlockRadixSort<MatchKey, block_threads, ItemsPerThread>;
    extern __shared__ __align__(16) unsigned char shared_raw[];
    __shared__ SearchWindow current_window;
    __shared__ PatchOrigin anchor;

    const int flat_task = static_cast<int>(blockIdx.x);
    const int group = flat_task / layout.total_tiles;
    const int task = flat_task - (group * layout.total_tiles);
    const int thread = static_cast<int>(threadIdx.x);
    if (group >= shape.groups) {
        return;
    }
    if (thread == 0) {
        anchor = batch.anchors[group];
    }
    __syncthreads();

    const DeviceTile candidate_tile =
        make_tile(layout, window_width, window_height, task);
    if (thread == 0) {
        current_window = make_window(shape, batch, anchor, group,
                                     candidate_tile.slot, temporal_count);
        current_window.x_low += candidate_tile.x_begin;
        current_window.y_low += candidate_tile.y_begin;
    }
    __syncthreads();

    const int patch = patch_size<StaticPatch>(shape);
    const int patch_area = patch * patch;
    const int match_channels = stage == Stage::Basic ? 1 : shape.channels;
    const int match_planes = match_channels * shape.patch_time;
    const int match_dim = match_planes * patch_area;
    const int max_tile_width = layout.candidate_width + patch - 1;
    const int max_tile_height = layout.candidate_height + patch - 1;
    const int max_tile_values = match_planes * max_tile_width * max_tile_height;
    const int tile_width = candidate_tile.width + patch - 1;
    const int tile_height = candidate_tile.height + patch - 1;
    const int tile_plane_values = tile_width * tile_height;
    const int tile_values = match_planes * tile_plane_values;
    const int tile_candidates = candidate_tile.width * candidate_tile.height;
    const DeviceVideoView search = matching_source<stage>(batch);

    float* reference = nullptr;
    float* tile = nullptr;
    float* distances = reinterpret_cast<float*>(shared_raw);
    if constexpr (cache_mode == CacheMode::Tile) {
        reference = reinterpret_cast<float*>(shared_raw);
        tile = reference + match_dim;
        distances = tile + max_tile_values;
    } else if constexpr (cache_mode == CacheMode::Reference) {
        reference = reinterpret_cast<float*>(shared_raw);
        distances = reference + match_dim;
    }

    if constexpr (cache_mode != CacheMode::Direct) {
        for (int index = thread; index < match_dim; index += block_threads) {
            const int plane = index / patch_area;
            const int position = index - (plane * patch_area);
            const int channel =
                stage == Stage::Basic ? 0 : plane / shape.patch_time;
            const int frame_delta = plane % shape.patch_time;
            const int py = position / patch;
            const int px = position - (py * patch);
            reference[index] =
                video_sample(search, anchor.frame + frame_delta, channel,
                             anchor.y + py, anchor.x + px);
        }
    }
    if constexpr (cache_mode == CacheMode::Tile) {
        for (int index = thread; index < tile_values; index += block_threads) {
            const int plane = index / tile_plane_values;
            const int position = index - (plane * tile_plane_values);
            const int channel =
                stage == Stage::Basic ? 0 : plane / shape.patch_time;
            const int frame_delta = plane % shape.patch_time;
            const int ty = position / tile_width;
            const int tx = position - (ty * tile_width);
            tile[index] = video_sample(
                search, current_window.frame + frame_delta, channel,
                current_window.y_low + ty, current_window.x_low + tx);
        }
    }
    if constexpr (cache_mode != CacheMode::Direct) {
        __syncthreads();
    }

    for (int local = thread; local < tile_candidates; local += block_threads) {
        const int candidate_y = local / candidate_tile.width;
        const int candidate_x = local - (candidate_y * candidate_tile.width);
        distances[local] = candidate_distance<stage, cache_mode, StaticPatch>(
            shape, search, anchor, current_window, candidate_x, candidate_y,
            reference, tile, tile_width, tile_plane_values);
    }
    __syncthreads();

    MatchKey thread_keys[ItemsPerThread];
#pragma unroll
    for (int item = 0; item < ItemsPerThread; ++item) {
        const int local = (thread * ItemsPerThread) + item;
        if (local < tile_candidates) {
            const int candidate_y = local / candidate_tile.width;
            const int candidate_x =
                local - (candidate_y * candidate_tile.width);
            const std::uint32_t candidate_id = static_cast<std::uint32_t>(
                (candidate_tile.slot * window_width * window_height) +
                ((candidate_tile.y_begin + candidate_y) * window_width) +
                candidate_tile.x_begin + candidate_x);
            thread_keys[item] = make_match_key(distances[local], candidate_id);
        } else {
            thread_keys[item] = padding_match_key;
        }
    }
    __syncthreads();

    auto& sort_storage =
        *reinterpret_cast<typename BlockSort::TempStorage*>(shared_raw);
    BlockSort(sort_storage).SortBlockedToStriped(thread_keys);
    const int task_output = partial_offsets[task];
    const int task_retained = partial_offsets[task + 1] - task_output;
    const std::size_t output_base =
        static_cast<std::size_t>(group) * partial_per_group + task_output;
#pragma unroll
    for (int item = 0; item < ItemsPerThread; ++item) {
        const int rank = (item * block_threads) + thread;
        if (rank < task_retained) {
            partial_keys[output_base + rank] = thread_keys[item];
        }
    }
}

template <Stage stage, CacheMode cache_mode, int StaticPatch>
__global__ void
match_distance_kernel(MatchBatchShape shape, DeviceMatchBatch batch,
                      TileLayout layout, int temporal_count, int window_width,
                      int window_height, int candidate_count, MatchKey* keys) {
    extern __shared__ __align__(16) unsigned char shared_raw[];
    __shared__ SearchWindow current_window;
    __shared__ PatchOrigin anchor;

    const int flat_task = static_cast<int>(blockIdx.x);
    const int group = flat_task / layout.total_tiles;
    const int task = flat_task - (group * layout.total_tiles);
    const int thread = static_cast<int>(threadIdx.x);
    if (group >= shape.groups) {
        return;
    }
    if (thread == 0) {
        anchor = batch.anchors[group];
    }
    __syncthreads();

    const DeviceTile candidate_tile =
        make_tile(layout, window_width, window_height, task);
    if (thread == 0) {
        current_window = make_window(shape, batch, anchor, group,
                                     candidate_tile.slot, temporal_count);
        current_window.x_low += candidate_tile.x_begin;
        current_window.y_low += candidate_tile.y_begin;
    }
    __syncthreads();

    const int patch = patch_size<StaticPatch>(shape);
    const int patch_area = patch * patch;
    const int match_channels = stage == Stage::Basic ? 1 : shape.channels;
    const int match_planes = match_channels * shape.patch_time;
    const int match_dim = match_planes * patch_area;
    const int tile_width = candidate_tile.width + patch - 1;
    const int tile_height = candidate_tile.height + patch - 1;
    const int tile_plane_values = tile_width * tile_height;
    const int tile_values = match_planes * tile_plane_values;
    const int tile_candidates = candidate_tile.width * candidate_tile.height;
    const DeviceVideoView search = matching_source<stage>(batch);

    float* reference = nullptr;
    float* tile = nullptr;
    if constexpr (cache_mode == CacheMode::Tile) {
        reference = reinterpret_cast<float*>(shared_raw);
        tile = reference + match_dim;
    } else if constexpr (cache_mode == CacheMode::Reference) {
        reference = reinterpret_cast<float*>(shared_raw);
    }

    if constexpr (cache_mode != CacheMode::Direct) {
        for (int index = thread; index < match_dim; index += block_threads) {
            const int plane = index / patch_area;
            const int position = index - (plane * patch_area);
            const int channel =
                stage == Stage::Basic ? 0 : plane / shape.patch_time;
            const int frame_delta = plane % shape.patch_time;
            const int py = position / patch;
            const int px = position - (py * patch);
            reference[index] =
                video_sample(search, anchor.frame + frame_delta, channel,
                             anchor.y + py, anchor.x + px);
        }
    }
    if constexpr (cache_mode == CacheMode::Tile) {
        for (int index = thread; index < tile_values; index += block_threads) {
            const int plane = index / tile_plane_values;
            const int position = index - (plane * tile_plane_values);
            const int channel =
                stage == Stage::Basic ? 0 : plane / shape.patch_time;
            const int frame_delta = plane % shape.patch_time;
            const int ty = position / tile_width;
            const int tx = position - (ty * tile_width);
            tile[index] = video_sample(
                search, current_window.frame + frame_delta, channel,
                current_window.y_low + ty, current_window.x_low + tx);
        }
    }
    if constexpr (cache_mode != CacheMode::Direct) {
        __syncthreads();
    }

    const std::size_t group_base =
        static_cast<std::size_t>(group) * candidate_count;
    for (int local = thread; local < tile_candidates; local += block_threads) {
        const int candidate_y = local / candidate_tile.width;
        const int candidate_x = local - (candidate_y * candidate_tile.width);
        const std::uint32_t candidate_id = static_cast<std::uint32_t>(
            (candidate_tile.slot * window_width * window_height) +
            ((candidate_tile.y_begin + candidate_y) * window_width) +
            candidate_tile.x_begin + candidate_x);
        const float distance =
            candidate_distance<stage, cache_mode, StaticPatch>(
                shape, search, anchor, current_window, candidate_x, candidate_y,
                reference, tile, tile_width, tile_plane_values);
        keys[group_base + candidate_id] =
            make_match_key(distance, candidate_id);
    }
}

template <int ItemsPerThread>
__global__ void merge_partial_kernel(const MatchKey* partial_keys, int groups,
                                     int partial_per_group, int retained_stride,
                                     MatchKey* selected_keys) {
    using BlockSort =
        cub::BlockRadixSort<MatchKey, block_threads, ItemsPerThread>;
    __shared__ typename BlockSort::TempStorage sort_storage;
    const int group = static_cast<int>(blockIdx.x);
    const int thread = static_cast<int>(threadIdx.x);
    if (group >= groups) {
        return;
    }
    const std::size_t input_base =
        static_cast<std::size_t>(group) * partial_per_group;
    MatchKey thread_keys[ItemsPerThread];
#pragma unroll
    for (int item = 0; item < ItemsPerThread; ++item) {
        const int rank = (thread * ItemsPerThread) + item;
        if (rank < partial_per_group) {
            thread_keys[item] = partial_keys[input_base + rank];
        } else {
            thread_keys[item] = padding_match_key;
        }
    }
    BlockSort(sort_storage).SortBlockedToStriped(thread_keys);
    const std::size_t output_base =
        static_cast<std::size_t>(group) * retained_stride;
#pragma unroll
    for (int item = 0; item < ItemsPerThread; ++item) {
        const int rank = (item * block_threads) + thread;
        if (rank < retained_stride) {
            selected_keys[output_base + rank] = thread_keys[item];
        }
    }
}

template <Stage stage, int StaticPatch>
__global__ void
selected_gather_kernel(MatchBatchShape shape, MatchParameters parameters,
                       DeviceMatchBatch batch, int temporal_count,
                       int window_width, int window_height, int candidate_count,
                       int sample_dim, const MatchKey* selected_keys,
                       int selected_stride) {
    __shared__ PatchOrigin anchor;
    __shared__ int retained;
    const int group = static_cast<int>(blockIdx.x);
    const int thread = static_cast<int>(threadIdx.x);
    if (group >= shape.groups) {
        return;
    }
    if (thread == 0) {
        anchor = batch.anchors[group];
        const std::size_t selected_base =
            static_cast<std::size_t>(group) * selected_stride;
        const int effective_similar =
            min(shape.requested_similar, candidate_count);
        const float kth = match_key_distance(
            selected_keys[selected_base + effective_similar - 1]);
        const float threshold = fmaxf(parameters.tau, kth);
        int count = effective_similar;
        while (count < shape.retained_stride &&
               match_key_distance(selected_keys[selected_base + count]) <=
                   threshold) {
            ++count;
        }
        retained = count;
        batch.retained_counts[group] = count;
    }
    __syncthreads();

    const int spatial_candidates = window_width * window_height;
    const std::size_t selected_base =
        static_cast<std::size_t>(group) * selected_stride;
    const std::size_t descriptor_base =
        static_cast<std::size_t>(group) * shape.retained_stride;
    for (int sample = thread; sample < retained; sample += block_threads) {
        const MatchKey key = selected_keys[selected_base + sample];
        const std::uint32_t candidate_id = match_key_candidate(key);
        const int slot = static_cast<int>(candidate_id) / spatial_candidates;
        const int local =
            static_cast<int>(candidate_id) - (slot * spatial_candidates);
        const int dy = local / window_width;
        const int dx = local - (dy * window_width);
        const SearchWindow window =
            make_window(shape, batch, anchor, group, slot, temporal_count);
        if (batch.candidate_ids != nullptr) {
            batch.candidate_ids[descriptor_base + sample] = candidate_id;
        }
        if (batch.matches != nullptr) {
            batch.matches[descriptor_base + sample] =
                PatchMatch{match_key_distance(key), window.x_low + dx,
                           window.y_low + dy, window.frame};
        }
    }

    constexpr int warp_threads = 32;
    constexpr int warps_per_block = block_threads / warp_threads;
    const int lane = thread & (warp_threads - 1);
    const int warp = thread / warp_threads;
    const int patch = patch_size<StaticPatch>(shape);
    const int patch_area = patch * patch;
    const int channel_patch_dim = shape.patch_time * patch_area;
    const std::size_t sample_group_base =
        static_cast<std::size_t>(group) * shape.retained_stride * sample_dim;
    for (int sample = warp; sample < retained; sample += warps_per_block) {
        int origin_x = 0;
        int origin_y = 0;
        int origin_frame = 0;
        if (lane == 0) {
            const std::uint32_t candidate_id =
                match_key_candidate(selected_keys[selected_base + sample]);
            const int slot =
                static_cast<int>(candidate_id) / spatial_candidates;
            const int local =
                static_cast<int>(candidate_id) - (slot * spatial_candidates);
            const int dy = local / window_width;
            const int dx = local - (dy * window_width);
            const SearchWindow window =
                make_window(shape, batch, anchor, group, slot, temporal_count);
            origin_x = window.x_low + dx;
            origin_y = window.y_low + dy;
            origin_frame = window.frame;
        }
        origin_x = __shfl_sync(0xffffffffU, origin_x, 0);
        origin_y = __shfl_sync(0xffffffffU, origin_y, 0);
        origin_frame = __shfl_sync(0xffffffffU, origin_frame, 0);
        const std::size_t output_base =
            sample_group_base + (static_cast<std::size_t>(sample) * sample_dim);
        for (int channel = 0; channel < shape.channels; ++channel) {
            const std::size_t channel_output =
                output_base +
                (static_cast<std::size_t>(channel) * channel_patch_dim);
            for (int frame_delta = 0; frame_delta < shape.patch_time;
                 ++frame_delta) {
                const std::size_t plane_output =
                    channel_output +
                    (static_cast<std::size_t>(frame_delta) * patch_area);
                for (int position = lane; position < patch_area;
                     position += warp_threads) {
                    const int py = position / patch;
                    const int px = position - (py * patch);
                    const std::size_t output = plane_output + position;
                    batch.noisy_samples[output] =
                        video_sample(batch.noisy, origin_frame + frame_delta,
                                     channel, origin_y + py, origin_x + px);
                    if constexpr (stage == Stage::Final) {
                        batch.basic_samples[output] = video_sample(
                            batch.basic, origin_frame + frame_delta, channel,
                            origin_y + py, origin_x + px);
                    }
                }
            }
        }
    }
}

template <int ItemsPerThread> std::size_t sort_storage_bytes() {
    using Sort = cub::BlockRadixSort<float, block_threads, ItemsPerThread,
                                     std::uint32_t>;
    return sizeof(typename Sort::TempStorage);
}

template <int ItemsPerThread> std::size_t key_sort_storage_bytes() {
    using Sort = cub::BlockRadixSort<MatchKey, block_threads, ItemsPerThread>;
    return sizeof(typename Sort::TempStorage);
}

template <int ItemsPerThread, CacheMode mode>
std::size_t required_shared_bytes(const MatchBatchShape& shape) {
    const int candidates = candidate_count_for(shape);
    const int match_dim = match_dimension_for(shape);
    const int window_width = window_width_for(shape);
    const int window_height = window_height_for(shape);
    const int tile_width = window_width + shape.patch_size - 1;
    const int tile_height = window_height + shape.patch_size - 1;
    const int match_planes =
        shape.stage == Stage::Basic
            ? shape.patch_time
            : checked_product_int(shape.channels, shape.patch_time,
                                  "match plane count overflows");
    const int tile_values =
        checked_product_int(match_planes,
                            checked_product_int(tile_width, tile_height,
                                                "search tile size overflows"),
                            "search tile size overflows");

    std::size_t distance_values = static_cast<std::size_t>(candidates);
    if constexpr (mode == CacheMode::Tile) {
        distance_values += static_cast<std::size_t>(match_dim) + tile_values;
    } else if constexpr (mode == CacheMode::Reference) {
        distance_values += static_cast<std::size_t>(match_dim);
    }
    const std::size_t distance_bytes = checked_product(
        distance_values, sizeof(float), "match shared memory size overflows");
    const std::size_t post_sort_bytes = checked_product(
        static_cast<std::size_t>(shape.retained_stride), sizeof(SelectedOrigin),
        "selection shared memory size overflows");
    return std::max({distance_bytes, sort_storage_bytes<ItemsPerThread>(),
                     post_sort_bytes});
}

template <Stage stage, int ItemsPerThread, CacheMode mode, int StaticPatch>
bool configure_kernel(std::size_t shared_bytes, int default_shared,
                      int optin_shared) {
    cudaFuncAttributes attributes{};
    check_cuda(
        cudaFuncGetAttributes(&attributes,
                              match_select_gather_kernel<stage, ItemsPerThread,
                                                         mode, StaticPatch>),
        "cudaFuncGetAttributes(block matcher)");
    const std::size_t total =
        shared_bytes + static_cast<std::size_t>(attributes.sharedSizeBytes);
    if (total > static_cast<std::size_t>(optin_shared)) {
        return false;
    }
    if (total > static_cast<std::size_t>(default_shared)) {
        const int maximum_dynamic = optin_shared - attributes.sharedSizeBytes;
        check_cuda(cudaFuncSetAttribute(
                       match_select_gather_kernel<stage, ItemsPerThread, mode,
                                                  StaticPatch>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       maximum_dynamic),
                   "cudaFuncSetAttribute(block matcher shared memory)");
    }
    return true;
}

template <Stage stage, int ItemsPerThread, CacheMode mode, int StaticPatch>
bool configure_chunk_kernel(std::size_t shared_bytes, int default_shared,
                            int optin_shared) {
    cudaFuncAttributes attributes{};
    check_cuda(cudaFuncGetAttributes(
                   &attributes, chunk_match_select_kernel<stage, ItemsPerThread,
                                                          mode, StaticPatch>),
               "cudaFuncGetAttributes(chunked matcher)");
    const std::size_t total =
        shared_bytes + static_cast<std::size_t>(attributes.sharedSizeBytes);
    if (total > static_cast<std::size_t>(optin_shared)) {
        return false;
    }
    if (total > static_cast<std::size_t>(default_shared)) {
        check_cuda(cudaFuncSetAttribute(
                       chunk_match_select_kernel<stage, ItemsPerThread, mode,
                                                 StaticPatch>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       optin_shared - attributes.sharedSizeBytes),
                   "cudaFuncSetAttribute(chunked matcher shared memory)");
    }
    return true;
}

template <Stage stage, CacheMode mode, int StaticPatch>
bool configure_distance_kernel(std::size_t shared_bytes, int default_shared,
                               int optin_shared) {
    cudaFuncAttributes attributes{};
    check_cuda(
        cudaFuncGetAttributes(&attributes,
                              match_distance_kernel<stage, mode, StaticPatch>),
        "cudaFuncGetAttributes(matcher distance)");
    const std::size_t total =
        shared_bytes + static_cast<std::size_t>(attributes.sharedSizeBytes);
    if (total > static_cast<std::size_t>(optin_shared)) {
        return false;
    }
    if (total > static_cast<std::size_t>(default_shared)) {
        check_cuda(cudaFuncSetAttribute(
                       match_distance_kernel<stage, mode, StaticPatch>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       optin_shared - attributes.sharedSizeBytes),
                   "cudaFuncSetAttribute(matcher distance shared memory)");
    }
    return true;
}

template <int ItemsPerThread, CacheMode mode>
std::size_t tiled_shared_bytes(const MatchBatchShape& shape,
                               const TileLayout& layout,
                               bool include_distances) {
    const int match_dim = match_dimension_for(shape);
    const int match_planes =
        shape.stage == Stage::Basic
            ? shape.patch_time
            : checked_product_int(shape.channels, shape.patch_time,
                                  "match plane count overflows");
    const int tile_width = layout.candidate_width + shape.patch_size - 1;
    const int tile_height = layout.candidate_height + shape.patch_size - 1;
    const int tile_values = checked_product_int(
        match_planes,
        checked_product_int(tile_width, tile_height,
                            "CUDA matcher tile size overflows"),
        "CUDA matcher tile size overflows");
    std::size_t values = include_distances
                             ? static_cast<std::size_t>(layout.candidate_width *
                                                        layout.candidate_height)
                             : 0;
    if constexpr (mode == CacheMode::Tile) {
        values += static_cast<std::size_t>(match_dim) + tile_values;
    } else if constexpr (mode == CacheMode::Reference) {
        values += static_cast<std::size_t>(match_dim);
    }
    const std::size_t bytes = checked_product(
        values, sizeof(float), "CUDA matcher tiled shared memory overflows");
    return include_distances
               ? std::max(bytes, key_sort_storage_bytes<ItemsPerThread>())
               : bytes;
}

template <Stage stage, int ItemsPerThread, CacheMode mode, int StaticPatch>
void launch_match(const MatchBatchShape& shape,
                  const MatchParameters& parameters,
                  const DeviceMatchBatch& batch, std::size_t shared_bytes,
                  cudaStream_t stream) {
    const int temporal_count = temporal_count_for(shape);
    const int window_width = window_width_for(shape);
    const int window_height = window_height_for(shape);
    const int candidates = candidate_count_for(shape);
    const int sample_dim = patch_dimension_for(shape);
    match_select_gather_kernel<stage, ItemsPerThread, mode, StaticPatch>
        <<<static_cast<unsigned int>(shape.groups), block_threads, shared_bytes,
           stream>>>(shape, parameters, batch, temporal_count, window_width,
                     window_height, candidates, sample_dim);
    check_cuda(cudaPeekAtLastError(), "launch fused block matcher");
}

template <Stage stage, int ItemsPerThread, CacheMode mode, int StaticPatch>
void launch_chunk_match(const MatchBatchShape& shape,
                        const DeviceMatchBatch& batch, const TileLayout& layout,
                        const int* partial_offsets, int partial_per_group,
                        MatchKey* partial_keys, std::size_t shared_bytes,
                        cudaStream_t stream) {
    const std::size_t blocks =
        checked_product(static_cast<std::size_t>(shape.groups),
                        static_cast<std::size_t>(layout.total_tiles),
                        "CUDA chunked matcher grid overflows");
    if (blocks > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::length_error("CUDA chunked matcher grid is too large");
    }
    chunk_match_select_kernel<stage, ItemsPerThread, mode, StaticPatch>
        <<<static_cast<unsigned int>(blocks), block_threads, shared_bytes,
           stream>>>(shape, batch, layout, temporal_count_for(shape),
                     window_width_for(shape), window_height_for(shape),
                     partial_offsets, partial_per_group, partial_keys);
    check_cuda(cudaPeekAtLastError(), "launch chunked block matcher");
}

template <Stage stage, CacheMode mode, int StaticPatch>
void launch_match_distances(const MatchBatchShape& shape,
                            const DeviceMatchBatch& batch,
                            const TileLayout& layout, MatchKey* keys,
                            std::size_t shared_bytes, cudaStream_t stream) {
    const std::size_t blocks =
        checked_product(static_cast<std::size_t>(shape.groups),
                        static_cast<std::size_t>(layout.total_tiles),
                        "CUDA matcher distance grid overflows");
    if (blocks > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::length_error("CUDA matcher distance grid is too large");
    }
    match_distance_kernel<stage, mode, StaticPatch>
        <<<static_cast<unsigned int>(blocks), block_threads, shared_bytes,
           stream>>>(shape, batch, layout, temporal_count_for(shape),
                     window_width_for(shape), window_height_for(shape),
                     candidate_count_for(shape), keys);
    check_cuda(cudaPeekAtLastError(), "launch matcher distance kernel");
}

template <int ItemsPerThread>
void launch_partial_merge(const MatchBatchShape& shape,
                          const MatchKey* partial_keys, int partial_per_group,
                          MatchKey* selected_keys, cudaStream_t stream) {
    merge_partial_kernel<ItemsPerThread>
        <<<static_cast<unsigned int>(shape.groups), block_threads, 0, stream>>>(
            partial_keys, shape.groups, partial_per_group,
            shape.retained_stride, selected_keys);
    check_cuda(cudaPeekAtLastError(), "launch matcher partial merge");
}

template <Stage stage, int StaticPatch>
void launch_selected_gather(const MatchBatchShape& shape,
                            const MatchParameters& parameters,
                            const DeviceMatchBatch& batch,
                            const MatchKey* selected_keys, int selected_stride,
                            cudaStream_t stream) {
    selected_gather_kernel<stage, StaticPatch>
        <<<static_cast<unsigned int>(shape.groups), block_threads, 0, stream>>>(
            shape, parameters, batch, temporal_count_for(shape),
            window_width_for(shape), window_height_for(shape),
            candidate_count_for(shape), patch_dimension_for(shape),
            selected_keys, selected_stride);
    check_cuda(cudaPeekAtLastError(), "launch matcher selected gather");
}

template <Stage stage, int StaticPatch>
void launch_single(const MatchBatchShape& shape, const DeviceMatchBatch& batch,
                   cudaStream_t stream) {
    single_match_gather_kernel<stage, StaticPatch>
        <<<static_cast<unsigned int>(shape.groups), block_threads, 0, stream>>>(
            shape, batch, window_width_for(shape), window_height_for(shape),
            patch_dimension_for(shape));
    check_cuda(cudaPeekAtLastError(), "launch single-patch gather");
}

template <typename Function>
void dispatch_items(int items, Function&& function) {
    switch (items) {
    case 1:
        function(std::integral_constant<int, 1>{});
        break;
    case 2:
        function(std::integral_constant<int, 2>{});
        break;
    case 3:
        function(std::integral_constant<int, 3>{});
        break;
    case 4:
        function(std::integral_constant<int, 4>{});
        break;
    case 5:
        function(std::integral_constant<int, 5>{});
        break;
    case 6:
        function(std::integral_constant<int, 6>{});
        break;
    case 7:
        function(std::integral_constant<int, 7>{});
        break;
    case 8:
        function(std::integral_constant<int, 8>{});
        break;
    case 9:
        function(std::integral_constant<int, 9>{});
        break;
    default:
        throw std::logic_error("unsupported CUDA matcher item count");
    }
}

template <typename Function>
void dispatch_stage(Stage stage, Function&& function) {
    if (stage == Stage::Basic) {
        function(std::integral_constant<Stage, Stage::Basic>{});
    } else {
        function(std::integral_constant<Stage, Stage::Final>{});
    }
}

template <typename Function>
void dispatch_patch(int patch, Function&& function) {
    if (patch == 7) {
        function(std::integral_constant<int, 7>{});
    } else if (patch == 8) {
        function(std::integral_constant<int, 8>{});
    } else if (patch == 10) {
        function(std::integral_constant<int, 10>{});
    } else {
        function(std::integral_constant<int, 0>{});
    }
}

template <typename Function>
void dispatch_cache(CacheMode mode, Function&& function) {
    switch (mode) {
    case CacheMode::Tile:
        function(std::integral_constant<CacheMode, CacheMode::Tile>{});
        break;
    case CacheMode::Reference:
        function(std::integral_constant<CacheMode, CacheMode::Reference>{});
        break;
    case CacheMode::Direct:
        function(std::integral_constant<CacheMode, CacheMode::Direct>{});
        break;
    }
}

} // namespace

class BlockMatcher::Impl {
  public:
    explicit Impl(int requested_device) {
        if (requested_device < 0) {
            check_cuda(cudaGetDevice(&device_), "cudaGetDevice");
        } else {
            device_ = requested_device;
            check_cuda(cudaSetDevice(device_), "cudaSetDevice");
        }
        check_cuda(cudaDeviceGetAttribute(&default_shared_memory_,
                                          cudaDevAttrMaxSharedMemoryPerBlock,
                                          device_),
                   "cudaDeviceGetAttribute(default shared memory)");
        check_cuda(cudaDeviceGetAttribute(
                       &optin_shared_memory_,
                       cudaDevAttrMaxSharedMemoryPerBlockOptin, device_),
                   "cudaDeviceGetAttribute(opt-in shared memory)");
        optin_shared_memory_ =
            std::max(optin_shared_memory_, default_shared_memory_);
    }

    ~Impl() { (void)cudaSetDevice(device_); }

    void reserve(const MatchBatchShape& shape) {
        validate_shape(shape);
        check_cuda(cudaSetDevice(device_), "cudaSetDevice");
        reserved_ = false;
        reset_workspace();
        shape_ = shape;
        temporal_count_ = temporal_count_for(shape);
        candidate_count_ = candidate_count_for(shape);
        sample_dim_ = patch_dimension_for(shape);
        tile_layout_ = tile_layout_for(shape);
        build_partial_offsets(shape);

        if (shape.requested_similar == 1 && shape.retained_stride == 1 &&
            (shape.strategy == MatchStrategy::Auto ||
             shape.strategy == MatchStrategy::Fused)) {
            strategy_ = MatchStrategy::Fused;
            items_per_thread_ = 0;
            merge_items_per_thread_ = 0;
            shared_bytes_ = 0;
            cache_mode_ = CacheMode::Direct;
            reserved_ = true;
            return;
        }

        const bool chunk_feasible =
            partial_per_group_ <= tile_candidate_capacity;
        strategy_ = shape.strategy;
        if (strategy_ == MatchStrategy::Auto) {
            if (candidate_count_ <= auto_fused_candidate_threshold) {
                strategy_ = MatchStrategy::Fused;
            } else if (chunk_feasible &&
                       auto_chunk_geometry(shape, tile_layout_,
                                           candidate_count_,
                                           partial_per_group_)) {
                strategy_ = MatchStrategy::Chunked;
            } else {
                strategy_ = MatchStrategy::FullSort;
            }
        }

        if (strategy_ == MatchStrategy::Fused &&
            candidate_count_ > fused_candidate_capacity) {
            throw std::invalid_argument(
                "forced fused CUDA matcher exceeds its 2304-candidate "
                "capacity");
        }
        if (strategy_ == MatchStrategy::Chunked && !chunk_feasible) {
            throw std::invalid_argument(
                "forced chunked CUDA matcher exceeds its 2304-partial merge "
                "capacity; use full-sort");
        }

        bool configured =
            strategy_ == MatchStrategy::Fused
                ? configure_fused(shape)
                : configure_tiled(shape, strategy_ == MatchStrategy::Chunked);
        if (!configured && shape.strategy == MatchStrategy::Auto &&
            strategy_ == MatchStrategy::Fused && chunk_feasible) {
            strategy_ = MatchStrategy::Chunked;
            configured = configure_tiled(shape, true);
        }
        if (!configured && shape.strategy == MatchStrategy::Auto &&
            strategy_ != MatchStrategy::FullSort) {
            strategy_ = MatchStrategy::FullSort;
            configured = configure_tiled(shape, false);
        }
        if (!configured) {
            throw std::runtime_error(
                "CUDA matcher does not fit device shared memory");
        }

        if (strategy_ == MatchStrategy::Chunked) {
            reserve_chunked(shape);
        } else if (strategy_ == MatchStrategy::FullSort) {
            reserve_full_sort(shape);
        }
        reserved_ = true;
    }

    bool configure_fused(const MatchBatchShape& shape) {
        items_per_thread_ = divide_round_up(candidate_count_, block_threads);
        bool configured = false;
        dispatch_stage(shape.stage, [&](auto stage_tag) {
            constexpr Stage stage = decltype(stage_tag)::value;
            dispatch_items(items_per_thread_, [&](auto items_tag) {
                constexpr int items = decltype(items_tag)::value;
                dispatch_patch(shape.patch_size, [&](auto patch_tag) {
                    constexpr int static_patch = decltype(patch_tag)::value;
                    for (CacheMode candidate :
                         {CacheMode::Tile, CacheMode::Reference,
                          CacheMode::Direct}) {
                        if (configured) {
                            break;
                        }
                        dispatch_cache(candidate, [&](auto cache_tag) {
                            constexpr CacheMode cache =
                                decltype(cache_tag)::value;
                            const std::size_t bytes =
                                required_shared_bytes<items, cache>(shape);
                            if (configure_kernel<stage, items, cache,
                                                 static_patch>(
                                    bytes, default_shared_memory_,
                                    optin_shared_memory_)) {
                                cache_mode_ = cache;
                                shared_bytes_ = bytes;
                                configured = true;
                            }
                        });
                    }
                });
            });
        });
        return configured;
    }

    bool configure_tiled(const MatchBatchShape& shape, bool chunked) {
        tile_items_per_thread_ = divide_round_up(
            checked_product_int(tile_layout_.candidate_width,
                                tile_layout_.candidate_height,
                                "CUDA matcher tile area overflows"),
            block_threads);
        bool configured = false;
        dispatch_stage(shape.stage, [&](auto stage_tag) {
            constexpr Stage stage = decltype(stage_tag)::value;
            dispatch_patch(shape.patch_size, [&](auto patch_tag) {
                constexpr int static_patch = decltype(patch_tag)::value;
                for (CacheMode candidate :
                     {CacheMode::Tile, CacheMode::Reference,
                      CacheMode::Direct}) {
                    if (configured) {
                        break;
                    }
                    dispatch_cache(candidate, [&](auto cache_tag) {
                        constexpr CacheMode cache = decltype(cache_tag)::value;
                        if (chunked) {
                            dispatch_items(
                                tile_items_per_thread_, [&](auto items_tag) {
                                    constexpr int items =
                                        decltype(items_tag)::value;
                                    const std::size_t bytes =
                                        tiled_shared_bytes<items, cache>(
                                            shape, tile_layout_, true);
                                    if (configure_chunk_kernel<
                                            stage, items, cache, static_patch>(
                                            bytes, default_shared_memory_,
                                            optin_shared_memory_)) {
                                        cache_mode_ = cache;
                                        shared_bytes_ = bytes;
                                        configured = true;
                                    }
                                });
                        } else {
                            const std::size_t bytes =
                                tiled_shared_bytes<1, cache>(
                                    shape, tile_layout_, false);
                            if (configure_distance_kernel<stage, cache,
                                                          static_patch>(
                                    bytes, default_shared_memory_,
                                    optin_shared_memory_)) {
                                cache_mode_ = cache;
                                shared_bytes_ = bytes;
                                configured = true;
                            }
                        }
                    });
                }
            });
        });
        return configured;
    }

    void build_partial_offsets(const MatchBatchShape& shape) {
        host_partial_offsets_.assign(
            static_cast<std::size_t>(tile_layout_.total_tiles) + 1, 0);
        int total = 0;
        for (int task = 0; task < tile_layout_.total_tiles; ++task) {
            total = checked_add(
                total,
                std::min(shape.retained_stride,
                         tile_candidate_count(shape, tile_layout_, task)),
                "CUDA matcher partial selection count overflows");
            host_partial_offsets_[static_cast<std::size_t>(task) + 1] = total;
        }
        if (total < shape.retained_stride) {
            throw std::logic_error(
                "CUDA matcher partial selection does not cover retained cap");
        }
        partial_per_group_ = total;
    }

    void reserve_chunked(const MatchBatchShape& shape) {
        merge_items_per_thread_ =
            divide_round_up(partial_per_group_, block_threads);
        const std::size_t partial_values =
            checked_product(static_cast<std::size_t>(shape.groups),
                            static_cast<std::size_t>(partial_per_group_),
                            "CUDA matcher partial workspace overflows");
        const std::size_t selected_values =
            checked_product(static_cast<std::size_t>(shape.groups),
                            static_cast<std::size_t>(shape.retained_stride),
                            "CUDA matcher selected workspace overflows");
        partial_offsets_.reserve(host_partial_offsets_.size(),
                                 "CUDA matcher partial offsets overflow");
        partial_keys_.reserve(partial_values,
                              "CUDA matcher partial keys overflow");
        selected_keys_.reserve(selected_values,
                               "CUDA matcher selected keys overflow");
        check_cuda(cudaMemcpy(partial_offsets_.data(),
                              host_partial_offsets_.data(),
                              host_partial_offsets_.size() * sizeof(int),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy(CUDA matcher partial offsets)");
    }

    void reserve_full_sort(const MatchBatchShape& shape) {
        merge_items_per_thread_ = 0;
        const std::size_t total_items =
            checked_product(static_cast<std::size_t>(shape.groups),
                            static_cast<std::size_t>(candidate_count_),
                            "CUDA matcher full-sort workspace overflows");
        if (total_items > static_cast<std::size_t>(
                              std::numeric_limits<std::int64_t>::max())) {
            throw std::length_error(
                "CUDA matcher full-sort item count exceeds int64");
        }
        full_keys_in_.reserve(total_items,
                              "CUDA matcher full-sort keys overflow");
        full_keys_out_.reserve(total_items,
                               "CUDA matcher full-sort keys overflow");

        host_segment_offsets_.resize(static_cast<std::size_t>(shape.groups) +
                                     1);
        for (int group = 0; group <= shape.groups; ++group) {
            host_segment_offsets_[static_cast<std::size_t>(group)] =
                static_cast<std::int64_t>(group) * candidate_count_;
        }
        segment_offsets_.reserve(host_segment_offsets_.size(),
                                 "CUDA matcher segment offsets overflow");
        check_cuda(
            cudaMemcpy(segment_offsets_.data(), host_segment_offsets_.data(),
                       host_segment_offsets_.size() * sizeof(std::int64_t),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy(CUDA matcher segment offsets)");

        full_sort_temp_bytes_ = 0;
        check_cuda(cub::DeviceSegmentedRadixSort::SortKeys(
                       nullptr, full_sort_temp_bytes_, full_keys_in_.data(),
                       full_keys_out_.data(),
                       static_cast<std::int64_t>(total_items),
                       static_cast<std::int64_t>(shape.groups),
                       segment_offsets_.data(), segment_offsets_.data() + 1, 0,
                       static_cast<int>(sizeof(MatchKey) * 8)),
                   "query CUDA matcher segmented radix sort workspace");
        full_sort_temp_.reserve(
            full_sort_temp_bytes_,
            "CUDA matcher segmented radix sort workspace overflows");
    }

    void reset_workspace() {
        partial_offsets_.reset();
        partial_keys_.reset();
        selected_keys_.reset();
        segment_offsets_.reset();
        full_keys_in_.reset();
        full_keys_out_.reset();
        full_sort_temp_.reset();
        host_partial_offsets_.clear();
        host_segment_offsets_.clear();
        partial_per_group_ = 0;
        full_sort_temp_bytes_ = 0;
    }

    void enqueue(const MatchBatchShape& shape,
                 const MatchParameters& parameters,
                 const DeviceMatchBatch& batch, cudaStream_t stream) {
        validate_shape(shape);
        validate_parameters(parameters);
        validate_batch(shape, batch);
        if (!reserved_ || shape.groups > shape_.groups ||
            !same_reserved_geometry(shape, shape_)) {
            throw std::logic_error(
                "CUDA BlockMatcher is not reserved for this shape");
        }
        check_cuda(cudaSetDevice(device_), "cudaSetDevice");

        dispatch_stage(shape.stage, [&](auto stage_tag) {
            constexpr Stage stage = decltype(stage_tag)::value;
            dispatch_patch(shape.patch_size, [&](auto patch_tag) {
                constexpr int static_patch = decltype(patch_tag)::value;
                if (strategy_ == MatchStrategy::Fused &&
                    shape.requested_similar == 1 &&
                    (shape.retained_stride == 1 || parameters.tau == 0.0F)) {
                    launch_single<stage, static_patch>(shape, batch, stream);
                    return;
                }
                if (strategy_ == MatchStrategy::Fused) {
                    dispatch_items(items_per_thread_, [&](auto items_tag) {
                        constexpr int items = decltype(items_tag)::value;
                        dispatch_cache(cache_mode_, [&](auto cache_tag) {
                            constexpr CacheMode cache =
                                decltype(cache_tag)::value;
                            launch_match<stage, items, cache, static_patch>(
                                shape, parameters, batch, shared_bytes_,
                                stream);
                        });
                    });
                    return;
                }
                if (strategy_ == MatchStrategy::Chunked) {
                    dispatch_items(tile_items_per_thread_, [&](auto items_tag) {
                        constexpr int items = decltype(items_tag)::value;
                        dispatch_cache(cache_mode_, [&](auto cache_tag) {
                            constexpr CacheMode cache =
                                decltype(cache_tag)::value;
                            launch_chunk_match<stage, items, cache,
                                               static_patch>(
                                shape, batch, tile_layout_,
                                partial_offsets_.data(), partial_per_group_,
                                partial_keys_.data(), shared_bytes_, stream);
                        });
                    });
                    dispatch_items(
                        merge_items_per_thread_, [&](auto items_tag) {
                            constexpr int items = decltype(items_tag)::value;
                            launch_partial_merge<items>(
                                shape, partial_keys_.data(), partial_per_group_,
                                selected_keys_.data(), stream);
                        });
                    launch_selected_gather<stage, static_patch>(
                        shape, parameters, batch, selected_keys_.data(),
                        shape.retained_stride, stream);
                    return;
                }

                dispatch_cache(cache_mode_, [&](auto cache_tag) {
                    constexpr CacheMode cache = decltype(cache_tag)::value;
                    launch_match_distances<stage, cache, static_patch>(
                        shape, batch, tile_layout_, full_keys_in_.data(),
                        shared_bytes_, stream);
                });
                std::size_t temp_bytes = full_sort_temp_bytes_;
                const std::int64_t total_items =
                    static_cast<std::int64_t>(shape.groups) * candidate_count_;
                check_cuda(
                    cub::DeviceSegmentedRadixSort::SortKeys(
                        full_sort_temp_.data(), temp_bytes,
                        full_keys_in_.data(), full_keys_out_.data(),
                        total_items, static_cast<std::int64_t>(shape.groups),
                        segment_offsets_.data(), segment_offsets_.data() + 1, 0,
                        static_cast<int>(sizeof(MatchKey) * 8), stream),
                    "run CUDA matcher segmented radix sort");
                launch_selected_gather<stage, static_patch>(
                    shape, parameters, batch, full_keys_out_.data(),
                    candidate_count_, stream);
            });
        });
    }

    [[nodiscard]] std::size_t workspace_bytes() const noexcept {
        return partial_offsets_.bytes() + partial_keys_.bytes() +
               selected_keys_.bytes() + segment_offsets_.bytes() +
               full_keys_in_.bytes() + full_keys_out_.bytes() +
               full_sort_temp_.bytes();
    }

    int device_ = 0;
    int default_shared_memory_ = 0;
    int optin_shared_memory_ = 0;
    MatchBatchShape shape_{};
    bool reserved_ = false;
    int temporal_count_ = 0;
    int candidate_count_ = 0;
    int sample_dim_ = 0;
    int items_per_thread_ = 0;
    int tile_items_per_thread_ = 0;
    int merge_items_per_thread_ = 0;
    int partial_per_group_ = 0;
    MatchStrategy strategy_ = MatchStrategy::Auto;
    TileLayout tile_layout_{};
    CacheMode cache_mode_ = CacheMode::Direct;
    std::size_t shared_bytes_ = 0;
    std::size_t full_sort_temp_bytes_ = 0;
    DeviceBuffer<int> partial_offsets_;
    DeviceBuffer<MatchKey> partial_keys_;
    DeviceBuffer<MatchKey> selected_keys_;
    DeviceBuffer<std::int64_t> segment_offsets_;
    DeviceBuffer<MatchKey> full_keys_in_;
    DeviceBuffer<MatchKey> full_keys_out_;
    DeviceBuffer<std::byte> full_sort_temp_;
    std::vector<int> host_partial_offsets_;
    std::vector<std::int64_t> host_segment_offsets_;
};

BlockMatcher::BlockMatcher(int device)
    : impl_(std::make_unique<Impl>(device)) {}

BlockMatcher::~BlockMatcher() = default;

BlockMatcher::BlockMatcher(BlockMatcher&&) noexcept = default;

BlockMatcher& BlockMatcher::operator=(BlockMatcher&&) noexcept = default;

void BlockMatcher::reserve(const MatchBatchShape& shape) {
    if (impl_ == nullptr) {
        throw std::logic_error("CUDA BlockMatcher was moved from");
    }
    impl_->reserve(shape);
}

void BlockMatcher::enqueue(const MatchBatchShape& shape,
                           const MatchParameters& parameters,
                           const DeviceMatchBatch& batch, cudaStream_t stream) {
    if (impl_ == nullptr) {
        throw std::logic_error("CUDA BlockMatcher was moved from");
    }
    impl_->enqueue(shape, parameters, batch, stream);
}

int BlockMatcher::candidate_count() const noexcept {
    return impl_ == nullptr ? 0 : impl_->candidate_count_;
}

int BlockMatcher::temporal_count() const noexcept {
    return impl_ == nullptr ? 0 : impl_->temporal_count_;
}

int BlockMatcher::sample_dim() const noexcept {
    return impl_ == nullptr ? 0 : impl_->sample_dim_;
}

MatchStrategy BlockMatcher::strategy() const noexcept {
    return impl_ == nullptr ? MatchStrategy::Auto : impl_->strategy_;
}

std::size_t BlockMatcher::workspace_bytes() const noexcept {
    return impl_ == nullptr ? 0 : impl_->workspace_bytes();
}

int BlockMatcher::device() const noexcept {
    return impl_ == nullptr ? -1 : impl_->device_;
}

} // namespace vnlbcu
