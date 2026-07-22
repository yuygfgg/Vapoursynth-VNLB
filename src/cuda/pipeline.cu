#include "cuda/pipeline.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace vnlbcu {
namespace {

constexpr int block_threads = 256;
constexpr int warp_width = 32;
constexpr int warps_per_block = block_threads / warp_width;
constexpr int max_match_candidates = 2048;

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
    if (result < 0 || result > std::numeric_limits<int>::max()) {
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

int patch_dim_for(const StagePipelineShape& shape) {
    return checked_product_int(
        checked_product_int(shape.patch_size, shape.patch_size,
                            "CUDA pipeline patch area overflows"),
        shape.patch_time, "CUDA pipeline patch dimension overflows");
}

int sample_dim_for(const StagePipelineShape& shape) {
    return checked_product_int(patch_dim_for(shape), shape.channels,
                               "CUDA pipeline sample dimension overflows");
}

int temporal_count_for(const StagePipelineShape& shape) {
    const int requested = checked_add_int(
        checked_add_int(shape.search_bwd, shape.search_fwd,
                        "CUDA pipeline temporal span overflows"),
        1, "CUDA pipeline temporal span overflows");
    const int temporal_origins = checked_add_int(
        shape.source_frames, 1 - shape.patch_time,
        "CUDA pipeline temporal origin count overflows");
    return std::min(requested, temporal_origins);
}

int candidate_count_for(const StagePipelineShape& shape) {
    const int x_origins = checked_add_int(
        shape.width, 1 - shape.patch_size,
        "CUDA pipeline horizontal origin count overflows");
    const int y_origins = checked_add_int(
        shape.height, 1 - shape.patch_size,
        "CUDA pipeline vertical origin count overflows");
    const int window_width = std::min(shape.search_window, x_origins);
    const int window_height = std::min(shape.search_window, y_origins);
    return checked_product_int(
        checked_product_int(window_width, window_height,
                            "CUDA pipeline spatial candidates overflow"),
        temporal_count_for(shape),
        "CUDA pipeline candidate count overflows");
}

int cached_frames_for(const StagePipelineShape& shape) {
    return checked_add_int(temporal_count_for(shape), shape.patch_time - 1,
                           "CUDA pipeline frame window overflows");
}

void validate_stage(Stage stage) {
    if (stage != Stage::Basic && stage != Stage::Final) {
        throw std::invalid_argument("invalid CUDA pipeline stage");
    }
}

void validate_shape(const StagePipelineShape& shape) {
    validate_stage(shape.stage);
    if (shape.max_groups <= 0 || shape.width <= 0 || shape.height <= 0 ||
        shape.channels <= 0 || shape.source_frames <= 0) {
        throw std::invalid_argument(
            "CUDA pipeline geometry and chunk size must be positive");
    }
    if (!shape.couple_channels) {
        throw std::invalid_argument(
            "CUDA pipeline currently requires coupled channel models");
    }
    if (shape.patch_size <= 0 || shape.patch_size > shape.width ||
        shape.patch_size > shape.height || shape.patch_time <= 0 ||
        shape.patch_time > shape.source_frames) {
        throw std::invalid_argument("invalid CUDA pipeline patch geometry");
    }
    if (shape.search_window <= 0 || (shape.search_window & 1) == 0 ||
        shape.search_bwd < 0 || shape.search_fwd < 0) {
        throw std::invalid_argument("invalid CUDA pipeline search geometry");
    }
    if (shape.requested_similar <= 0 || shape.retained_stride <= 0 ||
        shape.basis_similar <= 0 || shape.rank < 0) {
        throw std::invalid_argument("invalid CUDA pipeline group geometry");
    }
    if (!std::isfinite(shape.model_cap_factor) ||
        shape.model_cap_factor != 1.0F) {
        throw std::invalid_argument(
            "CUDA pipeline currently requires model_cap_factor=1");
    }

    const int candidates = candidate_count_for(shape);
    if (candidates > max_match_candidates) {
        throw std::invalid_argument(
            "CUDA pipeline matcher supports at most 2048 candidates");
    }
    const int effective_similar = std::min(shape.requested_similar, candidates);
    if (shape.basis_similar != effective_similar) {
        throw std::invalid_argument(
            "CUDA pipeline currently requires fixed B=min(K,candidates); "
            "model_cap_factor must be 1");
    }
    if (shape.retained_stride < effective_similar ||
        shape.retained_stride > candidates) {
        throw std::invalid_argument(
            "CUDA pipeline retained stride does not cover the selected group");
    }
    if (shape.rank > shape.basis_similar) {
        throw std::invalid_argument(
            "CUDA pipeline rank must not exceed the fixed model count");
    }
    if (shape.basis_similar >= sample_dim_for(shape)) {
        throw std::invalid_argument(
            "CUDA pipeline currently requires the dual-PCA path (B < D)");
    }

    const int expected_slots = checked_add_int(
        checked_add_int(shape.search_bwd, shape.search_fwd,
                        "CUDA pipeline contribution slots overflow"),
        shape.patch_time, "CUDA pipeline contribution slots overflow");
    if (shape.contribution_slots != expected_slots) {
        throw std::invalid_argument(
            "CUDA pipeline contribution slot count is inconsistent");
    }
}

bool same_shape(const StagePipelineShape& left,
                const StagePipelineShape& right) noexcept {
    return left.stage == right.stage && left.max_groups == right.max_groups &&
           left.width == right.width && left.height == right.height &&
           left.channels == right.channels &&
           left.source_frames == right.source_frames &&
           left.patch_size == right.patch_size &&
           left.patch_time == right.patch_time &&
           left.search_window == right.search_window &&
           left.search_bwd == right.search_bwd &&
           left.search_fwd == right.search_fwd &&
           left.requested_similar == right.requested_similar &&
           left.retained_stride == right.retained_stride &&
           left.basis_similar == right.basis_similar &&
           left.rank == right.rank &&
           left.contribution_slots == right.contribution_slots &&
           left.model_cap_factor == right.model_cap_factor &&
           left.couple_channels == right.couple_channels;
}

void validate_parameters(const StagePipelineShape& shape,
                         const StagePipelineParameters& parameters) {
    if (parameters.filter.stage != shape.stage) {
        throw std::invalid_argument(
            "CUDA pipeline filter stage does not match its reserved stage");
    }
    if (parameters.flat_areas && shape.stage != Stage::Final) {
        throw std::invalid_argument(
            "flat-area centering is only available in the Final stage");
    }
    if (!std::isfinite(parameters.flat_gamma) ||
        parameters.flat_gamma <= 0.0F) {
        throw std::invalid_argument(
            "CUDA pipeline flat gamma must be finite and positive");
    }
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

    void reserve(std::size_t count) {
        if (count <= capacity_) {
            return;
        }
        T* replacement = nullptr;
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&replacement),
                              checked_product(count, sizeof(T),
                                              "CUDA pipeline buffer overflows")),
                   "cudaMalloc(CUDA pipeline buffer)");
        if (data_ != nullptr) {
            check_cuda(cudaFree(data_), "cudaFree(CUDA pipeline buffer)");
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

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = warp_width / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    return value;
}

__global__ void set_anchor_frame_kernel(PatchOrigin* anchors, int groups,
                                        int frame) {
    const int group = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (group < groups) {
        anchors[group].frame = frame;
    }
}

// The flat-area test is intentionally separated from GroupFilter.  It is
// optional, reads the gathered group once, and leaves the common Basic/Final
// PCA kernels free of a branch and an unconditional variance pass.
__global__ void flat_flags_kernel(const float* __restrict__ noisy_samples,
                                  const int* __restrict__ retained_counts,
                                  int groups, int retained_stride,
                                  int channels, int patch_dim, float threshold,
                                  std::uint8_t* __restrict__ flat_flags) {
    __shared__ float warp_sums[warps_per_block];
    __shared__ float warp_squares[warps_per_block];
    __shared__ float variance_sum;

    const int group = static_cast<int>(blockIdx.x);
    const int thread = static_cast<int>(threadIdx.x);
    const int lane = thread & (warp_width - 1);
    const int warp = thread / warp_width;
    if (group >= groups) {
        return;
    }
    if (thread == 0) {
        variance_sum = 0.0F;
    }
    __syncthreads();

    const int similar = retained_counts[group];
    const int sample_dim = channels * patch_dim;
    const std::size_t group_base =
        static_cast<std::size_t>(group) * retained_stride * sample_dim;
    const float count =
        static_cast<float>(similar) * static_cast<float>(patch_dim);

    for (int channel = 0; channel < channels; ++channel) {
        float local_sum = 0.0F;
        float local_square = 0.0F;
        const int channel_base = channel * patch_dim;
        // Adjacent lanes read adjacent patch positions for every sample.  This
        // avoids integer division in the pass and keeps global loads coalesced.
        for (int position = thread; position < patch_dim;
             position += block_threads) {
            for (int sample = 0; sample < similar; ++sample) {
                const float value = noisy_samples[
                    group_base +
                    static_cast<std::size_t>(sample) * sample_dim +
                    channel_base + position];
                local_sum += value;
                local_square = fmaf(value, value, local_square);
            }
        }

        local_sum = warp_sum(local_sum);
        local_square = warp_sum(local_square);
        if (lane == 0) {
            warp_sums[warp] = local_sum;
            warp_squares[warp] = local_square;
        }
        __syncthreads();

        if (warp == 0) {
            float total_sum = lane < warps_per_block ? warp_sums[lane] : 0.0F;
            float total_square =
                lane < warps_per_block ? warp_squares[lane] : 0.0F;
            total_sum = warp_sum(total_sum);
            total_square = warp_sum(total_square);
            if (lane == 0) {
                const float numerator =
                    fmaxf(total_square - (total_sum * total_sum / count), 0.0F);
                variance_sum += count > 1.0F ? numerator / (count - 1.0F)
                                             : 0.0F;
            }
        }
        __syncthreads();
    }

    if (thread == 0) {
        flat_flags[group] =
            variance_sum / static_cast<float>(channels) < threshold ? 1U : 0U;
    }
}

MatchBatchShape make_match_shape(const StagePipelineShape& shape, int groups,
                                 const DeviceVideoView& video) {
    return MatchBatchShape{
        .stage = shape.stage,
        .groups = groups,
        .width = shape.width,
        .height = shape.height,
        .channels = shape.channels,
        .frames = video.frames,
        .first_frame = video.first_frame,
        .source_frames = shape.source_frames,
        .patch_size = shape.patch_size,
        .patch_time = shape.patch_time,
        .search_window = shape.search_window,
        .search_bwd = shape.search_bwd,
        .search_fwd = shape.search_fwd,
        .requested_similar = shape.requested_similar,
        .retained_stride = shape.retained_stride,
    };
}

GroupBatchShape make_group_shape(const StagePipelineShape& shape, int groups) {
    return GroupBatchShape{
        .groups = groups,
        .retained_stride = shape.retained_stride,
        .sample_dim = sample_dim_for(shape),
        .basis_similar = shape.basis_similar,
        .rank = shape.rank,
    };
}

AggregationShape make_aggregation_shape(const StagePipelineShape& shape) {
    return AggregationShape{
        .max_groups = shape.max_groups,
        .width = shape.width,
        .height = shape.height,
        .channels = shape.channels,
        .slots = shape.contribution_slots,
        .retained_stride = shape.retained_stride,
        .patch_size = shape.patch_size,
        .patch_time = shape.patch_time,
        .search_window = shape.search_window,
        .search_bwd = shape.search_bwd,
    };
}

} // namespace

class StagePipeline::Impl {
  public:
    explicit Impl(int requested_device)
        : matcher_(requested_device), filter_(matcher_.device()),
          aggregator_(matcher_.device()), device_(matcher_.device()) {}

    ~Impl() {
        if (device_ >= 0) {
            (void)cudaSetDevice(device_);
        }
    }

    void reserve(const StagePipelineShape& shape, float window_gamma) {
        validate_shape(shape);
        if (!std::isfinite(window_gamma) || window_gamma < 0.0F) {
            throw std::invalid_argument(
                "CUDA aggregation window gamma must be finite and non-negative");
        }
        check_cuda(cudaSetDevice(device_), "cudaSetDevice(CUDA pipeline)");
        if (reserved_ && same_shape(shape, shape_) &&
            window_gamma == window_gamma_) {
            return;
        }

        const int sample_dim = sample_dim_for(shape);
        const std::size_t descriptors = checked_product(
            static_cast<std::size_t>(shape.max_groups),
            static_cast<std::size_t>(shape.retained_stride),
            "CUDA pipeline descriptor count overflows");
        const std::size_t samples = checked_product(
            descriptors, static_cast<std::size_t>(sample_dim),
            "CUDA pipeline sample count overflows");

        MatchBatchShape match_shape = make_match_shape(
            shape, shape.max_groups,
            DeviceVideoView{
                .width = shape.width,
                .height = shape.height,
                .channels = shape.channels,
                .frames = cached_frames_for(shape),
                .first_frame = 0,
                .source_frames = shape.source_frames,
            });
        matcher_.reserve(match_shape);
        filter_.reserve(make_group_shape(shape, shape.max_groups));
        aggregator_.reserve(make_aggregation_shape(shape), window_gamma);

        retained_counts_.reserve(static_cast<std::size_t>(shape.max_groups));
        matches_.reserve(descriptors);
        noisy_samples_.reserve(samples);
        if (shape.stage == Stage::Final) {
            basic_samples_.reserve(samples);
            flat_flags_.reserve(static_cast<std::size_t>(shape.max_groups));
        }
        log_patch_weights_.reserve(descriptors);

        shape_ = shape;
        window_gamma_ = window_gamma;
        reserved_ = true;
    }

    void clear_contributions(float* numerators, float* weights,
                             cudaStream_t stream) const {
        require_reserved();
        check_cuda(cudaSetDevice(device_), "cudaSetDevice(CUDA pipeline)");
        aggregator_.enqueue_clear(make_contiguous_contribution_view(
                                      make_aggregation_shape(shape_), numerators,
                                      weights),
                                  stream);
    }

    void set_anchor_frame(PatchOrigin* anchors, int groups, int frame,
                          cudaStream_t stream) const {
        require_reserved();
        if (anchors == nullptr || groups <= 0) {
            throw std::invalid_argument(
                "CUDA pipeline anchor update requires a non-empty array");
        }
        if (frame < 0 || frame > shape_.source_frames - shape_.patch_time) {
            throw std::invalid_argument(
                "CUDA pipeline anchor update frame is not a valid origin");
        }
        check_cuda(cudaSetDevice(device_), "cudaSetDevice(CUDA pipeline)");
        const int blocks = (groups + block_threads - 1) / block_threads;
        set_anchor_frame_kernel<<<blocks, block_threads, 0, stream>>>(
            anchors, groups, frame);
        check_cuda(cudaPeekAtLastError(),
                   "launch CUDA anchor-frame update kernel");
    }

    void enqueue(const StagePipelineShape& shape,
                 const StagePipelineParameters& parameters,
                 const DeviceStageBatch& batch, cudaStream_t stream) {
        validate_shape(shape);
        validate_parameters(shape, parameters);
        require_reserved();
        if (!same_shape(shape, shape_)) {
            throw std::logic_error(
                "CUDA StagePipeline is not reserved for this shape");
        }
        if (batch.groups <= 0 || batch.groups > shape.max_groups) {
            throw std::invalid_argument(
                "CUDA pipeline chunk group count exceeds its reservation");
        }
        if (batch.anchors == nullptr || batch.solver_info == nullptr ||
            batch.contribution_numerators == nullptr ||
            batch.contribution_weights == nullptr) {
            throw std::invalid_argument(
                "CUDA pipeline requires anchors, solver status storage and "
                "contribution outputs");
        }
        if (batch.anchor_frame < 0 ||
            batch.anchor_frame > shape.source_frames - shape.patch_time) {
            throw std::invalid_argument(
                "CUDA pipeline anchor frame is not a valid patch origin");
        }
        check_cuda(cudaSetDevice(device_), "cudaSetDevice(CUDA pipeline)");

        const MatchBatchShape match_shape =
            make_match_shape(shape, batch.groups, batch.noisy);
        const DeviceMatchBatch match_batch{
            .noisy = batch.noisy,
            .basic = batch.basic,
            .anchors = batch.anchors,
            .search_centers = batch.search_centers,
            .retained_counts = retained_counts_.data(),
            .candidate_ids = nullptr,
            .matches = matches_.data(),
            .noisy_samples = noisy_samples_.data(),
            .basic_samples =
                shape.stage == Stage::Final ? basic_samples_.data() : nullptr,
        };
        matcher_.enqueue(match_shape, parameters.match, match_batch, stream);

        const std::uint8_t* flat_flags = nullptr;
        if (parameters.flat_areas) {
            const float threshold = parameters.filter.sigma *
                                    parameters.filter.sigma *
                                    parameters.flat_gamma;
            flat_flags_kernel<<<static_cast<unsigned int>(batch.groups),
                                block_threads, 0, stream>>>(
                noisy_samples_.data(), retained_counts_.data(), batch.groups,
                shape.retained_stride, shape.channels, patch_dim_for(shape),
                threshold, flat_flags_.data());
            check_cuda(cudaPeekAtLastError(),
                       "launch CUDA flat-area detection kernel");
            flat_flags = flat_flags_.data();
        }

        const GroupBatchShape group_shape =
            make_group_shape(shape, batch.groups);
        const DeviceGroupBatch group_batch{
            .noisy_samples = noisy_samples_.data(),
            .basic_samples =
                shape.stage == Stage::Final ? basic_samples_.data() : nullptr,
            .retained_counts = retained_counts_.data(),
            .flat_flags = flat_flags,
            // GroupFilter has completed every read of noisy_samples before its
            // final kernel writes estimates, so in-place output removes one
            // full G*C*D buffer without introducing a copy.
            .filtered_samples = noisy_samples_.data(),
            .log_patch_weights = log_patch_weights_.data(),
            .eigenvalues = nullptr,
            .basis_vectors = nullptr,
            .solver_info = batch.solver_info,
        };
        filter_.enqueue(group_shape, parameters.filter, group_batch, stream);

        const AggregationShape aggregation_shape =
            make_aggregation_shape(shape);
        aggregator_.enqueue_scatter(
            aggregation_shape,
            AggregationParameters{
                .anchor_frame = batch.anchor_frame,
                .log_weight_model_count = 1,
                .log_weight_model_stride = 0,
            },
            DeviceAggregationBatch{
                .filtered_samples = noisy_samples_.data(),
                .log_patch_weights = log_patch_weights_.data(),
                .retained_counts = retained_counts_.data(),
                .matches = matches_.data(),
                .groups = batch.groups,
                .contributions = make_contiguous_contribution_view(
                    aggregation_shape, batch.contribution_numerators,
                    batch.contribution_weights),
            },
            stream);
    }

    void pack_contributions(float* numerators, float* weights, float* packed,
                            cudaStream_t stream) const {
        require_reserved();
        check_cuda(cudaSetDevice(device_), "cudaSetDevice(CUDA pipeline)");
        aggregator_.enqueue_pack(make_contiguous_contribution_view(
                                     make_aggregation_shape(shape_),
                                     numerators, weights),
                                 packed, stream);
    }

    void normalize_contributions(float* numerators, float* weights, int slot,
                                 DeviceVideoView fallback_source,
                                 int source_frame,
                                 DeviceMutableFrameView output,
                                 cudaStream_t stream) const {
        require_reserved();
        check_cuda(cudaSetDevice(device_), "cudaSetDevice(CUDA pipeline)");
        aggregator_.enqueue_normalize(
            make_contiguous_contribution_view(make_aggregation_shape(shape_),
                                              numerators, weights),
            slot, fallback_source, source_frame, output, stream);
    }

    [[nodiscard]] std::size_t workspace_bytes() const noexcept {
        return retained_counts_.bytes() + matches_.bytes() +
               noisy_samples_.bytes() + basic_samples_.bytes() +
               flat_flags_.bytes() + log_patch_weights_.bytes() +
               matcher_.workspace_bytes() + filter_.workspace_bytes() +
               aggregator_.workspace_bytes();
    }

    void require_reserved() const {
        if (!reserved_) {
            throw std::logic_error("CUDA StagePipeline has not been reserved");
        }
    }

    BlockMatcher matcher_;
    GroupFilter filter_;
    Aggregator aggregator_;
    int device_ = -1;
    StagePipelineShape shape_{};
    float window_gamma_ = 0.0F;
    bool reserved_ = false;
    DeviceBuffer<int> retained_counts_;
    DeviceBuffer<PatchMatch> matches_;
    DeviceBuffer<float> noisy_samples_;
    DeviceBuffer<float> basic_samples_;
    DeviceBuffer<std::uint8_t> flat_flags_;
    DeviceBuffer<float> log_patch_weights_;
};

StagePipeline::StagePipeline(int device)
    : impl_(std::make_unique<Impl>(device)) {}

StagePipeline::~StagePipeline() = default;

StagePipeline::StagePipeline(StagePipeline&&) noexcept = default;

StagePipeline& StagePipeline::operator=(StagePipeline&&) noexcept = default;

void StagePipeline::reserve(const StagePipelineShape& shape,
                            float window_gamma) {
    if (impl_ == nullptr) {
        throw std::logic_error("CUDA StagePipeline was moved from");
    }
    impl_->reserve(shape, window_gamma);
}

void StagePipeline::clear_contributions(float* numerators, float* weights,
                                        cudaStream_t stream) const {
    if (impl_ == nullptr) {
        throw std::logic_error("CUDA StagePipeline was moved from");
    }
    impl_->clear_contributions(numerators, weights, stream);
}

void StagePipeline::set_anchor_frame(PatchOrigin* anchors, int groups,
                                     int frame, cudaStream_t stream) const {
    if (impl_ == nullptr) {
        throw std::logic_error("CUDA StagePipeline was moved from");
    }
    impl_->set_anchor_frame(anchors, groups, frame, stream);
}

void StagePipeline::enqueue(const StagePipelineShape& shape,
                            const StagePipelineParameters& parameters,
                            const DeviceStageBatch& batch,
                            cudaStream_t stream) {
    if (impl_ == nullptr) {
        throw std::logic_error("CUDA StagePipeline was moved from");
    }
    impl_->enqueue(shape, parameters, batch, stream);
}

void StagePipeline::pack_contributions(float* numerators, float* weights,
                                       float* packed,
                                       cudaStream_t stream) const {
    if (impl_ == nullptr) {
        throw std::logic_error("CUDA StagePipeline was moved from");
    }
    impl_->pack_contributions(numerators, weights, packed, stream);
}

void StagePipeline::normalize_contributions(
    float* numerators, float* weights, int slot,
    DeviceVideoView fallback_source, int source_frame,
    DeviceMutableFrameView output, cudaStream_t stream) const {
    if (impl_ == nullptr) {
        throw std::logic_error("CUDA StagePipeline was moved from");
    }
    impl_->normalize_contributions(numerators, weights, slot, fallback_source,
                                   source_frame, output, stream);
}

std::size_t StagePipeline::numerator_values() const noexcept {
    return impl_ == nullptr ? 0 : impl_->aggregator_.numerator_values();
}

std::size_t StagePipeline::weight_values() const noexcept {
    return impl_ == nullptr ? 0 : impl_->aggregator_.weight_values();
}

std::size_t StagePipeline::packed_values() const noexcept {
    return impl_ == nullptr ? 0 : impl_->aggregator_.packed_values();
}

std::size_t StagePipeline::workspace_bytes() const noexcept {
    return impl_ == nullptr ? 0 : impl_->workspace_bytes();
}

int StagePipeline::device() const noexcept {
    return impl_ == nullptr ? -1 : impl_->device_;
}

} // namespace vnlbcu
