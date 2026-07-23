#pragma once

#include "cuda/block_match.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace vnlbcu {

// Split device accumulator.  Strides are measured in float elements.
// Numerators use [channel][slot][y][x], while the common weight plane uses
// [slot][y][x].  Keeping weight outside the channel dimension avoids the CPU
// contribution format's redundant per-channel writes in the hot path.
struct DeviceContributionView {
    float* numerators = nullptr;
    float* weights = nullptr;
    int width = 0;
    int height = 0;
    int channels = 0;
    int slots = 0;
    std::ptrdiff_t numerator_row_stride = 0;
    std::ptrdiff_t numerator_slot_stride = 0;
    std::ptrdiff_t numerator_channel_stride = 0;
    std::ptrdiff_t weight_row_stride = 0;
    std::ptrdiff_t weight_slot_stride = 0;
};

// Device-resident descriptor for one slot selected from a contribution view.
// numerators and weights are already biased to that slot, keeping slot-stride
// arithmetic out of the per-pixel kernel.  Arrays can mix slots and backing
// allocations.  The descriptor itself is copied to device storage by the
// caller, allowing final normalization without an intermediate accumulator or
// host synchronization.
struct DeviceContributionSource {
    const float* numerators = nullptr;
    const float* weights = nullptr;
    std::ptrdiff_t numerator_row_stride = 0;
    std::ptrdiff_t numerator_channel_stride = 0;
    std::ptrdiff_t weight_row_stride = 0;
};

struct DeviceMutableFrameView {
    float* data = nullptr;
    int width = 0;
    int height = 0;
    int channels = 0;
    std::ptrdiff_t row_stride = 0;
    std::ptrdiff_t channel_stride = 0;
};

// Fixed geometry for one reusable aggregation worker.  max_groups is the
// largest chunk submitted at once.  A frame may be accumulated through any
// number of smaller chunks without clearing between chunks.
struct AggregationShape {
    int max_groups = 0;
    int width = 0;
    int height = 0;
    int channels = 0;
    int slots = 0;
    int retained_stride = 0;
    int patch_size = 0;
    int patch_time = 0;
    int search_window = 0;
    int search_bwd = 0;
};

struct AggregationParameters {
    int anchor_frame = 0;
    // One for coupled filtering.  For uncoupled filtering, logs are stored
    // model-major and the scatter kernel exponentiates their arithmetic mean,
    // reproducing the CPU geometric-average patch weight without an
    // intermediate reduction buffer.
    int log_weight_model_count = 1;
    // Distance in floats between model planes.  Zero is accepted for a
    // single model; multiple models require an explicit positive stride.
    std::ptrdiff_t log_weight_model_stride = 0;
};

// All pointers are device pointers.  Samples use the matcher/filter layout
// [group][retained_stride][channels * patch_time * patch_size^2].  Matches and
// logs use [group][retained_stride].  retained_counts may be null to select
// retained_stride rows for every group.  log_patch_weights may be null, in
// which case every patch weight is one.
struct DeviceAggregationBatch {
    const float* filtered_samples = nullptr;
    const float* log_patch_weights = nullptr;
    const int* retained_counts = nullptr;
    const PatchMatch* matches = nullptr;
    // Optional per-group activity mask. Inactive groups are intentionally
    // omitted from the contribution stack after CPU-compatible paste-mask
    // suppression, while the upstream matcher/filter may remain batched.
    const std::uint8_t* active_groups = nullptr;
    int groups = 0;
    DeviceContributionView contributions{};
};

[[nodiscard]] DeviceContributionView
make_contiguous_contribution_view(const AggregationShape& shape,
                                  float* numerators, float* weights);

[[nodiscard]] DeviceContributionSource
make_contribution_source(const DeviceContributionView& contributions, int slot);

class Aggregator final {
  public:
    explicit Aggregator(int device = -1);
    ~Aggregator();

    Aggregator(const Aggregator&) = delete;
    Aggregator& operator=(const Aggregator&) = delete;
    Aggregator(Aggregator&&) noexcept;
    Aggregator& operator=(Aggregator&&) noexcept;

    // Allocates/reuses the spatial window and uploads it once.  No allocation,
    // host transfer, or synchronization occurs in the enqueue methods after
    // reserve.  Changing window_gamma requires another reserve call.
    void reserve(const AggregationShape& shape, float window_gamma);

    void enqueue_clear(const DeviceContributionView& contributions,
                       cudaStream_t stream = nullptr) const;

    // One CTA handles one group; its warps walk the true per-group retained
    // count rather than padding S to a warp-friendly value.
    void enqueue_scatter(const AggregationShape& shape,
                         const AggregationParameters& parameters,
                         const DeviceAggregationBatch& batch,
                         cudaStream_t stream = nullptr) const;

    // Compatibility output for a VapourSynth contribution frame:
    // [channel][slot][numerator/weight][height][width].  Weight duplication is
    // confined to this boundary kernel and is not present in the accumulator.
    void enqueue_pack(const DeviceContributionView& contributions,
                      float* packed, cudaStream_t stream = nullptr) const;

    // Normalizes one slot directly into a packed device frame.  Pixels with
    // zero accumulated weight fall back to source_frame in fallback_source.
    void enqueue_normalize(const DeviceContributionView& contributions,
                           int slot, DeviceVideoView fallback_source,
                           int source_frame, DeviceMutableFrameView output,
                           cudaStream_t stream = nullptr) const;

    // Merges selected slots from any number of contribution buffers and
    // normalizes directly into output.  device_sources points to an array in
    // device memory.  A zero source_count is valid and writes fallback_source.
    // No allocation, descriptor copy, intermediate accumulator, or
    // synchronization is performed here.
    void enqueue_normalize_many(const DeviceContributionSource* device_sources,
                                int source_count,
                                DeviceVideoView fallback_source,
                                int source_frame, DeviceMutableFrameView output,
                                cudaStream_t stream = nullptr) const;

    [[nodiscard]] std::size_t numerator_values() const noexcept;
    [[nodiscard]] std::size_t weight_values() const noexcept;
    [[nodiscard]] std::size_t packed_values() const noexcept;
    [[nodiscard]] std::size_t workspace_bytes() const noexcept;
    [[nodiscard]] int device() const noexcept;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vnlbcu
