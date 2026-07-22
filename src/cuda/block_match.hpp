#pragma once

#include "cuda/group_filter.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace vnlbcu {

// A device-resident, uniformly-strided frame window.  Strides are measured in
// float elements, not bytes.  Frame coordinates in PatchOrigin/SearchCenter
// are absolute; first_frame maps them into this cached local window.
//
// The storage may either be one contiguous allocation rooted at data, or a
// device-resident table of per-frame base pointers in frame_data.  When
// frame_data is non-null it takes precedence and frame_stride is ignored;
// row/channel strides remain common to all pointed-to frames.  This lets a
// frame cache feed the matcher without assembling a copied temporal window.
struct DeviceVideoView {
    const float* data = nullptr;
    const float* const* frame_data = nullptr;
    int width = 0;
    int height = 0;
    int channels = 0;
    int frames = 0;
    int first_frame = 0;
    int source_frames = 0;
    std::ptrdiff_t row_stride = 0;
    std::ptrdiff_t channel_stride = 0;
    std::ptrdiff_t frame_stride = 0;
};

struct PatchOrigin {
    int x = 0;
    int y = 0;
    int frame = 0;
};

// Optional motion-compensated centers, layout [group][temporal_count].  The
// temporal order must be anchor, future ascending, then past descending, as in
// the CPU implementation.  If omitted, BlockMatcher generates same-location
// centers and the boundary-shifted temporal schedule on the device.
struct SearchCenter {
    int x = 0;
    int y = 0;
    int frame = 0;
};

struct PatchMatch {
    float distance = 0.0F;
    int x = 0;
    int y = 0;
    int frame = 0;
};

struct MatchBatchShape {
    Stage stage = Stage::Basic;
    int groups = 0;
    int width = 0;
    int height = 0;
    int channels = 0;
    int frames = 0;      // frames present in the device cache
    int first_frame = 0; // absolute index of cache frame zero
    int source_frames = 0;
    int patch_size = 0;
    int patch_time = 0;
    int search_window = 0;
    int search_bwd = 0;
    int search_fwd = 0;
    int requested_similar = 0; // configured K; effective K=min(K,candidates)
    // Actual output cap/stride C.  For CPU parameter parity, callers pass
    // min(ceil(K * cap_factor), candidate_count), with factor==0 selecting all
    // candidates.  It must cover min(K,candidate_count).  The kernel computes
    // only the true retained count S<=C.
    int retained_stride = 0;
};

struct MatchParameters {
    float tau = 0.0F;
};

struct DeviceMatchBatch {
    DeviceVideoView noisy{};
    // Required for Final.  It is both the all-channel matching reference and
    // the source gathered into basic_samples.
    DeviceVideoView basic{};
    // Device arrays.  Anchors must be valid full-source patch origins:
    // x in [0,width-P], y in [0,height-P], and frame in
    // [0,source_frames-Pt].  Search centers use absolute coordinates and must
    // name valid temporal patch origins; the kernel boundary-shifts their
    // spatial search windows.  Every referenced frame plus [0,Pt) must be
    // present in the noisy/basic DeviceVideoView cache.  These are caller
    // contracts and are deliberately not checked inside the hot kernel.
    const PatchOrigin* anchors = nullptr;         // [groups]
    const SearchCenter* search_centers = nullptr; // optional [groups][T]

    int* retained_counts = nullptr; // [groups]
    // Optional compact descriptors, layout [groups][retained_stride].
    std::uint32_t* candidate_ids = nullptr;
    PatchMatch* matches = nullptr;

    // Direct sample-major output consumed by GroupFilter:
    // [group][retained_stride][channels * patch_time * P * P].
    float* noisy_samples = nullptr;
    float* basic_samples = nullptr; // required for Final
};

class BlockMatcher final {
  public:
    explicit BlockMatcher(int device = -1);
    ~BlockMatcher();

    BlockMatcher(const BlockMatcher&) = delete;
    BlockMatcher& operator=(const BlockMatcher&) = delete;
    BlockMatcher(BlockMatcher&&) noexcept;
    BlockMatcher& operator=(BlockMatcher&&) noexcept;

    // Validates the fixed source geometry, records groups as the maximum batch
    // size, and configures the common fused kernels.  Enqueue accepts smaller
    // group counts and different first_frame/frames cache windows, since those
    // do not affect kernel selection or shared-memory sizing.  The current
    // fast path supports at most 2048 candidates per anchor; this covers the
    // default 19x19x3 search without a global distance stack.
    void reserve(const MatchBatchShape& shape);

    // Enqueues matching, stable (distance, scan-order) selection, tau
    // expansion and direct noisy/basic gather as one kernel per whole batch.
    // There are no host/device copies, allocations, or synchronizations here.
    // One instance is not thread-safe and must not be submitted concurrently.
    void enqueue(const MatchBatchShape& shape,
                 const MatchParameters& parameters,
                 const DeviceMatchBatch& batch, cudaStream_t stream = nullptr);

    [[nodiscard]] int candidate_count() const noexcept;
    [[nodiscard]] int temporal_count() const noexcept;
    [[nodiscard]] int sample_dim() const noexcept;
    [[nodiscard]] std::size_t workspace_bytes() const noexcept;
    [[nodiscard]] int device() const noexcept;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vnlbcu
