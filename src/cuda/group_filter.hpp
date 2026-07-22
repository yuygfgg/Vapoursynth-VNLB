#pragma once

// This header is intentionally only included by the optional CUDA backend.
// The CPU targets do not need a CUDA SDK installed to build VapourSynth-VNLB.

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace vnlbcu {

enum class Stage : std::uint8_t {
    Basic,
    Final,
};

// Parameters which affect the batched group model.  Geometry and layout are
// supplied separately in GroupBatchShape so that a workspace can be reused
// while noise/weight parameters change between frames.
struct FilterParameters {
    Stage stage = Stage::Basic;
    float sigma = 1.0F;
    float sigma_basic = 0.0F;
    float beta = 1.0F;
    float variance_threshold = 1.1F;
    float weight_alpha = 0.75F;
    float weight_beta = 0.35F;
    float weight_epsilon = 1.0e-6F;
    float membership_noise_floor = 0.25F;
    // Exact noisy-row equality is a CPU fast path.  It is disabled by
    // default because scanning every retained value solely to discover a
    // rare shortcut would add an unconditional pass to the GPU hot path.
    bool detect_equal_groups = false;
};

struct GroupBatchShape {
    int groups = 0;
    int retained_stride = 0; // Smax rows reserved per group in sample buffers.
    int sample_dim = 0;      // D
    int basis_similar = 0;   // B, identical for every group in this batch
    int rank = 0;            // R
};

// All pointers in this structure are device pointers.  Samples are row-major
// and have the layout [group][retained_stride][sample_dim].  retained_counts
// may be null, in which case every group contains retained_stride rows.  The
// first B rows are always used for the model; all S rows are projected and
// returned.  Every retained count must satisfy
// B <= retained_counts[group] <= retained_stride.
//
// For Final, noisy_samples and basic_samples are both required.  For Basic,
// basic_samples is ignored.  flat_flags is optional and only consulted for
// Final; a non-zero value selects the CPU implementation's flat-patch center
// semantics (basic mean is used for noisy centering and output center).
struct DeviceGroupBatch {
    const float* noisy_samples = nullptr;
    const float* basic_samples = nullptr;
    const int* retained_counts = nullptr;
    const std::uint8_t* flat_flags = nullptr;

    float* filtered_samples = nullptr;
    // Optional per-model log weights, overwritten at
    // [group][retained_stride].  Uncoupled multi-channel callers should sum
    // these logs across channel models and divide by the model count before
    // exponentiating, matching the CPU geometric-average semantics.
    float* log_patch_weights = nullptr;
    float* eigenvalues = nullptr;   // optional, layout [groups][rank]
    float* basis_vectors = nullptr; // optional, layout [groups][rank][D]
    // Optional per-submission cuSOLVER status storage, layout [groups].  Give
    // consecutive queued batches distinct buffers when every batch must be
    // diagnosed; otherwise GroupFilter reuses its internal latest-batch
    // buffer without adding a copy to the hot path.
    int* solver_info = nullptr;
};

class GroupFilter final {
  public:
    explicit GroupFilter(int device = -1);
    ~GroupFilter();

    GroupFilter(const GroupFilter&) = delete;
    GroupFilter& operator=(const GroupFilter&) = delete;
    GroupFilter(GroupFilter&&) noexcept;
    GroupFilter& operator=(GroupFilter&&) noexcept;

    // Allocates/reuses all intermediate device buffers.  This function is
    // deliberately separate from enqueue: no cudaMalloc/cudaFree occurs in
    // the frame hot path after the workspace has been reserved.
    void reserve(const GroupBatchShape& shape);

    // Enqueues the complete batch model/filter pipeline.  It does not perform
    // host/device copies and does not synchronize the stream.  A GroupFilter
    // owns one reusable workspace; callers may queue repeated batches on one
    // stream, but the instance is not thread-safe.  Every reserve/enqueue/check
    // host call must be serialized; a stream pool should own one GroupFilter
    // per stream/worker.
    void enqueue(const GroupBatchShape& shape, const FilterParameters& params,
                 const DeviceGroupBatch& batch, cudaStream_t stream = nullptr);

    // Synchronizes the latest submitted batch and checks every cuSOLVER info
    // value in a single small device-to-host copy.  The stream must be the
    // same stream used by the latest enqueue (or omitted).  Production callers
    // may instead provide DeviceGroupBatch::solver_info and integrate checks
    // into their own pipeline.
    void synchronize_and_check(cudaStream_t stream = nullptr);

    [[nodiscard]] const int* solver_info_device() const noexcept;
    [[nodiscard]] std::size_t workspace_bytes() const noexcept;
    [[nodiscard]] int device() const noexcept;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vnlbcu
