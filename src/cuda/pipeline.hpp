#pragma once

#include "cuda/aggregation.hpp"
#include "cuda/block_match.hpp"
#include "cuda/group_filter.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>

namespace vnlbcu {

// Fixed geometry for one reusable CUDA stage worker.  max_groups is the
// largest chunk submitted at once; a complete frame may contain any number of
// chunks.  The model count B is deliberately fixed per worker while the
// retained count S remains per-group and is read from retained_counts.  The
// matcher selects a fused, chunked, or full-sort implementation per geometry.
struct StagePipelineShape {
    Stage stage = Stage::Basic;
    int max_groups = 0;
    int width = 0;
    int height = 0;
    int channels = 0;
    int source_frames = 0;
    int patch_size = 0;
    int patch_time = 0;
    int search_window = 0;
    int search_bwd = 0;
    int search_fwd = 0;
    int requested_similar = 0;
    int retained_stride = 0;
    int basis_similar = 0;
    int rank = 0;
    int contribution_slots = 0;
    float model_cap_factor = 1.0F;
    // The current fast path builds one coupled PCA model over all channels.
    // Keep this field explicit so an uncoupled caller is rejected instead of
    // silently receiving a different estimator.
    bool couple_channels = true;
};

struct StagePipelineParameters {
    MatchParameters match{};
    FilterParameters filter{};
    // Final-stage flat-area detection is performed on the gathered noisy
    // group before PCA.  It is skipped entirely when flat_areas is false.
    bool flat_areas = false;
    float flat_gamma = 0.95F;
    // Reproduce the CPU's ordered paste-mask suppression between matching and
    // aggregation. Matching and PCA remain batched; inactive groups simply do
    // not contribute to the output.
    bool paste_mask = false;
};

// All pointers are device pointers.  The video window and anchor array may
// change for every chunk without reallocating the pipeline.  Contributions
// are shared by all chunks of an anchor frame and therefore must be cleared
// once before the first enqueue, not once per chunk.
struct DeviceStageBatch {
    DeviceVideoView noisy{};
    DeviceVideoView basic{};
    // Every anchor in this chunk must use anchor_frame as its temporal origin.
    const PatchOrigin* anchors = nullptr;         // [groups]
    const SearchCenter* search_centers = nullptr; // optional [groups][T]
    int groups = 0;
    int anchor_frame = 0;

    // Contiguous layouts:
    // numerator [channel][slot][height][width]
    // weight    [slot][height][width]
    float* contribution_numerators = nullptr;
    float* contribution_weights = nullptr;

    // Per-chunk cuSOLVER status destination [groups].  A frame scheduler gives
    // every chunk a distinct slice and downloads all status values once at the
    // final stream synchronization point; this avoids both lost diagnostics
    // and a synchronization after every chunk.
    int* solver_info = nullptr;
};

class StagePipeline final {
  public:
    explicit StagePipeline(int device = -1);
    ~StagePipeline();

    StagePipeline(const StagePipeline&) = delete;
    StagePipeline& operator=(const StagePipeline&) = delete;
    StagePipeline(StagePipeline&&) noexcept;
    StagePipeline& operator=(StagePipeline&&) noexcept;

    // Allocates all chunk-local workspaces and uploads the spatial aggregation
    // window.  No allocation occurs in clear/enqueue/pack after reserve.
    void reserve(const StagePipelineShape& shape, float window_gamma);

    // Clears a complete anchor-frame contribution stack and resets the
    // optional ordered paste mask asynchronously.
    void clear_contributions(float* numerators, float* weights,
                             cudaStream_t stream = nullptr);

    // Updates only the temporal coordinate of a device-resident spatial
    // anchor grid.  A scheduler can upload x/y once during worker creation
    // and avoid copying the full PatchOrigin array for every anchor frame.
    void set_anchor_frame(PatchOrigin* anchors, int groups, int frame,
                          cudaStream_t stream = nullptr) const;

    // Enqueues match -> optional flat detection -> batched PCA/filter ->
    // scatter-add for one chunk.  It performs no allocation, host transfer or
    // synchronization.
    void enqueue(const StagePipelineShape& shape,
                 const StagePipelineParameters& parameters,
                 const DeviceStageBatch& batch, cudaStream_t stream = nullptr);

    // Packs the split accumulator into the VapourSynth contribution-plane
    // layout [channel][slot][numerator/weight][height][width].
    void pack_contributions(float* numerators, float* weights, float* packed,
                            cudaStream_t stream = nullptr) const;

    // Direct device-resident exit for a fused scheduler which has accumulated
    // all contributing anchor frames into one view.  This avoids forcing the
    // compatibility contribution-stack format through host memory.
    void normalize_contributions(float* numerators, float* weights, int slot,
                                 DeviceVideoView fallback_source,
                                 int source_frame,
                                 DeviceMutableFrameView output,
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
