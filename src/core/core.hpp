#pragma once

#include "aggregate/aggregate.hpp"
#include "core/video.hpp"
#include "flow/flow.hpp"
#include "linalg/linalg.hpp"

#include <vector>

namespace vnlb::core {

enum class Stage {
    Basic,
    Final,
};

struct StageParameters {
    float sigma = 1.0F;
    int patch_size = 8;
    int patch_time = 1;
    int search_window = 19;
    int search_bwd = 1;
    int search_fwd = 1;
    int similar = 8;
    int rank = 8;
    float similar_cap_factor = 4.0F;
    float model_cap_factor = 1.0F;
    float beta = 1.0F;
    float tau = 0.0F;
    float variance_threshold = 1.1F;
    float sigma_basic = 0.0F;
    float gamma = 0.95F;
    int proc_step = 0;
    bool aggregation_window = false;
    bool order_invariance = false;
    bool flat_areas = false;
    bool couple_channels = false;
};

struct ProcessStats {
    int groups = 0;
};

struct FrameRange {
    int first = 0;
    int last = 0;
};

struct PatchMatch {
    float distance = 0.0F;
    int x = 0;
    int y = 0;
    int frame = 0;
};

class StageWorkspace {
  public:
    void prepare(VideoGeometry geometry, StageParameters parameters,
                 Stage stage);

    VideoGeometry geometry_{};
    StageParameters parameters_{};
    Stage stage_ = Stage::Basic;
    int patch_area_ = 0;
    int patch_dim_ = 0;
    int estimator_dim_ = 0;
    int max_similar_ = 0;

    std::vector<float> reference_patch_;
    std::vector<float> group_noisy_;
    std::vector<float> group_basic_;
    std::vector<float> mean_noisy_;
    std::vector<float> mean_basic_;
    std::vector<float> eigenvalues_;
    std::vector<float> filter_coefficients_;
    linalg::Matrix<float> samples_noisy_;
    linalg::Matrix<float> samples_basic_;
    linalg::Matrix<float> centered_noisy_;
    linalg::Matrix<float> centered_basic_;
    linalg::Matrix<float> covariance_;
    linalg::Matrix<float> gram_;
    linalg::Matrix<float> eigenvectors_;
    linalg::Matrix<float> dual_eigenvectors_;
    linalg::Matrix<float> filtered_;
    linalg::SymmetricEigenWorkspace<float> eigen_workspace_;
    std::vector<PatchMatch> matches_;
    std::vector<float> top_distances_;
    std::vector<const float*> distance_plane_bases_;
    std::vector<int> distance_plane_strides_;
    std::vector<int> scheduled_frames_;
    std::vector<unsigned char> processing_mask_;
    bool output_sample_major_ = false;
};

[[nodiscard]] StageParameters make_basic_parameters(float sigma);
[[nodiscard]] StageParameters make_final_parameters(float sigma);
[[nodiscard]] int default_proc_step(Stage stage, int patch_size) noexcept;

void validate_stage_parameters(StageParameters parameters, Stage stage);
void validate_stage_configuration(VideoGeometry geometry,
                                  StageParameters parameters, Stage stage);
[[nodiscard]] FrameRange
input_frame_range_for_anchor(VideoGeometry geometry, StageParameters parameters,
                             int anchor_frame);

ProcessStats process_basic_anchor_no_flow(
    ConstVideoView noisy, int anchor_frame, StageParameters parameters,
    const flow::SameLocationProvider& flow_provider,
    aggregate::ContributionStackView contributions, StageWorkspace& workspace);

ProcessStats process_final_anchor_no_flow(
    ConstVideoView noisy, ConstVideoView basic, int anchor_frame,
    StageParameters parameters, const flow::SameLocationProvider& flow_provider,
    aggregate::ContributionStackView contributions, StageWorkspace& workspace);

ProcessStats process_basic_anchor_mvtools(
    ConstVideoView noisy, int anchor_frame, StageParameters parameters,
    const flow::MVToolsFlowProvider& flow_provider,
    aggregate::ContributionStackView contributions, StageWorkspace& workspace);

ProcessStats process_final_anchor_mvtools(
    ConstVideoView noisy, ConstVideoView basic, int anchor_frame,
    StageParameters parameters, const flow::MVToolsFlowProvider& flow_provider,
    aggregate::ContributionStackView contributions, StageWorkspace& workspace);

} // namespace vnlb::core
