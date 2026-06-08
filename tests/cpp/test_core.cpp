#include "aggregate/aggregate.hpp"
#include "core/core.hpp"
#include "flow/flow.hpp"

#include <cstdlib>
#include <iostream>
#include <span>
#include <string_view>
#include <vector>

namespace {

void require(bool condition, std::string_view label) {
    if (!condition) {
        std::cerr << label << '\n';
        std::exit(EXIT_FAILURE);
    }
}

void require_close(float actual, float expected, float tolerance,
                   std::string_view label) {
    const float difference =
        actual > expected ? actual - expected : expected - actual;
    if (difference > tolerance) {
        std::cerr << label << ": expected " << expected << ", got " << actual
                  << '\n';
        std::exit(EXIT_FAILURE);
    }
}

void test_same_location_provider() {
    const vnlb::flow::SameLocationProvider provider;
    const auto center = provider.center_for({7, 9, 2, 4});
    require(center.x == 7 && center.y == 9, "same-location center mismatch");
}

void test_mvtools_flow_provider_clamps_center() {
    vnlb::flow::MVToolsVectorGrid grid;
    grid.analysis.block_size_x = 1;
    grid.analysis.block_size_y = 1;
    grid.analysis.pel = 1;
    grid.analysis.block_count_x = 1;
    grid.analysis.block_count_y = 1;
    grid.valid = true;
    grid.vectors.push_back(vnlb::flow::MVToolsVector{10, -10, 0});

    const std::vector<vnlb::flow::MVToolsVectorGrid> previous{grid, grid};
    const vnlb::flow::MVToolsFlowProvider provider(
        std::span<const vnlb::flow::MVToolsVectorGrid>(previous), 1, {}, 0);
    const auto center = provider.center_for({3, 3, 2, 0, 2, 5, 5});
    require(center.x == 4 && center.y == 0, "mvtools center clamp mismatch");
}

void test_aggregate_frame() {
    const vnlb::core::VideoGeometry geometry{2, 2, 2, 1};
    vnlb::core::VideoBuffer source(geometry, 7.0F);
    vnlb::core::VideoBuffer output(geometry, 0.0F);

    const auto layout =
        vnlb::aggregate::make_contribution_layout(geometry, 1, 1, 1);
    vnlb::aggregate::ContributionStack stack(layout);
    stack.clear();

    auto view = stack.view();
    const int slot = layout.slot_for_output_offset(1);
    view.numerator(slot, 0, 0, 0) = 10.0F;
    view.weight(slot, 0, 0) = 2.0F;

    const std::vector<vnlb::aggregate::ConstContributionStackView> stacks{
        stack.cview()};
    const std::vector<int> anchors{0};
    vnlb::aggregate::aggregate_frame(
        std::span<const vnlb::aggregate::ConstContributionStackView>(stacks),
        std::span<const int>(anchors), 1, source.cview(), output.view());

    require_close(output.cview().sample(0, 0, 1), 5.0F, 1.0e-6F,
                  "weighted aggregate");
    require_close(output.cview().sample(1, 0, 1), 7.0F, 1.0e-6F,
                  "aggregate source fallback");
}

void test_stage_parameter_defaults() {
    const auto basic = vnlb::core::make_basic_parameters(0.1F);
    require(basic.patch_size == 8, "basic default block size");
    require(basic.search_window == 19, "basic default search window");
    require(basic.search_bwd == 1 && basic.search_fwd == 1,
            "basic default temporal radius");
    require(basic.similar == 8, "basic default group size");
    require(basic.rank == 8, "basic default rank");
    require_close(basic.similar_cap_factor, 4.0F, 1.0e-6F,
                  "basic default cap factor");
    require_close(basic.model_cap_factor, 1.0F, 1.0e-6F,
                  "basic default model cap factor");
    require(basic.proc_step == 8, "basic default block step");
    require_close(basic.tau, 0.0F, 1.0e-9F, "basic default tau");

    const auto final = vnlb::core::make_final_parameters(0.1F);
    require(final.patch_size == 8, "final default block size");
    require(final.search_window == 19, "final default search window");
    require(final.search_bwd == 1 && final.search_fwd == 1,
            "final default temporal radius");
    require(final.similar == 8, "final default group size");
    require(final.rank == 8, "final default rank");
    require_close(final.similar_cap_factor, 4.0F, 1.0e-6F,
                  "final default cap factor");
    require_close(final.model_cap_factor, 1.0F, 1.0e-6F,
                  "final default model cap factor");
    require(final.proc_step == 8, "final default block step");
    require_close(final.tau, 400.0F / (255.0F * 255.0F), 1.0e-9F,
                  "final default tau");
}

vnlb::core::StageParameters constant_test_parameters() {
    auto parameters = vnlb::core::make_basic_parameters(0.1F);
    parameters.patch_size = 2;
    parameters.patch_time = 1;
    parameters.search_window = 3;
    parameters.search_bwd = 1;
    parameters.search_fwd = 1;
    parameters.similar = 4;
    parameters.rank = 4;
    parameters.beta = 2.0F;
    parameters.tau = 1000.0F;
    parameters.proc_step = 2;
    return parameters;
}

template <typename ProcessAnchor>
void run_constant_pipeline(ProcessAnchor process_anchor,
                           vnlb::core::ConstVideoView source,
                           vnlb::core::MutableVideoView output,
                           vnlb::core::StageParameters parameters) {
    const auto geometry = source.geometry();
    const auto layout = vnlb::aggregate::make_contribution_layout(
        geometry, parameters.search_bwd, parameters.search_fwd,
        parameters.patch_time);

    std::vector<vnlb::aggregate::ContributionStack> contribution_stacks;
    contribution_stacks.reserve(static_cast<std::size_t>(geometry.frames));

    vnlb::core::StageWorkspace workspace;
    const vnlb::flow::SameLocationProvider flow_provider;
    for (int frame = 0; frame < geometry.frames; ++frame) {
        contribution_stacks.emplace_back(layout);
        const auto stats = process_anchor(
            frame, flow_provider, contribution_stacks.back().view(), workspace);
        if (frame <= geometry.frames - parameters.patch_time) {
            require(stats.groups <= 9, "unexpected processed group count");
        }
    }

    std::vector<vnlb::aggregate::ConstContributionStackView> stack_views;
    std::vector<int> anchor_frames;
    stack_views.reserve(contribution_stacks.size());
    anchor_frames.reserve(contribution_stacks.size());
    for (int frame = 0; frame < geometry.frames; ++frame) {
        stack_views.push_back(
            contribution_stacks[static_cast<std::size_t>(frame)].cview());
        anchor_frames.push_back(frame);
    }

    for (int frame = 0; frame < geometry.frames; ++frame) {
        vnlb::aggregate::aggregate_frame(
            std::span<const vnlb::aggregate::ConstContributionStackView>(
                stack_views),
            std::span<const int>(anchor_frames), frame, source, output);
    }
}

void require_constant_video(vnlb::core::ConstVideoView video, float expected) {
    const auto geometry = video.geometry();
    for (int frame = 0; frame < geometry.frames; ++frame) {
        for (int channel = 0; channel < geometry.channels; ++channel) {
            for (int y = 0; y < geometry.height; ++y) {
                for (int x = 0; x < geometry.width; ++x) {
                    require_close(video.sample(x, y, frame, channel), expected,
                                  1.0e-5F, "constant video output");
                }
            }
        }
    }
}

void test_basic_constant_pipeline() {
    const vnlb::core::VideoGeometry geometry{5, 5, 3, 1};
    vnlb::core::VideoBuffer noisy(geometry, 0.25F);
    vnlb::core::VideoBuffer output(geometry, 0.0F);
    const auto parameters = constant_test_parameters();

    run_constant_pipeline(
        [&](int frame, const vnlb::flow::SameLocationProvider& flow_provider,
            vnlb::aggregate::ContributionStackView stack,
            vnlb::core::StageWorkspace& workspace) {
            return vnlb::core::process_basic_anchor_no_flow(
                noisy.cview(), frame, parameters, flow_provider, stack,
                workspace);
        },
        noisy.cview(), output.view(), parameters);

    require_constant_video(output.cview(), 0.25F);
}

void test_final_constant_pipeline() {
    const vnlb::core::VideoGeometry geometry{5, 5, 3, 1};
    vnlb::core::VideoBuffer noisy(geometry, 0.25F);
    vnlb::core::VideoBuffer basic(geometry, 0.25F);
    vnlb::core::VideoBuffer output(geometry, 0.0F);
    auto parameters = constant_test_parameters();
    parameters.tau = 1000.0F;

    run_constant_pipeline(
        [&](int frame, const vnlb::flow::SameLocationProvider& flow_provider,
            vnlb::aggregate::ContributionStackView stack,
            vnlb::core::StageWorkspace& workspace) {
            return vnlb::core::process_final_anchor_no_flow(
                noisy.cview(), basic.cview(), frame, parameters, flow_provider,
                stack, workspace);
        },
        noisy.cview(), output.view(), parameters);

    require_constant_video(output.cview(), 0.25F);
}

} // namespace

int main() {
    test_same_location_provider();
    test_mvtools_flow_provider_clamps_center();
    test_aggregate_frame();
    test_stage_parameter_defaults();
    test_basic_constant_pipeline();
    test_final_constant_pipeline();
    return EXIT_SUCCESS;
}
