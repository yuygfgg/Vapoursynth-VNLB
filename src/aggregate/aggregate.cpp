#include "aggregate.hpp"

#include "common/validation.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace vnlb::aggregate {
namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

void require_valid_layout(ContributionLayout layout) {
    require(layout.width > 0 && layout.height > 0 && layout.channels > 0,
            "contribution layout dimensions must be non-empty");
    require(layout.search_bwd >= 0 && layout.search_fwd >= 0 &&
                layout.patch_time > 0,
            "contribution temporal parameters are invalid");
    const int expected_slot_count = common::checked_add_int(
        common::checked_add_int(layout.search_bwd, layout.search_fwd,
                                "contribution slot count overflows int"),
        layout.patch_time, "contribution slot count overflows int");
    require(layout.slot_count == expected_slot_count,
            "contribution slot count does not match temporal parameters");
    require(layout.value_count() <=
                static_cast<std::size_t>(std::numeric_limits<int>::max()),
            "contribution stack size overflows int");
    (void)common::checked_mul_int(
        common::checked_mul_int(layout.slot_count, 2,
                                "contribution frame height overflows int"),
        layout.height, "contribution frame height overflows int");
}

void validate_layout(ContributionLayout layout) {
    require_valid_layout(layout);
}

} // namespace

ContributionLayout make_contribution_layout(core::VideoGeometry geometry,
                                            int search_bwd, int search_fwd,
                                            int patch_time) {
    require(geometry.valid(), "video geometry must be non-empty");
    ContributionLayout layout{};
    layout.width = geometry.width;
    layout.height = geometry.height;
    layout.channels = geometry.channels;
    layout.search_bwd = search_bwd;
    layout.search_fwd = search_fwd;
    layout.patch_time = patch_time;
    layout.slot_count = common::checked_add_int(
        common::checked_add_int(search_bwd, search_fwd,
                                "contribution slot count overflows int"),
        patch_time, "contribution slot count overflows int");
    require_valid_layout(layout);
    return layout;
}

void ContributionStack::resize(ContributionLayout layout) {
    require_valid_layout(layout);
    storage_.resize(layout.value_count());
    layout_ = layout;
}

void ContributionStack::clear() {
    std::fill(storage_.begin(), storage_.end(), 0.0F);
}

void clear_contributions(ContributionStackView stack) {
    const auto layout = stack.layout();
    validate_layout(layout);
    if (stack.data() != nullptr) {
        std::fill_n(stack.data(), layout.value_count(), 0.0F);
        return;
    }

    for (int channel = 0; channel < layout.channels; ++channel) {
        for (int slot = 0; slot < layout.slot_count; ++slot) {
            for (int y = 0; y < layout.height; ++y) {
                std::fill_n(stack.numerator_row(slot, channel, y), layout.width,
                            0.0F);
                std::fill_n(stack.channel_weight_row(slot, channel, y),
                            layout.width, 0.0F);
            }
        }
    }
}

void aggregate_frame(std::span<const ConstContributionStackView> stacks,
                     std::span<const int> anchor_frames, int output_frame,
                     core::ConstVideoView source,
                     core::MutableVideoView output) {
    require(stacks.size() == anchor_frames.size(),
            "stack and anchor arrays must have the same length");

    const core::VideoGeometry geometry = source.geometry();
    require(geometry.same_shape(output.geometry()),
            "source and output geometries must match");
    require(geometry.channels <= 4,
            "aggregate_frame currently supports up to four channels");
    require(output_frame >= 0 && output_frame < geometry.frames,
            "output frame is outside the video");

    for (ConstContributionStackView stack : stacks) {
        const ContributionLayout layout = stack.layout();
        validate_layout(layout);
        require(layout.width == geometry.width &&
                    layout.height == geometry.height &&
                    layout.channels == geometry.channels,
                "contribution stack shape does not match the source");
    }

    std::array<float, 4> numerator{};
    for (int y = 0; y < geometry.height; ++y) {
        for (int x = 0; x < geometry.width; ++x) {
            numerator.fill(0.0F);
            float weight = 0.0F;

            for (std::size_t stack_index = 0; stack_index < stacks.size();
                 ++stack_index) {
                const ConstContributionStackView stack = stacks[stack_index];
                const ContributionLayout layout = stack.layout();
                const int output_offset =
                    output_frame - anchor_frames[stack_index];
                if (!layout.contains_output_offset(output_offset)) {
                    continue;
                }

                const int slot = layout.slot_for_output_offset(output_offset);
                const float stack_weight = stack.weight(slot, x, y);
                if (stack_weight == 0.0F) {
                    continue;
                }

                weight += stack_weight;
                for (int channel = 0; channel < geometry.channels; ++channel) {
                    numerator[static_cast<std::size_t>(channel)] +=
                        stack.numerator(slot, channel, x, y);
                }
            }

            if (weight > 0.0F) {
                const float inv_weight = 1.0F / weight;
                for (int channel = 0; channel < geometry.channels; ++channel) {
                    output.sample(x, y, output_frame, channel) =
                        numerator[static_cast<std::size_t>(channel)] *
                        inv_weight;
                }
            } else {
                for (int channel = 0; channel < geometry.channels; ++channel) {
                    output.sample(x, y, output_frame, channel) =
                        source.sample(x, y, output_frame, channel);
                }
            }
        }
    }
}

} // namespace vnlb::aggregate
