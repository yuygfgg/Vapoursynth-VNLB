#pragma once

#include "common/arithmetic.hpp"
#include "common/validation.hpp"
#include "core/video.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace vnlb::aggregate {

struct ContributionPlaneView {
    float* data = nullptr;
    int stride = 0;
};

struct ContributionLayout {
    int width = 0;
    int height = 0;
    int channels = 0;
    int search_bwd = 0;
    int search_fwd = 0;
    int patch_time = 0;
    int slot_count = 0;

    [[nodiscard]] int plane_pixels() const {
        return common::checked_mul_int(
            width, height, "contribution plane pixel count overflows int");
    }
    [[nodiscard]] int slot_stride() const {
        return common::checked_mul_int(
            common::checked_add_int(channels, 1,
                                    "contribution channel count overflows int"),
            plane_pixels(), "contribution slot stride overflows int");
    }
    [[nodiscard]] int weight_plane_offset(int slot) const {
        return common::checked_add_int(
            common::checked_mul_int(slot, slot_stride(),
                                    "contribution slot offset overflows int"),
            common::checked_mul_int(
                channels, plane_pixels(),
                "contribution weight plane offset overflows int"),
            "contribution weight plane offset overflows int");
    }
    [[nodiscard]] int numerator_plane_offset(int slot, int channel) const {
        return common::checked_add_int(
            common::checked_mul_int(slot, slot_stride(),
                                    "contribution slot offset overflows int"),
            common::checked_mul_int(
                channel, plane_pixels(),
                "contribution numerator plane offset overflows int"),
            "contribution numerator plane offset overflows int");
    }
    [[nodiscard]] int slot_for_output_offset(int output_offset) const {
        return common::checked_add_int(output_offset, search_bwd,
                                       "contribution slot index overflows int");
    }
    [[nodiscard]] bool
    contains_output_offset(int output_offset) const noexcept {
        const auto slot = static_cast<long long>(output_offset) +
                          static_cast<long long>(search_bwd);
        return slot >= 0 && slot < static_cast<long long>(slot_count);
    }
    [[nodiscard]] std::size_t value_count() const {
        return common::checked_mul_size(
            static_cast<std::size_t>(slot_count),
            static_cast<std::size_t>(slot_stride()),
            "contribution stack size overflows size_t");
    }
};

[[nodiscard]] ContributionLayout
make_contribution_layout(core::VideoGeometry geometry, int search_bwd,
                         int search_fwd, int patch_time);

class ConstContributionStackView {
  public:
    constexpr ConstContributionStackView() noexcept = default;
    constexpr ConstContributionStackView(const float* data,
                                         ContributionLayout layout) noexcept
        : data_(data), layout_(layout) {}
    constexpr ConstContributionStackView(
        std::span<const ContributionPlaneView> planes,
        ContributionLayout layout) noexcept
        : planes_(planes.data()), layout_(layout) {}

    [[nodiscard]] constexpr const float* data() const noexcept { return data_; }
    [[nodiscard]] constexpr const ContributionPlaneView*
    planes() const noexcept {
        return planes_;
    }
    [[nodiscard]] constexpr bool has_storage() const noexcept {
        return data_ != nullptr || planes_ != nullptr;
    }
    [[nodiscard]] constexpr ContributionLayout layout() const noexcept {
        return layout_;
    }

    [[nodiscard]] float numerator(int slot, int channel, int x, int y) const {
        if (planes_ != nullptr) {
            const auto plane = planes_[channel];
            return plane.data[(((static_cast<std::ptrdiff_t>(slot) * 2 *
                                 layout_.height) +
                                y) *
                               plane.stride) +
                              x];
        }
        const int plane = layout_.numerator_plane_offset(slot, channel);
        return data_[plane + (y * layout_.width) + x];
    }

    [[nodiscard]] float weight(int slot, int x, int y) const {
        if (planes_ != nullptr) {
            const auto plane = planes_[0];
            return plane.data[(((static_cast<std::ptrdiff_t>(slot) * 2 *
                                 layout_.height) +
                                layout_.height + y) *
                               plane.stride) +
                              x];
        }
        const int plane = layout_.weight_plane_offset(slot);
        return data_[plane + (y * layout_.width) + x];
    }

  private:
    const float* data_ = nullptr;
    const ContributionPlaneView* planes_ = nullptr;
    ContributionLayout layout_{};
};

class ContributionStackView {
  public:
    constexpr ContributionStackView() noexcept = default;
    constexpr ContributionStackView(float* data,
                                    ContributionLayout layout) noexcept
        : data_(data), layout_(layout) {}
    constexpr ContributionStackView(std::span<ContributionPlaneView> planes,
                                    ContributionLayout layout) noexcept
        : planes_(planes.data()), layout_(layout) {}

    [[nodiscard]] constexpr float* data() const noexcept { return data_; }
    [[nodiscard]] constexpr ContributionPlaneView* planes() const noexcept {
        return planes_;
    }
    [[nodiscard]] constexpr bool has_storage() const noexcept {
        return data_ != nullptr || planes_ != nullptr;
    }
    [[nodiscard]] constexpr ContributionLayout layout() const noexcept {
        return layout_;
    }
    [[nodiscard]] constexpr bool has_plane_storage() const noexcept {
        return planes_ != nullptr;
    }

    [[nodiscard]] float& numerator(int slot, int channel, int x, int y) const {
        if (planes_ != nullptr) {
            const auto plane = planes_[channel];
            return plane.data[(((static_cast<std::ptrdiff_t>(slot) * 2 *
                                 layout_.height) +
                                y) *
                               plane.stride) +
                              x];
        }
        const int plane = layout_.numerator_plane_offset(slot, channel);
        return data_[plane + (y * layout_.width) + x];
    }

    [[nodiscard]] float& weight(int slot, int x, int y) const {
        if (planes_ != nullptr) {
            const auto plane = planes_[0];
            return plane.data[(((static_cast<std::ptrdiff_t>(slot) * 2 *
                                 layout_.height) +
                                layout_.height + y) *
                               plane.stride) +
                              x];
        }
        const int plane = layout_.weight_plane_offset(slot);
        return data_[plane + (y * layout_.width) + x];
    }

    [[nodiscard]] float& channel_weight(int slot, int channel, int x,
                                        int y) const {
        if (planes_ != nullptr) {
            const auto plane = planes_[channel];
            return plane.data[(((static_cast<std::ptrdiff_t>(slot) * 2 *
                                 layout_.height) +
                                layout_.height + y) *
                               plane.stride) +
                              x];
        }
        return weight(slot, x, y);
    }

    [[nodiscard]] float* numerator_row(int slot, int channel, int y) const {
        if (planes_ != nullptr) {
            const auto plane = planes_[channel];
            return plane.data +
                   (((static_cast<std::ptrdiff_t>(slot) * 2 * layout_.height) +
                     y) *
                    plane.stride);
        }
        const int plane = layout_.numerator_plane_offset(slot, channel);
        return data_ + plane + (y * layout_.width);
    }

    [[nodiscard]] float* weight_row(int slot, int y) const {
        if (planes_ != nullptr) {
            const auto plane = planes_[0];
            return plane.data +
                   (((static_cast<std::ptrdiff_t>(slot) * 2 * layout_.height) +
                     layout_.height + y) *
                    plane.stride);
        }
        const int plane = layout_.weight_plane_offset(slot);
        return data_ + plane + (y * layout_.width);
    }

    [[nodiscard]] float* channel_weight_row(int slot, int channel,
                                            int y) const {
        if (planes_ != nullptr) {
            const auto plane = planes_[channel];
            return plane.data +
                   (((static_cast<std::ptrdiff_t>(slot) * 2 * layout_.height) +
                     layout_.height + y) *
                    plane.stride);
        }
        return weight_row(slot, y);
    }

    void add_weight(int slot, int x, int y, float value) const {
        if (planes_ != nullptr) {
            for (int channel = 0; channel < layout_.channels; ++channel) {
                channel_weight(slot, channel, x, y) += value;
            }
            return;
        }
        weight(slot, x, y) += value;
    }

    [[nodiscard]] ConstContributionStackView as_const() const noexcept {
        if (planes_ != nullptr) {
            return ConstContributionStackView(
                std::span<const ContributionPlaneView>(
                    planes_, static_cast<std::size_t>(layout_.channels)),
                layout_);
        }
        return ConstContributionStackView(data_, layout_);
    }

  private:
    float* data_ = nullptr;
    ContributionPlaneView* planes_ = nullptr;
    ContributionLayout layout_{};
};

class ContributionStack {
  public:
    ContributionStack() = default;
    explicit ContributionStack(ContributionLayout layout) { resize(layout); }

    void resize(ContributionLayout layout);
    void clear();

    [[nodiscard]] ContributionLayout layout() const noexcept { return layout_; }
    [[nodiscard]] std::span<float> values() noexcept { return storage_; }
    [[nodiscard]] std::span<const float> values() const noexcept {
        return storage_;
    }
    [[nodiscard]] ContributionStackView view() noexcept {
        return ContributionStackView(storage_.data(), layout_);
    }
    [[nodiscard]] ConstContributionStackView view() const noexcept {
        return ConstContributionStackView(storage_.data(), layout_);
    }
    [[nodiscard]] ConstContributionStackView cview() const noexcept {
        return ConstContributionStackView(storage_.data(), layout_);
    }

  private:
    ContributionLayout layout_{};
    std::vector<float> storage_;
};

void clear_contributions(ContributionStackView stack);

void aggregate_frame(std::span<const ConstContributionStackView> stacks,
                     std::span<const int> anchor_frames, int output_frame,
                     core::ConstVideoView source,
                     core::MutableVideoView output);

} // namespace vnlb::aggregate
