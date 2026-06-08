#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace vnlb::core {

struct ConstPlaneView {
    const float* data = nullptr;
    int stride = 0;
};

struct VideoGeometry {
    int width = 0;
    int height = 0;
    int frames = 0;
    int channels = 0;
    int first_frame = 0;
    int total_frames = 0;

    [[nodiscard]] bool valid() const noexcept {
        return width > 0 && height > 0 && frames > 0 && channels > 0;
    }

    [[nodiscard]] int plane_pixels() const noexcept { return width * height; }
    [[nodiscard]] int frame_samples() const noexcept {
        return plane_pixels() * channels;
    }
    [[nodiscard]] int sample_count() const noexcept {
        return frame_samples() * frames;
    }
    [[nodiscard]] int source_frames() const noexcept {
        return total_frames > 0 ? total_frames : frames;
    }

    [[nodiscard]] bool same_shape(VideoGeometry other) const noexcept {
        return width == other.width && height == other.height &&
               frames == other.frames && channels == other.channels &&
               first_frame == other.first_frame &&
               source_frames() == other.source_frames();
    }

    [[nodiscard]] std::size_t checked_sample_count() const {
        if (!valid()) {
            throw std::invalid_argument("video geometry must be non-empty");
        }

        const auto checked_mul = [](std::size_t a, std::size_t b) {
            if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
                throw std::length_error("video geometry overflows size_t");
            }
            return a * b;
        };

        return checked_mul(checked_mul(static_cast<std::size_t>(width),
                                       static_cast<std::size_t>(height)),
                           checked_mul(static_cast<std::size_t>(frames),
                                       static_cast<std::size_t>(channels)));
    }

    [[nodiscard]] int local_frame(int frame) const noexcept {
        return frame - first_frame;
    }

    [[nodiscard]] std::size_t index_local(int x, int y, int frame,
                                          int channel) const noexcept {
        return static_cast<std::size_t>(frame) *
                   static_cast<std::size_t>(frame_samples()) +
               static_cast<std::size_t>(channel) *
                   static_cast<std::size_t>(plane_pixels()) +
               static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
               static_cast<std::size_t>(x);
    }

    [[nodiscard]] std::size_t index(int x, int y, int frame,
                                    int channel) const noexcept {
        return index_local(x, y, local_frame(frame), channel);
    }
};

class ConstVideoView {
  public:
    constexpr ConstVideoView() noexcept = default;
    constexpr ConstVideoView(const float* data, VideoGeometry geometry) noexcept
        : data_(data), geometry_(geometry) {}
    constexpr ConstVideoView(std::span<const ConstPlaneView> planes,
                             VideoGeometry geometry) noexcept
        : planes_(planes.data()), geometry_(geometry) {}

    [[nodiscard]] constexpr const float* data() const noexcept { return data_; }
    [[nodiscard]] constexpr bool has_storage() const noexcept {
        return data_ != nullptr || planes_ != nullptr;
    }
    [[nodiscard]] constexpr VideoGeometry geometry() const noexcept {
        return geometry_;
    }

    [[nodiscard]] float sample(int x, int y, int frame,
                               int channel = 0) const noexcept {
        if (planes_ != nullptr) {
            const int local_frame = geometry_.local_frame(frame);
            const ConstPlaneView plane =
                planes_[local_frame * geometry_.channels + channel];
            return plane.data[y * plane.stride + x];
        }
        return data_[geometry_.index(x, y, frame, channel)];
    }

    [[nodiscard]] const float* plane_data(int frame,
                                          int channel = 0) const noexcept {
        if (planes_ != nullptr) {
            const int local_frame = geometry_.local_frame(frame);
            return planes_[local_frame * geometry_.channels + channel].data;
        }
        return data_ + geometry_.index(0, 0, frame, channel);
    }

    [[nodiscard]] int plane_stride(int frame, int channel = 0) const noexcept {
        if (planes_ != nullptr) {
            const int local_frame = geometry_.local_frame(frame);
            return planes_[local_frame * geometry_.channels + channel].stride;
        }
        return geometry_.width;
    }

    [[nodiscard]] const float* row_data(int frame, int channel, int y,
                                        int x = 0) const noexcept {
        return plane_data(frame, channel) +
               static_cast<std::ptrdiff_t>(y) * plane_stride(frame, channel) +
               x;
    }

  private:
    const float* data_ = nullptr;
    const ConstPlaneView* planes_ = nullptr;
    VideoGeometry geometry_{};
};

class MutableVideoView {
  public:
    constexpr MutableVideoView() noexcept = default;
    constexpr MutableVideoView(float* data, VideoGeometry geometry) noexcept
        : data_(data), geometry_(geometry) {}

    [[nodiscard]] constexpr float* data() const noexcept { return data_; }
    [[nodiscard]] constexpr VideoGeometry geometry() const noexcept {
        return geometry_;
    }

    [[nodiscard]] float& sample(int x, int y, int frame,
                                int channel = 0) const noexcept {
        return data_[geometry_.index(x, y, frame, channel)];
    }

    [[nodiscard]] float* plane_data(int frame, int channel = 0) const noexcept {
        return data_ + geometry_.index(0, 0, frame, channel);
    }

    [[nodiscard]] ConstVideoView as_const() const noexcept {
        return ConstVideoView(data_, geometry_);
    }

  private:
    float* data_ = nullptr;
    VideoGeometry geometry_{};
};

class VideoBuffer {
  public:
    VideoBuffer() = default;
    explicit VideoBuffer(VideoGeometry geometry) { resize(geometry); }
    VideoBuffer(VideoGeometry geometry, float value) {
        resize(geometry);
        fill(value);
    }

    void resize(VideoGeometry geometry) {
        storage_.resize(geometry.checked_sample_count());
        geometry_ = geometry;
    }

    void fill(float value) {
        std::fill(storage_.begin(), storage_.end(), value);
    }

    [[nodiscard]] VideoGeometry geometry() const noexcept { return geometry_; }
    [[nodiscard]] std::span<float> values() noexcept { return storage_; }
    [[nodiscard]] std::span<const float> values() const noexcept {
        return storage_;
    }
    [[nodiscard]] float* data() noexcept { return storage_.data(); }
    [[nodiscard]] const float* data() const noexcept { return storage_.data(); }

    [[nodiscard]] MutableVideoView view() noexcept {
        return MutableVideoView(storage_.data(), geometry_);
    }
    [[nodiscard]] ConstVideoView view() const noexcept {
        return ConstVideoView(storage_.data(), geometry_);
    }
    [[nodiscard]] ConstVideoView cview() const noexcept {
        return ConstVideoView(storage_.data(), geometry_);
    }

  private:
    VideoGeometry geometry_{};
    std::vector<float> storage_;
};

} // namespace vnlb::core
