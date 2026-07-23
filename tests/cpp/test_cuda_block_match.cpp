#include "cuda/block_match.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr int skipped = 77;

void check_cuda(cudaError_t status, std::string_view operation) {
    if (status != cudaSuccess) {
        std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

template <typename T> class DeviceArray {
  public:
    explicit DeviceArray(std::size_t count) : count_(count) {
        if (count_ != 0) {
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_),
                                  count_ * sizeof(T)),
                       "cudaMalloc");
        }
    }

    ~DeviceArray() {
        if (data_ != nullptr) {
            (void)cudaFree(data_);
        }
    }

    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    [[nodiscard]] T* data() noexcept { return data_; }
    [[nodiscard]] const T* data() const noexcept { return data_; }

    void upload(std::span<const T> source, cudaStream_t stream) {
        if (source.size() != count_) {
            throw std::invalid_argument("device upload size mismatch");
        }
        check_cuda(cudaMemcpyAsync(data_, source.data(), count_ * sizeof(T),
                                   cudaMemcpyHostToDevice, stream),
                   "cudaMemcpyAsync host to device");
    }

    void download(std::span<T> destination) const {
        if (destination.size() != count_) {
            throw std::invalid_argument("device download size mismatch");
        }
        check_cuda(cudaMemcpy(destination.data(), data_, count_ * sizeof(T),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy device to host");
    }

  private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

class Stream {
  public:
    Stream() {
        check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                   "cudaStreamCreateWithFlags");
    }
    ~Stream() {
        if (stream_ != nullptr) {
            (void)cudaStreamDestroy(stream_);
        }
    }
    [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

  private:
    cudaStream_t stream_ = nullptr;
};

struct HostWindow {
    int x_low;
    int y_low;
    int frame;
};

struct HostResult {
    std::vector<int> counts;
    std::vector<std::uint32_t> ids;
    std::vector<vnlbcu::PatchMatch> matches;
    std::vector<float> noisy_samples;
    std::vector<float> basic_samples;
};

int origin_count(int axis, int patch) { return axis - patch + 1; }

int temporal_count(const vnlbcu::MatchBatchShape& shape) {
    const long long requested = static_cast<long long>(shape.search_bwd) +
                                shape.search_fwd + 1;
    return static_cast<int>(std::min<long long>(
        requested, origin_count(shape.source_frames, shape.patch_time)));
}

int window_width(const vnlbcu::MatchBatchShape& shape) {
    return std::min(shape.search_window,
                    origin_count(shape.width, shape.patch_size));
}

int window_height(const vnlbcu::MatchBatchShape& shape) {
    return std::min(shape.search_window,
                    origin_count(shape.height, shape.patch_size));
}

int shifted_low(int center, int half, int max_origin) {
    const long long start = static_cast<long long>(center) - half;
    const long long end = static_cast<long long>(center) + half;
    const long long shift =
        std::min(0LL, start) + std::max(0LL, end - max_origin);
    return static_cast<int>(
        std::clamp(start - shift, 0LL, static_cast<long long>(max_origin)));
}

int scheduled_frame(const vnlbcu::MatchBatchShape& shape, int anchor,
                    int slot) {
    if (slot == 0) {
        return anchor;
    }
    const int max_origin = shape.source_frames - shape.patch_time;
    const long long start =
        static_cast<long long>(anchor) - shape.search_bwd;
    const long long end =
        static_cast<long long>(anchor) + shape.search_fwd;
    const long long shift =
        std::min(0LL, start) + std::max(0LL, end - max_origin);
    const long long high = std::min<long long>(max_origin, end - shift);
    const long long low = std::max(0LL, start - shift);
    const long long forward = high - anchor;
    const long long frame = slot <= forward
                                ? static_cast<long long>(anchor) + slot
                                : static_cast<long long>(anchor) -
                                      (slot - forward);
    return static_cast<int>(std::clamp(frame, low, high));
}

HostWindow make_window(const vnlbcu::MatchBatchShape& shape,
                       std::span<const vnlbcu::SearchCenter> centers,
                       vnlbcu::PatchOrigin anchor, int group, int slot) {
    int center_x = anchor.x;
    int center_y = anchor.y;
    int frame = scheduled_frame(shape, anchor.frame, slot);
    if (!centers.empty()) {
        const vnlbcu::SearchCenter center =
            centers[(static_cast<std::size_t>(group) * temporal_count(shape)) +
                    slot];
        center_x = center.x;
        center_y = center.y;
        frame = center.frame;
    }
    const int half = (shape.search_window - 1) / 2;
    return HostWindow{
        shifted_low(center_x, half, shape.width - shape.patch_size),
        shifted_low(center_y, half, shape.height - shape.patch_size), frame};
}

std::size_t video_index(const vnlbcu::MatchBatchShape& shape, int frame,
                        int channel, int y, int x) {
    const int local = frame - shape.first_frame;
    return (((static_cast<std::size_t>(local) * shape.channels + channel) *
                 shape.height +
             y) *
            shape.width) +
           x;
}

float video_sample(std::span<const float> video,
                   const vnlbcu::MatchBatchShape& shape, int frame, int channel,
                   int y, int x) {
    return video[video_index(shape, frame, channel, y, x)];
}

HostResult reference_match(const vnlbcu::MatchBatchShape& shape,
                           const vnlbcu::MatchParameters& parameters,
                           std::span<const float> noisy,
                           std::span<const float> basic,
                           std::span<const vnlbcu::PatchOrigin> anchors,
                           std::span<const vnlbcu::SearchCenter> centers) {
    const int temporal = temporal_count(shape);
    const int wx = window_width(shape);
    const int wy = window_height(shape);
    const int spatial = wx * wy;
    const int candidates = temporal * spatial;
    const int patch_area = shape.patch_size * shape.patch_size;
    const int channel_patch_dim = patch_area * shape.patch_time;
    const int sample_dim = channel_patch_dim * shape.channels;
    const std::size_t descriptor_count =
        static_cast<std::size_t>(shape.groups) * shape.retained_stride;
    const std::size_t sample_count = descriptor_count * sample_dim;
    HostResult result{
        .counts = std::vector<int>(static_cast<std::size_t>(shape.groups)),
        .ids = std::vector<std::uint32_t>(descriptor_count),
        .matches = std::vector<vnlbcu::PatchMatch>(descriptor_count),
        .noisy_samples = std::vector<float>(sample_count),
        .basic_samples = std::vector<float>(sample_count),
    };

    for (int group = 0; group < shape.groups; ++group) {
        const vnlbcu::PatchOrigin anchor =
            anchors[static_cast<std::size_t>(group)];
        const std::size_t descriptor_base =
            static_cast<std::size_t>(group) * shape.retained_stride;
        const std::size_t sample_base = descriptor_base * sample_dim;

        if (shape.requested_similar == 1 &&
            (shape.retained_stride == 1 || parameters.tau == 0.0F)) {
            result.counts[static_cast<std::size_t>(group)] = 1;
            const int half = (shape.search_window - 1) / 2;
            const int x_low =
                shifted_low(anchor.x, half, shape.width - shape.patch_size);
            const int y_low =
                shifted_low(anchor.y, half, shape.height - shape.patch_size);
            result.ids[descriptor_base] = static_cast<std::uint32_t>(
                ((anchor.y - y_low) * wx) + anchor.x - x_low);
            result.matches[descriptor_base] =
                vnlbcu::PatchMatch{0.0F, anchor.x, anchor.y, anchor.frame};
        } else {
            struct Candidate {
                float distance;
                std::uint32_t id;
                int x;
                int y;
                int frame;
            };
            std::vector<Candidate> sorted;
            sorted.reserve(static_cast<std::size_t>(candidates));
            const std::span<const float> search =
                shape.stage == vnlbcu::Stage::Basic ? noisy : basic;
            const int match_channels =
                shape.stage == vnlbcu::Stage::Basic ? 1 : shape.channels;
            for (int slot = 0; slot < temporal; ++slot) {
                const HostWindow window =
                    make_window(shape, centers, anchor, group, slot);
                for (int dy = 0; dy < wy; ++dy) {
                    for (int dx = 0; dx < wx; ++dx) {
                        float distance = 0.0F;
                        for (int channel = 0; channel < match_channels;
                             ++channel) {
                            for (int dt = 0; dt < shape.patch_time; ++dt) {
                                for (int py = 0; py < shape.patch_size; ++py) {
                                    for (int px = 0; px < shape.patch_size;
                                         ++px) {
                                        const float reference = video_sample(
                                            search, shape, anchor.frame + dt,
                                            channel, anchor.y + py,
                                            anchor.x + px);
                                        const float candidate = video_sample(
                                            search, shape, window.frame + dt,
                                            channel, window.y_low + dy + py,
                                            window.x_low + dx + px);
                                        const float difference =
                                            reference - candidate;
                                        distance += difference * difference;
                                    }
                                }
                            }
                        }
                        const std::uint32_t id = static_cast<std::uint32_t>(
                            (slot * spatial) + (dy * wx) + dx);
                        sorted.push_back(
                            Candidate{distance, id, window.x_low + dx,
                                      window.y_low + dy, window.frame});
                    }
                }
            }
            std::stable_sort(sorted.begin(), sorted.end(),
                             [](const Candidate& left, const Candidate& right) {
                                 return left.distance < right.distance;
                             });
            const int effective_similar =
                std::min(shape.requested_similar, candidates);
            const float threshold =
                std::max(parameters.tau,
                         sorted[static_cast<std::size_t>(effective_similar - 1)]
                             .distance);
            int retained = effective_similar;
            while (retained < shape.retained_stride &&
                   sorted[static_cast<std::size_t>(retained)].distance <=
                       threshold) {
                ++retained;
            }
            result.counts[static_cast<std::size_t>(group)] = retained;
            for (int sample = 0; sample < retained; ++sample) {
                const Candidate candidate =
                    sorted[static_cast<std::size_t>(sample)];
                result.ids[descriptor_base + sample] = candidate.id;
                result.matches[descriptor_base + sample] =
                    vnlbcu::PatchMatch{candidate.distance, candidate.x,
                                       candidate.y, candidate.frame};
            }
        }

        const int retained = result.counts[static_cast<std::size_t>(group)];
        for (int sample = 0; sample < retained; ++sample) {
            const vnlbcu::PatchMatch match =
                result.matches[descriptor_base + sample];
            for (int channel = 0; channel < shape.channels; ++channel) {
                for (int dt = 0; dt < shape.patch_time; ++dt) {
                    for (int py = 0; py < shape.patch_size; ++py) {
                        for (int px = 0; px < shape.patch_size; ++px) {
                            const int position =
                                (((channel * shape.patch_time) + dt) *
                                     shape.patch_size +
                                 py) *
                                    shape.patch_size +
                                px;
                            const std::size_t output =
                                sample_base +
                                static_cast<std::size_t>(sample) * sample_dim +
                                position;
                            result.noisy_samples[output] = video_sample(
                                noisy, shape, match.frame + dt, channel,
                                match.y + py, match.x + px);
                            if (shape.stage == vnlbcu::Stage::Final) {
                                result.basic_samples[output] = video_sample(
                                    basic, shape, match.frame + dt, channel,
                                    match.y + py, match.x + px);
                            }
                        }
                    }
                }
            }
        }
    }
    return result;
}

void require_equal(float actual, float expected, std::string_view label) {
    if (actual != expected) {
        std::cerr << label << ": expected " << expected << ", got " << actual
                  << '\n';
        std::exit(EXIT_FAILURE);
    }
}

struct CaseGeometry {
    int width = 8;
    int height = 7;
    int channels = 3;
    int frames = 5;
    int first_frame = 1;
    int source_frames = 7;
    int patch_size = 3;
    int patch_time = 2;
    int search_window = 3;
    int search_bwd = 1;
    int search_fwd = 1;
    int anchor_frame = 3;
    int second_anchor_frame = -1;
    int row_padding = 0;
    int channel_padding = 0;
    int frame_padding = 0;
    bool constant_video = false;
    bool pointer_table = false;
};

void run_case(
    vnlbcu::Stage stage, int requested_similar, bool use_centers,
    bool expand_to_cap, int retained_stride = 0, int reserved_groups = 2,
    CaseGeometry geometry = {},
    vnlbcu::MatchStrategy strategy = vnlbcu::MatchStrategy::Auto,
    vnlbcu::MatchStrategy expected_strategy = vnlbcu::MatchStrategy::Fused) {
    const vnlbcu::MatchBatchShape shape{
        .stage = stage,
        .groups = 2,
        .width = geometry.width,
        .height = geometry.height,
        .channels = geometry.channels,
        .frames = geometry.frames,
        .first_frame = geometry.first_frame,
        .source_frames = geometry.source_frames,
        .patch_size = geometry.patch_size,
        .patch_time = geometry.patch_time,
        .search_window = geometry.search_window,
        .search_bwd = geometry.search_bwd,
        .search_fwd = geometry.search_fwd,
        .requested_similar = requested_similar,
        .retained_stride =
            retained_stride > 0
                ? retained_stride
                : (requested_similar == 1 ? 1 : requested_similar + 4),
        .strategy = strategy,
    };
    const std::size_t video_values = static_cast<std::size_t>(shape.frames) *
                                     shape.channels * shape.height *
                                     shape.width;
    std::vector<float> noisy(video_values);
    std::vector<float> basic(video_values);
    for (int local = 0; local < shape.frames; ++local) {
        const int frame = shape.first_frame + local;
        for (int channel = 0; channel < shape.channels; ++channel) {
            for (int y = 0; y < shape.height; ++y) {
                for (int x = 0; x < shape.width; ++x) {
                    const std::size_t index =
                        video_index(shape, frame, channel, y, x);
                    noisy[index] =
                        geometry.constant_video
                            ? 0.25F
                            : static_cast<float>((frame * 37 + channel * 71 +
                                                  y * 13 + x * 17 + x * y * 3 +
                                                  frame * x * 5) %
                                                 31);
                    // Deliberately unrelated chroma/reference content: Basic
                    // matching must ignore channels 1/2, Final must use them.
                    basic[index] =
                        geometry.constant_video
                            ? 0.5F
                            : static_cast<float>((frame * 19 + channel * 43 +
                                                  y * 11 + x * 7 + x * y * 5 +
                                                  frame * y * 3) %
                                                 23);
                }
            }
        }
    }

    const std::vector<vnlbcu::PatchOrigin> anchors{
        vnlbcu::PatchOrigin{0, 0, geometry.anchor_frame},
        vnlbcu::PatchOrigin{
            shape.width - shape.patch_size, shape.height - shape.patch_size,
            geometry.second_anchor_frame >= 0 ? geometry.second_anchor_frame
                                              : geometry.anchor_frame},
    };
    std::vector<vnlbcu::SearchCenter> centers;
    if (use_centers) {
        const int count = temporal_count(shape);
        centers.resize(static_cast<std::size_t>(shape.groups) * count);
        for (int group = 0; group < shape.groups; ++group) {
            for (int slot = 0; slot < count; ++slot) {
                centers[(static_cast<std::size_t>(group) * count) + slot] =
                    vnlbcu::SearchCenter{
                        group == 0 ? slot : shape.width - 1 - slot,
                        group == 0 ? slot : shape.height - 1 - slot,
                        scheduled_frame(
                            shape,
                            anchors[static_cast<std::size_t>(group)].frame,
                            slot),
                    };
            }
        }
    }

    const vnlbcu::MatchParameters parameters{
        .tau = expand_to_cap ? 1.0e30F : 0.0F,
    };
    const HostResult expected =
        reference_match(shape, parameters, std::span<const float>(noisy),
                        std::span<const float>(basic),
                        std::span<const vnlbcu::PatchOrigin>(anchors),
                        std::span<const vnlbcu::SearchCenter>(centers));

    const int sample_dim =
        shape.channels * shape.patch_time * shape.patch_size * shape.patch_size;
    const std::size_t descriptors =
        static_cast<std::size_t>(shape.groups) * shape.retained_stride;
    const std::size_t samples = descriptors * sample_dim;
    const std::size_t row_stride =
        static_cast<std::size_t>(shape.width + geometry.row_padding);
    const std::size_t channel_stride =
        row_stride * shape.height + geometry.channel_padding;
    const std::size_t frame_stride =
        channel_stride * shape.channels + geometry.frame_padding;
    const std::size_t storage_values =
        static_cast<std::size_t>(shape.frames) * frame_stride;
    std::vector<float> padded_noisy(storage_values, -12345.0F);
    std::vector<float> padded_basic(storage_values, -23456.0F);
    for (int local = 0; local < shape.frames; ++local) {
        const int frame = shape.first_frame + local;
        for (int channel = 0; channel < shape.channels; ++channel) {
            for (int y = 0; y < shape.height; ++y) {
                for (int x = 0; x < shape.width; ++x) {
                    const std::size_t source =
                        video_index(shape, frame, channel, y, x);
                    const std::size_t destination =
                        static_cast<std::size_t>(local) * frame_stride +
                        static_cast<std::size_t>(channel) * channel_stride +
                        static_cast<std::size_t>(y) * row_stride + x;
                    padded_noisy[destination] = noisy[source];
                    padded_basic[destination] = basic[source];
                }
            }
        }
    }
    Stream stream;
    DeviceArray<float> device_noisy(geometry.pointer_table ? 0
                                                           : storage_values);
    DeviceArray<float> device_basic(geometry.pointer_table ? 0
                                                           : storage_values);
    std::vector<std::unique_ptr<DeviceArray<float>>> device_noisy_frames;
    std::vector<std::unique_ptr<DeviceArray<float>>> device_basic_frames;
    std::vector<const float*> noisy_frame_pointers;
    std::vector<const float*> basic_frame_pointers;
    DeviceArray<const float*> device_noisy_frame_pointers(
        geometry.pointer_table ? static_cast<std::size_t>(shape.frames) : 0);
    DeviceArray<const float*> device_basic_frame_pointers(
        geometry.pointer_table ? static_cast<std::size_t>(shape.frames) : 0);
    DeviceArray<vnlbcu::PatchOrigin> device_anchors(anchors.size());
    DeviceArray<vnlbcu::SearchCenter> device_centers(centers.size());
    DeviceArray<int> device_counts(static_cast<std::size_t>(shape.groups));
    DeviceArray<std::uint32_t> device_ids(descriptors);
    DeviceArray<vnlbcu::PatchMatch> device_matches(descriptors);
    DeviceArray<float> device_noisy_samples(samples);
    DeviceArray<float> device_basic_samples(samples);
    if (!geometry.pointer_table) {
        device_noisy.upload(std::span<const float>(padded_noisy), stream.get());
        device_basic.upload(std::span<const float>(padded_basic), stream.get());
    } else {
        device_noisy_frames.reserve(static_cast<std::size_t>(shape.frames));
        device_basic_frames.reserve(static_cast<std::size_t>(shape.frames));
        noisy_frame_pointers.resize(static_cast<std::size_t>(shape.frames));
        basic_frame_pointers.resize(static_cast<std::size_t>(shape.frames));
        // Allocate frames independently and store them in different physical
        // orders.  The logical frame table is therefore neither contiguous
        // nor monotonic, while row/channel padding remains active.
        for (int physical = 0; physical < shape.frames; ++physical) {
            device_noisy_frames.push_back(
                std::make_unique<DeviceArray<float>>(frame_stride));
            device_basic_frames.push_back(
                std::make_unique<DeviceArray<float>>(frame_stride));
        }
        for (int local = 0; local < shape.frames; ++local) {
            const int noisy_physical = shape.frames - 1 - local;
            const int basic_physical = (local + 2) % shape.frames;
            const std::size_t host_offset =
                static_cast<std::size_t>(local) * frame_stride;
            device_noisy_frames[static_cast<std::size_t>(noisy_physical)]
                ->upload(std::span<const float>(padded_noisy)
                             .subspan(host_offset, frame_stride),
                         stream.get());
            device_basic_frames[static_cast<std::size_t>(basic_physical)]
                ->upload(std::span<const float>(padded_basic)
                             .subspan(host_offset, frame_stride),
                         stream.get());
            noisy_frame_pointers[static_cast<std::size_t>(local)] =
                device_noisy_frames[static_cast<std::size_t>(noisy_physical)]
                    ->data();
            basic_frame_pointers[static_cast<std::size_t>(local)] =
                device_basic_frames[static_cast<std::size_t>(basic_physical)]
                    ->data();
        }
        device_noisy_frame_pointers.upload(
            std::span<const float* const>(noisy_frame_pointers), stream.get());
        device_basic_frame_pointers.upload(
            std::span<const float* const>(basic_frame_pointers), stream.get());
    }
    device_anchors.upload(std::span<const vnlbcu::PatchOrigin>(anchors),
                          stream.get());
    if (!centers.empty()) {
        device_centers.upload(std::span<const vnlbcu::SearchCenter>(centers),
                              stream.get());
    }

    const auto make_view = [&](const float* data,
                               const float* const* frame_data) {
        return vnlbcu::DeviceVideoView{
            .data = data,
            .frame_data = frame_data,
            .width = shape.width,
            .height = shape.height,
            .channels = shape.channels,
            .frames = shape.frames,
            .first_frame = shape.first_frame,
            .source_frames = shape.source_frames,
            .row_stride = static_cast<std::ptrdiff_t>(row_stride),
            .channel_stride = static_cast<std::ptrdiff_t>(channel_stride),
            .frame_stride = frame_data == nullptr
                                ? static_cast<std::ptrdiff_t>(frame_stride)
                                : 0,
        };
    };
    const vnlbcu::DeviceMatchBatch batch{
        .noisy =
            make_view(device_noisy.data(), device_noisy_frame_pointers.data()),
        .basic =
            make_view(device_basic.data(), device_basic_frame_pointers.data()),
        .anchors = device_anchors.data(),
        .search_centers = centers.empty() ? nullptr : device_centers.data(),
        .retained_counts = device_counts.data(),
        .candidate_ids = device_ids.data(),
        .matches = device_matches.data(),
        .noisy_samples = device_noisy_samples.data(),
        .basic_samples = stage == vnlbcu::Stage::Final
                             ? device_basic_samples.data()
                             : nullptr,
    };
    vnlbcu::BlockMatcher matcher;
    vnlbcu::MatchBatchShape reserved_shape = shape;
    reserved_shape.groups = reserved_groups;
    reserved_shape.first_frame = 0;
    reserved_shape.frames = shape.first_frame + shape.frames;
    matcher.reserve(reserved_shape);
    if (matcher.strategy() != expected_strategy) {
        throw std::runtime_error("CUDA matcher strategy query mismatch");
    }
    if (matcher.temporal_count() != temporal_count(shape) ||
        matcher.candidate_count() != temporal_count(shape) *
                                         window_width(shape) *
                                         window_height(shape) ||
        matcher.sample_dim() != sample_dim) {
        throw std::runtime_error("CUDA matcher geometry query mismatch");
    }
    matcher.enqueue(shape, parameters, batch, stream.get());
    check_cuda(cudaStreamSynchronize(stream.get()), "cudaStreamSynchronize");

    std::vector<int> actual_counts(static_cast<std::size_t>(shape.groups));
    std::vector<std::uint32_t> actual_ids(descriptors);
    std::vector<vnlbcu::PatchMatch> actual_matches(descriptors);
    std::vector<float> actual_noisy(samples);
    std::vector<float> actual_basic(samples);
    device_counts.download(std::span<int>(actual_counts));
    device_ids.download(std::span<std::uint32_t>(actual_ids));
    device_matches.download(std::span<vnlbcu::PatchMatch>(actual_matches));
    device_noisy_samples.download(std::span<float>(actual_noisy));
    if (stage == vnlbcu::Stage::Final) {
        device_basic_samples.download(std::span<float>(actual_basic));
    }

    for (int group = 0; group < shape.groups; ++group) {
        const int count = expected.counts[static_cast<std::size_t>(group)];
        if (actual_counts[static_cast<std::size_t>(group)] != count) {
            std::cerr << "retained count mismatch\n";
            std::exit(EXIT_FAILURE);
        }
        const std::size_t descriptor_base =
            static_cast<std::size_t>(group) * shape.retained_stride;
        const std::size_t sample_base = descriptor_base * sample_dim;
        for (int sample = 0; sample < count; ++sample) {
            const std::size_t descriptor = descriptor_base + sample;
            if (actual_ids[descriptor] != expected.ids[descriptor]) {
                std::cerr << "candidate id mismatch\n";
                std::exit(EXIT_FAILURE);
            }
            const vnlbcu::PatchMatch actual = actual_matches[descriptor];
            const vnlbcu::PatchMatch wanted = expected.matches[descriptor];
            require_equal(actual.distance, wanted.distance, "match distance");
            if (actual.x != wanted.x || actual.y != wanted.y ||
                actual.frame != wanted.frame) {
                std::cerr << "match coordinate mismatch\n";
                std::exit(EXIT_FAILURE);
            }
            for (int dimension = 0; dimension < sample_dim; ++dimension) {
                const std::size_t index =
                    sample_base +
                    static_cast<std::size_t>(sample) * sample_dim + dimension;
                require_equal(actual_noisy[index],
                              expected.noisy_samples[index],
                              "gathered noisy sample");
                if (stage == vnlbcu::Stage::Final) {
                    require_equal(actual_basic[index],
                                  expected.basic_samples[index],
                                  "gathered basic sample");
                }
            }
        }
    }
}

} // namespace

int main() {
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        device_count == 0) {
        std::cout << "CUDA device unavailable; skipping vnlbcu matcher test\n";
        return skipped;
    }
    check_cuda(status, "cudaGetDeviceCount");

    try {
        run_case(vnlbcu::Stage::Basic, 15, false, false);
        run_case(vnlbcu::Stage::Basic, 17, false, true);
        run_case(vnlbcu::Stage::Final, 15, false, false);
        run_case(vnlbcu::Stage::Final, 17, true, true);
        // Exercise every DeviceVideoView stride independently.  The CPU
        // reference remains tightly packed while both Final-stage device
        // inputs contain sentinel-filled padding between rows, channels, and
        // frames.
        run_case(vnlbcu::Stage::Final, 17, false, true, 0, 2,
                 CaseGeometry{.row_padding = 5,
                              .channel_padding = 7,
                              .frame_padding = 11});
        run_case(vnlbcu::Stage::Final, 17, false, true, 0, 2,
                 CaseGeometry{.row_padding = 5,
                              .channel_padding = 7,
                              .frame_padding = 11,
                              .pointer_table = true});
        // Without motion-compensated centers, boundary shifting must preserve
        // the full asymmetric temporal search.  The two groups cover the
        // first and last legal patch-origin frames in one enqueue.
        run_case(vnlbcu::Stage::Basic, 15, false, false, 0, 2,
                 CaseGeometry{.frames = 9,
                              .first_frame = 0,
                              .source_frames = 9,
                              .patch_time = 2,
                              .search_bwd = 3,
                              .search_fwd = 1,
                              .anchor_frame = 0,
                              .second_anchor_frame = 7});
        run_case(vnlbcu::Stage::Basic, 1, false, true);
        // K=1 still needs the full selector when tau expansion is allowed to
        // retain more than one match.
        run_case(vnlbcu::Stage::Basic, 1, false, true, 4);
        // The public K may exceed the available candidates; the effective K
        // is clamped without changing the retained output stride contract.
        run_case(vnlbcu::Stage::Basic, 31, false, true, 27);
        // All candidate distances tie.  CUB selection must preserve scan
        // order and tau expansion must stop exactly at the retained cap.
        run_case(vnlbcu::Stage::Basic, 15, false, false, 20, 2,
                 CaseGeometry{.constant_video = true});
        // More than one block of temporal slots catches partial initialization
        // and scheduling bugs without increasing the candidate count much.
        run_case(vnlbcu::Stage::Basic, 17, false, true, 68, 2,
                 CaseGeometry{.width = 3,
                              .height = 3,
                              .channels = 1,
                              .frames = 300,
                              .first_frame = 0,
                              .source_frames = 300,
                              .patch_size = 3,
                              .patch_time = 1,
                              .search_window = 1,
                              .search_bwd = 149,
                              .search_fwd = 150,
                              .anchor_frame = 149});
        // A legal but extremely asymmetric radius must be clamped before any
        // device-side addition, so anchor+search_fwd cannot wrap around.
        run_case(vnlbcu::Stage::Basic, 3, false, true, 5, 2,
                 CaseGeometry{
                     .width = 3,
                     .height = 3,
                     .channels = 1,
                     .frames = 5,
                     .first_frame = 0,
                     .source_frames = 5,
                     .patch_size = 3,
                     .patch_time = 1,
                     .search_window = 1,
                     .search_bwd = 0,
                     .search_fwd = std::numeric_limits<int>::max() - 1,
                     .anchor_frame = 2,
                     .second_anchor_frame = 4,
                 });
        // All three selectors must agree at 2048 candidates.  This also
        // exercises ItemsPerThread==8, the static P=8
        // gather path, and enqueueing a smaller final chunk than reserve.
        run_case(vnlbcu::Stage::Basic, 17, false, true, 0, 3,
                 CaseGeometry{.width = 39,
                              .height = 39,
                              .patch_size = 8,
                              .search_window = 33,
                              .search_bwd = 0,
                              .search_fwd = 1},
                 vnlbcu::MatchStrategy::Fused, vnlbcu::MatchStrategy::Fused);
        run_case(vnlbcu::Stage::Basic, 17, false, true, 0, 3,
                 CaseGeometry{.width = 39,
                              .height = 39,
                              .patch_size = 8,
                              .search_window = 33,
                              .search_bwd = 0,
                              .search_fwd = 1},
                 vnlbcu::MatchStrategy::Chunked,
                 vnlbcu::MatchStrategy::Chunked);
        run_case(vnlbcu::Stage::Basic, 17, false, true, 0, 3,
                 CaseGeometry{.width = 39,
                              .height = 39,
                              .patch_size = 8,
                              .search_window = 33,
                              .search_bwd = 0,
                              .search_fwd = 1},
                 vnlbcu::MatchStrategy::FullSort,
                 vnlbcu::MatchStrategy::FullSort);

        // The default 27x27x3 search has N=2187 candidates.  Constant input
        // makes every distance tie, so these cases also verify that the newly
        // admitted 9-item fused selector and both fallback selectors preserve
        // the same candidate scan order.
        const CaseGeometry default_search_geometry{
            .width = 27,
            .height = 27,
            .channels = 1,
            .frames = 3,
            .first_frame = 0,
            .source_frames = 3,
            .patch_size = 1,
            .patch_time = 1,
            .search_window = 27,
            .search_bwd = 1,
            .search_fwd = 1,
            .anchor_frame = 1,
            .constant_video = true,
        };
        run_case(vnlbcu::Stage::Basic, 17, false, true, 68, 2,
                 default_search_geometry, vnlbcu::MatchStrategy::Auto,
                 vnlbcu::MatchStrategy::Fused);
        run_case(vnlbcu::Stage::Basic, 17, false, true, 68, 2,
                 default_search_geometry, vnlbcu::MatchStrategy::Chunked,
                 vnlbcu::MatchStrategy::Chunked);
        run_case(vnlbcu::Stage::Basic, 17, false, true, 68, 2,
                 default_search_geometry, vnlbcu::MatchStrategy::FullSort,
                 vnlbcu::MatchStrategy::FullSort);

        // A non-power-of-two candidate count splits the spatial window into
        // unequal tiles.  Equal distances verify stable scan-order selection
        // across the tile boundary.
        run_case(vnlbcu::Stage::Basic, 17, false, true, 68, 2,
                 CaseGeometry{.width = 47,
                              .height = 47,
                              .channels = 1,
                              .frames = 1,
                              .first_frame = 0,
                              .source_frames = 1,
                              .patch_size = 1,
                              .patch_time = 1,
                              .search_window = 47,
                              .search_bwd = 0,
                              .search_fwd = 0,
                              .anchor_frame = 0,
                              .constant_video = true},
                 vnlbcu::MatchStrategy::Chunked,
                 vnlbcu::MatchStrategy::Chunked);

        // Crossover guards for Auto: 29x29 tiles still favor the
        // segmented sort, while 31x31 tiles amortize the local radix sort.
        run_case(vnlbcu::Stage::Basic, 17, false, true, 68, 2,
                 CaseGeometry{.width = 29,
                              .height = 29,
                              .channels = 1,
                              .frames = 4,
                              .first_frame = 0,
                              .source_frames = 4,
                              .patch_size = 1,
                              .patch_time = 1,
                              .search_window = 29,
                              .search_bwd = 1,
                              .search_fwd = 2,
                              .anchor_frame = 1,
                              .second_anchor_frame = 3},
                 vnlbcu::MatchStrategy::Auto,
                 vnlbcu::MatchStrategy::FullSort);
        run_case(vnlbcu::Stage::Basic, 17, false, true, 68, 2,
                 CaseGeometry{.width = 31,
                              .height = 31,
                              .channels = 1,
                              .frames = 4,
                              .first_frame = 0,
                              .source_frames = 4,
                              .patch_size = 1,
                              .patch_time = 1,
                              .search_window = 31,
                              .search_bwd = 1,
                              .search_fwd = 2,
                              .anchor_frame = 1,
                              .second_anchor_frame = 3},
                 vnlbcu::MatchStrategy::Auto,
                 vnlbcu::MatchStrategy::Chunked);

        // Large tiles remain profitable beyond ten tasks; partial-count
        // feasibility, rather than a hard task limit, bounds this path.
        run_case(vnlbcu::Stage::Basic, 17, false, true, 68, 2,
                 CaseGeometry{.width = 33,
                              .height = 33,
                              .channels = 1,
                              .frames = 11,
                              .first_frame = 0,
                              .source_frames = 11,
                              .patch_size = 1,
                              .patch_time = 1,
                              .search_window = 33,
                              .search_bwd = 5,
                              .search_fwd = 5,
                              .anchor_frame = 5},
                 vnlbcu::MatchStrategy::Auto,
                 vnlbcu::MatchStrategy::Chunked);

        // Paper-scale N=8192: the default-like K/C geometry automatically
        // sends four local tiles into the maximum-size bounded merge.
        run_case(vnlbcu::Stage::Basic, 128, false, true, 512, 2,
                 CaseGeometry{.width = 64,
                              .height = 64,
                              .channels = 1,
                              .frames = 2,
                              .first_frame = 0,
                              .source_frames = 2,
                              .patch_size = 1,
                              .patch_time = 1,
                              .search_window = 65,
                              .search_bwd = 0,
                              .search_fwd = 1,
                              .anchor_frame = 0},
                 vnlbcu::MatchStrategy::Auto,
                 vnlbcu::MatchStrategy::Chunked);

        // Full-sort must also support retaining every paper-scale candidate,
        // rather than only the usual small cap.
        run_case(vnlbcu::Stage::Basic, 17, false, true, 8192, 2,
                 CaseGeometry{.width = 64,
                              .height = 64,
                              .channels = 1,
                              .frames = 2,
                              .first_frame = 0,
                              .source_frames = 2,
                              .patch_size = 1,
                              .patch_time = 1,
                              .search_window = 65,
                              .search_bwd = 0,
                              .search_fwd = 1,
                              .anchor_frame = 0},
                 vnlbcu::MatchStrategy::FullSort,
                 vnlbcu::MatchStrategy::FullSort);

        // N=9477 must automatically select the general full-sort fallback;
        // this also covers Final matching, motion centers, pointer tables, and
        // padded frame storage.
        run_case(vnlbcu::Stage::Final, 60, true, true, 240, 2,
                 CaseGeometry{.width = 27,
                              .height = 27,
                              .channels = 3,
                              .frames = 13,
                              .first_frame = 0,
                              .source_frames = 13,
                              .patch_size = 1,
                              .patch_time = 1,
                              .search_window = 27,
                              .search_bwd = 6,
                              .search_fwd = 6,
                              .anchor_frame = 6,
                              .row_padding = 2,
                              .channel_padding = 3,
                              .frame_padding = 5,
                              .pointer_table = true},
                 vnlbcu::MatchStrategy::Auto,
                 vnlbcu::MatchStrategy::FullSort);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
