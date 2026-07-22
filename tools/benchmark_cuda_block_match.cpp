#include "cuda/block_match.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
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

    void download(std::span<T> values) const {
        if (values.size() != count_) {
            throw std::invalid_argument("download size mismatch");
        }
        check_cuda(cudaMemcpy(values.data(), data_, count_ * sizeof(T),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy");
    }

    void upload(std::span<const T> values, cudaStream_t stream) {
        if (values.size() != count_) {
            throw std::invalid_argument("upload size mismatch");
        }
        check_cuda(cudaMemcpyAsync(data_, values.data(), count_ * sizeof(T),
                                   cudaMemcpyHostToDevice, stream),
                   "cudaMemcpyAsync");
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

class Event {
  public:
    Event() { check_cuda(cudaEventCreate(&event_), "cudaEventCreate"); }
    ~Event() {
        if (event_ != nullptr) {
            (void)cudaEventDestroy(event_);
        }
    }
    [[nodiscard]] cudaEvent_t get() const noexcept { return event_; }

  private:
    cudaEvent_t event_ = nullptr;
};

int parse_positive(const char* value, const char* name) {
    const int parsed = std::stoi(value);
    if (parsed <= 0) {
        throw std::invalid_argument(std::string(name) + " must be positive");
    }
    return parsed;
}

vnlbcu::Stage parse_stage(const char* value) {
    const std::string stage(value);
    if (stage == "basic") {
        return vnlbcu::Stage::Basic;
    }
    if (stage == "final") {
        return vnlbcu::Stage::Final;
    }
    throw std::invalid_argument("stage must be basic or final");
}

bool parse_expand_to_cap(const char* value) {
    const std::string mode(value);
    if (mode == "topk") {
        return false;
    }
    if (mode == "cap") {
        return true;
    }
    throw std::invalid_argument("retention mode must be topk or cap");
}

enum class VideoStorage { Contiguous, Pointers };

VideoStorage parse_video_storage(const char* value) {
    const std::string storage(value);
    if (storage == "contiguous") {
        return VideoStorage::Contiguous;
    }
    if (storage == "pointers") {
        return VideoStorage::Pointers;
    }
    throw std::invalid_argument("video storage must be contiguous or pointers");
}

float next_random(std::uint32_t& state) {
    state = (state * 1664525U) + 1013904223U;
    return static_cast<float>((state >> 8U) & 0x00ffffffU) /
           static_cast<float>(0x01000000U);
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc > 8) {
            throw std::invalid_argument(
                "usage: benchmark_cuda_block_match [K] [groups] "
                "[iterations] [basic|final] [topk|cap] [patch_time] "
                "[contiguous|pointers]");
        }
        int device_count = 0;
        check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        if (device_count == 0) {
            std::cerr << "No CUDA device is available\n";
            return EXIT_FAILURE;
        }

        const int requested = argc > 1 ? parse_positive(argv[1], "K") : 17;
        const int groups = argc > 2 ? parse_positive(argv[2], "groups") : 256;
        const int iterations =
            argc > 3 ? parse_positive(argv[3], "iterations") : 100;
        const vnlbcu::Stage stage =
            argc > 4 ? parse_stage(argv[4]) : vnlbcu::Stage::Basic;
        const bool expand_to_cap =
            argc > 5 ? parse_expand_to_cap(argv[5]) : false;
        const int patch_time =
            argc > 6 ? parse_positive(argv[6], "patch_time") : 1;
        const VideoStorage video_storage =
            argc > 7 ? parse_video_storage(argv[7]) : VideoStorage::Contiguous;

        constexpr int width = 640;
        constexpr int height = 360;
        constexpr int channels = 3;
        constexpr int patch = 8;
        constexpr int search_window = 19;
        constexpr int temporal = 3;
        const int frames = patch_time + temporal - 1;
        constexpr int candidates = search_window * search_window * temporal;
        if (requested > candidates) {
            throw std::invalid_argument("K exceeds the benchmark candidates");
        }
        const int retained = std::min(candidates, requested * 4);
        const vnlbcu::MatchBatchShape shape{
            .stage = stage,
            .groups = groups,
            .width = width,
            .height = height,
            .channels = channels,
            .frames = frames,
            .first_frame = 0,
            .source_frames = frames,
            .patch_size = patch,
            .patch_time = patch_time,
            .search_window = search_window,
            .search_bwd = 1,
            .search_fwd = 1,
            .requested_similar = requested,
            .retained_stride = retained,
        };
        const int sample_dim = channels * patch * patch * patch_time;
        const std::size_t video_count =
            static_cast<std::size_t>(frames) * channels * height * width;
        const std::size_t sample_count =
            static_cast<std::size_t>(groups) * retained * sample_dim;

        std::vector<float> noisy(video_count);
        std::vector<float> basic(video_count);
        std::uint32_t random_state = 0x12345678U;
        for (std::size_t index = 0; index < video_count; ++index) {
            noisy[index] = next_random(random_state);
            basic[index] =
                0.8F * noisy[index] + 0.1F * next_random(random_state);
        }
        std::vector<vnlbcu::PatchOrigin> anchors(
            static_cast<std::size_t>(groups));
        const int max_x = width - patch;
        const int max_y = height - patch;
        for (int group = 0; group < groups; ++group) {
            anchors[static_cast<std::size_t>(group)] = vnlbcu::PatchOrigin{
                (group * 29) % (max_x + 1), (group * 47) % (max_y + 1),
                group % temporal};
        }

        Stream stream;
        const std::size_t frame_values =
            static_cast<std::size_t>(channels) * height * width;
        DeviceArray<float> device_noisy(
            video_storage == VideoStorage::Contiguous ? video_count : 0);
        DeviceArray<float> device_basic(
            video_storage == VideoStorage::Contiguous ? video_count : 0);
        std::vector<std::unique_ptr<DeviceArray<float>>> device_noisy_frames;
        std::vector<std::unique_ptr<DeviceArray<float>>> device_basic_frames;
        std::vector<const float*> noisy_frame_pointers;
        std::vector<const float*> basic_frame_pointers;
        DeviceArray<const float*> device_noisy_frame_pointers(
            video_storage == VideoStorage::Pointers
                ? static_cast<std::size_t>(frames)
                : 0);
        DeviceArray<const float*> device_basic_frame_pointers(
            video_storage == VideoStorage::Pointers
                ? static_cast<std::size_t>(frames)
                : 0);
        DeviceArray<vnlbcu::PatchOrigin> device_anchors(anchors.size());
        DeviceArray<int> device_counts(static_cast<std::size_t>(groups));
        DeviceArray<std::uint32_t> device_candidate_ids(
            static_cast<std::size_t>(groups) * retained);
        DeviceArray<vnlbcu::PatchMatch> device_matches(
            static_cast<std::size_t>(groups) * retained);
        DeviceArray<float> device_noisy_samples(sample_count);
        DeviceArray<float> device_basic_samples(sample_count);
        if (video_storage == VideoStorage::Contiguous) {
            device_noisy.upload(std::span<const float>(noisy), stream.get());
            device_basic.upload(std::span<const float>(basic), stream.get());
        } else {
            device_noisy_frames.reserve(static_cast<std::size_t>(frames));
            device_basic_frames.reserve(static_cast<std::size_t>(frames));
            noisy_frame_pointers.reserve(static_cast<std::size_t>(frames));
            basic_frame_pointers.reserve(static_cast<std::size_t>(frames));
            for (int frame = 0; frame < frames; ++frame) {
                auto noisy_frame =
                    std::make_unique<DeviceArray<float>>(frame_values);
                auto basic_frame =
                    std::make_unique<DeviceArray<float>>(frame_values);
                const std::size_t offset =
                    static_cast<std::size_t>(frame) * frame_values;
                noisy_frame->upload(
                    std::span<const float>(noisy).subspan(offset, frame_values),
                    stream.get());
                basic_frame->upload(
                    std::span<const float>(basic).subspan(offset, frame_values),
                    stream.get());
                noisy_frame_pointers.push_back(noisy_frame->data());
                basic_frame_pointers.push_back(basic_frame->data());
                device_noisy_frames.push_back(std::move(noisy_frame));
                device_basic_frames.push_back(std::move(basic_frame));
            }
            device_noisy_frame_pointers.upload(
                std::span<const float* const>(noisy_frame_pointers),
                stream.get());
            device_basic_frame_pointers.upload(
                std::span<const float* const>(basic_frame_pointers),
                stream.get());
        }
        device_anchors.upload(std::span<const vnlbcu::PatchOrigin>(anchors),
                              stream.get());

        const auto make_view = [&](const float* data,
                                   const float* const* frame_data) {
            return vnlbcu::DeviceVideoView{
                .data = data,
                .frame_data = frame_data,
                .width = width,
                .height = height,
                .channels = channels,
                .frames = frames,
                .first_frame = 0,
                .source_frames = frames,
                .row_stride = width,
                .channel_stride = width * height,
                .frame_stride =
                    frame_data == nullptr ? width * height * channels : 0,
            };
        };
        const vnlbcu::DeviceMatchBatch batch{
            .noisy = make_view(device_noisy.data(),
                               device_noisy_frame_pointers.data()),
            .basic = make_view(device_basic.data(),
                               device_basic_frame_pointers.data()),
            .anchors = device_anchors.data(),
            .retained_counts = device_counts.data(),
            .candidate_ids = device_candidate_ids.data(),
            .matches = device_matches.data(),
            .noisy_samples = device_noisy_samples.data(),
            .basic_samples = stage == vnlbcu::Stage::Final
                                 ? device_basic_samples.data()
                                 : nullptr,
        };
        const vnlbcu::MatchParameters parameters{
            .tau = expand_to_cap ? std::numeric_limits<float>::max() : 0.0F,
        };
        vnlbcu::BlockMatcher matcher;
        matcher.reserve(shape);

        for (int warmup = 0; warmup < 10; ++warmup) {
            matcher.enqueue(shape, parameters, batch, stream.get());
        }
        check_cuda(cudaStreamSynchronize(stream.get()),
                   "cudaStreamSynchronize(warmup)");

        Event begin;
        Event end;
        check_cuda(cudaEventRecord(begin.get(), stream.get()),
                   "cudaEventRecord(begin)");
        for (int iteration = 0; iteration < iterations; ++iteration) {
            matcher.enqueue(shape, parameters, batch, stream.get());
        }
        check_cuda(cudaEventRecord(end.get(), stream.get()),
                   "cudaEventRecord(end)");
        check_cuda(cudaEventSynchronize(end.get()), "cudaEventSynchronize");
        float elapsed_ms = 0.0F;
        check_cuda(cudaEventElapsedTime(&elapsed_ms, begin.get(), end.get()),
                   "cudaEventElapsedTime");

        const double batch_ms = static_cast<double>(elapsed_ms) / iterations;
        const double anchors_per_second =
            (static_cast<double>(groups) * iterations * 1000.0) / elapsed_ms;
        const double candidates_per_second =
            anchors_per_second * matcher.candidate_count();
        std::vector<int> retained_counts(static_cast<std::size_t>(groups));
        device_counts.download(std::span<int>(retained_counts));
        const auto [minimum_count, maximum_count] =
            std::minmax_element(retained_counts.begin(), retained_counts.end());
        const double average_count =
            static_cast<double>(std::accumulate(retained_counts.begin(),
                                                retained_counts.end(), 0LL)) /
            groups;
        std::cout << std::fixed << std::setprecision(3)
                  << "vnlbcu matcher: stage="
                  << (stage == vnlbcu::Stage::Basic ? "basic" : "final")
                  << " retention=" << (expand_to_cap ? "cap" : "topk")
                  << " storage="
                  << (video_storage == VideoStorage::Contiguous ? "contiguous"
                                                                : "pointers")
                  << " P=" << patch << " Pt=" << patch_time
                  << " K=" << requested << " C=" << retained
                  << " candidates=" << matcher.candidate_count()
                  << " groups=" << groups << '\n'
                  << "retained S: min=" << *minimum_count
                  << " average=" << average_count << " max=" << *maximum_count
                  << '\n'
                  << "average batch: " << batch_ms << " ms\n"
                  << "throughput: " << anchors_per_second << " anchors/s, "
                  << candidates_per_second / 1.0e6
                  << " M candidate patches/s\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
