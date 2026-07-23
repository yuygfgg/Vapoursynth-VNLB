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
#include <string_view>
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
    const std::string text(value);
    std::size_t consumed = 0;
    const int parsed = std::stoi(text, &consumed);
    if (consumed != text.size()) {
        throw std::invalid_argument(std::string(name) + " must be an integer");
    }
    if (parsed <= 0) {
        throw std::invalid_argument(std::string(name) + " must be positive");
    }
    return parsed;
}

int parse_nonnegative(const char* value, const char* name) {
    const std::string text(value);
    std::size_t consumed = 0;
    const int parsed = std::stoi(text, &consumed);
    if (consumed != text.size()) {
        throw std::invalid_argument(std::string(name) + " must be an integer");
    }
    if (parsed < 0) {
        throw std::invalid_argument(std::string(name) +
                                    " must be non-negative");
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

vnlbcu::MatchStrategy parse_strategy(const char* value) {
    const std::string strategy(value);
    if (strategy == "auto") {
        return vnlbcu::MatchStrategy::Auto;
    }
    if (strategy == "fused") {
        return vnlbcu::MatchStrategy::Fused;
    }
    if (strategy == "chunked") {
        return vnlbcu::MatchStrategy::Chunked;
    }
    if (strategy == "full-sort") {
        return vnlbcu::MatchStrategy::FullSort;
    }
    throw std::invalid_argument(
        "strategy must be auto, fused, chunked, or full-sort");
}

const char* strategy_name(vnlbcu::MatchStrategy strategy) {
    switch (strategy) {
    case vnlbcu::MatchStrategy::Auto:
        return "auto";
    case vnlbcu::MatchStrategy::Fused:
        return "fused";
    case vnlbcu::MatchStrategy::Chunked:
        return "chunked";
    case vnlbcu::MatchStrategy::FullSort:
        return "full-sort";
    }
    return "unknown";
}

void print_usage(std::ostream& output) {
    output << "usage: benchmark_cuda_block_match [K] [groups] [iterations] "
              "[basic|final] [topk|cap] [patch_time] "
              "[contiguous|pointers] [options]\n"
           << "options:\n"
           << "  --patch N\n"
           << "  --channels N\n"
           << "  --window N\n"
           << "  --search-bwd N\n"
           << "  --search-fwd N\n"
           << "  --retained N|all\n"
           << "  --strategy auto|fused|chunked|full-sort\n"
           << "  --csv\n";
}

float next_random(std::uint32_t& state) {
    state = (state * 1664525U) + 1013904223U;
    return static_cast<float>((state >> 8U) & 0x00ffffffU) /
           static_cast<float>(0x01000000U);
}

} // namespace

int main(int argc, char** argv) {
    try {
        int requested = 17;
        int groups = 256;
        int iterations = 100;
        vnlbcu::Stage stage = vnlbcu::Stage::Basic;
        bool expand_to_cap = false;
        int patch_time = 1;
        VideoStorage video_storage = VideoStorage::Contiguous;

        int channels = 3;
        int patch = 8;
        int search_window = 19;
        int search_bwd = 1;
        int search_fwd = 1;
        int retained_option = 0;
        bool retain_all = false;
        vnlbcu::MatchStrategy requested_strategy = vnlbcu::MatchStrategy::Auto;
        bool csv = false;

        int argument = 1;
        int positional = 0;
        while (argument < argc &&
               !std::string_view(argv[argument]).starts_with("--")) {
            switch (positional) {
            case 0:
                requested = parse_positive(argv[argument], "K");
                break;
            case 1:
                groups = parse_positive(argv[argument], "groups");
                break;
            case 2:
                iterations = parse_positive(argv[argument], "iterations");
                break;
            case 3:
                stage = parse_stage(argv[argument]);
                break;
            case 4:
                expand_to_cap = parse_expand_to_cap(argv[argument]);
                break;
            case 5:
                patch_time = parse_positive(argv[argument], "patch_time");
                break;
            case 6:
                video_storage = parse_video_storage(argv[argument]);
                break;
            default:
                throw std::invalid_argument(
                    "at most seven positional arguments are accepted");
            }
            ++argument;
            ++positional;
        }

        const auto option_value = [&](std::string_view option) -> const char* {
            if (++argument >= argc) {
                throw std::invalid_argument(std::string(option) +
                                            " requires a value");
            }
            return argv[argument];
        };
        while (argument < argc) {
            const std::string_view option(argv[argument]);
            if (option == "--help") {
                print_usage(std::cout);
                return EXIT_SUCCESS;
            }
            if (option == "--csv") {
                csv = true;
            } else if (option == "--patch") {
                patch = parse_positive(option_value(option), "patch");
            } else if (option == "--channels") {
                channels = parse_positive(option_value(option), "channels");
            } else if (option == "--window") {
                search_window = parse_positive(option_value(option), "window");
            } else if (option == "--search-bwd") {
                search_bwd =
                    parse_nonnegative(option_value(option), "search-bwd");
            } else if (option == "--search-fwd") {
                search_fwd =
                    parse_nonnegative(option_value(option), "search-fwd");
            } else if (option == "--retained") {
                const char* value = option_value(option);
                if (std::string_view(value) == "all") {
                    retain_all = true;
                    retained_option = 0;
                } else {
                    retained_option = parse_positive(value, "retained");
                    retain_all = false;
                }
            } else if (option == "--strategy") {
                requested_strategy = parse_strategy(option_value(option));
            } else {
                throw std::invalid_argument("unknown option: " +
                                            std::string(option));
            }
            ++argument;
        }

        int device_count = 0;
        check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        if (device_count == 0) {
            std::cerr << "No CUDA device is available\n";
            return EXIT_FAILURE;
        }

        constexpr int width = 640;
        constexpr int height = 360;
        if (patch > width || patch > height) {
            throw std::invalid_argument(
                "patch exceeds the benchmark frame dimensions");
        }
        if ((search_window & 1) == 0) {
            throw std::invalid_argument("window must be odd");
        }
        const long long temporal_wide =
            static_cast<long long>(search_bwd) + search_fwd + 1;
        const long long frames_wide =
            static_cast<long long>(patch_time) + temporal_wide - 1;
        if (frames_wide > std::numeric_limits<int>::max()) {
            throw std::length_error("benchmark frame count overflows int");
        }
        const int temporal = static_cast<int>(temporal_wide);
        const int frames = static_cast<int>(frames_wide);
        const int window_width = std::min(search_window, width - patch + 1);
        const int window_height = std::min(search_window, height - patch + 1);
        const long long candidates_wide =
            static_cast<long long>(window_width) * window_height * temporal;
        if (candidates_wide > std::numeric_limits<int>::max()) {
            throw std::length_error("benchmark candidate count overflows int");
        }
        const int candidates = static_cast<int>(candidates_wide);
        if (requested > candidates) {
            throw std::invalid_argument("K exceeds the benchmark candidates");
        }
        const int retained =
            retain_all ? candidates
            : retained_option > 0
                ? retained_option
                : static_cast<int>(std::min<long long>(
                      candidates, static_cast<long long>(requested) * 4));
        if (retained < requested || retained > candidates) {
            throw std::invalid_argument(
                "retained must be between K and the candidate count");
        }
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
            .search_bwd = search_bwd,
            .search_fwd = search_fwd,
            .requested_similar = requested,
            .retained_stride = retained,
            .strategy = requested_strategy,
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
        const char* const stage_name =
            stage == vnlbcu::Stage::Basic ? "basic" : "final";
        const char* const retention_name = expand_to_cap ? "cap" : "topk";
        const char* const storage_name =
            video_storage == VideoStorage::Contiguous ? "contiguous"
                                                      : "pointers";
        if (csv) {
            std::cout
                << "requested_strategy,chosen_strategy,stage,retention,storage,"
                   "channels,patch,patch_time,window,window_width,window_"
                   "height,"
                   "search_bwd,search_fwd,temporal,candidates,K,C,groups,"
                   "iterations,retained_min,retained_average,retained_max,"
                   "batch_ms,workspace_bytes,anchors_per_second,"
                   "candidates_per_second\n"
                << strategy_name(requested_strategy) << ','
                << strategy_name(matcher.strategy()) << ',' << stage_name << ','
                << retention_name << ',' << storage_name << ',' << channels
                << ',' << patch << ',' << patch_time << ',' << search_window
                << ',' << window_width << ',' << window_height << ','
                << search_bwd << ',' << search_fwd << ',' << temporal << ','
                << matcher.candidate_count() << ',' << requested << ','
                << retained << ',' << groups << ',' << iterations << ','
                << *minimum_count << ',' << std::fixed << std::setprecision(6)
                << average_count << ',' << *maximum_count << ',' << batch_ms
                << ',' << matcher.workspace_bytes() << ',' << anchors_per_second
                << ',' << candidates_per_second << '\n';
        } else {
            std::cout << std::fixed << std::setprecision(3)
                      << "vnlbcu matcher: stage=" << stage_name
                      << " retention=" << retention_name
                      << " storage=" << storage_name
                      << " strategy=" << strategy_name(requested_strategy)
                      << "->" << strategy_name(matcher.strategy())
                      << " Ch=" << channels << " P=" << patch
                      << " Pt=" << patch_time << " W=" << search_window
                      << " Wx=" << window_width << " Wy=" << window_height
                      << " Tb=" << search_bwd << " Tf=" << search_fwd
                      << " T=" << temporal << " K=" << requested
                      << " C=" << retained
                      << " candidates=" << matcher.candidate_count()
                      << " groups=" << groups << '\n'
                      << "retained S: min=" << *minimum_count
                      << " average=" << average_count
                      << " max=" << *maximum_count << '\n'
                      << "average batch: " << batch_ms << " ms\n"
                      << "workspace: " << matcher.workspace_bytes()
                      << " bytes\n"
                      << "throughput: " << anchors_per_second << " anchors/s, "
                      << candidates_per_second / 1.0e6
                      << " M candidate patches/s\n";
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
