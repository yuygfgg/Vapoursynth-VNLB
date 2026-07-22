#include "cuda/aggregation.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
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

enum class RetentionMode { K, Cap };
enum class MatchPattern { Spread, Cluster };

int parse_positive(const char* value, const char* name) {
    const int parsed = std::stoi(value);
    if (parsed <= 0) {
        throw std::invalid_argument(std::string(name) + " must be positive");
    }
    return parsed;
}

RetentionMode parse_retention_mode(const char* value) {
    const std::string mode(value);
    if (mode == "k") {
        return RetentionMode::K;
    }
    if (mode == "cap") {
        return RetentionMode::Cap;
    }
    throw std::invalid_argument("retention mode must be k or cap");
}

MatchPattern parse_match_pattern(const char* value) {
    const std::string pattern(value);
    if (pattern == "spread") {
        return MatchPattern::Spread;
    }
    if (pattern == "cluster") {
        return MatchPattern::Cluster;
    }
    throw std::invalid_argument("match pattern must be spread or cluster");
}

float next_random(std::uint32_t& state) {
    state = (state * 1664525U) + 1013904223U;
    return static_cast<float>((state >> 8U) & 0x00ffffffU) /
           static_cast<float>(0x01000000U);
}

int retained_count(int group, int requested, int retained_stride,
                   RetentionMode mode) {
    if (mode == RetentionMode::K || retained_stride == requested) {
        return requested;
    }
    if (group == 0) {
        return requested;
    }
    if (group == 1) {
        return retained_stride;
    }
    const std::uint32_t mixed =
        (static_cast<std::uint32_t>(group) * 747796405U) + 2891336453U;
    return requested +
           static_cast<int>(mixed % static_cast<std::uint32_t>(retained_stride -
                                                               requested + 1));
}

template <typename Enqueue>
double measure_average_ms(int iterations, cudaStream_t stream,
                          Enqueue&& enqueue) {
    Event begin;
    Event end;
    check_cuda(cudaEventRecord(begin.get(), stream), "cudaEventRecord(begin)");
    for (int iteration = 0; iteration < iterations; ++iteration) {
        enqueue();
    }
    check_cuda(cudaEventRecord(end.get(), stream), "cudaEventRecord(end)");
    check_cuda(cudaEventSynchronize(end.get()), "cudaEventSynchronize");
    float elapsed_ms = 0.0F;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, begin.get(), end.get()),
               "cudaEventElapsedTime");
    return static_cast<double>(elapsed_ms) / iterations;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc > 8) {
            throw std::invalid_argument(
                "usage: benchmark_cuda_aggregation [K] [C] [groups] "
                "[iterations] [patch_time] [k|cap] [spread|cluster]");
        }

        int device_count = 0;
        check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        if (device_count == 0) {
            std::cerr << "No CUDA device is available\n";
            return EXIT_FAILURE;
        }

        const int requested = argc > 1 ? parse_positive(argv[1], "K") : 17;
        const int retained_stride =
            argc > 2 ? parse_positive(argv[2], "C") : 68;
        const int groups = argc > 3 ? parse_positive(argv[3], "groups") : 256;
        const int iterations =
            argc > 4 ? parse_positive(argv[4], "iterations") : 100;
        const int patch_time =
            argc > 5 ? parse_positive(argv[5], "patch_time") : 1;
        const RetentionMode retention_mode =
            argc > 6 ? parse_retention_mode(argv[6]) : RetentionMode::Cap;
        const MatchPattern match_pattern =
            argc > 7 ? parse_match_pattern(argv[7]) : MatchPattern::Spread;

        constexpr int width = 640;
        constexpr int height = 360;
        constexpr int channels = 3;
        constexpr int patch = 8;
        constexpr int search_window = 19;
        constexpr int search_bwd = 1;
        constexpr int search_fwd = 1;
        constexpr int temporal_origins = search_bwd + search_fwd + 1;
        constexpr int spatial_candidates = search_window * search_window;
        constexpr int candidates = spatial_candidates * temporal_origins;
        if (patch_time != 1 && patch_time != 2) {
            throw std::invalid_argument("patch_time must be 1 or 2");
        }
        if (requested > retained_stride) {
            throw std::invalid_argument("K must not exceed C");
        }
        if (retained_stride > candidates) {
            throw std::invalid_argument(
                "C exceeds the 19x19x3 benchmark candidate set");
        }

        const int slots = search_bwd + search_fwd + patch_time;
        const int sample_dim = channels * patch_time * patch * patch;
        const vnlbcu::AggregationShape shape{
            .max_groups = groups,
            .width = width,
            .height = height,
            .channels = channels,
            .slots = slots,
            .retained_stride = retained_stride,
            .patch_size = patch,
            .patch_time = patch_time,
            .search_window = search_window,
            .search_bwd = search_bwd,
        };
        const std::size_t descriptor_count =
            static_cast<std::size_t>(groups) * retained_stride;
        const std::size_t sample_count = descriptor_count * sample_dim;
        std::vector<float> filtered_samples(sample_count);
        std::vector<float> log_patch_weights(descriptor_count);
        std::vector<int> retained_counts(static_cast<std::size_t>(groups));
        std::vector<vnlbcu::PatchMatch> matches(descriptor_count);

        std::uint32_t random_state = 0x12345678U;
        for (float& sample : filtered_samples) {
            sample = next_random(random_state);
        }
        for (float& log_weight : log_patch_weights) {
            log_weight = -0.5F * next_random(random_state);
        }

        const int half_window = (search_window - 1) / 2;
        const int anchor_x_count = width - patch - (2 * half_window) + 1;
        const int anchor_y_count = height - patch - (2 * half_window) + 1;
        for (int group = 0; group < groups; ++group) {
            retained_counts[static_cast<std::size_t>(group)] = retained_count(
                group, requested, retained_stride, retention_mode);
            const int anchor_x = half_window + ((group * 29) % anchor_x_count);
            const int anchor_y = half_window + ((group * 47) % anchor_y_count);
            const std::size_t base =
                static_cast<std::size_t>(group) * retained_stride;
            for (int sample = 0; sample < retained_stride; ++sample) {
                int temporal = 0;
                int offset_x = 0;
                int offset_y = 0;
                if (match_pattern == MatchPattern::Spread) {
                    // The first K descriptors from successive groups walk a
                    // contiguous sequence modulo W*W*T.  Thus even fixed-K
                    // runs cover all temporal origins and the full window.
                    const int candidate =
                        ((group * requested) + sample) % candidates;
                    temporal = candidate / spatial_candidates;
                    const int spatial = candidate % spatial_candidates;
                    offset_x = (spatial % search_window) - half_window;
                    offset_y = (spatial / search_window) - half_window;
                } else {
                    // A unique 5x5xT neighborhood approximates the common
                    // case where the best matches remain close to the anchor.
                    constexpr int cluster_width = 5;
                    constexpr int cluster_area = cluster_width * cluster_width;
                    const int candidate =
                        sample % (cluster_area * temporal_origins);
                    temporal = candidate / cluster_area;
                    const int spatial = candidate % cluster_area;
                    offset_x = (spatial % cluster_width) - cluster_width / 2;
                    offset_y = (spatial / cluster_width) - cluster_width / 2;
                }
                matches[base + sample] = vnlbcu::PatchMatch{
                    0.0F,
                    anchor_x + offset_x,
                    anchor_y + offset_y,
                    temporal,
                };
            }
        }

        Stream stream;
        DeviceArray<float> device_filtered_samples(sample_count);
        DeviceArray<float> device_log_patch_weights(descriptor_count);
        DeviceArray<int> device_retained_counts(retained_counts.size());
        DeviceArray<vnlbcu::PatchMatch> device_matches(matches.size());
        device_filtered_samples.upload(std::span<const float>(filtered_samples),
                                       stream.get());
        device_log_patch_weights.upload(
            std::span<const float>(log_patch_weights), stream.get());
        device_retained_counts.upload(std::span<const int>(retained_counts),
                                      stream.get());
        device_matches.upload(std::span<const vnlbcu::PatchMatch>(matches),
                              stream.get());

        vnlbcu::Aggregator aggregator;
        aggregator.reserve(shape, 1.0F);
        DeviceArray<float> device_numerators(aggregator.numerator_values());
        DeviceArray<float> device_weights(aggregator.weight_values());
        const vnlbcu::DeviceContributionView contributions =
            vnlbcu::make_contiguous_contribution_view(
                shape, device_numerators.data(), device_weights.data());
        const vnlbcu::AggregationParameters parameters{
            .anchor_frame = search_bwd,
            .log_weight_model_count = 1,
            .log_weight_model_stride = 0,
        };
        const vnlbcu::DeviceAggregationBatch batch{
            .filtered_samples = device_filtered_samples.data(),
            .log_patch_weights = device_log_patch_weights.data(),
            .retained_counts = device_retained_counts.data(),
            .matches = device_matches.data(),
            .groups = groups,
            .contributions = contributions,
        };

        for (int warmup = 0; warmup < 10; ++warmup) {
            aggregator.enqueue_clear(contributions, stream.get());
            aggregator.enqueue_scatter(shape, parameters, batch, stream.get());
        }
        check_cuda(cudaStreamSynchronize(stream.get()),
                   "cudaStreamSynchronize(warmup)");

        const double clear_ms =
            measure_average_ms(iterations, stream.get(), [&] {
                aggregator.enqueue_clear(contributions, stream.get());
            });
        const double scatter_ms =
            measure_average_ms(iterations, stream.get(), [&] {
                aggregator.enqueue_scatter(shape, parameters, batch,
                                           stream.get());
            });
        const double combined_ms =
            measure_average_ms(iterations, stream.get(), [&] {
                aggregator.enqueue_clear(contributions, stream.get());
                aggregator.enqueue_scatter(shape, parameters, batch,
                                           stream.get());
            });
        const auto [minimum_count, maximum_count] =
            std::minmax_element(retained_counts.begin(), retained_counts.end());
        const std::int64_t total_retained = std::accumulate(
            retained_counts.begin(), retained_counts.end(), std::int64_t{0});
        const double average_count =
            static_cast<double>(total_retained) / groups;
        const double contributions_per_batch =
            static_cast<double>(total_retained) * patch_time * patch * patch;
        const double scatter_groups_per_second =
            static_cast<double>(groups) * 1000.0 / scatter_ms;
        const double scatter_contributions_per_second =
            contributions_per_batch * 1000.0 / scatter_ms;
        const double combined_groups_per_second =
            static_cast<double>(groups) * 1000.0 / combined_ms;
        const double combined_contributions_per_second =
            contributions_per_batch * 1000.0 / combined_ms;
        const std::size_t numerator_bytes =
            aggregator.numerator_values() * sizeof(float);
        const std::size_t weight_bytes =
            aggregator.weight_values() * sizeof(float);
        const std::size_t accumulator_bytes = numerator_bytes + weight_bytes;
        const double clear_gigabytes_per_second =
            static_cast<double>(accumulator_bytes) / (clear_ms * 1.0e6);
        constexpr double bytes_per_mib = 1024.0 * 1024.0;

        std::cout
            << std::fixed << std::setprecision(3)
            << "vnlbcu aggregation: P=" << patch << " Pt=" << patch_time
            << " W=" << search_window << " K=" << requested
            << " C=" << retained_stride << " groups=" << groups << " retention="
            << (retention_mode == RetentionMode::K ? "k" : "cap") << " matches="
            << (match_pattern == MatchPattern::Spread ? "spread" : "cluster")
            << '\n'
            << "retained S: min=" << *minimum_count
            << " average=" << average_count << " max=" << *maximum_count << '\n'
            << "average clear: " << clear_ms << " ms ("
            << clear_gigabytes_per_second << " GB/s)\n"
            << "average direct-global scatter: " << scatter_ms << " ms\n"
            << "average clear+scatter: " << combined_ms << " ms\n"
            << "scatter throughput: " << scatter_groups_per_second
            << " groups/s, " << scatter_contributions_per_second / 1.0e6
            << " M contributions/s\n"
            << "clear+scatter throughput: " << combined_groups_per_second
            << " groups/s, " << combined_contributions_per_second / 1.0e6
            << " M contributions/s\n"
            << "accumulator: " << accumulator_bytes << " bytes ("
            << static_cast<double>(accumulator_bytes) / bytes_per_mib
            << " MiB; numerators "
            << static_cast<double>(numerator_bytes) / bytes_per_mib
            << " MiB, weights "
            << static_cast<double>(weight_bytes) / bytes_per_mib << " MiB)\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
