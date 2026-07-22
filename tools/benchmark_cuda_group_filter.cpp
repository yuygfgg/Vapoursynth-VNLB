#include "cuda/group_filter.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
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

float next_random(std::uint32_t& state) {
    state = (state * 1664525U) + 1013904223U;
    return static_cast<float>((state >> 8U) & 0x00ffffffU) /
           static_cast<float>(0x01000000U);
}

int parse_positive(const char* value, const char* name) {
    const int parsed = std::stoi(value);
    if (parsed <= 0) {
        throw std::invalid_argument(std::string(name) + " must be positive");
    }
    return parsed;
}

} // namespace

int main(int argc, char** argv) {
    try {
        int device_count = 0;
        check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        if (device_count == 0) {
            std::cerr << "No CUDA device is available\n";
            return EXIT_FAILURE;
        }

        const int basis_similar = argc > 1 ? parse_positive(argv[1], "B") : 17;
        const int sample_dim = argc > 2 ? parse_positive(argv[2], "D") : 192;
        const int retained = argc > 3 ? parse_positive(argv[3], "S") : 68;
        const int groups = argc > 4 ? parse_positive(argv[4], "groups") : 256;
        const int iterations =
            argc > 5 ? parse_positive(argv[5], "iterations") : 100;
        const int rank =
            argc > 6 ? parse_positive(argv[6], "rank")
                     : std::min(basis_similar, 8);
        if (rank > basis_similar) {
            throw std::invalid_argument("rank must not exceed B");
        }

        const vnlbcu::GroupBatchShape shape{
            .groups = groups,
            .retained_stride = retained,
            .sample_dim = sample_dim,
            .basis_similar = basis_similar,
            .rank = rank,
        };
        const std::size_t sample_count =
            static_cast<std::size_t>(groups) * retained * sample_dim;
        const std::size_t weight_count =
            static_cast<std::size_t>(groups) * retained;

        std::vector<float> samples(sample_count);
        std::uint32_t random_state = 0x12345678U;
        for (float& value : samples) {
            value = next_random(random_state);
        }

        Stream stream;
        DeviceArray<float> device_samples(sample_count);
        DeviceArray<float> device_filtered(sample_count);
        DeviceArray<float> device_weights(weight_count);
        device_samples.upload(std::span<const float>(samples), stream.get());

        vnlbcu::GroupFilter filter;
        filter.reserve(shape);
        vnlbcu::FilterParameters parameters{};
        parameters.stage = vnlbcu::Stage::Basic;
        parameters.sigma = 0.08F;
        parameters.detect_equal_groups = false;
        const vnlbcu::DeviceGroupBatch batch{
            .noisy_samples = device_samples.data(),
            .filtered_samples = device_filtered.data(),
            .log_patch_weights = device_weights.data(),
        };

        for (int warmup = 0; warmup < 10; ++warmup) {
            filter.enqueue(shape, parameters, batch, stream.get());
        }
        filter.synchronize_and_check(stream.get());

        Event begin;
        Event end;
        check_cuda(cudaEventRecord(begin.get(), stream.get()),
                   "cudaEventRecord(begin)");
        for (int iteration = 0; iteration < iterations; ++iteration) {
            filter.enqueue(shape, parameters, batch, stream.get());
        }
        check_cuda(cudaEventRecord(end.get(), stream.get()),
                   "cudaEventRecord(end)");
        check_cuda(cudaEventSynchronize(end.get()), "cudaEventSynchronize");

        float elapsed_ms = 0.0F;
        check_cuda(cudaEventElapsedTime(&elapsed_ms, begin.get(), end.get()),
                   "cudaEventElapsedTime");
        filter.synchronize_and_check(stream.get());

        const double batch_ms = static_cast<double>(elapsed_ms) / iterations;
        const double groups_per_second =
            (static_cast<double>(groups) * iterations * 1000.0) / elapsed_ms;
        const double workspace_mib =
            static_cast<double>(filter.workspace_bytes()) / (1024.0 * 1024.0);

        std::cout << std::fixed << std::setprecision(3)
                  << "vnlbcu batch: B=" << basis_similar << " D=" << sample_dim
                  << " S=" << retained << " groups=" << groups
                  << " rank=" << rank << '\n'
                  << "average batch: " << batch_ms << " ms\n"
                  << "throughput: " << groups_per_second << " groups/s\n"
                  << "workspace: " << workspace_mib << " MiB\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
