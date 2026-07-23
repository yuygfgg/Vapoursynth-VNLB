#include "cuda/group_filter.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <math_constants.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace vnlbcu {
namespace {

constexpr int warp_width = 32;
constexpr int block_threads = 256;
constexpr int warps_per_block = block_threads / warp_width;
constexpr int max_basis_similar = 128;
constexpr float dual_eigenvalue_floor = 128.0F * FLT_EPSILON;
constexpr double small_group_jacobi_tolerance = 1.0e-7;
constexpr double large_group_jacobi_tolerance = 1.0e-6;
// A solver-only diagonal regularizer for exact/zero-signal groups.  The
// filtered path masks these groups' eigenvalues back to zero, so this cannot
// change their pixels; it merely keeps cuSOLVER away from a fully zero Gram
// matrix, which can otherwise report a non-convergence status on some GPUs.
constexpr float degenerate_gram_jitter = 1.0e-8F;
constexpr int exact_equal_group = 1;
constexpr int zero_signal_group = 2;

[[noreturn]] void throw_cuda(cudaError_t status, const char* operation) {
    std::ostringstream message;
    message << operation << " failed: " << cudaGetErrorString(status);
    throw std::runtime_error(message.str());
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw_cuda(status, operation);
    }
}

[[noreturn]] void throw_cublas(cublasStatus_t status,
                               const char* operation) {
    std::ostringstream message;
    message << operation << " failed with cuBLAS status "
            << static_cast<int>(status);
    throw std::runtime_error(message.str());
}

void check_cublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw_cublas(status, operation);
    }
}

[[noreturn]] void throw_cusolver(cusolverStatus_t status,
                                 const char* operation) {
    std::ostringstream message;
    message << operation << " failed with cuSOLVER status "
            << static_cast<int>(status);
    throw std::runtime_error(message.str());
}

void check_cusolver(cusolverStatus_t status, const char* operation) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        throw_cusolver(status, operation);
    }
}

template <typename Kernel>
bool configure_dynamic_shared(Kernel kernel, std::size_t dynamic_bytes,
                              int default_shared_memory,
                              int optin_shared_memory) {
    cudaFuncAttributes attributes{};
    check_cuda(cudaFuncGetAttributes(&attributes, kernel),
               "cudaFuncGetAttributes");
    const std::size_t total =
        dynamic_bytes + static_cast<std::size_t>(attributes.sharedSizeBytes);
    if (total > static_cast<std::size_t>(optin_shared_memory)) {
        return false;
    }
    if (total > static_cast<std::size_t>(default_shared_memory)) {
        const int maximum_dynamic =
            optin_shared_memory - attributes.sharedSizeBytes;
        check_cuda(cudaFuncSetAttribute(
                       kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       maximum_dynamic),
                   "cudaFuncSetAttribute(max dynamic shared memory)");
    }
    return true;
}

std::size_t checked_product(std::size_t left, std::size_t right,
                            const char* label) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::length_error(label);
    }
    return left * right;
}

std::size_t checked_sum(std::size_t left, std::size_t right,
                        const char* label) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        throw std::length_error(label);
    }
    return left + right;
}

template <typename T> class DeviceBuffer {
  public:
    DeviceBuffer() = default;
    ~DeviceBuffer() { reset_noexcept(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          capacity_(std::exchange(other.capacity_, 0)) {}

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            reset_noexcept();
            data_ = std::exchange(other.data_, nullptr);
            capacity_ = std::exchange(other.capacity_, 0);
        }
        return *this;
    }

    void reserve(std::size_t count) {
        if (count <= capacity_) {
            return;
        }

        T* replacement = nullptr;
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&replacement),
                              checked_product(count, sizeof(T),
                                              "CUDA allocation overflows")),
                   "cudaMalloc");
        if (data_ != nullptr) {
            const cudaError_t status = cudaFree(data_);
            if (status != cudaSuccess) {
                (void)cudaFree(replacement);
                throw_cuda(status, "cudaFree");
            }
        }
        data_ = replacement;
        capacity_ = count;
    }

    [[nodiscard]] T* data() noexcept { return data_; }
    [[nodiscard]] const T* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] std::size_t bytes() const noexcept {
        return capacity_ * sizeof(T);
    }

  private:
    void reset_noexcept() noexcept {
        if (data_ != nullptr) {
            (void)cudaFree(data_);
        }
        data_ = nullptr;
        capacity_ = 0;
    }

    T* data_ = nullptr;
    std::size_t capacity_ = 0;
};

__device__ __forceinline__ float warp_sum(float value) {
    constexpr unsigned int mask = 0xffffffffU;
    for (int offset = warp_width / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(mask, value, offset);
    }
    return value;
}

__device__ __forceinline__ float warp_min(float value) {
    constexpr unsigned int mask = 0xffffffffU;
    for (int offset = warp_width / 2; offset > 0; offset >>= 1) {
        value = fminf(value, __shfl_down_sync(mask, value, offset));
    }
    return value;
}

template <int StaticB>
__device__ __forceinline__ int basis_count(int runtime_basis_count) {
    if constexpr (StaticB != 0) {
        return StaticB;
    }
    return runtime_basis_count;
}

__device__ __forceinline__ int retained_count(const int* counts, int group,
                                              int retained_stride) {
    return counts == nullptr ? retained_stride : counts[group];
}

template <int StaticB, bool UseSharedMean>
__global__ void prepare_basic_kernel(
    const float* __restrict__ noisy, const int* __restrict__ retained_counts,
    const std::uint8_t* __restrict__ active_groups,
    int groups, int retained_stride, int sample_dim, int runtime_basis_count,
    bool detect_equal_groups, float model_noise2, float* __restrict__ centers,
    float* __restrict__ centered_noisy, int* __restrict__ equal_flags) {
    extern __shared__ float shared[];
    __shared__ int equal_by_warp[warps_per_block];
    __shared__ float energy_by_warp[warps_per_block];

    const int group = static_cast<int>(blockIdx.x);
    if (group >= groups) {
        return;
    }
    if (active_groups != nullptr && active_groups[group] == 0U) {
        if (threadIdx.x == 0) {
            equal_flags[group] = 0;
        }
        return;
    }
    const int similar = retained_count(retained_counts, group, retained_stride);
    const int model_count = basis_count<StaticB>(runtime_basis_count);
    const std::size_t group_stride =
        static_cast<std::size_t>(retained_stride) * sample_dim;
    const std::size_t group_base =
        static_cast<std::size_t>(group) * group_stride;
    float* const center =
        centers + static_cast<std::size_t>(group) * sample_dim;
    float* const mean = UseSharedMean ? shared : center;

    if (threadIdx.x == 0) {
        equal_flags[group] = 0;
    }

    for (int dimension = static_cast<int>(threadIdx.x); dimension < sample_dim;
         dimension += static_cast<int>(blockDim.x)) {
        float sum = 0.0F;
        for (int sample = 0; sample < model_count; ++sample) {
            sum += noisy[group_base +
                         static_cast<std::size_t>(sample) * sample_dim +
                         dimension];
        }
        const float value = sum / static_cast<float>(model_count);
        mean[dimension] = value;
        center[dimension] = value;
    }
    __syncthreads();

    int local_equal = detect_equal_groups && similar > 1 ? 1 : 0;
    float local_model_energy = 0.0F;
    const std::size_t centered_count =
        static_cast<std::size_t>(similar) * sample_dim;
    for (std::size_t index = threadIdx.x; index < centered_count;
         index += blockDim.x) {
        const int sample = static_cast<int>(index / sample_dim);
        const int dimension = static_cast<int>(index % sample_dim);
        const float value = noisy[group_base + index];
        if (detect_equal_groups && sample > 0 &&
            value != noisy[group_base + dimension]) {
            local_equal = 0;
        }
        const float centered = value - mean[dimension];
        centered_noisy[group_base + index] = centered;
        if (detect_equal_groups && sample < model_count) {
            local_model_energy =
                fmaf(centered, centered, local_model_energy);
        }
    }

    if (detect_equal_groups) {
        const int lane = static_cast<int>(threadIdx.x) & (warp_width - 1);
        const int warp = static_cast<int>(threadIdx.x) / warp_width;
        const int warp_equal =
            __all_sync(0xffffffffU, local_equal != 0) ? 1 : 0;
        const float warp_energy = warp_sum(local_model_energy);
        if (lane == 0) {
            equal_by_warp[warp] = warp_equal;
            energy_by_warp[warp] = warp_energy;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int all_equal = similar > 1 ? 1 : 0;
            float model_energy = 0.0F;
            for (int candidate = 0; candidate < warps_per_block; ++candidate) {
                all_equal &= equal_by_warp[candidate];
                model_energy += energy_by_warp[candidate];
            }
            const float trace = model_energy / static_cast<float>(model_count);
            equal_flags[group] =
                all_equal ? exact_equal_group
                          : (trace <= model_noise2 ? zero_signal_group : 0);
        }
    }
}

template <int StaticB, bool UseSharedMean>
__global__ void prepare_final_kernel(
    const float* __restrict__ noisy, const float* __restrict__ basic,
    const int* __restrict__ retained_counts,
    const std::uint8_t* __restrict__ active_groups,
    const std::uint8_t* __restrict__ flat_flags, int groups,
    int retained_stride, int sample_dim, int runtime_basis_count,
    bool detect_equal_groups, float model_noise2,
    float* __restrict__ model_means,
    float* __restrict__ centers, float* __restrict__ centered_noisy,
    float* __restrict__ centered_model, int* __restrict__ equal_flags) {
    extern __shared__ float shared[];
    __shared__ int equal_by_warp[warps_per_block];
    __shared__ float energy_by_warp[warps_per_block];

    const int group = static_cast<int>(blockIdx.x);
    if (group >= groups) {
        return;
    }
    if (active_groups != nullptr && active_groups[group] == 0U) {
        if (threadIdx.x == 0) {
            equal_flags[group] = 0;
        }
        return;
    }
    const int similar = retained_count(retained_counts, group, retained_stride);
    const int model_count = basis_count<StaticB>(runtime_basis_count);
    const bool flat = flat_flags != nullptr && flat_flags[group] != 0;
    const std::size_t group_stride =
        static_cast<std::size_t>(retained_stride) * sample_dim;
    const std::size_t group_base =
        static_cast<std::size_t>(group) * group_stride;
    float* const model_mean_global =
        model_means + static_cast<std::size_t>(group) * sample_dim;
    float* const center_global =
        centers + static_cast<std::size_t>(group) * sample_dim;
    float* const model_mean = UseSharedMean ? shared : model_mean_global;
    float* const center = UseSharedMean ? shared + sample_dim : center_global;

    if (threadIdx.x == 0) {
        equal_flags[group] = 0;
    }

    for (int dimension = static_cast<int>(threadIdx.x); dimension < sample_dim;
         dimension += static_cast<int>(blockDim.x)) {
        float basic_sum = 0.0F;
        float noisy_sum = 0.0F;
        for (int sample = 0; sample < model_count; ++sample) {
            const std::size_t index =
                group_base + static_cast<std::size_t>(sample) * sample_dim +
                dimension;
            basic_sum += basic[index];
            if (!flat) {
                noisy_sum += noisy[index];
            }
        }
        const float basic_value = basic_sum / static_cast<float>(model_count);
        const float center_value =
            flat ? basic_value : noisy_sum / static_cast<float>(model_count);
        model_mean[dimension] = basic_value;
        center[dimension] = center_value;
        if constexpr (!UseSharedMean) {
            model_mean_global[dimension] = basic_value;
        }
        center_global[dimension] = center_value;
    }
    __syncthreads();

    int local_equal = detect_equal_groups && similar > 1 ? 1 : 0;
    float local_model_energy = 0.0F;
    const std::size_t centered_count =
        static_cast<std::size_t>(similar) * sample_dim;
    for (std::size_t index = threadIdx.x; index < centered_count;
         index += blockDim.x) {
        const int sample = static_cast<int>(index / sample_dim);
        const int dimension = static_cast<int>(index % sample_dim);
        const float noisy_value = noisy[group_base + index];
        if (detect_equal_groups && sample > 0 &&
            noisy_value != noisy[group_base + dimension]) {
            local_equal = 0;
        }
        const float model_value =
            basic[group_base + index] - model_mean[dimension];
        centered_model[group_base + index] = model_value;
        centered_noisy[group_base + index] = noisy_value - center[dimension];
        if (detect_equal_groups && sample < model_count) {
            local_model_energy =
                fmaf(model_value, model_value, local_model_energy);
        }
    }

    if (detect_equal_groups) {
        const int lane = static_cast<int>(threadIdx.x) & (warp_width - 1);
        const int warp = static_cast<int>(threadIdx.x) / warp_width;
        const int warp_equal =
            __all_sync(0xffffffffU, local_equal != 0) ? 1 : 0;
        const float warp_energy = warp_sum(local_model_energy);
        if (lane == 0) {
            equal_by_warp[warp] = warp_equal;
            energy_by_warp[warp] = warp_energy;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int all_equal = similar > 1 ? 1 : 0;
            float model_energy = 0.0F;
            for (int candidate = 0; candidate < warps_per_block; ++candidate) {
                all_equal &= equal_by_warp[candidate];
                model_energy += energy_by_warp[candidate];
            }
            const float trace = model_energy / static_cast<float>(model_count);
            equal_flags[group] =
                all_equal ? exact_equal_group
                          : (trace <= model_noise2 ? zero_signal_group : 0);
        }
    }
}

template <int StaticB, bool UseSharedSamples>
__global__ void gram_kernel(const float* __restrict__ centered_model,
                            const int* __restrict__ equal_flags,
                            const std::uint8_t* __restrict__ active_groups,
                            int groups,
                            int retained_stride, int sample_dim,
                            int runtime_basis_count, float* __restrict__ gram) {
    extern __shared__ float shared_samples[];

    const int group = static_cast<int>(blockIdx.x);
    if (group >= groups) {
        return;
    }
    const int model_count = basis_count<StaticB>(runtime_basis_count);
    const std::size_t sample_group_stride =
        static_cast<std::size_t>(retained_stride) * sample_dim;
    const std::size_t sample_group_base =
        static_cast<std::size_t>(group) * sample_group_stride;
    const std::size_t gram_group_base =
        static_cast<std::size_t>(group) * model_count * model_count;

    const bool inactive =
        active_groups != nullptr && active_groups[group] == 0U;
    if (inactive || equal_flags[group] != 0) {
        const int matrix_elements = model_count * model_count;
        for (int index = static_cast<int>(threadIdx.x); index < matrix_elements;
             index += static_cast<int>(blockDim.x)) {
            gram[gram_group_base + index] =
                (index % (model_count + 1)) == 0
                    ? degenerate_gram_jitter
                    : 0.0F;
        }
        return;
    }

    if constexpr (UseSharedSamples) {
        const std::size_t sample_count =
            static_cast<std::size_t>(model_count) * sample_dim;
        for (std::size_t index = threadIdx.x; index < sample_count;
             index += blockDim.x) {
            shared_samples[index] = centered_model[sample_group_base + index];
        }
        __syncthreads();
    }

    const int lane = static_cast<int>(threadIdx.x) & (warp_width - 1);
    const int warp = static_cast<int>(threadIdx.x) / warp_width;
    const int pair_count = model_count * (model_count + 1) / 2;

    int row = 0;
    int row_start = 0;
    for (int pair = warp; pair < pair_count; pair += warps_per_block) {
        while (pair >= row_start + row + 1) {
            row_start += row + 1;
            ++row;
        }
        const int col = pair - row_start;

        float dot = 0.0F;
        for (int dimension = lane; dimension < sample_dim;
             dimension += warp_width) {
            const std::size_t left_index =
                static_cast<std::size_t>(row) * sample_dim + dimension;
            const std::size_t right_index =
                static_cast<std::size_t>(col) * sample_dim + dimension;
            if constexpr (UseSharedSamples) {
                dot += shared_samples[left_index] * shared_samples[right_index];
            } else {
                dot += centered_model[sample_group_base + left_index] *
                       centered_model[sample_group_base + right_index];
            }
        }
        dot = warp_sum(dot);
        if (lane == 0) {
            const float value = dot / static_cast<float>(model_count);
            // cuSOLVER consumes column-major matrices.
            gram[gram_group_base + static_cast<std::size_t>(col) * model_count +
                 row] = value;
        }
    }
}

__global__ void regularize_flagged_gram_kernel(
    float* __restrict__ gram, const int* __restrict__ flags,
    const std::uint8_t* __restrict__ active_groups, int groups,
    int basis_similar, float diagonal_jitter) {
    const int group = static_cast<int>(blockIdx.x);
    if (group >= groups) {
        return;
    }
    const bool inactive =
        active_groups != nullptr && active_groups[group] == 0U;
    if (!inactive && flags[group] == 0) {
        return;
    }
    const int matrix_elements = basis_similar * basis_similar;
    const std::size_t group_base =
        static_cast<std::size_t>(group) * matrix_elements;
    for (int index = static_cast<int>(threadIdx.x); index < matrix_elements;
         index += static_cast<int>(blockDim.x)) {
        gram[group_base + index] =
            (index % (basis_similar + 1)) == 0 ? diagonal_jitter : 0.0F;
    }
}

template <int StaticB>
__global__ void map_dual_basis_kernel(
    const float* __restrict__ centered_model,
    const float* __restrict__ eigenvectors_column_major,
    const float* __restrict__ eigenvalues_ascending,
    const int* __restrict__ equal_flags,
    const std::uint8_t* __restrict__ active_groups, int groups,
    int retained_stride,
    int sample_dim, int runtime_basis_count, int rank,
    float* __restrict__ selected_eigenvalues, float* __restrict__ basis) {
    const int group = static_cast<int>(blockIdx.x);
    if (group >= groups ||
        (active_groups != nullptr && active_groups[group] == 0U)) {
        return;
    }
    const int model_count = basis_count<StaticB>(runtime_basis_count);
    const int lane = static_cast<int>(threadIdx.x) & (warp_width - 1);
    const int warp = static_cast<int>(threadIdx.x) / warp_width;
    const std::size_t sample_group_base =
        static_cast<std::size_t>(group) * retained_stride * sample_dim;
    const std::size_t matrix_group_base =
        static_cast<std::size_t>(group) * model_count * model_count;
    const std::size_t eigen_group_base =
        static_cast<std::size_t>(group) * model_count;
    const std::size_t selected_group_base =
        static_cast<std::size_t>(group) * rank;
    const std::size_t basis_group_base =
        static_cast<std::size_t>(group) * rank * sample_dim;

    for (int component = warp; component < rank; component += warps_per_block) {
        const int source = model_count - 1 - component;
        const float eigenvalue =
            equal_flags[group] != 0
                ? 0.0F
                : eigenvalues_ascending[eigen_group_base + source];
        if (lane == 0) {
            selected_eigenvalues[selected_group_base + component] = eigenvalue;
        }

        const std::size_t basis_row =
            basis_group_base + static_cast<std::size_t>(component) * sample_dim;
        if (eigenvalue <= dual_eigenvalue_floor) {
            for (int dimension = lane; dimension < sample_dim;
                 dimension += warp_width) {
                basis[basis_row + dimension] = 0.0F;
            }
            continue;
        }

        const float inv_sqrt_eigenvalue = rsqrtf(eigenvalue);
        float norm2 = 0.0F;
        for (int dimension = lane; dimension < sample_dim;
             dimension += warp_width) {
            float value = 0.0F;
            for (int sample = 0; sample < model_count; ++sample) {
                const float coefficient =
                    eigenvectors_column_major[matrix_group_base +
                                              static_cast<std::size_t>(source) *
                                                  model_count +
                                              sample];
                value += coefficient *
                         centered_model[sample_group_base +
                                        static_cast<std::size_t>(sample) *
                                            sample_dim +
                                        dimension];
            }
            value *= inv_sqrt_eigenvalue;
            basis[basis_row + dimension] = value;
            norm2 += value * value;
        }
        norm2 = warp_sum(norm2);
        norm2 = __shfl_sync(0xffffffffU, norm2, 0);
        const float inv_norm =
            norm2 > dual_eigenvalue_floor ? rsqrtf(norm2) : 0.0F;
        for (int dimension = lane; dimension < sample_dim;
             dimension += warp_width) {
            basis[basis_row + dimension] *= inv_norm;
        }
    }
}

template <Stage stage, bool UseSharedBasis, bool UseSharedScores>
__global__ void filter_and_weight_kernel(
    const float* __restrict__ noisy_samples,
    const float* __restrict__ centered_noisy,
    const float* __restrict__ centered_model, const float* __restrict__ centers,
    const float* __restrict__ selected_eigenvalues,
    const float* __restrict__ basis, const int* __restrict__ retained_counts,
    const int* __restrict__ equal_flags,
    const std::uint8_t* __restrict__ active_groups, int groups,
    int retained_stride,
    int sample_dim, int model_count, int rank, FilterParameters parameters,
    float* __restrict__ filtered, float* __restrict__ scores,
    float* __restrict__ log_patch_weights) {
    extern __shared__ float shared[];
    float* const filter_coefficients = shared;
    float* const signal_eigenvalues = filter_coefficients + rank;
    float* const shared_basis = signal_eigenvalues + rank;
    float* const shared_scores =
        shared_basis +
        (UseSharedBasis ? static_cast<std::size_t>(rank) * sample_dim : 0U);
    __shared__ float warp_minima[warps_per_block];
    __shared__ float minimum_score;
    __shared__ float log_group_weight;
    __shared__ float membership_volume;

    const int group = static_cast<int>(blockIdx.x);
    if (group >= groups ||
        (active_groups != nullptr && active_groups[group] == 0U)) {
        return;
    }
    const int similar = retained_count(retained_counts, group, retained_stride);
    const int lane = static_cast<int>(threadIdx.x) & (warp_width - 1);
    const int warp = static_cast<int>(threadIdx.x) / warp_width;
    const std::size_t group_stride =
        static_cast<std::size_t>(retained_stride) * sample_dim;
    const std::size_t group_base =
        static_cast<std::size_t>(group) * group_stride;
    const std::size_t center_base =
        static_cast<std::size_t>(group) * sample_dim;
    const std::size_t eigen_base = static_cast<std::size_t>(group) * rank;
    const std::size_t basis_base =
        static_cast<std::size_t>(group) * rank * sample_dim;
    const std::size_t score_base =
        static_cast<std::size_t>(group) * retained_stride;
    float* const patch_scores =
        UseSharedScores ? shared_scores : scores + score_base;

    if (equal_flags[group] == exact_equal_group) {
        const std::size_t count =
            static_cast<std::size_t>(similar) * sample_dim;
        for (std::size_t index = threadIdx.x; index < count;
             index += blockDim.x) {
            filtered[group_base + index] = noisy_samples[group_base + index];
        }
        if (log_patch_weights != nullptr) {
            const float tau =
                parameters.weight_epsilon + static_cast<float>(sample_dim) /
                                                static_cast<float>(model_count);
            const float default_log =
                parameters.weight_alpha == 0.0F
                    ? 0.0F
                    : -parameters.weight_alpha * logf(tau);
            for (int patch = static_cast<int>(threadIdx.x); patch < similar;
                 patch += static_cast<int>(blockDim.x)) {
                log_patch_weights[score_base + patch] = default_log;
            }
        }
        return;
    }

    const float* group_basis = rank > 0 ? basis + basis_base : basis;
    if constexpr (UseSharedBasis) {
        const std::size_t basis_values =
            static_cast<std::size_t>(rank) * sample_dim;
        for (std::size_t index = threadIdx.x; index < basis_values;
             index += blockDim.x) {
            shared_basis[index] = group_basis[index];
        }
        __syncthreads();
        group_basis = shared_basis;
    }

    const float estimate_noise2 =
        parameters.beta * parameters.sigma * parameters.sigma;
    float model_noise2 = estimate_noise2;
    if constexpr (stage == Stage::Final) {
        model_noise2 =
            parameters.beta * parameters.sigma_basic * parameters.sigma_basic;
    }
    const float filter_threshold =
        parameters.variance_threshold * estimate_noise2;

    for (int component = static_cast<int>(threadIdx.x); component < rank;
         component += static_cast<int>(blockDim.x)) {
        const float observed = selected_eigenvalues[eigen_base + component];
        const float variance = fmaxf(observed - model_noise2, 0.0F);
        filter_coefficients[component] =
            variance > filter_threshold
                ? 1.0F / (1.0F + (estimate_noise2 / variance))
                : 0.0F;
        signal_eigenvalues[component] = variance;
    }
    __syncwarp();

    if (threadIdx.x == 0) {
        float trace = 0.0F;
        float volume = 0.0F;
        const float match_sigma =
            stage == Stage::Final ? parameters.sigma_basic : parameters.sigma;
        const float membership_noise2 = fmaxf(
            match_sigma * match_sigma, parameters.membership_noise_floor *
                                           parameters.sigma * parameters.sigma);
        for (int component = 0; component < rank; ++component) {
            const float lambda = signal_eigenvalues[component];
            if (lambda > 0.0F) {
                trace += lambda / (lambda + estimate_noise2);
                if (parameters.weight_beta != 0.0F) {
                    volume += log1pf(lambda / membership_noise2);
                }
            }
        }
        const float tau =
            parameters.weight_epsilon +
            static_cast<float>(sample_dim) / static_cast<float>(model_count) +
            trace;
        log_group_weight = parameters.weight_alpha == 0.0F
                               ? 0.0F
                               : -parameters.weight_alpha * logf(tau);
        membership_volume = volume / static_cast<float>(sample_dim);
    }
    __syncthreads();

    const float match_sigma =
        stage == Stage::Final ? parameters.sigma_basic : parameters.sigma;
    const float membership_noise2 = fmaxf(
        match_sigma * match_sigma, parameters.membership_noise_floor *
                                       parameters.sigma * parameters.sigma);

    for (int patch = warp; patch < similar; patch += warps_per_block) {
        const std::size_t sample_base =
            group_base + static_cast<std::size_t>(patch) * sample_dim;
        float owned_projection = 0.0F;

        for (int component = 0; component < rank; ++component) {
            float projection = 0.0F;
            const std::size_t basis_row =
                static_cast<std::size_t>(component) * sample_dim;
            for (int dimension = lane; dimension < sample_dim;
                 dimension += warp_width) {
                projection += centered_noisy[sample_base + dimension] *
                              group_basis[basis_row + dimension];
            }
            projection = warp_sum(projection);
            projection = __shfl_sync(0xffffffffU, projection, 0);
            if (lane == component) {
                owned_projection = projection;
            }
        }

        // Keep every lane participating in each shuffle, including the final
        // partial row when sample_dim is not a multiple of the warp width.
        for (int dimension_base = 0; dimension_base < sample_dim;
             dimension_base += warp_width) {
            const int dimension = dimension_base + lane;
            float estimate = dimension < sample_dim
                                 ? centers[center_base + dimension]
                                 : 0.0F;
            for (int component = 0; component < rank; ++component) {
                const float projection =
                    __shfl_sync(0xffffffffU, owned_projection, component);
                const std::size_t basis_row =
                    static_cast<std::size_t>(component) * sample_dim;
                if (dimension < sample_dim) {
                    estimate += filter_coefficients[component] * projection *
                                group_basis[basis_row + dimension];
                }
            }
            if (dimension < sample_dim) {
                filtered[sample_base + dimension] = estimate;
            }
        }

        if (log_patch_weights == nullptr) {
            continue;
        }
        if (parameters.weight_beta == 0.0F) {
            if (lane == 0) {
                log_patch_weights[score_base + patch] = log_group_weight;
            }
            continue;
        }

        float norm2 = 0.0F;
        for (int dimension = lane; dimension < sample_dim;
             dimension += warp_width) {
            const float value = centered_model[sample_base + dimension];
            norm2 += value * value;
        }
        norm2 = warp_sum(norm2);

        if constexpr (stage == Stage::Final) {
            for (int component = 0; component < rank; ++component) {
                float projection = 0.0F;
                const std::size_t basis_row =
                    static_cast<std::size_t>(component) * sample_dim;
                for (int dimension = lane; dimension < sample_dim;
                     dimension += warp_width) {
                    projection += centered_model[sample_base + dimension] *
                                  group_basis[basis_row + dimension];
                }
                projection = warp_sum(projection);
                projection = __shfl_sync(0xffffffffU, projection, 0);
                if (lane == component) {
                    owned_projection = projection;
                }
            }
        }

        float distance = 0.0F;
        float projected_norm2 = 0.0F;
        for (int component = 0; component < rank; ++component) {
            const float projection =
                __shfl_sync(0xffffffffU, owned_projection, component);
            if (lane == 0) {
                const float projection2 = projection * projection;
                distance += projection2 /
                            (signal_eigenvalues[component] + membership_noise2);
                projected_norm2 += projection2;
            }
        }
        if (lane == 0) {
            const float residual_norm2 = fmaxf(norm2 - projected_norm2, 0.0F);
            distance += residual_norm2 / membership_noise2;
            patch_scores[patch] =
                (distance - static_cast<float>(sample_dim)) /
                sqrtf(2.0F * static_cast<float>(sample_dim));
        }
    }

    if (log_patch_weights == nullptr || parameters.weight_beta == 0.0F) {
        return;
    }
    __syncthreads();

    float local_minimum = CUDART_INF_F;
    for (int patch = static_cast<int>(threadIdx.x); patch < similar;
         patch += static_cast<int>(blockDim.x)) {
        local_minimum = fminf(local_minimum, patch_scores[patch]);
    }
    local_minimum = warp_min(local_minimum);
    if (lane == 0) {
        warp_minima[warp] = local_minimum;
    }
    __syncthreads();

    if (warp == 0) {
        float block_minimum =
            lane < warps_per_block ? warp_minima[lane] : CUDART_INF_F;
        block_minimum = warp_min(block_minimum);
        if (lane == 0) {
            minimum_score = block_minimum;
        }
    }
    __syncthreads();

    for (int patch = static_cast<int>(threadIdx.x); patch < similar;
         patch += static_cast<int>(blockDim.x)) {
        const float membership =
            fmaxf(patch_scores[patch] - minimum_score, 0.0F) +
            membership_volume;
        log_patch_weights[score_base + patch] =
            log_group_weight - parameters.weight_beta * membership;
    }
}

template <Stage stage>
__global__ void projected_filter_and_weight_kernel(
    const float* __restrict__ centered_model,
    const float* __restrict__ selected_eigenvalues,
    const int* __restrict__ retained_counts,
    const int* __restrict__ equal_flags,
    const std::uint8_t* __restrict__ active_groups, int groups,
    int retained_stride,
    int sample_dim, int model_count, int rank, FilterParameters parameters,
    float* __restrict__ noisy_projections,
    const float* __restrict__ model_projections,
    float* __restrict__ scores, float* __restrict__ log_patch_weights) {
    extern __shared__ float shared[];
    float* const filter_coefficients = shared;
    float* const signal_eigenvalues = filter_coefficients + rank;
    __shared__ float warp_minima[warps_per_block];
    __shared__ float minimum_score;
    __shared__ float log_group_weight;
    __shared__ float membership_volume;

    const int group = static_cast<int>(blockIdx.x);
    if (group >= groups ||
        (active_groups != nullptr && active_groups[group] == 0U)) {
        return;
    }
    const int similar = retained_count(retained_counts, group, retained_stride);
    const int lane = static_cast<int>(threadIdx.x) & (warp_width - 1);
    const int warp = static_cast<int>(threadIdx.x) / warp_width;
    const std::size_t sample_group_base =
        static_cast<std::size_t>(group) * retained_stride * sample_dim;
    const std::size_t projection_group_base =
        static_cast<std::size_t>(group) * retained_stride * rank;
    const std::size_t eigen_base = static_cast<std::size_t>(group) * rank;
    const std::size_t score_base =
        static_cast<std::size_t>(group) * retained_stride;

    const float estimate_noise2 =
        parameters.beta * parameters.sigma * parameters.sigma;
    float model_noise2 = estimate_noise2;
    if constexpr (stage == Stage::Final) {
        model_noise2 =
            parameters.beta * parameters.sigma_basic * parameters.sigma_basic;
    }
    const float filter_threshold =
        parameters.variance_threshold * estimate_noise2;

    for (int component = static_cast<int>(threadIdx.x); component < rank;
         component += static_cast<int>(blockDim.x)) {
        const float observed = selected_eigenvalues[eigen_base + component];
        const float variance = fmaxf(observed - model_noise2, 0.0F);
        filter_coefficients[component] =
            variance > filter_threshold
                ? 1.0F / (1.0F + (estimate_noise2 / variance))
                : 0.0F;
        signal_eigenvalues[component] = variance;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float trace = 0.0F;
        float volume = 0.0F;
        const float match_sigma =
            stage == Stage::Final ? parameters.sigma_basic : parameters.sigma;
        const float membership_noise2 = fmaxf(
            match_sigma * match_sigma,
            parameters.membership_noise_floor * parameters.sigma *
                parameters.sigma);
        for (int component = 0; component < rank; ++component) {
            const float lambda = signal_eigenvalues[component];
            if (lambda > 0.0F) {
                trace += lambda / (lambda + estimate_noise2);
                if (parameters.weight_beta != 0.0F) {
                    volume += log1pf(lambda / membership_noise2);
                }
            }
        }
        const float tau =
            parameters.weight_epsilon +
            static_cast<float>(sample_dim) / static_cast<float>(model_count) +
            trace;
        log_group_weight = parameters.weight_alpha == 0.0F
                               ? 0.0F
                               : -parameters.weight_alpha * logf(tau);
        membership_volume = volume / static_cast<float>(sample_dim);
    }
    __syncthreads();

    if (equal_flags[group] == exact_equal_group) {
        const std::size_t projection_count =
            static_cast<std::size_t>(similar) * rank;
        for (std::size_t index = threadIdx.x; index < projection_count;
             index += blockDim.x) {
            noisy_projections[projection_group_base + index] = 0.0F;
        }
        if (log_patch_weights != nullptr) {
            for (int patch = static_cast<int>(threadIdx.x); patch < similar;
                 patch += static_cast<int>(blockDim.x)) {
                log_patch_weights[score_base + patch] = log_group_weight;
            }
        }
        return;
    }

    if (log_patch_weights != nullptr && parameters.weight_beta != 0.0F) {
        const float match_sigma = stage == Stage::Final
                                      ? parameters.sigma_basic
                                      : parameters.sigma;
        const float membership_noise2 = fmaxf(
            match_sigma * match_sigma,
            parameters.membership_noise_floor * parameters.sigma *
                parameters.sigma);
        const float* const membership_projections =
            stage == Stage::Final ? model_projections : noisy_projections;

        for (int patch = warp; patch < similar; patch += warps_per_block) {
            const std::size_t sample_base =
                sample_group_base +
                static_cast<std::size_t>(patch) * sample_dim;
            float norm2 = 0.0F;
            for (int dimension = lane; dimension < sample_dim;
                 dimension += warp_width) {
                const float value = centered_model[sample_base + dimension];
                norm2 = fmaf(value, value, norm2);
            }
            norm2 = warp_sum(norm2);

            if (lane == 0) {
                const std::size_t projection_base =
                    projection_group_base +
                    static_cast<std::size_t>(patch) * rank;
                float distance = 0.0F;
                float projected_norm2 = 0.0F;
                for (int component = 0; component < rank; ++component) {
                    const float projection =
                        membership_projections[projection_base + component];
                    const float projection2 = projection * projection;
                    distance += projection2 /
                                (signal_eigenvalues[component] +
                                 membership_noise2);
                    projected_norm2 += projection2;
                }
                const float residual_norm2 =
                    fmaxf(norm2 - projected_norm2, 0.0F);
                distance += residual_norm2 / membership_noise2;
                scores[score_base + patch] =
                    (distance - static_cast<float>(sample_dim)) /
                    sqrtf(2.0F * static_cast<float>(sample_dim));
            }
        }
        __syncthreads();

        float local_minimum = CUDART_INF_F;
        for (int patch = static_cast<int>(threadIdx.x); patch < similar;
             patch += static_cast<int>(blockDim.x)) {
            local_minimum =
                fminf(local_minimum, scores[score_base + patch]);
        }
        local_minimum = warp_min(local_minimum);
        if (lane == 0) {
            warp_minima[warp] = local_minimum;
        }
        __syncthreads();

        if (warp == 0) {
            float block_minimum =
                lane < warps_per_block ? warp_minima[lane] : CUDART_INF_F;
            block_minimum = warp_min(block_minimum);
            if (lane == 0) {
                minimum_score = block_minimum;
            }
        }
        __syncthreads();

        for (int patch = static_cast<int>(threadIdx.x); patch < similar;
             patch += static_cast<int>(blockDim.x)) {
            const float membership =
                fmaxf(scores[score_base + patch] - minimum_score, 0.0F) +
                membership_volume;
            log_patch_weights[score_base + patch] =
                log_group_weight - parameters.weight_beta * membership;
        }
    } else if (log_patch_weights != nullptr) {
        for (int patch = static_cast<int>(threadIdx.x); patch < similar;
             patch += static_cast<int>(blockDim.x)) {
            log_patch_weights[score_base + patch] = log_group_weight;
        }
    }
    __syncthreads();

    const std::size_t projection_count =
        static_cast<std::size_t>(similar) * rank;
    for (std::size_t index = threadIdx.x; index < projection_count;
         index += blockDim.x) {
        const int component = static_cast<int>(index % rank);
        noisy_projections[projection_group_base + index] *=
            filter_coefficients[component];
    }
}

__global__ void add_filter_centers_kernel(
    const float* __restrict__ centered_noisy,
    const float* __restrict__ centers,
    const int* __restrict__ retained_counts,
    const int* __restrict__ equal_flags,
    const std::uint8_t* __restrict__ active_groups, int groups,
    int retained_stride,
    int sample_dim, float* __restrict__ filtered) {
    const int group = static_cast<int>(blockIdx.x);
    if (group >= groups ||
        (active_groups != nullptr && active_groups[group] == 0U)) {
        return;
    }
    const int similar = retained_count(retained_counts, group, retained_stride);
    const std::size_t group_base =
        static_cast<std::size_t>(group) * retained_stride * sample_dim;
    const std::size_t center_base =
        static_cast<std::size_t>(group) * sample_dim;
    const std::size_t count = static_cast<std::size_t>(similar) * sample_dim;
    const bool exact = equal_flags[group] == exact_equal_group;
    for (std::size_t index = threadIdx.x; index < count;
         index += blockDim.x) {
        const int dimension = static_cast<int>(index % sample_dim);
        const float center = centers[center_base + dimension];
        filtered[group_base + index] =
            exact ? centered_noisy[group_base + index] + center
                  : filtered[group_base + index] + center;
    }
}

void validate_shape(const GroupBatchShape& shape) {
    if (shape.groups <= 0) {
        throw std::invalid_argument("CUDA group batch must contain groups");
    }
    if (shape.retained_stride <= 0 || shape.sample_dim <= 0) {
        throw std::invalid_argument(
            "CUDA group batch dimensions must be positive");
    }
    if (shape.basis_similar <= 0 || shape.basis_similar > max_basis_similar) {
        throw std::invalid_argument(
            "CUDA dual PCA currently supports 1..128 model samples");
    }
    if (shape.basis_similar > shape.retained_stride) {
        throw std::invalid_argument(
            "basis_similar exceeds retained sample stride");
    }
    if (shape.basis_similar >= shape.sample_dim) {
        throw std::invalid_argument(
            "CUDA group filter currently requires the dual-PCA path (B < D)");
    }
    if (shape.rank < 0 || shape.rank > shape.basis_similar) {
        throw std::invalid_argument("CUDA group rank must be in [0, B]");
    }
}

void validate_parameters(const FilterParameters& parameters) {
    if (parameters.stage != Stage::Basic && parameters.stage != Stage::Final) {
        throw std::invalid_argument("invalid CUDA group filter stage");
    }
    const auto finite = [](float value) { return std::isfinite(value); };
    if (!finite(parameters.sigma) || parameters.sigma <= 0.0F ||
        !finite(parameters.sigma_basic) || parameters.sigma_basic < 0.0F ||
        !finite(parameters.beta) || parameters.beta <= 0.0F ||
        !finite(parameters.variance_threshold) ||
        !finite(parameters.weight_alpha) || parameters.weight_alpha < 0.0F ||
        !finite(parameters.weight_beta) || parameters.weight_beta < 0.0F ||
        !finite(parameters.weight_epsilon) ||
        parameters.weight_epsilon <= 0.0F ||
        !finite(parameters.membership_noise_floor) ||
        parameters.membership_noise_floor <= 0.0F) {
        throw std::invalid_argument("invalid CUDA group filter parameters");
    }
}

void validate_batch(const FilterParameters& parameters,
                    const DeviceGroupBatch& batch) {
    if (batch.noisy_samples == nullptr || batch.filtered_samples == nullptr) {
        throw std::invalid_argument(
            "CUDA group filter requires noisy input and filtered output");
    }
    if (parameters.stage == Stage::Final && batch.basic_samples == nullptr) {
        throw std::invalid_argument(
            "CUDA Final group filter requires basic samples");
    }
}

template <int StaticB>
void launch_prepare(const GroupBatchShape& shape,
                    const FilterParameters& parameters,
                    const DeviceGroupBatch& batch, float* model_means,
                    float* centers, float* centered_noisy,
                    float* centered_model, int* equal_flags, bool basic_shared,
                    bool final_shared, cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned int>(shape.groups));
    const dim3 block(block_threads);
    const float estimate_noise2 =
        parameters.beta * parameters.sigma * parameters.sigma;
    const float model_noise2 =
        parameters.stage == Stage::Final
            ? parameters.beta * parameters.sigma_basic * parameters.sigma_basic
            : estimate_noise2;
    if (parameters.stage == Stage::Basic) {
        const std::size_t shared_bytes =
            static_cast<std::size_t>(shape.sample_dim) * sizeof(float);
        if (basic_shared) {
            prepare_basic_kernel<StaticB, true>
                <<<grid, block, shared_bytes, stream>>>(
                    batch.noisy_samples, batch.retained_counts,
                    batch.active_groups, shape.groups, shape.retained_stride,
                    shape.sample_dim,
                    shape.basis_similar, parameters.detect_equal_groups,
                    model_noise2, centers, centered_noisy, equal_flags);
        } else {
            prepare_basic_kernel<StaticB, false><<<grid, block, 0, stream>>>(
                batch.noisy_samples, batch.retained_counts,
                batch.active_groups, shape.groups, shape.retained_stride,
                shape.sample_dim, shape.basis_similar,
                parameters.detect_equal_groups, model_noise2, centers,
                centered_noisy, equal_flags);
        }
    } else {
        const std::size_t shared_bytes =
            static_cast<std::size_t>(shape.sample_dim) * 2 * sizeof(float);
        if (final_shared) {
            prepare_final_kernel<StaticB, true>
                <<<grid, block, shared_bytes, stream>>>(
                    batch.noisy_samples, batch.basic_samples,
                    batch.retained_counts, batch.active_groups,
                    batch.flat_flags, shape.groups, shape.retained_stride,
                    shape.sample_dim,
                    shape.basis_similar, parameters.detect_equal_groups,
                    model_noise2, model_means, centers, centered_noisy,
                    centered_model, equal_flags);
        } else {
            prepare_final_kernel<StaticB, false><<<grid, block, 0, stream>>>(
                batch.noisy_samples, batch.basic_samples, batch.retained_counts,
                batch.active_groups, batch.flat_flags, shape.groups,
                shape.retained_stride, shape.sample_dim, shape.basis_similar,
                parameters.detect_equal_groups, model_noise2, model_means,
                centers, centered_noisy, centered_model, equal_flags);
        }
    }
    check_cuda(cudaPeekAtLastError(), "launch group preparation kernel");
}

void launch_regularize_flagged_gram(
    const GroupBatchShape& shape, float* gram, const int* flags,
    const std::uint8_t* active_groups, cudaStream_t stream) {
    regularize_flagged_gram_kernel<<<static_cast<unsigned int>(shape.groups),
                                     block_threads, 0, stream>>>(
        gram, flags, active_groups, shape.groups, shape.basis_similar,
        degenerate_gram_jitter);
    check_cuda(cudaPeekAtLastError(),
               "launch flagged Gram regularization kernel");
}

template <int StaticB>
void launch_gram(const GroupBatchShape& shape, const float* centered_model,
                 const int* equal_flags,
                 const std::uint8_t* active_groups, float* gram,
                 bool use_shared_samples, cudaStream_t stream) {
    const std::size_t shared_bytes =
        static_cast<std::size_t>(shape.basis_similar) * shape.sample_dim *
        sizeof(float);
    const dim3 grid(static_cast<unsigned int>(shape.groups));
    const dim3 block(block_threads);
    if (use_shared_samples) {
        gram_kernel<StaticB, true><<<grid, block, shared_bytes, stream>>>(
            centered_model, equal_flags, active_groups, shape.groups,
            shape.retained_stride, shape.sample_dim, shape.basis_similar, gram);
    } else {
        gram_kernel<StaticB, false><<<grid, block, 0, stream>>>(
            centered_model, equal_flags, active_groups, shape.groups,
            shape.retained_stride, shape.sample_dim, shape.basis_similar, gram);
    }
    check_cuda(cudaPeekAtLastError(), "launch Gram kernel");
}

template <int StaticB>
void launch_map_basis(const GroupBatchShape& shape, const float* centered_model,
                      const float* gram, const float* raw_eigenvalues,
                      const int* equal_flags,
                      const std::uint8_t* active_groups,
                      float* selected_eigenvalues, float* basis,
                      cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned int>(shape.groups));
    const dim3 block(block_threads);
    map_dual_basis_kernel<StaticB><<<grid, block, 0, stream>>>(
        centered_model, gram, raw_eigenvalues, equal_flags, active_groups,
        shape.groups, shape.retained_stride, shape.sample_dim,
        shape.basis_similar, shape.rank, selected_eigenvalues, basis);
    check_cuda(cudaPeekAtLastError(), "launch dual-basis mapping kernel");
}

template <Stage stage>
void launch_filter(const GroupBatchShape& shape,
                   const FilterParameters& parameters,
                   const DeviceGroupBatch& batch, const float* centered_noisy,
                   const float* centered_model, const float* centers,
    const float* selected_eigenvalues, const float* basis,
    const int* equal_flags, float* scores, bool use_shared_basis,
    bool use_shared_scores, cudaStream_t stream) {
    const std::size_t control_values = checked_product(
        static_cast<std::size_t>(shape.rank), 2U,
        "CUDA filter control cache size overflows");
    std::size_t shared_values = control_values;
    if (use_shared_basis) {
        shared_values = checked_sum(
            shared_values,
            checked_product(static_cast<std::size_t>(shape.rank),
                            static_cast<std::size_t>(shape.sample_dim),
                            "CUDA shared basis size overflows"),
            "CUDA filter shared memory size overflows");
    }
    if (use_shared_scores) {
        shared_values = checked_sum(
            shared_values, static_cast<std::size_t>(shape.retained_stride),
            "CUDA filter shared memory size overflows");
    }
    const std::size_t shared_bytes =
        checked_product(shared_values, sizeof(float),
                        "CUDA filter shared memory size overflows");
    const dim3 grid(static_cast<unsigned int>(shape.groups));
    const dim3 block(block_threads);

    if (use_shared_basis) {
        if (use_shared_scores) {
            filter_and_weight_kernel<stage, true, true>
                <<<grid, block, shared_bytes, stream>>>(
                    batch.noisy_samples, centered_noisy, centered_model,
                    centers, selected_eigenvalues, basis,
                    batch.retained_counts, equal_flags, batch.active_groups,
                    shape.groups, shape.retained_stride, shape.sample_dim,
                    shape.basis_similar, shape.rank, parameters,
                    batch.filtered_samples, scores, batch.log_patch_weights);
        } else {
            filter_and_weight_kernel<stage, true, false>
                <<<grid, block, shared_bytes, stream>>>(
                    batch.noisy_samples, centered_noisy, centered_model,
                    centers, selected_eigenvalues, basis,
                    batch.retained_counts, equal_flags, batch.active_groups,
                    shape.groups, shape.retained_stride, shape.sample_dim,
                    shape.basis_similar, shape.rank, parameters,
                    batch.filtered_samples, scores, batch.log_patch_weights);
        }
    } else {
        if (use_shared_scores) {
            filter_and_weight_kernel<stage, false, true>
                <<<grid, block, shared_bytes, stream>>>(
                    batch.noisy_samples, centered_noisy, centered_model,
                    centers, selected_eigenvalues, basis,
                    batch.retained_counts, equal_flags, batch.active_groups,
                    shape.groups, shape.retained_stride, shape.sample_dim,
                    shape.basis_similar, shape.rank, parameters,
                    batch.filtered_samples, scores, batch.log_patch_weights);
        } else {
            filter_and_weight_kernel<stage, false, false>
                <<<grid, block, shared_bytes, stream>>>(
                    batch.noisy_samples, centered_noisy, centered_model,
                    centers, selected_eigenvalues, basis,
                    batch.retained_counts, equal_flags, batch.active_groups,
                    shape.groups, shape.retained_stride, shape.sample_dim,
                    shape.basis_similar, shape.rank, parameters,
                    batch.filtered_samples, scores, batch.log_patch_weights);
        }
    }
    check_cuda(cudaPeekAtLastError(), "launch filter/weight kernel");
}

template <Stage stage>
void launch_projected_filter(
    const GroupBatchShape& shape, const FilterParameters& parameters,
    const DeviceGroupBatch& batch, const float* centered_model,
    const float* selected_eigenvalues, const int* equal_flags,
    float* noisy_projections, const float* model_projections, float* scores,
    cudaStream_t stream) {
    const std::size_t shared_bytes = checked_product(
        checked_product(static_cast<std::size_t>(shape.rank), 2U,
                        "CUDA projected filter cache size overflows"),
        sizeof(float), "CUDA projected filter cache bytes overflow");
    projected_filter_and_weight_kernel<stage>
        <<<static_cast<unsigned int>(shape.groups), block_threads,
           shared_bytes, stream>>>(
            centered_model, selected_eigenvalues, batch.retained_counts,
            equal_flags, batch.active_groups, shape.groups,
            shape.retained_stride,
            shape.sample_dim, shape.basis_similar, shape.rank, parameters,
            noisy_projections, model_projections, scores,
            batch.log_patch_weights);
    check_cuda(cudaPeekAtLastError(),
               "launch projected filter/weight kernel");
}

void launch_add_filter_centers(const GroupBatchShape& shape,
                               const DeviceGroupBatch& batch,
                               const float* centered_noisy,
                               const float* centers,
                               const int* equal_flags,
                               cudaStream_t stream) {
    add_filter_centers_kernel<<<static_cast<unsigned int>(shape.groups),
                                block_threads, 0, stream>>>(
        centered_noisy, centers, batch.retained_counts, equal_flags,
        batch.active_groups, shape.groups, shape.retained_stride,
        shape.sample_dim,
        batch.filtered_samples);
    check_cuda(cudaPeekAtLastError(), "launch filter center-add kernel");
}

template <typename Function>
void dispatch_basis_count(int basis_similar, Function&& function) {
    switch (basis_similar) {
    case 8:
        function(std::integral_constant<int, 8>{});
        break;
    case 15:
        function(std::integral_constant<int, 15>{});
        break;
    case 16:
        function(std::integral_constant<int, 16>{});
        break;
    case 17:
        function(std::integral_constant<int, 17>{});
        break;
    case 32:
        function(std::integral_constant<int, 32>{});
        break;
    default:
        function(std::integral_constant<int, 0>{});
        break;
    }
}

} // namespace

class GroupFilter::Impl {
  public:
    explicit Impl(int requested_device) {
        if (requested_device < 0) {
            check_cuda(cudaGetDevice(&device_), "cudaGetDevice");
        } else {
            device_ = requested_device;
            check_cuda(cudaSetDevice(device_), "cudaSetDevice");
        }
        check_cuda(cudaDeviceGetAttribute(&default_shared_memory_,
                                          cudaDevAttrMaxSharedMemoryPerBlock,
                                          device_),
                   "cudaDeviceGetAttribute(default shared memory)");
        check_cuda(cudaDeviceGetAttribute(
                       &optin_shared_memory_,
                       cudaDevAttrMaxSharedMemoryPerBlockOptin, device_),
                   "cudaDeviceGetAttribute(opt-in shared memory)");
        optin_shared_memory_ =
            std::max(optin_shared_memory_, default_shared_memory_);
        check_cusolver(cusolverDnCreate(&solver_), "cusolverDnCreate");
        try {
            check_cublas(cublasCreate(&blas_), "cublasCreate");
            check_cublas(cublasSetMathMode(blas_, CUBLAS_PEDANTIC_MATH),
                         "cublasSetMathMode(CUBLAS_PEDANTIC_MATH)");
            check_cusolver(cusolverDnCreateSyevjInfo(&jacobi_info_),
                           "cusolverDnCreateSyevjInfo");
            check_cusolver(cusolverDnXsyevjSetMaxSweeps(jacobi_info_, 100),
                           "cusolverDnXsyevjSetMaxSweeps");
            check_cusolver(cusolverDnXsyevjSetSortEig(jacobi_info_, 1),
                           "cusolverDnXsyevjSetSortEig");
        } catch (...) {
            if (jacobi_info_ != nullptr) {
                (void)cusolverDnDestroySyevjInfo(jacobi_info_);
                jacobi_info_ = nullptr;
            }
            if (blas_ != nullptr) {
                (void)cublasDestroy(blas_);
                blas_ = nullptr;
            }
            (void)cusolverDnDestroy(solver_);
            solver_ = nullptr;
            throw;
        }
    }

    ~Impl() {
        (void)cudaSetDevice(device_);
        if (jacobi_info_ != nullptr) {
            (void)cusolverDnDestroySyevjInfo(jacobi_info_);
        }
        if (blas_ != nullptr) {
            (void)cublasDestroy(blas_);
        }
        if (solver_ != nullptr) {
            (void)cusolverDnDestroy(solver_);
        }
    }

    void reserve(const GroupBatchShape& shape) {
        validate_shape(shape);
        check_cuda(cudaSetDevice(device_), "cudaSetDevice");

        const std::size_t groups = static_cast<std::size_t>(shape.groups);
        const std::size_t samples_per_group =
            checked_product(static_cast<std::size_t>(shape.retained_stride),
                            static_cast<std::size_t>(shape.sample_dim),
                            "CUDA sample group size overflows");
        const std::size_t sample_values = checked_product(
            groups, samples_per_group, "CUDA sample batch size overflows");
        const std::size_t mean_values =
            checked_product(groups, static_cast<std::size_t>(shape.sample_dim),
                            "CUDA mean batch size overflows");
        const std::size_t gram_per_group =
            checked_product(static_cast<std::size_t>(shape.basis_similar),
                            static_cast<std::size_t>(shape.basis_similar),
                            "CUDA Gram size overflows");
        const std::size_t gram_values = checked_product(
            groups, gram_per_group, "CUDA Gram batch size overflows");
        const std::size_t raw_eigen_values = checked_product(
            groups, static_cast<std::size_t>(shape.basis_similar),
            "CUDA eigenvalue batch size overflows");
        const std::size_t selected_values =
            checked_product(groups, static_cast<std::size_t>(shape.rank),
                            "CUDA selected eigenvalue batch size overflows");
        const std::size_t basis_values = checked_product(
            selected_values, static_cast<std::size_t>(shape.sample_dim),
            "CUDA basis batch size overflows");
        const std::size_t score_values = checked_product(
            groups, static_cast<std::size_t>(shape.retained_stride),
            "CUDA score batch size overflows");
        const std::size_t projection_values = checked_product(
            score_values, static_cast<std::size_t>(shape.rank),
            "CUDA projection batch size overflows");

        centered_noisy_.reserve(sample_values);
        centered_model_.reserve(sample_values);
        model_means_.reserve(mean_values);
        centers_.reserve(mean_values);
        gram_.reserve(gram_values);
        raw_eigenvalues_.reserve(raw_eigen_values);
        selected_eigenvalues_.reserve(selected_values);
        basis_.reserve(basis_values);
        scores_.reserve(score_values);
        use_cublas_filter_ =
            shape.basis_similar > warp_width && shape.rank > 0;
        if (use_cublas_filter_) {
            noisy_projections_.reserve(projection_values);
            model_projections_.reserve(projection_values);
        }
        equal_flags_.reserve(groups);
        solver_info_.reserve(groups);

        dispatch_basis_count(shape.basis_similar, [&](auto basis_tag) {
            constexpr int static_basis_count = decltype(basis_tag)::value;
            const std::size_t basic_bytes =
                static_cast<std::size_t>(shape.sample_dim) * sizeof(float);
            const std::size_t final_bytes = basic_bytes * 2;
            const std::size_t gram_bytes =
                static_cast<std::size_t>(shape.basis_similar) *
                shape.sample_dim * sizeof(float);
            prepare_basic_shared_ = configure_dynamic_shared(
                prepare_basic_kernel<static_basis_count, true>, basic_bytes,
                default_shared_memory_, optin_shared_memory_);
            prepare_final_shared_ = configure_dynamic_shared(
                prepare_final_kernel<static_basis_count, true>, final_bytes,
                default_shared_memory_, optin_shared_memory_);
            gram_shared_ = configure_dynamic_shared(
                gram_kernel<static_basis_count, true>, gram_bytes,
                default_shared_memory_, optin_shared_memory_);
        });
        const std::size_t filter_control_values = checked_product(
            static_cast<std::size_t>(shape.rank), 2U,
            "CUDA filter control cache size overflows");
        const std::size_t filter_control_bytes =
            checked_product(filter_control_values, sizeof(float),
                            "CUDA filter control cache size overflows");
        const std::size_t filter_basis_bytes = checked_product(
            checked_product(static_cast<std::size_t>(shape.rank),
                            static_cast<std::size_t>(shape.sample_dim),
                            "CUDA shared basis size overflows"),
            sizeof(float), "CUDA shared basis size overflows");
        const std::size_t filter_with_basis_bytes = checked_sum(
            filter_control_bytes, filter_basis_bytes,
            "CUDA filter shared memory size overflows");
        const std::size_t filter_score_bytes = checked_product(
            static_cast<std::size_t>(shape.retained_stride), sizeof(float),
            "CUDA shared score size overflows");
        const std::size_t filter_with_scores_bytes = checked_sum(
            filter_control_bytes, filter_score_bytes,
            "CUDA filter shared memory size overflows");
        const std::size_t filter_with_basis_scores_bytes = checked_sum(
            filter_with_basis_bytes, filter_score_bytes,
            "CUDA filter shared memory size overflows");
        const bool filter_basic_control = configure_dynamic_shared(
            filter_and_weight_kernel<Stage::Basic, false, false>,
            filter_control_bytes, default_shared_memory_,
            optin_shared_memory_);
        const bool filter_final_control = configure_dynamic_shared(
            filter_and_weight_kernel<Stage::Final, false, false>,
            filter_control_bytes, default_shared_memory_,
            optin_shared_memory_);
        if (!filter_basic_control || !filter_final_control) {
            throw std::runtime_error(
                "CUDA filter rank cache does not fit device shared memory");
        }
        filter_basic_shared_ = false;
        filter_basic_scores_ = false;
        if (shape.rank > 0 && configure_dynamic_shared(
                                  filter_and_weight_kernel<Stage::Basic, true,
                                                           true>,
                                  filter_with_basis_scores_bytes,
                                  default_shared_memory_,
                                  optin_shared_memory_)) {
            filter_basic_shared_ = true;
            filter_basic_scores_ = true;
        } else {
            filter_basic_shared_ =
                shape.rank > 0 &&
                configure_dynamic_shared(
                    filter_and_weight_kernel<Stage::Basic, true, false>,
                    filter_with_basis_bytes, default_shared_memory_,
                    optin_shared_memory_);
            if (!filter_basic_shared_) {
                filter_basic_scores_ = configure_dynamic_shared(
                    filter_and_weight_kernel<Stage::Basic, false, true>,
                    filter_with_scores_bytes, default_shared_memory_,
                    optin_shared_memory_);
            }
        }

        filter_final_shared_ = false;
        filter_final_scores_ = false;
        if (shape.rank > 0 && configure_dynamic_shared(
                                  filter_and_weight_kernel<Stage::Final, true,
                                                           true>,
                                  filter_with_basis_scores_bytes,
                                  default_shared_memory_,
                                  optin_shared_memory_)) {
            filter_final_shared_ = true;
            filter_final_scores_ = true;
        } else {
            filter_final_shared_ =
                shape.rank > 0 &&
                configure_dynamic_shared(
                    filter_and_weight_kernel<Stage::Final, true, false>,
                    filter_with_basis_bytes, default_shared_memory_,
                    optin_shared_memory_);
            if (!filter_final_shared_) {
                filter_final_scores_ = configure_dynamic_shared(
                    filter_and_weight_kernel<Stage::Final, false, true>,
                    filter_with_scores_bytes, default_shared_memory_,
                    optin_shared_memory_);
            }
        }

        use_cublas_gram_ = shape.basis_similar > warp_width;
        solver_lwork_ = 0;
        if (shape.rank > 0) {
            // The tighter small-K setting preserves the established path.
            // cuSOLVER's batched Jacobi solver needs a slightly looser
            // stopping threshold to converge reliably for K=60/100 while
            // still matching the double-precision filter oracle.
            const double jacobi_tolerance =
                shape.basis_similar > warp_width
                    ? large_group_jacobi_tolerance
                    : small_group_jacobi_tolerance;
            check_cusolver(
                cusolverDnXsyevjSetTolerance(jacobi_info_, jacobi_tolerance),
                "cusolverDnXsyevjSetTolerance");
            int required_lwork = 0;
            check_cusolver(cusolverDnSsyevjBatched_bufferSize(
                               solver_, CUSOLVER_EIG_MODE_VECTOR,
                               CUBLAS_FILL_MODE_LOWER,
                               shape.basis_similar, gram_.data(),
                               shape.basis_similar, raw_eigenvalues_.data(),
                               &required_lwork, jacobi_info_, shape.groups),
                           "cusolverDnSsyevjBatched_bufferSize");
            if (required_lwork < 0) {
                throw std::runtime_error(
                    "cuSOLVER returned a negative workspace size");
            }
            solver_workspace_.reserve(
                static_cast<std::size_t>(required_lwork));
            solver_lwork_ = required_lwork;
        }

        reserved_shape_ = shape;
        reserved_ = true;
        host_solver_info_.resize(groups);
    }

    void enqueue_projected_filter(
        const GroupBatchShape& shape, const FilterParameters& parameters,
        const DeviceGroupBatch& batch, const float* centered_noisy,
        const float* centered_model, const float* centers,
        const float* selected_eigenvalues, const float* basis, float* scores,
        cudaStream_t stream) {
        constexpr float alpha = 1.0F;
        constexpr float beta = 0.0F;
        const long long sample_group_stride =
            static_cast<long long>(shape.retained_stride) * shape.sample_dim;
        const long long basis_group_stride =
            static_cast<long long>(shape.rank) * shape.sample_dim;
        const long long projection_group_stride =
            static_cast<long long>(shape.retained_stride) * shape.rank;

        check_cublas(
            cublasSgemmStridedBatched(
                blas_, CUBLAS_OP_T, CUBLAS_OP_N, shape.rank,
                shape.retained_stride, shape.sample_dim, &alpha, basis,
                shape.sample_dim, basis_group_stride, centered_noisy,
                shape.sample_dim, sample_group_stride, &beta,
                noisy_projections_.data(), shape.rank,
                projection_group_stride, shape.groups),
            "cublasSgemmStridedBatched(noisy projection)");

        const bool needs_model_projection =
            parameters.stage == Stage::Final &&
            batch.log_patch_weights != nullptr &&
            parameters.weight_beta != 0.0F;
        if (needs_model_projection) {
            check_cublas(
                cublasSgemmStridedBatched(
                    blas_, CUBLAS_OP_T, CUBLAS_OP_N, shape.rank,
                    shape.retained_stride, shape.sample_dim, &alpha, basis,
                    shape.sample_dim, basis_group_stride, centered_model,
                    shape.sample_dim, sample_group_stride, &beta,
                    model_projections_.data(), shape.rank,
                    projection_group_stride, shape.groups),
                "cublasSgemmStridedBatched(model projection)");
        }

        if (parameters.stage == Stage::Basic) {
            launch_projected_filter<Stage::Basic>(
                shape, parameters, batch, centered_model,
                selected_eigenvalues, equal_flags_.data(),
                noisy_projections_.data(), nullptr, scores, stream);
        } else {
            launch_projected_filter<Stage::Final>(
                shape, parameters, batch, centered_model,
                selected_eigenvalues, equal_flags_.data(),
                noisy_projections_.data(),
                needs_model_projection ? model_projections_.data() : nullptr,
                scores, stream);
        }

        check_cublas(
            cublasSgemmStridedBatched(
                blas_, CUBLAS_OP_N, CUBLAS_OP_N, shape.sample_dim,
                shape.retained_stride, shape.rank, &alpha, basis,
                shape.sample_dim, basis_group_stride,
                noisy_projections_.data(), shape.rank,
                projection_group_stride, &beta, batch.filtered_samples,
                shape.sample_dim, sample_group_stride, shape.groups),
            "cublasSgemmStridedBatched(filtered reconstruction)");
        launch_add_filter_centers(shape, batch, centered_noisy, centers,
                                  equal_flags_.data(), stream);
    }

    void enqueue(const GroupBatchShape& shape,
                 const FilterParameters& parameters,
                 const DeviceGroupBatch& batch, cudaStream_t stream) {
        validate_shape(shape);
        validate_parameters(parameters);
        validate_batch(parameters, batch);
        if (!reserved_ ||
            shape.retained_stride != reserved_shape_.retained_stride ||
            shape.sample_dim != reserved_shape_.sample_dim ||
            shape.basis_similar != reserved_shape_.basis_similar ||
            shape.rank != reserved_shape_.rank ||
            shape.groups > reserved_shape_.groups) {
            throw std::logic_error(
                "CUDA GroupFilter workspace is not reserved for this shape");
        }

        check_cuda(cudaSetDevice(device_), "cudaSetDevice");
        if (shape.rank > 0 &&
            (!solver_stream_bound_ || solver_stream_ != stream)) {
            check_cusolver(cusolverDnSetStream(solver_, stream),
                           "cusolverDnSetStream");
            solver_stream_ = stream;
            solver_stream_bound_ = true;
        }
        if (shape.rank > 0 && use_cublas_gram_ &&
            (!blas_stream_bound_ || blas_stream_ != stream)) {
            check_cublas(cublasSetStream(blas_, stream), "cublasSetStream");
            blas_stream_ = stream;
            blas_stream_bound_ = true;
        }
        last_groups_ = shape.groups;
        last_stream_ = stream;
        last_solver_info_ = batch.solver_info != nullptr ? batch.solver_info
                                                         : solver_info_.data();

        float* const selected_eigenvalues = batch.eigenvalues != nullptr
                                                ? batch.eigenvalues
                                                : selected_eigenvalues_.data();
        float* const basis = batch.basis_vectors != nullptr
                                 ? batch.basis_vectors
                                 : basis_.data();
        float* const centered_model = parameters.stage == Stage::Basic
                                          ? centered_noisy_.data()
                                          : centered_model_.data();

        dispatch_basis_count(shape.basis_similar, [&](auto basis_tag) {
            constexpr int static_basis_count = decltype(basis_tag)::value;
            launch_prepare<static_basis_count>(
                shape, parameters, batch, model_means_.data(), centers_.data(),
                centered_noisy_.data(), centered_model, equal_flags_.data(),
                prepare_basic_shared_, prepare_final_shared_, stream);
        });

        if (shape.rank > 0) {
            if (use_cublas_gram_) {
                const float alpha =
                    1.0F / static_cast<float>(shape.basis_similar);
                constexpr float beta = 0.0F;
                const long long sample_group_stride =
                    static_cast<long long>(shape.retained_stride) *
                    shape.sample_dim;
                const long long gram_group_stride =
                    static_cast<long long>(shape.basis_similar) *
                    shape.basis_similar;
                check_cublas(
                    cublasSgemmStridedBatched(
                        blas_, CUBLAS_OP_T, CUBLAS_OP_N,
                        shape.basis_similar, shape.basis_similar,
                        shape.sample_dim, &alpha, centered_model,
                        shape.sample_dim, sample_group_stride,
                        centered_model, shape.sample_dim,
                        sample_group_stride, &beta, gram_.data(),
                        shape.basis_similar, gram_group_stride, shape.groups),
                    "cublasSgemmStridedBatched(group Gram)");
            } else {
                dispatch_basis_count(shape.basis_similar,
                                     [&](auto basis_tag) {
                    constexpr int static_basis_count =
                        decltype(basis_tag)::value;
                    launch_gram<static_basis_count>(
                        shape, centered_model, equal_flags_.data(),
                        batch.active_groups, gram_.data(), gram_shared_,
                        stream);
                });
            }

            if (use_cublas_gram_ &&
                (parameters.detect_equal_groups ||
                 batch.active_groups != nullptr)) {
                launch_regularize_flagged_gram(
                    shape, gram_.data(), equal_flags_.data(),
                    batch.active_groups, stream);
            }

            check_cusolver(cusolverDnSsyevjBatched(
                               solver_, CUSOLVER_EIG_MODE_VECTOR,
                               CUBLAS_FILL_MODE_LOWER,
                               shape.basis_similar, gram_.data(),
                               shape.basis_similar, raw_eigenvalues_.data(),
                               solver_workspace_.data(), solver_lwork_,
                               last_solver_info_, jacobi_info_, shape.groups),
                           "cusolverDnSsyevjBatched");

            dispatch_basis_count(shape.basis_similar, [&](auto basis_tag) {
                constexpr int static_basis_count = decltype(basis_tag)::value;
                launch_map_basis<static_basis_count>(
                    shape, centered_model, gram_.data(),
                    raw_eigenvalues_.data(), equal_flags_.data(),
                    batch.active_groups, selected_eigenvalues, basis, stream);
            });
        } else {
            check_cuda(cudaMemsetAsync(last_solver_info_, 0,
                                       static_cast<std::size_t>(shape.groups) *
                                           sizeof(int),
                                       stream),
                       "cudaMemsetAsync(solver info)");
        }

        if (use_cublas_filter_) {
            enqueue_projected_filter(
                shape, parameters, batch, centered_noisy_.data(),
                centered_model, centers_.data(), selected_eigenvalues, basis,
                scores_.data(), stream);
        } else if (parameters.stage == Stage::Basic) {
            launch_filter<Stage::Basic>(
                shape, parameters, batch, centered_noisy_.data(),
                centered_model, centers_.data(), selected_eigenvalues, basis,
                equal_flags_.data(), scores_.data(), filter_basic_shared_,
                filter_basic_scores_, stream);
        } else {
            launch_filter<Stage::Final>(
                shape, parameters, batch, centered_noisy_.data(),
                centered_model, centers_.data(), selected_eigenvalues, basis,
                equal_flags_.data(), scores_.data(), filter_final_shared_,
                filter_final_scores_, stream);
        }
    }

    void synchronize_and_check(cudaStream_t stream) {
        if (last_groups_ <= 0) {
            throw std::logic_error("no CUDA group batch has been submitted");
        }
        if (stream != nullptr && stream != last_stream_) {
            throw std::invalid_argument(
                "solver status must be checked on the enqueue stream");
        }
        check_cuda(cudaSetDevice(device_), "cudaSetDevice");
        const cudaStream_t check_stream = last_stream_;
        check_cuda(cudaMemcpyAsync(host_solver_info_.data(), last_solver_info_,
                                   static_cast<std::size_t>(last_groups_) *
                                       sizeof(int),
                                   cudaMemcpyDeviceToHost, check_stream),
                   "cudaMemcpyAsync(solver info)");
        check_cuda(cudaStreamSynchronize(check_stream),
                   "cudaStreamSynchronize");
        for (int group = 0; group < last_groups_; ++group) {
            const int info = host_solver_info_[static_cast<std::size_t>(group)];
            if (info != 0) {
                std::ostringstream message;
                message << "cuSOLVER eigensolver failed for group " << group
                        << " with info=" << info;
                throw std::runtime_error(message.str());
            }
        }
    }

    [[nodiscard]] std::size_t workspace_bytes() const noexcept {
        return centered_noisy_.bytes() + centered_model_.bytes() +
               model_means_.bytes() + centers_.bytes() + gram_.bytes() +
               raw_eigenvalues_.bytes() + selected_eigenvalues_.bytes() +
               basis_.bytes() + scores_.bytes() +
               noisy_projections_.bytes() + model_projections_.bytes() +
               equal_flags_.bytes() + solver_info_.bytes() +
               solver_workspace_.bytes();
    }

    int device_ = 0;
    int default_shared_memory_ = 0;
    int optin_shared_memory_ = 0;
    cublasHandle_t blas_ = nullptr;
    cusolverDnHandle_t solver_ = nullptr;
    syevjInfo_t jacobi_info_ = nullptr;
    GroupBatchShape reserved_shape_{};
    bool reserved_ = false;
    int solver_lwork_ = 0;
    int last_groups_ = 0;
    cudaStream_t last_stream_ = nullptr;
    int* last_solver_info_ = nullptr;
    cudaStream_t solver_stream_ = nullptr;
    cudaStream_t blas_stream_ = nullptr;
    bool solver_stream_bound_ = false;
    bool blas_stream_bound_ = false;
    bool prepare_basic_shared_ = false;
    bool prepare_final_shared_ = false;
    bool gram_shared_ = false;
    bool filter_basic_shared_ = false;
    bool filter_final_shared_ = false;
    bool filter_basic_scores_ = false;
    bool filter_final_scores_ = false;
    bool use_cublas_gram_ = false;
    bool use_cublas_filter_ = false;

    DeviceBuffer<float> centered_noisy_;
    DeviceBuffer<float> centered_model_;
    DeviceBuffer<float> model_means_;
    DeviceBuffer<float> centers_;
    DeviceBuffer<float> gram_;
    DeviceBuffer<float> raw_eigenvalues_;
    DeviceBuffer<float> selected_eigenvalues_;
    DeviceBuffer<float> basis_;
    DeviceBuffer<float> scores_;
    DeviceBuffer<float> noisy_projections_;
    DeviceBuffer<float> model_projections_;
    DeviceBuffer<float> solver_workspace_;
    DeviceBuffer<int> equal_flags_;
    DeviceBuffer<int> solver_info_;
    std::vector<int> host_solver_info_;
};

GroupFilter::GroupFilter(int device) : impl_(std::make_unique<Impl>(device)) {}

GroupFilter::~GroupFilter() = default;

GroupFilter::GroupFilter(GroupFilter&&) noexcept = default;

GroupFilter& GroupFilter::operator=(GroupFilter&&) noexcept = default;

void GroupFilter::reserve(const GroupBatchShape& shape) {
    if (impl_ == nullptr) {
        throw std::logic_error("CUDA GroupFilter was moved from");
    }
    impl_->reserve(shape);
}

void GroupFilter::enqueue(const GroupBatchShape& shape,
                          const FilterParameters& parameters,
                          const DeviceGroupBatch& batch, cudaStream_t stream) {
    if (impl_ == nullptr) {
        throw std::logic_error("CUDA GroupFilter was moved from");
    }
    impl_->enqueue(shape, parameters, batch, stream);
}

void GroupFilter::synchronize_and_check(cudaStream_t stream) {
    if (impl_ == nullptr) {
        throw std::logic_error("CUDA GroupFilter was moved from");
    }
    impl_->synchronize_and_check(stream);
}

const int* GroupFilter::solver_info_device() const noexcept {
    return impl_ == nullptr ? nullptr : impl_->last_solver_info_;
}

std::size_t GroupFilter::workspace_bytes() const noexcept {
    return impl_ == nullptr ? 0 : impl_->workspace_bytes();
}

int GroupFilter::device() const noexcept {
    return impl_ == nullptr ? -1 : impl_->device_;
}

} // namespace vnlbcu
