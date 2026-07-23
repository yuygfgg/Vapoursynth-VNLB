#include "cuda/group_filter.hpp"

#include "linalg/linalg.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr int skipped = 77;

struct FailureContext {
    int basis_similar = 0;
    vnlbcu::Stage stage = vnlbcu::Stage::Basic;
    int group = -1;
    int sample = -1;
    int dimension = -1;
    int component = -1;
};

FailureContext failure_context{};

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

    void upload(std::span<const T> source) {
        if (source.size() != count_) {
            throw std::invalid_argument("device upload size mismatch");
        }
        check_cuda(cudaMemcpy(data_, source.data(), count_ * sizeof(T),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy host to device");
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

struct HostResult {
    std::vector<float> filtered;
    std::vector<float> log_weights;
    std::vector<float> eigenvalues;
};

bool rows_are_equal(const float* samples, int similar, int sample_dim) {
    if (similar <= 1) {
        return false;
    }
    for (int sample = 1; sample < similar; ++sample) {
        for (int dimension = 0; dimension < sample_dim; ++dimension) {
            if (samples[dimension] !=
                samples[(sample * sample_dim) + dimension]) {
                return false;
            }
        }
    }
    return true;
}

template <typename T>
HostResult reference_filter(const vnlbcu::GroupBatchShape& shape,
                            const vnlbcu::FilterParameters& parameters,
                            std::span<const float> noisy_samples,
                            std::span<const float> basic_samples,
                            std::span<const int> retained_counts,
                            std::span<const std::uint8_t> flat_flags) {
    const std::size_t sample_group_stride =
        static_cast<std::size_t>(shape.retained_stride) * shape.sample_dim;
    HostResult result{
        .filtered = std::vector<float>(static_cast<std::size_t>(shape.groups) *
                                       sample_group_stride),
        .log_weights = std::vector<float>(
            static_cast<std::size_t>(shape.groups) * shape.retained_stride),
        .eigenvalues = std::vector<float>(
            static_cast<std::size_t>(shape.groups) * shape.rank),
    };

    for (int group = 0; group < shape.groups; ++group) {
        const int similar =
            retained_counts.empty()
                ? shape.retained_stride
                : retained_counts[static_cast<std::size_t>(group)];
        const std::size_t sample_base =
            static_cast<std::size_t>(group) * sample_group_stride;
        const std::size_t weight_base =
            static_cast<std::size_t>(group) * shape.retained_stride;
        const std::size_t eigen_base =
            static_cast<std::size_t>(group) * shape.rank;
        const float* const noisy = noisy_samples.data() + sample_base;

        if (rows_are_equal(noisy, similar, shape.sample_dim)) {
            std::copy_n(noisy,
                        static_cast<std::size_t>(similar) * shape.sample_dim,
                        result.filtered.data() + sample_base);
            const T tau = static_cast<T>(parameters.weight_epsilon) +
                          static_cast<T>(shape.sample_dim) /
                              static_cast<T>(shape.basis_similar);
            const T log_weight =
                parameters.weight_alpha == 0.0F
                    ? T{}
                    : -static_cast<T>(parameters.weight_alpha) * std::log(tau);
            std::fill_n(result.log_weights.data() + weight_base, similar,
                        static_cast<float>(log_weight));
            continue;
        }

        const bool final = parameters.stage == vnlbcu::Stage::Final;
        const bool flat =
            final && flat_flags[static_cast<std::size_t>(group)] != 0;
        const float* const model =
            final ? basic_samples.data() + sample_base : noisy;

        std::vector<T> model_mean(static_cast<std::size_t>(shape.sample_dim));
        std::vector<T> center(static_cast<std::size_t>(shape.sample_dim));
        for (int dimension = 0; dimension < shape.sample_dim; ++dimension) {
            T model_sum{};
            T noisy_sum{};
            for (int sample = 0; sample < shape.basis_similar; ++sample) {
                model_sum += static_cast<T>(
                    model[(sample * shape.sample_dim) + dimension]);
                noisy_sum += static_cast<T>(
                    noisy[(sample * shape.sample_dim) + dimension]);
            }
            model_mean[static_cast<std::size_t>(dimension)] =
                model_sum / static_cast<T>(shape.basis_similar);
            center[static_cast<std::size_t>(dimension)] =
                flat ? model_mean[static_cast<std::size_t>(dimension)]
                     : noisy_sum / static_cast<T>(shape.basis_similar);
        }

        vnlb::linalg::Matrix<T> centered_model(
            static_cast<std::size_t>(similar),
            static_cast<std::size_t>(shape.sample_dim));
        vnlb::linalg::Matrix<T> centered_noisy(
            static_cast<std::size_t>(similar),
            static_cast<std::size_t>(shape.sample_dim));
        for (int sample = 0; sample < similar; ++sample) {
            for (int dimension = 0; dimension < shape.sample_dim; ++dimension) {
                centered_model(static_cast<std::size_t>(sample),
                               static_cast<std::size_t>(dimension)) =
                    static_cast<T>(
                        model[(sample * shape.sample_dim) + dimension]) -
                    model_mean[static_cast<std::size_t>(dimension)];
                centered_noisy(static_cast<std::size_t>(sample),
                               static_cast<std::size_t>(dimension)) =
                    static_cast<T>(
                        noisy[(sample * shape.sample_dim) + dimension]) -
                    center[static_cast<std::size_t>(dimension)];
            }
        }

        const vnlb::linalg::ConstMatrixView<T> model_basis(
            centered_model.data(),
            static_cast<std::size_t>(shape.basis_similar),
            static_cast<std::size_t>(shape.sample_dim),
            static_cast<std::size_t>(shape.sample_dim));
        vnlb::linalg::Matrix<T> gram(
            static_cast<std::size_t>(shape.basis_similar),
            static_cast<std::size_t>(shape.basis_similar));
        vnlb::linalg::compute_gram_centered(
            model_basis, gram.view(),
            T{1} / static_cast<T>(shape.basis_similar));

        std::vector<T> eigenvalues(static_cast<std::size_t>(shape.rank));
        vnlb::linalg::Matrix<T> basis(
            static_cast<std::size_t>(shape.rank),
            static_cast<std::size_t>(shape.sample_dim));
        vnlb::linalg::Matrix<T> dual_vectors(
            static_cast<std::size_t>(shape.rank),
            static_cast<std::size_t>(shape.basis_similar));
        vnlb::linalg::SymmetricEigenWorkspace<T> eigen_workspace;
        (void)vnlb::linalg::topk_symmetric_eigen(
            gram.cview(), std::span<T>(eigenvalues), dual_vectors.view(),
            eigen_workspace);
        vnlb::linalg::map_dual_eigenvectors_to_basis(
            model_basis, std::span<const T>(eigenvalues), dual_vectors.cview(),
            basis.view());
        std::transform(eigenvalues.begin(), eigenvalues.end(),
                       result.eigenvalues.begin() +
                           static_cast<std::ptrdiff_t>(eigen_base),
                       [](T value) { return static_cast<float>(value); });

        const T estimate_noise2 = static_cast<T>(parameters.beta) *
                                  static_cast<T>(parameters.sigma) *
                                  static_cast<T>(parameters.sigma);
        const T model_noise2 =
            final ? static_cast<T>(parameters.beta) *
                        static_cast<T>(parameters.sigma_basic) *
                        static_cast<T>(parameters.sigma_basic)
                  : estimate_noise2;
        const T threshold =
            static_cast<T>(parameters.variance_threshold) * estimate_noise2;
        std::vector<T> coefficients(static_cast<std::size_t>(shape.rank));
        std::vector<T> signal_eigenvalues(static_cast<std::size_t>(shape.rank));
        for (int component = 0; component < shape.rank; ++component) {
            const T signal = std::max(
                eigenvalues[static_cast<std::size_t>(component)] - model_noise2,
                T{});
            signal_eigenvalues[static_cast<std::size_t>(component)] = signal;
            coefficients[static_cast<std::size_t>(component)] =
                signal > threshold ? T{1} / (T{1} + (estimate_noise2 / signal))
                                   : T{};
        }

        for (int sample = 0; sample < similar; ++sample) {
            for (int dimension = 0; dimension < shape.sample_dim; ++dimension) {
                T value = center[static_cast<std::size_t>(dimension)];
                for (int component = 0; component < shape.rank; ++component) {
                    T projection{};
                    for (int inner = 0; inner < shape.sample_dim; ++inner) {
                        projection +=
                            centered_noisy(static_cast<std::size_t>(sample),
                                           static_cast<std::size_t>(inner)) *
                            basis(static_cast<std::size_t>(component),
                                  static_cast<std::size_t>(inner));
                    }
                    value += coefficients[static_cast<std::size_t>(component)] *
                             projection *
                             basis(static_cast<std::size_t>(component),
                                   static_cast<std::size_t>(dimension));
                }
                result.filtered[sample_base +
                                static_cast<std::size_t>(sample) *
                                    shape.sample_dim +
                                dimension] = static_cast<float>(value);
            }
        }

        T trace{};
        T volume{};
        const T match_sigma =
            static_cast<T>(final ? parameters.sigma_basic : parameters.sigma);
        const T membership_noise2 = std::max(
            match_sigma * match_sigma, parameters.membership_noise_floor *
                                           static_cast<T>(parameters.sigma) *
                                           static_cast<T>(parameters.sigma));
        for (T signal : signal_eigenvalues) {
            if (signal > T{}) {
                trace += signal / (signal + estimate_noise2);
                volume += std::log1p(signal / membership_noise2);
            }
        }
        volume /= static_cast<T>(shape.sample_dim);
        const T tau = static_cast<T>(parameters.weight_epsilon) +
                      static_cast<T>(shape.sample_dim) /
                          static_cast<T>(shape.basis_similar) +
                      trace;
        const T group_log =
            parameters.weight_alpha == 0.0F
                ? T{}
                : -static_cast<T>(parameters.weight_alpha) * std::log(tau);
        std::vector<T> scores(static_cast<std::size_t>(similar));
        T minimum_score = std::numeric_limits<T>::infinity();
        for (int sample = 0; sample < similar; ++sample) {
            T norm2{};
            T distance{};
            T projected_norm2{};
            for (int dimension = 0; dimension < shape.sample_dim; ++dimension) {
                const T value =
                    centered_model(static_cast<std::size_t>(sample),
                                   static_cast<std::size_t>(dimension));
                norm2 += value * value;
            }
            for (int component = 0; component < shape.rank; ++component) {
                T projection{};
                for (int dimension = 0; dimension < shape.sample_dim;
                     ++dimension) {
                    projection +=
                        centered_model(static_cast<std::size_t>(sample),
                                       static_cast<std::size_t>(dimension)) *
                        basis(static_cast<std::size_t>(component),
                              static_cast<std::size_t>(dimension));
                }
                const T projection2 = projection * projection;
                distance +=
                    projection2 /
                    (signal_eigenvalues[static_cast<std::size_t>(component)] +
                     membership_noise2);
                projected_norm2 += projection2;
            }
            distance +=
                std::max(norm2 - projected_norm2, T{}) / membership_noise2;
            const T score = (distance - static_cast<T>(shape.sample_dim)) /
                            std::sqrt(T{2} * static_cast<T>(shape.sample_dim));
            scores[static_cast<std::size_t>(sample)] = score;
            minimum_score = std::min(minimum_score, score);
        }
        for (int sample = 0; sample < similar; ++sample) {
            const T membership =
                std::max(scores[static_cast<std::size_t>(sample)] -
                             minimum_score,
                         T{}) +
                volume;
            result.log_weights[weight_base + sample] = static_cast<float>(
                group_log -
                static_cast<T>(parameters.weight_beta) * membership);
        }
    }
    return result;
}

float next_random(std::uint32_t& state) {
    state = (state * 1664525U) + 1013904223U;
    const float unit = static_cast<float>((state >> 8U) & 0x00ffffffU) /
                       static_cast<float>(0x01000000U);
    return (2.0F * unit) - 1.0F;
}

void require_close(float actual, float expected, float tolerance,
                   std::string_view label) {
    const float error = std::abs(actual - expected);
    const float limit = tolerance * (1.0F + std::abs(expected));
    if (!(error <= limit)) {
        std::cerr << label << " (B=" << failure_context.basis_similar
                  << ", stage="
                  << (failure_context.stage == vnlbcu::Stage::Basic ? "Basic"
                                                                    : "Final")
                  << ", group=" << failure_context.group;
        if (failure_context.sample >= 0) {
            std::cerr << ", sample=" << failure_context.sample;
        }
        if (failure_context.dimension >= 0) {
            std::cerr << ", dimension=" << failure_context.dimension;
        }
        if (failure_context.component >= 0) {
            std::cerr << ", component=" << failure_context.component;
        }
        std::cerr << "): expected " << expected << ", got " << actual
                  << ", error " << error << '\n';
        std::exit(EXIT_FAILURE);
    }
}

enum class DataProfile {
    SingleDominant,
    BoundedMultiComponent,
    SubNoise,
};

struct CaseOptions {
    int rank = 8;
    int sample_dim = 64;
    bool use_retained_counts = true;
    bool output_log_weights = true;
    bool output_eigenvalues = true;
    bool output_basis = true;
    bool external_solver_info = false;
    bool use_active_groups = false;
    DataProfile profile = DataProfile::SingleDominant;
};

void run_case(int basis_similar, vnlbcu::Stage stage,
              CaseOptions options = {}) {
    failure_context.basis_similar = basis_similar;
    failure_context.stage = stage;
    failure_context.group = -1;
    failure_context.sample = -1;
    failure_context.dimension = -1;
    failure_context.component = -1;
    const vnlbcu::GroupBatchShape shape{
        .groups = 3,
        .retained_stride = basis_similar + 3,
        .sample_dim = options.sample_dim,
        .basis_similar = basis_similar,
        .rank = options.rank,
    };
    const std::size_t sample_count = static_cast<std::size_t>(shape.groups) *
                                     shape.retained_stride * shape.sample_dim;
    const std::size_t weight_count =
        static_cast<std::size_t>(shape.groups) * shape.retained_stride;
    const std::size_t eigen_count =
        static_cast<std::size_t>(shape.groups) * shape.rank;
    const std::size_t basis_count = eigen_count * shape.sample_dim;

    std::vector<float> noisy(sample_count);
    std::vector<float> basic(sample_count);
    std::vector<int> retained_counts;
    if (options.use_retained_counts) {
        retained_counts = {basis_similar, basis_similar + 1, basis_similar + 3};
    }
    std::vector<std::uint8_t> flat_flags{0, 1, 0};
    std::vector<std::uint8_t> active_groups{1, 0, 1};
    std::uint32_t random_state = 0x5eed1234U;
    for (int group = 0; group < shape.groups; ++group) {
        const std::size_t group_base = static_cast<std::size_t>(group) *
                                       shape.retained_stride * shape.sample_dim;
        for (int sample = 0; sample < shape.retained_stride; ++sample) {
            for (int dimension = 0; dimension < shape.sample_dim; ++dimension) {
                const std::size_t index =
                    group_base +
                    static_cast<std::size_t>(sample) * shape.sample_dim +
                    dimension;
                if (options.profile == DataProfile::SubNoise) {
                    const float tiny =
                        1.0e-5F * static_cast<float>(
                                       ((sample * 17) + (dimension * 13) +
                                        (group * 7)) %
                                           23 -
                                       11);
                    noisy[index] = 0.25F + tiny;
                } else if (options.profile == DataProfile::SingleDominant) {
                    const float structured =
                        0.03F * static_cast<float>((sample + 1) *
                                                   ((dimension % 11) - 5));
                    noisy[index] =
                        0.5F + structured + 0.12F * next_random(random_state);
                } else {
                    const float sample_position =
                        static_cast<float>(sample + 1);
                    const float dimension_position =
                        static_cast<float>(dimension + 1);
                    const float group_phase = 0.31F * static_cast<float>(group);
                    const float component0 =
                        0.18F *
                        std::sin(0.17F * sample_position + group_phase) *
                        std::sin(0.13F * dimension_position);
                    const float component1 =
                        0.14F *
                        std::cos(0.11F * sample_position + 0.7F * group_phase) *
                        std::cos(0.19F * dimension_position);
                    const float component2 =
                        0.10F *
                        std::sin(0.07F * sample_position + 0.6F +
                                 1.3F * group_phase) *
                        std::sin(0.07F * dimension_position + 0.6F);
                    noisy[index] = 0.5F + component0 + component1 + component2 +
                                   0.03F * next_random(random_state);
                }
                basic[index] =
                    noisy[index] * 0.82F + 0.03F * next_random(random_state);
            }
        }
    }

    // Group 2 exercises the CPU-compatible exact-equality bypass.
    const std::size_t equal_base =
        static_cast<std::size_t>(2) * shape.retained_stride * shape.sample_dim;
    for (int sample = 1; sample < shape.retained_stride; ++sample) {
        std::copy_n(noisy.data() + equal_base,
                    static_cast<std::size_t>(shape.sample_dim),
                    noisy.data() + equal_base +
                        static_cast<std::size_t>(sample) * shape.sample_dim);
    }

    vnlbcu::FilterParameters parameters{};
    parameters.stage = stage;
    parameters.sigma = 0.08F;
    parameters.sigma_basic = 0.045F;
    parameters.beta = 1.1F;
    parameters.variance_threshold = 0.9F;
    parameters.weight_alpha = 0.7F;
    parameters.weight_beta = 0.4F;
    parameters.weight_epsilon = 1.0e-6F;
    parameters.membership_noise_floor = 0.25F;
    parameters.detect_equal_groups = true;

    // Preserve the established float CPU comparison for the tuned small-K
    // path.  Large-K cases use a double oracle throughout the model, filter
    // and weight calculation so a second FP32 eigensolver does not bless the
    // CUDA path's numerical drift.
    const HostResult expected =
        basis_similar > 32
            ? reference_filter<double>(
                  shape, parameters, std::span<const float>(noisy),
                  std::span<const float>(basic),
                  std::span<const int>(retained_counts),
                  std::span<const std::uint8_t>(flat_flags))
            : reference_filter<float>(
                  shape, parameters, std::span<const float>(noisy),
                  std::span<const float>(basic),
                  std::span<const int>(retained_counts),
                  std::span<const std::uint8_t>(flat_flags));

    DeviceArray<float> device_noisy(sample_count);
    DeviceArray<float> device_basic(sample_count);
    DeviceArray<int> device_counts(retained_counts.size());
    DeviceArray<std::uint8_t> device_flat(flat_flags.size());
    DeviceArray<std::uint8_t> device_active(
        options.use_active_groups ? active_groups.size() : 0);
    DeviceArray<float> device_filtered(sample_count);
    DeviceArray<float> device_weights(weight_count);
    DeviceArray<float> device_eigenvalues(eigen_count);
    DeviceArray<float> device_basis(basis_count);
    DeviceArray<int> device_solver_info(
        options.external_solver_info ? static_cast<std::size_t>(shape.groups)
                                     : 0);
    device_noisy.upload(std::span<const float>(noisy));
    device_basic.upload(std::span<const float>(basic));
    if (!retained_counts.empty()) {
        device_counts.upload(std::span<const int>(retained_counts));
    }
    device_flat.upload(std::span<const std::uint8_t>(flat_flags));
    if (options.use_active_groups) {
        device_active.upload(std::span<const std::uint8_t>(active_groups));
        device_filtered.upload(std::span<const float>(
            std::vector<float>(sample_count, -123.25F)));
        device_weights.upload(std::span<const float>(
            std::vector<float>(weight_count, -124.25F)));
        device_eigenvalues.upload(std::span<const float>(
            std::vector<float>(eigen_count, -125.25F)));
    }
    if (options.external_solver_info) {
        check_cuda(
            cudaMemset(device_solver_info.data(), 0x7f,
                       static_cast<std::size_t>(shape.groups) * sizeof(int)),
            "cudaMemset external solver info");
    }

    vnlbcu::GroupFilter filter;
    filter.reserve(shape);
    const vnlbcu::DeviceGroupBatch batch{
        .noisy_samples = device_noisy.data(),
        .basic_samples =
            stage == vnlbcu::Stage::Final ? device_basic.data() : nullptr,
        .retained_counts =
            options.use_retained_counts ? device_counts.data() : nullptr,
        .active_groups =
            options.use_active_groups ? device_active.data() : nullptr,
        .flat_flags =
            stage == vnlbcu::Stage::Final ? device_flat.data() : nullptr,
        .filtered_samples = device_filtered.data(),
        .log_patch_weights =
            options.output_log_weights ? device_weights.data() : nullptr,
        .eigenvalues =
            options.output_eigenvalues ? device_eigenvalues.data() : nullptr,
        .basis_vectors = options.output_basis ? device_basis.data() : nullptr,
        .solver_info =
            options.external_solver_info ? device_solver_info.data() : nullptr,
    };
    filter.enqueue(shape, parameters, batch);
    if (options.external_solver_info &&
        filter.solver_info_device() != device_solver_info.data()) {
        throw std::runtime_error("external CUDA solver info was not selected");
    }
    filter.synchronize_and_check();

    std::vector<float> actual_filtered(sample_count);
    std::vector<float> actual_weights(weight_count);
    std::vector<float> actual_eigenvalues(eigen_count);
    device_filtered.download(std::span<float>(actual_filtered));
    if (options.output_log_weights) {
        device_weights.download(std::span<float>(actual_weights));
    }
    if (options.output_eigenvalues) {
        device_eigenvalues.download(std::span<float>(actual_eigenvalues));
    }
    if (options.external_solver_info) {
        std::vector<int> solver_info(static_cast<std::size_t>(shape.groups));
        device_solver_info.download(std::span<int>(solver_info));
        if (!std::all_of(solver_info.begin(), solver_info.end(),
                         [](int info) { return info == 0; })) {
            throw std::runtime_error("external CUDA solver info was not reset");
        }
    }

    if (options.output_eigenvalues) {
        for (int group = 0; group < shape.groups; ++group) {
            if (group == 2 ||
                (options.use_active_groups && active_groups[group] == 0U)) {
                continue;
            }
            failure_context.group = group;
            failure_context.sample = -1;
            failure_context.dimension = -1;
            const std::size_t eigen_base =
                static_cast<std::size_t>(group) * shape.rank;
            for (int component = 0; component < shape.rank; ++component) {
                failure_context.component = component;
                require_close(actual_eigenvalues[eigen_base + component],
                              expected.eigenvalues[eigen_base + component],
                              4.0e-3F, "eigenvalue");
            }
        }
    }

    for (int group = 0; group < shape.groups; ++group) {
        failure_context.group = group;
        failure_context.component = -1;
        const int similar =
            retained_counts.empty()
                ? shape.retained_stride
                : retained_counts[static_cast<std::size_t>(group)];
        const std::size_t sample_base = static_cast<std::size_t>(group) *
                                        shape.retained_stride *
                                        shape.sample_dim;
        const std::size_t weight_base =
            static_cast<std::size_t>(group) * shape.retained_stride;
        if (options.use_active_groups && active_groups[group] == 0U) {
            for (int sample = 0; sample < similar; ++sample) {
                const std::size_t inactive_sample_base =
                    sample_base +
                    static_cast<std::size_t>(sample) * shape.sample_dim;
                for (int dimension = 0; dimension < shape.sample_dim;
                     ++dimension) {
                    require_close(actual_filtered[inactive_sample_base +
                                                  dimension],
                                  -123.25F, 0.0F,
                                  "inactive filtered sample");
                }
                if (options.output_log_weights) {
                    require_close(actual_weights[weight_base + sample],
                                  -124.25F, 0.0F,
                                  "inactive log patch weight");
                }
            }
            if (options.output_eigenvalues) {
                const std::size_t eigen_base =
                    static_cast<std::size_t>(group) * shape.rank;
                for (int component = 0; component < shape.rank; ++component) {
                    require_close(actual_eigenvalues[eigen_base + component],
                                  -125.25F, 0.0F,
                                  "inactive selected eigenvalue");
                }
            }
            continue;
        }
        for (int sample = 0; sample < similar; ++sample) {
            failure_context.sample = sample;
            for (int dimension = 0; dimension < shape.sample_dim; ++dimension) {
                failure_context.dimension = dimension;
                const std::size_t index =
                    sample_base +
                    static_cast<std::size_t>(sample) * shape.sample_dim +
                    dimension;
                require_close(actual_filtered[index], expected.filtered[index],
                              4.0e-3F, "filtered sample");
            }
            if (options.output_log_weights) {
                failure_context.dimension = -1;
                require_close(actual_weights[weight_base + sample],
                              expected.log_weights[weight_base + sample],
                              6.0e-3F, "log patch weight");
            }
        }
    }
}

} // namespace

int main() {
    int device_count = 0;
    const cudaError_t device_status = cudaGetDeviceCount(&device_count);
    if (device_status == cudaErrorNoDevice ||
        device_status == cudaErrorInsufficientDriver || device_count == 0) {
        std::cout << "CUDA device unavailable; skipping vnlbcu runtime test\n";
        return skipped;
    }
    check_cuda(device_status, "cudaGetDeviceCount");

    try {
        run_case(15, vnlbcu::Stage::Basic);
        run_case(17, vnlbcu::Stage::Basic);
        run_case(15, vnlbcu::Stage::Final);
        run_case(17, vnlbcu::Stage::Final);
        run_case(15, vnlbcu::Stage::Basic,
                 CaseOptions{.external_solver_info = true,
                             .use_active_groups = true});
        run_case(15, vnlbcu::Stage::Final,
                 CaseOptions{.external_solver_info = true,
                             .use_active_groups = true});
        // The tuned reference RGB profile uses K=100/60 and rank 39.  These
        // cases exercise the runtime-size path beyond a single warp and check
        // that batched cuSOLVER converges without a per-group fallback.
        run_case(100, vnlbcu::Stage::Basic,
                 CaseOptions{.rank = 39,
                             .sample_dim = 128,
                             .external_solver_info = true});
        run_case(100, vnlbcu::Stage::Basic,
                 CaseOptions{
                     .rank = 39,
                     .sample_dim = 128,
                     .external_solver_info = true,
                     .profile = DataProfile::BoundedMultiComponent,
                 });
        // Non-identical groups whose complete covariance trace is below the
        // model-noise floor have no positive signal eigenvalue.  They bypass
        // the solver but still use the normal center/weight path.
        run_case(100, vnlbcu::Stage::Basic,
                 CaseOptions{
                     .rank = 39,
                     .sample_dim = 128,
                     .external_solver_info = true,
                     .profile = DataProfile::SubNoise,
                 });
        run_case(60, vnlbcu::Stage::Final,
                 CaseOptions{.rank = 39,
                             .sample_dim = 128,
                             .external_solver_info = true});
        // B=7 selects the runtime/generic kernel path.  This case also checks
        // internal eigen/basis fallbacks, null retained counts and weights,
        // plus caller-owned cuSOLVER status for an actual eigensolve.
        run_case(7, vnlbcu::Stage::Final,
                 CaseOptions{.rank = 5,
                             .use_retained_counts = false,
                             .output_log_weights = false,
                             .output_eigenvalues = false,
                             .output_basis = false,
                             .external_solver_info = true});
        // Rank zero bypasses cuSOLVER and must still clear external status and
        // produce the CPU-reference center-only estimate.
        run_case(7, vnlbcu::Stage::Basic,
                 CaseOptions{.rank = 0,
                             .output_eigenvalues = false,
                             .output_basis = false,
                             .external_solver_info = true});
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
