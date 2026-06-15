#include "core.hpp"

#include "common/arithmetic.hpp"
#include "common/compiler.hpp"
#include "common/validation.hpp"
#include "distance_highway.hpp"
#include "linalg/kernels_highway.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>

namespace vnlb::core {
namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

[[nodiscard]] bool same_processing_shape(StageParameters lhs,
                                         StageParameters rhs) noexcept {
    return lhs.patch_size == rhs.patch_size &&
           lhs.patch_time == rhs.patch_time &&
           lhs.search_window == rhs.search_window &&
           lhs.search_bwd == rhs.search_bwd &&
           lhs.search_fwd == rhs.search_fwd && lhs.similar == rhs.similar &&
           lhs.rank == rhs.rank &&
           lhs.similar_cap_factor == rhs.similar_cap_factor &&
           lhs.model_cap_factor == rhs.model_cap_factor &&
           lhs.couple_channels == rhs.couple_channels &&
           lhs.weight_gamma == rhs.weight_gamma;
}

[[nodiscard]] int normalized_proc_step(StageParameters parameters) noexcept {
    if (parameters.proc_step > 0) {
        return parameters.proc_step;
    }
    if (parameters.patch_size <= 1) {
        return 1;
    }
    return parameters.patch_size;
}

[[nodiscard]] int patch_position(int patch_size, int x, int y, int t) noexcept {
    return (((t * patch_size) + y) * patch_size) + x;
}

[[nodiscard]] int checked_patch_area(StageParameters parameters) {
    return common::checked_mul_int(parameters.patch_size, parameters.patch_size,
                                   "patch area overflows int");
}

[[nodiscard]] int checked_patch_dim(StageParameters parameters) {
    return common::checked_mul_int(checked_patch_area(parameters),
                                   parameters.patch_time,
                                   "patch dimension overflows int");
}

[[nodiscard]] int checked_estimator_dim(StageParameters parameters,
                                        int channels) {
    return common::checked_mul_int(checked_patch_dim(parameters),
                                   parameters.couple_channels ? channels : 1,
                                   "estimator dimension overflows int");
}

[[nodiscard]] int checked_temporal_candidate_count(StageParameters parameters) {
    return common::checked_add_int(
        common::checked_add_int(parameters.search_bwd, parameters.search_fwd,
                                "temporal candidate count overflows int"),
        1, "temporal candidate count overflows int");
}

[[nodiscard]] int checked_raw_candidate_count(StageParameters parameters) {
    const int spatial_candidates = common::checked_mul_int(
        parameters.search_window, parameters.search_window,
        "spatial candidate count overflows int");
    return common::checked_mul_int(spatial_candidates,
                                   checked_temporal_candidate_count(parameters),
                                   "candidate count overflows int");
}

[[nodiscard]] std::size_t checked_group_capacity(int channels, int patch_dim,
                                                 int similar) {
    const std::size_t capacity = common::checked_mul_size(
        common::checked_mul_size(static_cast<std::size_t>(channels),
                                 static_cast<std::size_t>(patch_dim),
                                 "group channel dimension overflows size_t"),
        static_cast<std::size_t>(similar), "group capacity overflows size_t");
    if (capacity > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::length_error("group capacity overflows int");
    }
    return capacity;
}

void copy_float_row(const float* VNLB_RESTRICT source,
                    float* VNLB_RESTRICT destination, int count) noexcept {
    if (count <= 0) {
        return;
    }
    std::memcpy(destination, source,
                static_cast<std::size_t>(count) * sizeof(float));
}

[[nodiscard]] bool group_is_equal(const std::vector<float>& group, int channels,
                                  int patch_dim, int similar) {
    if (similar <= 1) {
        return false;
    }

    for (int channel = 0; channel < channels; ++channel) {
        const int channel_base = channel * patch_dim * similar;
        for (int position = 0; position < patch_dim; ++position) {
            const float value = group[static_cast<std::size_t>(
                channel_base + (position * similar))];
            for (int patch = 1; patch < similar; ++patch) {
                if (value !=
                    group[static_cast<std::size_t>(
                        channel_base + (position * similar) + patch)]) {
                    return false;
                }
            }
        }
    }
    return true;
}

[[nodiscard]] int effective_rank(StageParameters parameters,
                                 int patch_dim) noexcept {
    (void)patch_dim;
    return parameters.rank;
}

[[nodiscard]] int soft_similar_cap(StageParameters parameters) noexcept {
    if (parameters.similar_cap_factor == 0.0F) {
        return std::numeric_limits<int>::max();
    }
    const float max_factor =
        static_cast<float>(std::numeric_limits<int>::max()) /
        static_cast<float>(parameters.similar);
    if (parameters.similar_cap_factor >= max_factor) {
        return std::numeric_limits<int>::max();
    }
    const float cap = std::ceil(static_cast<float>(parameters.similar) *
                                parameters.similar_cap_factor);
    return std::max(parameters.similar, static_cast<int>(cap));
}

[[nodiscard]] int checked_retained_group_count(StageParameters parameters) {
    return std::min(soft_similar_cap(parameters),
                    checked_raw_candidate_count(parameters));
}

[[nodiscard]] int model_similar_cap(StageParameters parameters) noexcept {
    if (parameters.model_cap_factor == 0.0F) {
        return std::numeric_limits<int>::max();
    }
    const float max_factor =
        static_cast<float>(std::numeric_limits<int>::max()) /
        static_cast<float>(parameters.similar);
    if (parameters.model_cap_factor >= max_factor) {
        return std::numeric_limits<int>::max();
    }
    const float cap = std::ceil(static_cast<float>(parameters.similar) *
                                parameters.model_cap_factor);
    return std::max(parameters.similar, static_cast<int>(cap));
}

[[nodiscard]] bool
aggregation_weights_enabled(StageParameters parameters) noexcept {
    return parameters.weight_alpha != 0.0F || parameters.weight_beta != 0.0F ||
           parameters.weight_gamma != 0.0F;
}

[[nodiscard]] float positive_or_zero(float value) noexcept {
    return value > 0.0F ? value : 0.0F;
}

[[nodiscard]] float safe_log_weight_exp(float value) noexcept {
    constexpr float min_log_weight = -80.0F;
    constexpr float max_log_weight = 80.0F;
    return std::exp(std::clamp(value, min_log_weight, max_log_weight));
}

[[nodiscard]] float signal_eigenvalue(float observed_eigenvalue,
                                      float model_noise2) noexcept {
    return positive_or_zero(observed_eigenvalue - model_noise2);
}

template <Stage stage>
[[nodiscard]] float model_noise_variance(StageParameters parameters) noexcept {
    if constexpr (stage == Stage::Final) {
        return parameters.beta * parameters.sigma_basic *
               parameters.sigma_basic;
    }
    return parameters.beta * parameters.sigma * parameters.sigma;
}

[[nodiscard]] float
estimate_noise_variance(StageParameters parameters) noexcept {
    return parameters.beta * parameters.sigma * parameters.sigma;
}

template <Stage stage>
[[nodiscard]] float
membership_noise_variance(StageParameters parameters) noexcept {
    float match_sigma = parameters.sigma;
    if constexpr (stage == Stage::Final) {
        match_sigma = parameters.sigma_basic;
    }
    const float match_noise2 = match_sigma * match_sigma;
    const float floor_noise2 =
        parameters.membership_noise_floor * parameters.sigma * parameters.sigma;
    return std::max(match_noise2, floor_noise2);
}

template <Stage stage>
[[nodiscard]] float group_precision_tau(StageParameters parameters,
                                        std::span<const float> eigenvalues,
                                        int rank, int sample_dim,
                                        int basis_similar) noexcept {
    const float model_noise2 = model_noise_variance<stage>(parameters);
    const float estimate_noise2 = estimate_noise_variance(parameters);
    float trace = 0.0F;
    for (int component = 0; component < rank; ++component) {
        const float lambda = signal_eigenvalue(
            eigenvalues[static_cast<std::size_t>(component)], model_noise2);
        if (lambda > 0.0F) {
            trace += lambda / (lambda + estimate_noise2);
        }
    }

    const float mean_term =
        static_cast<float>(sample_dim) / static_cast<float>(basis_similar);
    return parameters.weight_epsilon + mean_term + trace;
}

void reset_aggregation_weights(StageWorkspace& workspace, int similar) {
    workspace.aggregation_log_patch_weights_.assign(
        static_cast<std::size_t>(similar), 0.0F);
    workspace.aggregation_patch_weights_.assign(
        static_cast<std::size_t>(similar), 1.0F);
    workspace.aggregation_scores_.assign(static_cast<std::size_t>(similar),
                                         0.0F);
    workspace.aggregation_weight_model_count_ = 0;
}

void accumulate_default_aggregation_model(StageWorkspace& workspace,
                                          int similar) {
    const StageParameters parameters = workspace.parameters_;
    const int channels = workspace.geometry_.channels;
    const int sample_dim =
        workspace.patch_dim_ *
        (parameters.couple_channels && channels > 1 ? channels : 1);
    const int basis_similar = std::min(similar, model_similar_cap(parameters));
    const float tau =
        parameters.weight_epsilon +
        (static_cast<float>(sample_dim) / static_cast<float>(basis_similar));
    const float log_group_weight =
        parameters.weight_alpha == 0.0F
            ? 0.0F
            : -parameters.weight_alpha * std::log(tau);
    for (int patch = 0; patch < similar; ++patch) {
        workspace
            .aggregation_log_patch_weights_[static_cast<std::size_t>(patch)] +=
            log_group_weight;
    }
    ++workspace.aggregation_weight_model_count_;
}

void finalize_aggregation_weights(StageWorkspace& workspace, int similar) {
    if (!aggregation_weights_enabled(workspace.parameters_)) {
        std::fill_n(workspace.aggregation_patch_weights_.begin(),
                    static_cast<std::size_t>(similar), 1.0F);
        return;
    }

    if (workspace.aggregation_weight_model_count_ == 0) {
        accumulate_default_aggregation_model(workspace, similar);
    }

    const float inv_models =
        1.0F / static_cast<float>(workspace.aggregation_weight_model_count_);
    for (int patch = 0; patch < similar; ++patch) {
        const std::size_t index = static_cast<std::size_t>(patch);
        workspace.aggregation_patch_weights_[index] = safe_log_weight_exp(
            workspace.aggregation_log_patch_weights_[index] * inv_models);
    }
}

void prepare_aggregation_window(StageWorkspace& workspace) {
    const int patch_size = workspace.parameters_.patch_size;
    const float gamma = workspace.parameters_.weight_gamma;
    workspace.aggregation_window_weights_.assign(
        static_cast<std::size_t>(patch_size * patch_size), 1.0F);
    if (gamma == 0.0F || patch_size <= 1) {
        return;
    }

    const float center = 0.5F * static_cast<float>(patch_size - 1);
    const float radius = std::max(center, 0.5F);
    float max_weight = 0.0F;
    for (int y = 0; y < patch_size; ++y) {
        const float dy = (static_cast<float>(y) - center) / radius;
        for (int x = 0; x < patch_size; ++x) {
            const float dx = (static_cast<float>(x) - center) / radius;
            const float raw = std::exp(-0.5F * ((dx * dx) + (dy * dy)));
            const std::size_t index =
                static_cast<std::size_t>((y * patch_size) + x);
            workspace.aggregation_window_weights_[index] = raw;
            max_weight = std::max(max_weight, raw);
        }
    }

    const float inv_max = max_weight > 0.0F ? 1.0F / max_weight : 1.0F;
    for (float& value : workspace.aggregation_window_weights_) {
        value = std::pow(value * inv_max, gamma);
    }
}

template <Stage stage>
void accumulate_aggregation_model(StageWorkspace& workspace,
                                  linalg::ConstMatrixView<float> centered,
                                  std::span<const float> eigenvalues, int rank,
                                  int basis_similar) {
    const StageParameters parameters = workspace.parameters_;
    if (parameters.weight_alpha == 0.0F && parameters.weight_beta == 0.0F) {
        ++workspace.aggregation_weight_model_count_;
        return;
    }

    const int similar = static_cast<int>(centered.rows());
    const int sample_dim = static_cast<int>(centered.cols());
    const float tau = group_precision_tau<stage>(parameters, eigenvalues, rank,
                                                 sample_dim, basis_similar);
    const float log_group_weight =
        parameters.weight_alpha == 0.0F
            ? 0.0F
            : -parameters.weight_alpha * std::log(tau);

    float volume = 0.0F;
    if (parameters.weight_beta != 0.0F) {
        const float model_noise2 = model_noise_variance<stage>(parameters);
        const float membership_noise2 =
            membership_noise_variance<stage>(parameters);
        for (int component = 0; component < rank; ++component) {
            const float lambda = signal_eigenvalue(
                eigenvalues[static_cast<std::size_t>(component)], model_noise2);
            if (lambda > 0.0F) {
                volume += std::log1p(lambda / membership_noise2);
            }
        }
        volume /= static_cast<float>(sample_dim);

        float min_score = std::numeric_limits<float>::infinity();
        const float score_scale =
            std::sqrt(2.0F * static_cast<float>(sample_dim));
        for (int patch = 0; patch < similar; ++patch) {
            const float* VNLB_RESTRICT row =
                centered.row_data(static_cast<std::size_t>(patch));
            const float norm2 = linalg::kernels::dot_contiguous_highway(
                row, row, static_cast<std::size_t>(sample_dim));
            float distance = 0.0F;
            float projected_norm2 = 0.0F;
            for (int component = 0; component < rank; ++component) {
                const float* VNLB_RESTRICT basis =
                    workspace.eigenvectors_.row_data(
                        static_cast<std::size_t>(component));
                const float projection =
                    linalg::kernels::dot_contiguous_highway(
                        row, basis, static_cast<std::size_t>(sample_dim));
                const float projection2 = projection * projection;
                const float lambda = signal_eigenvalue(
                    eigenvalues[static_cast<std::size_t>(component)],
                    model_noise2);
                distance += projection2 / (lambda + membership_noise2);
                projected_norm2 += projection2;
            }
            const float residual_norm2 =
                positive_or_zero(norm2 - projected_norm2);
            distance += residual_norm2 / membership_noise2;

            const float score =
                (distance - static_cast<float>(sample_dim)) / score_scale;
            workspace.aggregation_scores_[static_cast<std::size_t>(patch)] =
                score;
            min_score = std::min(min_score, score);
        }

        for (int patch = 0; patch < similar; ++patch) {
            const std::size_t index = static_cast<std::size_t>(patch);
            const float membership =
                positive_or_zero(workspace.aggregation_scores_[index] -
                                 min_score) +
                volume;
            workspace.aggregation_log_patch_weights_[index] +=
                log_group_weight - (parameters.weight_beta * membership);
        }
    } else {
        for (int patch = 0; patch < similar; ++patch) {
            workspace.aggregation_log_patch_weights_[static_cast<std::size_t>(
                patch)] += log_group_weight;
        }
    }

    ++workspace.aggregation_weight_model_count_;
}

[[nodiscard]] linalg::ConstMatrixView<float>
first_matrix_rows(const linalg::Matrix<float>& matrix, int rows) noexcept {
    return linalg::ConstMatrixView<float>(matrix.data(),
                                          static_cast<std::size_t>(rows),
                                          matrix.cols(), matrix.cols());
}

void copy_group_channel_to_samples(const std::vector<float>& group, int channel,
                                   int patch_dim, int similar,
                                   linalg::Matrix<float>& samples) {
    samples.resize(static_cast<std::size_t>(similar),
                   static_cast<std::size_t>(patch_dim));
    const int channel_base = channel * patch_dim * similar;
    for (int patch = 0; patch < similar; ++patch) {
        float* VNLB_RESTRICT row =
            samples.row_data(static_cast<std::size_t>(patch));
        for (int position = 0; position < patch_dim; ++position) {
            row[position] = group[static_cast<std::size_t>(
                channel_base + (position * similar) + patch)];
        }
    }
}

void copy_samples_to_group_channel(const linalg::Matrix<float>& samples,
                                   int channel, int patch_dim, int similar,
                                   std::vector<float>& group) {
    const int channel_base = channel * patch_dim * similar;
    for (int patch = 0; patch < similar; ++patch) {
        const float* VNLB_RESTRICT row =
            samples.row_data(static_cast<std::size_t>(patch));
        for (int position = 0; position < patch_dim; ++position) {
            group[static_cast<std::size_t>(channel_base + (position * similar) +
                                           patch)] = row[position];
        }
    }
}

void copy_group_coupled_to_samples(const std::vector<float>& group,
                                   int channels, int patch_dim, int similar,
                                   linalg::Matrix<float>& samples) {
    const int sample_dim = channels * patch_dim;
    samples.resize(static_cast<std::size_t>(similar),
                   static_cast<std::size_t>(sample_dim));
    for (int patch = 0; patch < similar; ++patch) {
        float* VNLB_RESTRICT row =
            samples.row_data(static_cast<std::size_t>(patch));
        for (int channel = 0; channel < channels; ++channel) {
            const int channel_base = channel * patch_dim * similar;
            const int output_base = channel * patch_dim;
            for (int position = 0; position < patch_dim; ++position) {
                row[output_base + position] = group[static_cast<std::size_t>(
                    channel_base + (position * similar) + patch)];
            }
        }
    }
}

void copy_samples_to_group_coupled(const linalg::Matrix<float>& samples,
                                   int channels, int patch_dim, int similar,
                                   std::vector<float>& group) {
    for (int patch = 0; patch < similar; ++patch) {
        const float* VNLB_RESTRICT row =
            samples.row_data(static_cast<std::size_t>(patch));
        for (int channel = 0; channel < channels; ++channel) {
            const int channel_base = channel * patch_dim * similar;
            const int input_base = channel * patch_dim;
            for (int position = 0; position < patch_dim; ++position) {
                group[static_cast<std::size_t>(channel_base +
                                               (position * similar) + patch)] =
                    row[input_base + position];
            }
        }
    }
}

void copy_samples(linalg::ConstMatrixView<float> input,
                  linalg::Matrix<float>& output) {
    output.resize(input.rows(), input.cols());
    if (input.cols() == 0) {
        return;
    }
    for (std::size_t row = 0; row < input.rows(); ++row) {
        const float* VNLB_RESTRICT input_row = input.row_data(row);
        float* VNLB_RESTRICT output_row = output.row_data(row);
        std::memcpy(output_row, input_row, input.cols() * sizeof(float));
    }
}

void center_samples_with_mean(const linalg::Matrix<float>& samples,
                              std::span<const float> mean,
                              linalg::Matrix<float>& centered) {
    centered.resize(samples.rows(), samples.cols());
    linalg::center_rows(samples.cview(), mean, centered.view());
}

[[nodiscard]] float group_variance(const std::vector<float>& group,
                                   int channels, int patch_dim, int similar) {
    float variance = 0.0F;
    const int values_per_channel = patch_dim * similar;
    if (values_per_channel <= 1) {
        return 0.0F;
    }

    for (int channel = 0; channel < channels; ++channel) {
        const int channel_base = channel * values_per_channel;
        double sum = 0.0;
        double sum2 = 0.0;
        for (int index = 0; index < values_per_channel; ++index) {
            const float value =
                group[static_cast<std::size_t>(channel_base + index)];
            sum += value;
            sum2 += static_cast<double>(value) * static_cast<double>(value);
        }
        const double count = static_cast<double>(values_per_channel);
        variance +=
            static_cast<float>((sum2 - (sum * sum / count)) / (count - 1.0));
    }
    return variance / static_cast<float>(channels);
}

[[nodiscard]] bool samples_are_equal(linalg::ConstMatrixView<float> samples) {
    if (samples.rows() <= 1) {
        return false;
    }

    const float* VNLB_RESTRICT first = samples.row_data(0);
    for (std::size_t row = 1; row < samples.rows(); ++row) {
        const float* VNLB_RESTRICT current = samples.row_data(row);
        for (std::size_t col = 0; col < samples.cols(); ++col) {
            if (current[col] != first[col]) {
                return false;
            }
        }
    }
    return true;
}

[[nodiscard]] float
samples_channel_variance(linalg::ConstMatrixView<float> samples, int channels,
                         int patch_dim) {
    float variance = 0.0F;
    const int values_per_channel = patch_dim * static_cast<int>(samples.rows());
    if (values_per_channel <= 1) {
        return 0.0F;
    }

    for (int channel = 0; channel < channels; ++channel) {
        const int channel_base = channel * patch_dim;
        double sum = 0.0;
        double sum2 = 0.0;
        for (std::size_t row = 0; row < samples.rows(); ++row) {
            const float* VNLB_RESTRICT row_data = samples.row_data(row);
            for (int position = 0; position < patch_dim; ++position) {
                const float value = row_data[channel_base + position];
                sum += value;
                sum2 += static_cast<double>(value) * static_cast<double>(value);
            }
        }
        const double count = static_cast<double>(values_per_channel);
        variance +=
            static_cast<float>((sum2 - (sum * sum / count)) / (count - 1.0));
    }
    return variance / static_cast<float>(channels);
}

[[nodiscard]] bool is_flat_group(const StageWorkspace& workspace, int similar) {
    const StageParameters parameters = workspace.parameters_;
    const float threshold =
        parameters.sigma * parameters.sigma * parameters.gamma;
    return group_variance(workspace.group_noisy_, workspace.geometry_.channels,
                          workspace.patch_dim_, similar) < threshold;
}

[[nodiscard]] bool is_flat_samples_group(const StageWorkspace& workspace) {
    const StageParameters parameters = workspace.parameters_;
    const float threshold =
        parameters.sigma * parameters.sigma * parameters.gamma;
    return samples_channel_variance(workspace.samples_noisy_.cview(),
                                    workspace.geometry_.channels,
                                    workspace.patch_dim_) < threshold;
}

void project_bayes_estimate(StageWorkspace& workspace, int rank,
                            std::span<const float> center) {
    const int similar = static_cast<int>(workspace.centered_noisy_.rows());
    const int patch_dim = static_cast<int>(workspace.centered_noisy_.cols());
    workspace.filtered_.resize(static_cast<std::size_t>(similar),
                               static_cast<std::size_t>(patch_dim));

    for (int patch = 0; patch < similar; ++patch) {
        const float* VNLB_RESTRICT input =
            workspace.centered_noisy_.row_data(static_cast<std::size_t>(patch));
        float* VNLB_RESTRICT output =
            workspace.filtered_.row_data(static_cast<std::size_t>(patch));
        std::fill_n(output, static_cast<std::size_t>(patch_dim), 0.0F);

        for (int component = 0; component < rank; ++component) {
            const float coefficient =
                workspace
                    .filter_coefficients_[static_cast<std::size_t>(component)];
            if (coefficient == 0.0F) {
                continue;
            }

            const float* VNLB_RESTRICT basis = workspace.eigenvectors_.row_data(
                static_cast<std::size_t>(component));
            const float projection = linalg::kernels::dot_contiguous_highway(
                input, basis, static_cast<std::size_t>(patch_dim));

            const float scaled_projection = coefficient * projection;
            linalg::kernels::add_scaled_contiguous_highway(
                output, basis, scaled_projection,
                static_cast<std::size_t>(patch_dim));
        }

        linalg::kernels::add_contiguous_highway(
            output, center.data(), static_cast<std::size_t>(patch_dim));
    }
}

template <Stage stage>
void compute_filter_coefficients(StageWorkspace& workspace,
                                 std::span<const float> eigenvalues, int rank) {
    const StageParameters parameters = workspace.parameters_;
    const float sigma2 = parameters.beta * parameters.sigma * parameters.sigma;
    float sigma_basic2 = sigma2;
    if constexpr (stage == Stage::Final) {
        sigma_basic2 =
            parameters.beta * parameters.sigma_basic * parameters.sigma_basic;
    }
    const float threshold = parameters.variance_threshold * sigma2;

    for (int component = 0; component < rank; ++component) {
        float variance = eigenvalues[static_cast<std::size_t>(component)];
        if (sigma_basic2 > 0.0F) {
            variance -= std::min(variance, sigma_basic2);
        }
        workspace.filter_coefficients_[static_cast<std::size_t>(component)] =
            variance > threshold ? 1.0F / (1.0F + (sigma2 / variance)) : 0.0F;
    }
}

template <Stage stage>
void filter_vnlb_samples(StageWorkspace& workspace, int similar, int sample_dim,
                         bool flat_patch) {
    const int rank = effective_rank(workspace.parameters_, sample_dim);
    const int basis_similar =
        std::min(similar, model_similar_cap(workspace.parameters_));
    if (basis_similar < std::min(sample_dim, rank)) {
        throw std::runtime_error(
            "group_size must cover the effective VNLB rank");
    }

    auto mean_noisy = std::span<float>(workspace.mean_noisy_.data(),
                                       static_cast<std::size_t>(sample_dim));
    [[maybe_unused]] auto mean_basic = std::span<float>(
        workspace.mean_basic_.data(), static_cast<std::size_t>(sample_dim));
    if constexpr (stage == Stage::Basic) {
        (void)flat_patch;
    }
    constexpr int dual_pca_min_dimension = 64;
    const bool use_dual_pca =
        sample_dim >= dual_pca_min_dimension && basis_similar < sample_dim;

    const linalg::ConstMatrixView<float> noisy_basis =
        first_matrix_rows(workspace.samples_noisy_, basis_similar);

    if constexpr (stage == Stage::Final) {
        const linalg::ConstMatrixView<float> basic_basis =
            first_matrix_rows(workspace.samples_basic_, basis_similar);
        linalg::compute_mean(basic_basis, mean_basic);
        workspace.centered_basic_.resize(workspace.samples_basic_.rows(),
                                         workspace.samples_basic_.cols());
        linalg::center_rows(
            workspace.samples_basic_.cview(),
            std::span<const float>(mean_basic.data(), mean_basic.size()),
            workspace.centered_basic_.view());

        if (flat_patch) {
            center_samples_with_mean(
                workspace.samples_noisy_,
                std::span<const float>(mean_basic.data(), mean_basic.size()),
                workspace.centered_noisy_);
        } else {
            linalg::compute_mean(noisy_basis, mean_noisy);
            workspace.centered_noisy_.resize(workspace.samples_noisy_.rows(),
                                             workspace.samples_noisy_.cols());
            linalg::center_rows(
                workspace.samples_noisy_.cview(),
                std::span<const float>(mean_noisy.data(), mean_noisy.size()),
                workspace.centered_noisy_.view());
        }
    } else {
        linalg::compute_mean(noisy_basis, mean_noisy);
        workspace.centered_noisy_.resize(workspace.samples_noisy_.rows(),
                                         workspace.samples_noisy_.cols());
        linalg::center_rows(
            workspace.samples_noisy_.cview(),
            std::span<const float>(mean_noisy.data(), mean_noisy.size()),
            workspace.centered_noisy_.view());
    }

    if (rank == 0) {
        workspace.filtered_.resize(workspace.samples_noisy_.rows(),
                                   workspace.samples_noisy_.cols());
        std::span<const float> center = mean_noisy;
        if constexpr (stage == Stage::Final) {
            if (flat_patch) {
                center = mean_basic;
            }
        }
        if (!center.empty()) {
            for (std::size_t row = 0; row < workspace.filtered_.rows(); ++row) {
                std::memcpy(workspace.filtered_.row_data(row), center.data(),
                            center.size() * sizeof(float));
            }
        }
        const linalg::ConstMatrixView<float> membership_centered = [&]() {
            if constexpr (stage == Stage::Final) {
                return workspace.centered_basic_.cview();
            } else {
                return workspace.centered_noisy_.cview();
            }
        }();
        accumulate_aggregation_model<stage>(workspace, membership_centered,
                                            std::span<const float>{}, rank,
                                            basis_similar);
        return;
    }

    workspace.eigenvectors_.resize(static_cast<std::size_t>(rank),
                                   static_cast<std::size_t>(sample_dim));
    auto eigenvalues = std::span<float>(workspace.eigenvalues_.data(),
                                        static_cast<std::size_t>(rank));
    const float covariance_scale = 1.0F / static_cast<float>(basis_similar);
    const linalg::ConstMatrixView<float> basis_samples = [&]() {
        if constexpr (stage == Stage::Basic) {
            return first_matrix_rows(workspace.centered_noisy_, basis_similar);
        } else {
            return first_matrix_rows(workspace.centered_basic_, basis_similar);
        }
    }();
    if (use_dual_pca) {
        workspace.gram_.resize(static_cast<std::size_t>(basis_similar),
                               static_cast<std::size_t>(basis_similar));
        linalg::compute_gram_centered(basis_samples, workspace.gram_.view(),
                                      covariance_scale);

        workspace.dual_eigenvectors_.resize(
            static_cast<std::size_t>(rank),
            static_cast<std::size_t>(basis_similar));
        // TODO: Report or recover from eigensolver non-convergence when a
        // robust cross-platform fallback is available.
        linalg::topk_symmetric_eigen(workspace.gram_.cview(), eigenvalues,
                                     workspace.dual_eigenvectors_.view(),
                                     workspace.eigen_workspace_);
        const auto gram_eigenvalues =
            std::span<const float>(eigenvalues.data(), eigenvalues.size());
        linalg::map_dual_eigenvectors_to_basis(
            basis_samples, gram_eigenvalues,
            workspace.dual_eigenvectors_.cview(),
            workspace.eigenvectors_.view());
    } else {
        workspace.covariance_.resize(static_cast<std::size_t>(sample_dim),
                                     static_cast<std::size_t>(sample_dim));
        linalg::compute_covariance_centered(
            basis_samples, workspace.covariance_.view(), covariance_scale);

        // TODO: Report or recover from eigensolver non-convergence when a
        // robust cross-platform fallback is available.
        linalg::topk_symmetric_eigen(workspace.covariance_.cview(), eigenvalues,
                                     workspace.eigenvectors_.view(),
                                     workspace.eigen_workspace_);
    }

    const auto eigenvalue_span =
        std::span<const float>(eigenvalues.data(), eigenvalues.size());
    compute_filter_coefficients<stage>(workspace, eigenvalue_span, rank);

    const linalg::ConstMatrixView<float> membership_centered = [&]() {
        if constexpr (stage == Stage::Final) {
            return workspace.centered_basic_.cview();
        } else {
            return workspace.centered_noisy_.cview();
        }
    }();
    accumulate_aggregation_model<stage>(workspace, membership_centered,
                                        eigenvalue_span, rank, basis_similar);

    std::span<const float> center = mean_noisy;
    if constexpr (stage == Stage::Final) {
        if (flat_patch) {
            center = mean_basic;
        }
    }
    project_bayes_estimate(workspace, rank, center);
}

template <Stage stage>
void transform_vnlb_channel(StageWorkspace& workspace, int channel, int similar,
                            bool flat_patch) {
    const int patch_dim = workspace.patch_dim_;
    copy_group_channel_to_samples(workspace.group_noisy_, channel, patch_dim,
                                  similar, workspace.samples_noisy_);
    if constexpr (stage == Stage::Final) {
        copy_group_channel_to_samples(workspace.group_basic_, channel,
                                      patch_dim, similar,
                                      workspace.samples_basic_);
    }

    filter_vnlb_samples<stage>(workspace, similar, patch_dim, flat_patch);
    copy_samples_to_group_channel(workspace.filtered_, channel, patch_dim,
                                  similar, workspace.group_noisy_);
}

template <Stage stage>
void transform_vnlb_coupled(StageWorkspace& workspace, int similar,
                            bool flat_patch) {
    const int channels = workspace.geometry_.channels;
    const int patch_dim = workspace.patch_dim_;
    const int sample_dim = channels * patch_dim;
    copy_group_coupled_to_samples(workspace.group_noisy_, channels, patch_dim,
                                  similar, workspace.samples_noisy_);
    if constexpr (stage == Stage::Final) {
        copy_group_coupled_to_samples(workspace.group_basic_, channels,
                                      patch_dim, similar,
                                      workspace.samples_basic_);
    }

    filter_vnlb_samples<stage>(workspace, similar, sample_dim, flat_patch);
    copy_samples_to_group_coupled(workspace.filtered_, channels, patch_dim,
                                  similar, workspace.group_noisy_);
}

template <Stage stage>
void transform_vnlb_coupled_samples(StageWorkspace& workspace, int similar) {
    const int channels = workspace.geometry_.channels;
    const int patch_dim = workspace.patch_dim_;
    const int sample_dim = channels * patch_dim;
    workspace.output_sample_major_ = true;

    if (samples_are_equal(workspace.samples_noisy_.cview())) {
        copy_samples(workspace.samples_noisy_.cview(), workspace.filtered_);
        return;
    }

    bool flat_patch = false;
    if constexpr (stage == Stage::Final) {
        flat_patch = workspace.parameters_.flat_areas &&
                     is_flat_samples_group(workspace);
    }
    filter_vnlb_samples<stage>(workspace, similar, sample_dim, flat_patch);
}

template <Stage stage>
void transform_vnlb_group(StageWorkspace& workspace, int similar) {
    const int channels = workspace.geometry_.channels;
    const int patch_dim = workspace.patch_dim_;

    if (group_is_equal(workspace.group_noisy_, channels, patch_dim, similar)) {
        return;
    }

    bool flat_patch = false;
    if constexpr (stage == Stage::Final) {
        flat_patch = workspace.parameters_.flat_areas &&
                     is_flat_group(workspace, similar);
    }

    if (workspace.parameters_.couple_channels && channels > 1) {
        transform_vnlb_coupled<stage>(workspace, similar, flat_patch);
    } else {
        for (int channel = 0; channel < channels; ++channel) {
            transform_vnlb_channel<stage>(workspace, channel, similar,
                                          flat_patch);
        }
    }
}

void compute_temporal_range(VideoGeometry geometry, StageParameters parameters,
                            int anchor_frame, int& low, int& high) {
    const int max_origin_frame =
        geometry.source_frames() - parameters.patch_time;
    const int search_start =
        common::checked_add_int(anchor_frame, -parameters.search_bwd,
                                "temporal search range overflows int");
    const int search_end =
        common::checked_add_int(anchor_frame, parameters.search_fwd,
                                "temporal search range overflows int");
    const int shift = common::checked_add_int(
        std::min(0, search_start),
        std::max(
            0, common::checked_add_int(search_end, -max_origin_frame,
                                       "temporal search range overflows int")),
        "temporal search range overflows int");
    low = std::max(
        0, common::checked_add_int(search_start, -shift,
                                   "temporal search range overflows int"));
    high = std::min(
        max_origin_frame,
        common::checked_add_int(search_end, -shift,
                                "temporal search range overflows int"));
}

void schedule_temporal_range(int anchor_frame, int low, int high,
                             std::vector<int>& scheduled_frames) {
    scheduled_frames.clear();
    scheduled_frames.push_back(anchor_frame);
    for (int frame = anchor_frame + 1; frame <= high; ++frame) {
        scheduled_frames.push_back(frame);
    }
    for (int frame = anchor_frame - 1; frame >= low; --frame) {
        scheduled_frames.push_back(frame);
    }
}

void compute_spatial_range(VideoGeometry geometry, StageParameters parameters,
                           int center, int axis_size, int& low, int& high) {
    const int max_origin = axis_size - parameters.patch_size;
    const int half = (parameters.search_window - 1) / 2;
    const int search_start = common::checked_add_int(
        center, -half, "spatial search range overflows int");
    const int search_end = common::checked_add_int(
        center, half, "spatial search range overflows int");
    const int shift = common::checked_add_int(
        std::min(0, search_start),
        std::max(0,
                 common::checked_add_int(search_end, -max_origin,
                                         "spatial search range overflows int")),
        "spatial search range overflows int");
    low = std::max(
        0, common::checked_add_int(search_start, -shift,
                                   "spatial search range overflows int"));
    high =
        std::min(max_origin,
                 common::checked_add_int(search_end, -shift,
                                         "spatial search range overflows int"));
    (void)geometry;
}

[[nodiscard]] bool
reference_processing_mask_selected(VideoGeometry geometry,
                                   StageParameters parameters, int anchor_frame,
                                   int x, int y) noexcept {
    const int step = normalized_proc_step(parameters);
    const int max_x = geometry.width - parameters.patch_size;
    const int max_y = geometry.height - parameters.patch_size;
    const int max_frame = geometry.source_frames() - parameters.patch_time;

    const int phase_y = anchor_frame == max_frame ? 0 : anchor_frame;
    if (y != 0 && y != max_y && y % step != phase_y % step) {
        return false;
    }

    const int phase_x = y == max_y ? 0 : phase_y + (y / step);
    return x == 0 || x == max_x || x % step == phase_x % step;
}

void initialize_frame_processing_mask(VideoGeometry geometry,
                                      StageParameters parameters,
                                      int anchor_frame,
                                      std::vector<unsigned char>& mask) {
    mask.assign(static_cast<std::size_t>(geometry.plane_pixels()), 0);
    const int max_x = geometry.width - parameters.patch_size;
    const int max_y = geometry.height - parameters.patch_size;
    for (int y = 0; y <= max_y; ++y) {
        for (int x = 0; x <= max_x; ++x) {
            if (reference_processing_mask_selected(geometry, parameters,
                                                   anchor_frame, x, y)) {
                mask[static_cast<std::size_t>((y * geometry.width) + x)] = 1;
            }
        }
    }
}

void clear_processing_mask_position(std::vector<unsigned char>& mask,
                                    VideoGeometry geometry, int anchor_frame,
                                    int frame, int x, int y) noexcept {
    if (frame != anchor_frame || x < 0 || x >= geometry.width || y < 0 ||
        y >= geometry.height) {
        return;
    }
    const int index = (y * geometry.width) + x;
    mask[static_cast<std::size_t>(index)] = 0;
}

void apply_frame_paste_mask(StageWorkspace& workspace, int anchor_frame,
                            int similar) {
    const VideoGeometry geometry = workspace.geometry_;
    const int patch_size = workspace.parameters_.patch_size;
    std::vector<unsigned char>& mask = workspace.processing_mask_;
    for (int patch = 0; patch < similar; ++patch) {
        const PatchMatch match =
            workspace.matches_[static_cast<std::size_t>(patch)];

        clear_processing_mask_position(mask, geometry, anchor_frame,
                                       match.frame, match.x, match.y);
        if (match.y > 2 * patch_size) {
            clear_processing_mask_position(mask, geometry, anchor_frame,
                                           match.frame, match.x, match.y - 1);
        }
        if (match.y < geometry.height - (2 * patch_size)) {
            clear_processing_mask_position(mask, geometry, anchor_frame,
                                           match.frame, match.x, match.y + 1);
        }
        if (match.x > 2 * patch_size) {
            clear_processing_mask_position(mask, geometry, anchor_frame,
                                           match.frame, match.x - 1, match.y);
        }
        if (match.x < geometry.width - (2 * patch_size)) {
            clear_processing_mask_position(mask, geometry, anchor_frame,
                                           match.frame, match.x + 1, match.y);
        }
    }
}

void prepare_distance_planes(ConstVideoView view, int candidate_frame,
                             int channels, StageParameters parameters,
                             StageWorkspace& workspace) {
    const int plane_count = channels * parameters.patch_time;
    workspace.distance_plane_bases_.resize(
        static_cast<std::size_t>(plane_count));
    workspace.distance_plane_strides_.resize(
        static_cast<std::size_t>(plane_count));

    for (int channel = 0; channel < channels; ++channel) {
        const int channel_base = channel * parameters.patch_time;
        for (int frame_delta = 0; frame_delta < parameters.patch_time;
             ++frame_delta) {
            const int frame = candidate_frame + frame_delta;
            const int index = channel_base + frame_delta;
            workspace.distance_plane_bases_[static_cast<std::size_t>(index)] =
                view.plane_data(frame, channel);
            workspace.distance_plane_strides_[static_cast<std::size_t>(index)] =
                view.plane_stride(frame, channel);
        }
    }
}

template <typename DistanceLimitFn, typename RecordCandidateFn>
void scan_basic_spatial_distances(ConstVideoView noisy, int x_low, int x_high,
                                  int y_low, int y_high, int candidate_frame,
                                  StageParameters parameters,
                                  StageWorkspace& workspace,
                                  DistanceLimitFn distance_limit_fn,
                                  RecordCandidateFn record_candidate_fn) {
    const int patch_size = parameters.patch_size;
    const int patch_area = patch_size * patch_size;
    prepare_distance_planes(noisy, candidate_frame, 1, parameters, workspace);
    const std::vector<const float*>& plane_bases =
        workspace.distance_plane_bases_;
    const std::vector<int>& plane_strides = workspace.distance_plane_strides_;

    for (int candidate_y = y_low; candidate_y <= y_high; ++candidate_y) {
        for (int candidate_x = x_low; candidate_x <= x_high; ++candidate_x) {
            const float distance_limit = distance_limit_fn();
            float distance = 0.0F;

            for (int frame_delta = 0; frame_delta < parameters.patch_time;
                 ++frame_delta) {
                const float* VNLB_RESTRICT reference_row =
                    workspace.reference_patch_.data() +
                    (frame_delta * patch_area);
                const std::size_t plane_index =
                    static_cast<std::size_t>(frame_delta);
                const int candidate_stride = plane_strides[plane_index];
                const float* VNLB_RESTRICT candidate_row =
                    plane_bases[plane_index] +
                    (static_cast<std::ptrdiff_t>(candidate_y) *
                     candidate_stride) +
                    candidate_x;

                for (int y = 0; y < patch_size; ++y) {
                    distance = distance::add_squared_row_distance_highway(
                        distance, reference_row, candidate_row, patch_size);
                    if (distance > distance_limit) {
                        break;
                    }
                    reference_row += patch_size;
                    candidate_row += candidate_stride;
                }
                if (distance > distance_limit) {
                    break;
                }
            }

            if (distance <= distance_limit) {
                record_candidate_fn(candidate_x, candidate_y, candidate_frame,
                                    distance);
            }
        }
    }
}

template <typename DistanceLimitFn, typename RecordCandidateFn>
void scan_final_spatial_distances(
    ConstVideoView reference, int x_low, int x_high, int y_low, int y_high,
    int candidate_frame, StageParameters parameters, StageWorkspace& workspace,
    DistanceLimitFn distance_limit_fn, RecordCandidateFn record_candidate_fn) {
    const int patch_size = parameters.patch_size;
    const int patch_area = patch_size * patch_size;
    const int patch_dim = patch_area * parameters.patch_time;
    const int channels = reference.geometry().channels;
    prepare_distance_planes(reference, candidate_frame, channels, parameters,
                            workspace);
    const std::vector<const float*>& plane_bases =
        workspace.distance_plane_bases_;
    const std::vector<int>& plane_strides = workspace.distance_plane_strides_;

    for (int candidate_y = y_low; candidate_y <= y_high; ++candidate_y) {
        for (int candidate_x = x_low; candidate_x <= x_high; ++candidate_x) {
            const float distance_limit = distance_limit_fn();
            float distance = 0.0F;

            for (int channel = 0; channel < channels; ++channel) {
                const int channel_patch_base = channel * patch_dim;
                const int channel_plane_base = channel * parameters.patch_time;
                for (int frame_delta = 0; frame_delta < parameters.patch_time;
                     ++frame_delta) {
                    const int plane_index = channel_plane_base + frame_delta;
                    const float* VNLB_RESTRICT reference_row =
                        workspace.reference_patch_.data() + channel_patch_base +
                        (frame_delta * patch_area);
                    const std::size_t plane_offset =
                        static_cast<std::size_t>(plane_index);
                    const int candidate_stride = plane_strides[plane_offset];
                    const float* VNLB_RESTRICT candidate_row =
                        plane_bases[plane_offset] +
                        (static_cast<std::ptrdiff_t>(candidate_y) *
                         candidate_stride) +
                        candidate_x;

                    for (int y = 0; y < patch_size; ++y) {
                        distance = distance::add_squared_row_distance_highway(
                            distance, reference_row, candidate_row, patch_size);
                        if (distance > distance_limit) {
                            break;
                        }
                        reference_row += patch_size;
                        candidate_row += candidate_stride;
                    }
                    if (distance > distance_limit) {
                        break;
                    }
                }
                if (distance > distance_limit) {
                    break;
                }
            }

            if (distance <= distance_limit) {
                record_candidate_fn(candidate_x, candidate_y, candidate_frame,
                                    distance);
            }
        }
    }
}

void load_reference_patch_basic(ConstVideoView noisy, int anchor_x,
                                int anchor_y, int anchor_frame,
                                StageParameters parameters,
                                StageWorkspace& workspace) {
    const int patch_size = parameters.patch_size;
    workspace.reference_patch_.resize(
        static_cast<std::size_t>(workspace.patch_dim_));
    for (int frame_delta = 0; frame_delta < parameters.patch_time;
         ++frame_delta) {
        const int frame = anchor_frame + frame_delta;
        for (int y = 0; y < patch_size; ++y) {
            const int position = patch_position(patch_size, 0, y, frame_delta);
            const float* VNLB_RESTRICT input_row =
                noisy.row_data(frame, 0, anchor_y + y, anchor_x);
            float* VNLB_RESTRICT output_row =
                workspace.reference_patch_.data() + position;
            copy_float_row(input_row, output_row, patch_size);
        }
    }
}

void load_reference_patch_final(ConstVideoView reference, int anchor_x,
                                int anchor_y, int anchor_frame,
                                StageParameters parameters,
                                StageWorkspace& workspace) {
    const int patch_size = parameters.patch_size;
    const int patch_dim = workspace.patch_dim_;
    const int channels = reference.geometry().channels;
    workspace.reference_patch_.resize(
        static_cast<std::size_t>(patch_dim * channels));
    for (int channel = 0; channel < channels; ++channel) {
        const int channel_base = channel * patch_dim;
        for (int frame_delta = 0; frame_delta < parameters.patch_time;
             ++frame_delta) {
            const int frame = anchor_frame + frame_delta;
            for (int y = 0; y < patch_size; ++y) {
                const int position =
                    patch_position(patch_size, 0, y, frame_delta);
                const float* VNLB_RESTRICT input_row =
                    reference.row_data(frame, channel, anchor_y + y, anchor_x);
                float* VNLB_RESTRICT output_row =
                    workspace.reference_patch_.data() + channel_base + position;
                copy_float_row(input_row, output_row, patch_size);
            }
        }
    }
}

int select_similar_patches(StageWorkspace& workspace,
                           StageParameters parameters) {
    if (parameters.similar <= 1) {
        workspace.matches_.resize(1);
        return 1;
    }

    const int selected = std::min(parameters.similar,
                                  static_cast<int>(workspace.matches_.size()));
    const auto first = workspace.matches_.begin();
    const auto selected_end = first + selected;
    const auto last = workspace.matches_.end();
    std::partial_sort(first, selected_end, last,
                      [](PatchMatch lhs, PatchMatch rhs) {
                          return lhs.distance < rhs.distance;
                      });

    const float kth_distance =
        workspace.matches_[static_cast<std::size_t>(selected - 1)].distance;
    const float threshold = std::max(parameters.tau, kth_distance);
    auto output = first;
    for (auto match = first; match != last; ++match) {
        if (match->distance <= threshold) {
            *output = *match;
            ++output;
        }
    }

    const int output_cap =
        std::min(soft_similar_cap(parameters),
                 static_cast<int>(workspace.matches_.size()));
    const auto output_cap_end = first + output_cap;
    if (output > output_cap_end) {
        std::partial_sort(first, output_cap_end, output,
                          [](PatchMatch lhs, PatchMatch rhs) {
                              return lhs.distance < rhs.distance;
                          });
        output = output_cap_end;
    }

    const int similar = static_cast<int>(output - first);
    workspace.matches_.resize(static_cast<std::size_t>(similar));
    return similar;
}

template <typename FlowProvider, typename ScanDistanceFn>
int find_similar_patches(ConstVideoView search_source, int anchor_x,
                         int anchor_y, int anchor_frame,
                         StageParameters parameters,
                         const FlowProvider& flow_provider,
                         StageWorkspace& workspace,
                         ScanDistanceFn scan_distance_fn) {
    const VideoGeometry geometry = search_source.geometry();
    if (parameters.similar <= 1) {
        workspace.matches_.assign(
            1, PatchMatch{0.0F, anchor_x, anchor_y, anchor_frame});
        return 1;
    }

    int temporal_low = 0;
    int temporal_high = 0;
    compute_temporal_range(geometry, parameters, anchor_frame, temporal_low,
                           temporal_high);
    schedule_temporal_range(anchor_frame, temporal_low, temporal_high,
                            workspace.scheduled_frames_);

    workspace.matches_.clear();
    const int output_cap =
        std::min(soft_similar_cap(parameters),
                 common::checked_mul_int(
                     common::checked_mul_int(
                         parameters.search_window, parameters.search_window,
                         "spatial candidate count overflows int"),
                     static_cast<int>(workspace.scheduled_frames_.size()),
                     "candidate count overflows int"));
    workspace.matches_.reserve(static_cast<std::size_t>(output_cap));

    std::vector<float>& top_distances = workspace.top_distances_;
    top_distances.clear();
    top_distances.reserve(static_cast<std::size_t>(parameters.similar));
    float soft_worst_distance = std::numeric_limits<float>::infinity();
    std::size_t soft_worst_index = 0;

    const auto refresh_soft_worst_distance = [&]() {
        const auto first = workspace.matches_.begin();
        const auto worst =
            std::max_element(first, workspace.matches_.end(),
                             [](PatchMatch lhs, PatchMatch rhs) {
                                 return lhs.distance < rhs.distance;
                             });
        soft_worst_index = static_cast<std::size_t>(worst - first);
        soft_worst_distance = worst->distance;
    };
    const auto farther_distance = [](float lhs, float rhs) {
        return lhs < rhs;
    };
    const auto record_candidate_distance = [&](float distance) {
        if (static_cast<int>(top_distances.size()) < parameters.similar) {
            top_distances.push_back(distance);
            std::push_heap(top_distances.begin(), top_distances.end(),
                           farther_distance);
            return;
        }

        if (distance < top_distances.front()) {
            std::pop_heap(top_distances.begin(), top_distances.end(),
                          farther_distance);
            top_distances.back() = distance;
            std::push_heap(top_distances.begin(), top_distances.end(),
                           farther_distance);
        }
    };
    const auto record_candidate_match = [&](PatchMatch match) {
        if (static_cast<int>(workspace.matches_.size()) < output_cap) {
            workspace.matches_.push_back(match);
            if (static_cast<int>(workspace.matches_.size()) == output_cap) {
                refresh_soft_worst_distance();
            }
            return;
        }

        if (match.distance < soft_worst_distance) {
            workspace.matches_[soft_worst_index] = match;
            refresh_soft_worst_distance();
        }
    };

    for (int candidate_frame : workspace.scheduled_frames_) {
        const flow::SearchCenter center =
            flow_provider.center_for(flow::SearchCenterRequest{
                anchor_x, anchor_y, anchor_frame, candidate_frame,
                parameters.patch_size, geometry.width, geometry.height});
        int x_low = 0;
        int x_high = 0;
        int y_low = 0;
        int y_high = 0;
        compute_spatial_range(geometry, parameters, center.x, geometry.width,
                              x_low, x_high);
        compute_spatial_range(geometry, parameters, center.y, geometry.height,
                              y_low, y_high);

        const auto distance_limit_for_next_candidate = [&]() {
            float distance_limit =
                static_cast<int>(top_distances.size()) >= parameters.similar
                    ? std::max(parameters.tau, top_distances.front())
                    : std::numeric_limits<float>::infinity();
            if (static_cast<int>(workspace.matches_.size()) >= output_cap) {
                distance_limit = std::min(distance_limit, soft_worst_distance);
            }
            return distance_limit;
        };
        const auto record_candidate = [&](int candidate_x, int candidate_y,
                                          int frame, float distance) {
            record_candidate_match(
                PatchMatch{distance, candidate_x, candidate_y, frame});
            record_candidate_distance(distance);
        };
        scan_distance_fn(x_low, x_high, y_low, y_high, candidate_frame,
                         distance_limit_for_next_candidate, record_candidate);
    }

    return select_similar_patches(workspace, parameters);
}

void gather_basic_group(ConstVideoView noisy, StageWorkspace& workspace,
                        int similar) {
    const VideoGeometry geometry = noisy.geometry();
    const int patch_size = workspace.parameters_.patch_size;
    const int patch_dim = workspace.patch_dim_;
    for (int channel = 0; channel < geometry.channels; ++channel) {
        const int channel_base = channel * patch_dim * similar;
        for (int position = 0; position < patch_dim * similar; ++position) {
            workspace.group_noisy_[static_cast<std::size_t>(channel_base +
                                                            position)] = 0.0F;
        }
        for (int frame_delta = 0;
             frame_delta < workspace.parameters_.patch_time; ++frame_delta) {
            for (int y = 0; y < patch_size; ++y) {
                for (int x = 0; x < patch_size; ++x) {
                    const int position =
                        patch_position(patch_size, x, y, frame_delta);
                    for (int patch = 0; patch < similar; ++patch) {
                        const PatchMatch match =
                            workspace.matches_[static_cast<std::size_t>(patch)];
                        workspace.group_noisy_[static_cast<std::size_t>(
                            channel_base + (position * similar) + patch)] =
                            noisy.sample(match.x + x, match.y + y,
                                         match.frame + frame_delta, channel);
                    }
                }
            }
        }
    }
}

void gather_basic_samples_coupled_planes(
    VideoGeometry geometry, const ConstPlaneView* VNLB_RESTRICT planes,
    StageWorkspace& workspace, int similar) {
    const int channels = geometry.channels;
    const int patch_size = workspace.parameters_.patch_size;
    const int patch_dim = workspace.patch_dim_;
    const int sample_dim = channels * patch_dim;
    workspace.samples_noisy_.resize(static_cast<std::size_t>(similar),
                                    static_cast<std::size_t>(sample_dim));

    for (int patch = 0; patch < similar; ++patch) {
        const PatchMatch match =
            workspace.matches_[static_cast<std::size_t>(patch)];
        float* VNLB_RESTRICT row =
            workspace.samples_noisy_.row_data(static_cast<std::size_t>(patch));
        for (int channel = 0; channel < channels; ++channel) {
            const int channel_base = channel * patch_dim;
            for (int frame_delta = 0;
                 frame_delta < workspace.parameters_.patch_time;
                 ++frame_delta) {
                const int local_frame =
                    geometry.local_frame(match.frame + frame_delta);
                const ConstPlaneView plane =
                    planes[(local_frame * channels) + channel];
                for (int y = 0; y < patch_size; ++y) {
                    const int position =
                        patch_position(patch_size, 0, y, frame_delta);
                    const float* VNLB_RESTRICT input_row =
                        plane.data +
                        (static_cast<std::ptrdiff_t>(match.y + y) *
                         plane.stride) +
                        match.x;
                    copy_float_row(input_row, row + channel_base + position,
                                   patch_size);
                }
            }
        }
    }
}

void gather_basic_samples_coupled(ConstVideoView noisy,
                                  StageWorkspace& workspace, int similar) {
    if (const ConstPlaneView* planes = noisy.planes(); planes != nullptr) {
        gather_basic_samples_coupled_planes(noisy.geometry(), planes, workspace,
                                            similar);
        return;
    }

    const VideoGeometry geometry = noisy.geometry();
    const int channels = geometry.channels;
    const int patch_size = workspace.parameters_.patch_size;
    const int patch_dim = workspace.patch_dim_;
    const int sample_dim = channels * patch_dim;
    workspace.samples_noisy_.resize(static_cast<std::size_t>(similar),
                                    static_cast<std::size_t>(sample_dim));

    for (int patch = 0; patch < similar; ++patch) {
        const PatchMatch match =
            workspace.matches_[static_cast<std::size_t>(patch)];
        float* VNLB_RESTRICT row =
            workspace.samples_noisy_.row_data(static_cast<std::size_t>(patch));
        for (int channel = 0; channel < channels; ++channel) {
            const int channel_base = channel * patch_dim;
            for (int frame_delta = 0;
                 frame_delta < workspace.parameters_.patch_time;
                 ++frame_delta) {
                const int frame = match.frame + frame_delta;
                const float* VNLB_RESTRICT plane =
                    noisy.plane_data(frame, channel);
                const int stride = noisy.plane_stride(frame, channel);
                for (int y = 0; y < patch_size; ++y) {
                    const int position =
                        patch_position(patch_size, 0, y, frame_delta);
                    const float* VNLB_RESTRICT input_row =
                        plane +
                        (static_cast<std::ptrdiff_t>(match.y + y) * stride) +
                        match.x;
                    copy_float_row(input_row, row + channel_base + position,
                                   patch_size);
                }
            }
        }
    }
}

void gather_final_group(ConstVideoView noisy, ConstVideoView basic,
                        StageWorkspace& workspace, int similar) {
    const VideoGeometry geometry = noisy.geometry();
    const int patch_size = workspace.parameters_.patch_size;
    const int patch_dim = workspace.patch_dim_;
    for (int channel = 0; channel < geometry.channels; ++channel) {
        const int channel_base = channel * patch_dim * similar;
        for (int frame_delta = 0;
             frame_delta < workspace.parameters_.patch_time; ++frame_delta) {
            for (int y = 0; y < patch_size; ++y) {
                for (int x = 0; x < patch_size; ++x) {
                    const int position =
                        patch_position(patch_size, x, y, frame_delta);
                    for (int patch = 0; patch < similar; ++patch) {
                        const PatchMatch match =
                            workspace.matches_[static_cast<std::size_t>(patch)];
                        const std::size_t group_index =
                            static_cast<std::size_t>(
                                channel_base + (position * similar) + patch);
                        workspace.group_noisy_[group_index] =
                            noisy.sample(match.x + x, match.y + y,
                                         match.frame + frame_delta, channel);
                        workspace.group_basic_[group_index] =
                            basic.sample(match.x + x, match.y + y,
                                         match.frame + frame_delta, channel);
                    }
                }
            }
        }
    }
}

void gather_final_samples_coupled_planes(
    VideoGeometry geometry, const ConstPlaneView* VNLB_RESTRICT noisy_planes,
    const ConstPlaneView* VNLB_RESTRICT basic_planes, StageWorkspace& workspace,
    int similar) {
    const int channels = geometry.channels;
    const int patch_size = workspace.parameters_.patch_size;
    const int patch_dim = workspace.patch_dim_;
    const int sample_dim = channels * patch_dim;
    workspace.samples_noisy_.resize(static_cast<std::size_t>(similar),
                                    static_cast<std::size_t>(sample_dim));
    workspace.samples_basic_.resize(static_cast<std::size_t>(similar),
                                    static_cast<std::size_t>(sample_dim));

    for (int patch = 0; patch < similar; ++patch) {
        const PatchMatch match =
            workspace.matches_[static_cast<std::size_t>(patch)];
        float* VNLB_RESTRICT noisy_row =
            workspace.samples_noisy_.row_data(static_cast<std::size_t>(patch));
        float* VNLB_RESTRICT basic_row =
            workspace.samples_basic_.row_data(static_cast<std::size_t>(patch));
        for (int channel = 0; channel < channels; ++channel) {
            const int channel_base = channel * patch_dim;
            for (int frame_delta = 0;
                 frame_delta < workspace.parameters_.patch_time;
                 ++frame_delta) {
                const int local_frame =
                    geometry.local_frame(match.frame + frame_delta);
                const int plane_index = (local_frame * channels) + channel;
                const ConstPlaneView noisy_plane = noisy_planes[plane_index];
                const ConstPlaneView basic_plane = basic_planes[plane_index];
                for (int y = 0; y < patch_size; ++y) {
                    const int position =
                        patch_position(patch_size, 0, y, frame_delta);
                    const int sample_position = channel_base + position;
                    const float* VNLB_RESTRICT noisy_input_row =
                        noisy_plane.data +
                        (static_cast<std::ptrdiff_t>(match.y + y) *
                         noisy_plane.stride) +
                        match.x;
                    const float* VNLB_RESTRICT basic_input_row =
                        basic_plane.data +
                        (static_cast<std::ptrdiff_t>(match.y + y) *
                         basic_plane.stride) +
                        match.x;
                    copy_float_row(noisy_input_row, noisy_row + sample_position,
                                   patch_size);
                    copy_float_row(basic_input_row, basic_row + sample_position,
                                   patch_size);
                }
            }
        }
    }
}

void gather_final_samples_coupled(ConstVideoView noisy, ConstVideoView basic,
                                  StageWorkspace& workspace, int similar) {
    if (const ConstPlaneView* noisy_planes = noisy.planes();
        noisy_planes != nullptr) {
        if (const ConstPlaneView* basic_planes = basic.planes();
            basic_planes != nullptr) {
            gather_final_samples_coupled_planes(noisy.geometry(), noisy_planes,
                                                basic_planes, workspace,
                                                similar);
            return;
        }
    }

    const VideoGeometry geometry = noisy.geometry();
    const int channels = geometry.channels;
    const int patch_size = workspace.parameters_.patch_size;
    const int patch_dim = workspace.patch_dim_;
    const int sample_dim = channels * patch_dim;
    workspace.samples_noisy_.resize(static_cast<std::size_t>(similar),
                                    static_cast<std::size_t>(sample_dim));
    workspace.samples_basic_.resize(static_cast<std::size_t>(similar),
                                    static_cast<std::size_t>(sample_dim));

    for (int patch = 0; patch < similar; ++patch) {
        const PatchMatch match =
            workspace.matches_[static_cast<std::size_t>(patch)];
        float* VNLB_RESTRICT noisy_row =
            workspace.samples_noisy_.row_data(static_cast<std::size_t>(patch));
        float* VNLB_RESTRICT basic_row =
            workspace.samples_basic_.row_data(static_cast<std::size_t>(patch));
        for (int channel = 0; channel < channels; ++channel) {
            const int channel_base = channel * patch_dim;
            for (int frame_delta = 0;
                 frame_delta < workspace.parameters_.patch_time;
                 ++frame_delta) {
                const int frame = match.frame + frame_delta;
                const float* VNLB_RESTRICT noisy_plane =
                    noisy.plane_data(frame, channel);
                const int noisy_stride = noisy.plane_stride(frame, channel);
                const float* VNLB_RESTRICT basic_plane =
                    basic.plane_data(frame, channel);
                const int basic_stride = basic.plane_stride(frame, channel);
                for (int y = 0; y < patch_size; ++y) {
                    const int position =
                        patch_position(patch_size, 0, y, frame_delta);
                    const int sample_position = channel_base + position;
                    const float* VNLB_RESTRICT noisy_input_row =
                        noisy_plane +
                        (static_cast<std::ptrdiff_t>(match.y + y) *
                         noisy_stride) +
                        match.x;
                    const float* VNLB_RESTRICT basic_input_row =
                        basic_plane +
                        (static_cast<std::ptrdiff_t>(match.y + y) *
                         basic_stride) +
                        match.x;
                    copy_float_row(noisy_input_row, noisy_row + sample_position,
                                   patch_size);
                    copy_float_row(basic_input_row, basic_row + sample_position,
                                   patch_size);
                }
            }
        }
    }
}

void write_group_contributions(StageWorkspace& workspace, int anchor_frame,
                               int similar,
                               aggregate::ContributionStackView stack) {
    const auto layout = stack.layout();
    const int channels = workspace.geometry_.channels;
    const int patch_size = workspace.parameters_.patch_size;
    const int patch_dim = workspace.patch_dim_;
    const std::vector<float>& group = workspace.group_noisy_;

    for (int patch = 0; patch < similar; ++patch) {
        const PatchMatch match =
            workspace.matches_[static_cast<std::size_t>(patch)];
        const float patch_weight =
            workspace
                .aggregation_patch_weights_[static_cast<std::size_t>(patch)];
        for (int frame_delta = 0;
             frame_delta < workspace.parameters_.patch_time; ++frame_delta) {
            const int output_offset = match.frame + frame_delta - anchor_frame;
            if (!layout.contains_output_offset(output_offset)) {
                continue;
            }
            const int slot = layout.slot_for_output_offset(output_offset);
            for (int y = 0; y < patch_size; ++y) {
                const int output_y = match.y + y;
                for (int x = 0; x < patch_size; ++x) {
                    const int output_x = match.x + x;
                    const int position =
                        patch_position(patch_size, x, y, frame_delta);
                    const std::size_t window_index =
                        static_cast<std::size_t>((y * patch_size) + x);
                    const float contribution_weight =
                        patch_weight *
                        workspace.aggregation_window_weights_[window_index];
                    stack.add_weight(slot, output_x, output_y,
                                     contribution_weight);
                    for (int channel = 0; channel < channels; ++channel) {
                        const int channel_base = channel * patch_dim * similar;
                        stack.numerator(slot, channel, output_x, output_y) +=
                            contribution_weight *
                            group[static_cast<std::size_t>(
                                channel_base + (position * similar) + patch)];
                    }
                }
            }
        }
    }
}

void write_sample_contributions_coupled_contiguous(
    StageWorkspace& workspace, int anchor_frame, int similar,
    aggregate::ContributionLayout layout, float* VNLB_RESTRICT data) {
    const int channels = workspace.geometry_.channels;
    const int patch_size = workspace.parameters_.patch_size;
    const int patch_dim = workspace.patch_dim_;
    const linalg::Matrix<float>& samples = workspace.filtered_;
    const int plane_pixels = layout.plane_pixels();
    const int slot_stride = layout.slot_stride();
    const int weight_plane = channels * plane_pixels;
    const auto count = static_cast<std::size_t>(patch_size);
    const bool windowed = workspace.parameters_.weight_gamma != 0.0F;

    for (int patch = 0; patch < similar; ++patch) {
        const PatchMatch match =
            workspace.matches_[static_cast<std::size_t>(patch)];
        const float patch_weight =
            workspace
                .aggregation_patch_weights_[static_cast<std::size_t>(patch)];
        const float* VNLB_RESTRICT row =
            samples.row_data(static_cast<std::size_t>(patch));
        for (int frame_delta = 0;
             frame_delta < workspace.parameters_.patch_time; ++frame_delta) {
            const int output_offset = match.frame + frame_delta - anchor_frame;
            if (!layout.contains_output_offset(output_offset)) {
                continue;
            }
            const int slot = layout.slot_for_output_offset(output_offset);
            for (int y = 0; y < patch_size; ++y) {
                const int output_y = match.y + y;
                const int position =
                    patch_position(patch_size, 0, y, frame_delta);
                const int row_offset = (output_y * layout.width) + match.x;
                float* VNLB_RESTRICT weight_row =
                    data + (slot * slot_stride) + weight_plane + row_offset;

                for (int channel = 0; channel < channels; ++channel) {
                    const float* VNLB_RESTRICT sample_row =
                        row + (channel * patch_dim) + position;
                    float* VNLB_RESTRICT numerator_row =
                        data + (slot * slot_stride) + (channel * plane_pixels) +
                        row_offset;
                    if (!windowed) {
                        linalg::kernels::add_scaled_contiguous_highway(
                            numerator_row, sample_row, patch_weight, count);
                    } else {
                        for (int x = 0; x < patch_size; ++x) {
                            const std::size_t window_index =
                                static_cast<std::size_t>((y * patch_size) + x);
                            const float contribution_weight =
                                patch_weight *
                                workspace
                                    .aggregation_window_weights_[window_index];
                            numerator_row[x] +=
                                contribution_weight * sample_row[x];
                        }
                    }
                }

                if (!windowed) {
                    linalg::kernels::add_scalar_contiguous_highway(
                        weight_row, patch_weight, count);
                } else {
                    for (int x = 0; x < patch_size; ++x) {
                        const std::size_t window_index =
                            static_cast<std::size_t>((y * patch_size) + x);
                        weight_row[x] +=
                            patch_weight *
                            workspace.aggregation_window_weights_[window_index];
                    }
                }
            }
        }
    }
}

void write_sample_contributions_coupled_planes(
    StageWorkspace& workspace, int anchor_frame, int similar,
    aggregate::ContributionLayout layout,
    aggregate::ContributionPlaneView* VNLB_RESTRICT planes) {
    const int channels = workspace.geometry_.channels;
    const int patch_size = workspace.parameters_.patch_size;
    const int patch_dim = workspace.patch_dim_;
    const linalg::Matrix<float>& samples = workspace.filtered_;
    const auto count = static_cast<std::size_t>(patch_size);
    const bool windowed = workspace.parameters_.weight_gamma != 0.0F;

    for (int patch = 0; patch < similar; ++patch) {
        const PatchMatch match =
            workspace.matches_[static_cast<std::size_t>(patch)];
        const float patch_weight =
            workspace
                .aggregation_patch_weights_[static_cast<std::size_t>(patch)];
        const float* VNLB_RESTRICT row =
            samples.row_data(static_cast<std::size_t>(patch));
        for (int frame_delta = 0;
             frame_delta < workspace.parameters_.patch_time; ++frame_delta) {
            const int output_offset = match.frame + frame_delta - anchor_frame;
            if (!layout.contains_output_offset(output_offset)) {
                continue;
            }
            const int slot = layout.slot_for_output_offset(output_offset);
            for (int y = 0; y < patch_size; ++y) {
                const int output_y = match.y + y;
                const int position =
                    patch_position(patch_size, 0, y, frame_delta);
                const int numerator_row = (slot * 2 * layout.height) + output_y;
                const int weight_row = numerator_row + layout.height;

                for (int channel = 0; channel < channels; ++channel) {
                    const auto plane = planes[channel];
                    const float* VNLB_RESTRICT sample_row =
                        row + (channel * patch_dim) + position;
                    float* VNLB_RESTRICT numerator =
                        plane.data +
                        (static_cast<std::ptrdiff_t>(numerator_row) *
                         plane.stride) +
                        match.x;
                    float* VNLB_RESTRICT weight =
                        plane.data +
                        (static_cast<std::ptrdiff_t>(weight_row) *
                         plane.stride) +
                        match.x;
                    if (!windowed) {
                        linalg::kernels::add_scaled_contiguous_highway(
                            numerator, sample_row, patch_weight, count);
                        linalg::kernels::add_scalar_contiguous_highway(
                            weight, patch_weight, count);
                    } else {
                        for (int x = 0; x < patch_size; ++x) {
                            const std::size_t window_index =
                                static_cast<std::size_t>((y * patch_size) + x);
                            const float contribution_weight =
                                patch_weight *
                                workspace
                                    .aggregation_window_weights_[window_index];
                            numerator[x] += contribution_weight * sample_row[x];
                            weight[x] += contribution_weight;
                        }
                    }
                }
            }
        }
    }
}

void write_sample_contributions_coupled(
    StageWorkspace& workspace, int anchor_frame, int similar,
    aggregate::ContributionStackView stack) {
    const auto layout = stack.layout();
    if (aggregate::ContributionPlaneView* planes = stack.planes();
        planes != nullptr) {
        write_sample_contributions_coupled_planes(workspace, anchor_frame,
                                                  similar, layout, planes);
        return;
    }
    if (float* data = stack.data(); data != nullptr) {
        write_sample_contributions_coupled_contiguous(workspace, anchor_frame,
                                                      similar, layout, data);
    }
}

template <Stage stage>
void validate_video_for_stage(ConstVideoView video,
                              StageParameters parameters) {
    require(video.has_storage(), "video data pointer must not be null");
    validate_stage_configuration(video.geometry(), parameters);
}

void validate_contribution_layout(aggregate::ContributionStackView stack,
                                  VideoGeometry geometry,
                                  StageParameters parameters) {
    const auto layout = stack.layout();
    require(stack.has_storage(),
            "contribution stack data pointer must not be null");
    require(layout.width == geometry.width &&
                layout.height == geometry.height &&
                layout.channels == geometry.channels,
            "contribution stack dimensions must match the video");
    require(layout.search_bwd == parameters.search_bwd &&
                layout.search_fwd == parameters.search_fwd &&
                layout.patch_time == parameters.patch_time,
            "contribution stack temporal layout does not match parameters");
}

template <Stage stage, typename FlowProvider, typename ProcessGroupFn>
ProcessStats process_anchor_impl(ConstVideoView noisy, ConstVideoView reference,
                                 int anchor_frame, StageParameters parameters,
                                 const FlowProvider& flow_provider,
                                 aggregate::ContributionStackView contributions,
                                 StageWorkspace& workspace,
                                 ProcessGroupFn process_group) {
    validate_video_for_stage<stage>(noisy, parameters);
    if constexpr (stage == Stage::Final) {
        require(noisy.geometry().same_shape(reference.geometry()),
                "noisy and reference videos must have the same shape");
    }
    if constexpr (stage == Stage::Basic) {
        (void)reference;
    }
    validate_contribution_layout(contributions, noisy.geometry(), parameters);
    aggregate::clear_contributions(contributions);

    const VideoGeometry geometry = noisy.geometry();
    if (anchor_frame < 0 ||
        anchor_frame > geometry.source_frames() - parameters.patch_time) {
        return ProcessStats{};
    }
    workspace.prepare<stage>(geometry, parameters);

    const int max_x = geometry.width - parameters.patch_size;
    const int max_y = geometry.height - parameters.patch_size;
    initialize_frame_processing_mask(geometry, parameters, anchor_frame,
                                     workspace.processing_mask_);

    ProcessStats stats{};
    for (int anchor_y = 0; anchor_y <= max_y; ++anchor_y) {
        for (int anchor_x = 0; anchor_x <= max_x; ++anchor_x) {
            if (workspace.processing_mask_[static_cast<std::size_t>(
                    (anchor_y * geometry.width) + anchor_x)] == 0) {
                continue;
            }

            int similar = 0;
            if constexpr (stage == Stage::Basic) {
                load_reference_patch_basic(noisy, anchor_x, anchor_y,
                                           anchor_frame, parameters, workspace);
                similar = find_similar_patches(
                    noisy, anchor_x, anchor_y, anchor_frame, parameters,
                    flow_provider, workspace,
                    [&](int x_low, int x_high, int y_low, int y_high,
                        int candidate_frame, auto distance_limit_fn,
                        auto record_candidate_fn) {
                        scan_basic_spatial_distances(
                            noisy, x_low, x_high, y_low, y_high,
                            candidate_frame, parameters, workspace,
                            distance_limit_fn, record_candidate_fn);
                    });
            } else {
                load_reference_patch_final(reference, anchor_x, anchor_y,
                                           anchor_frame, parameters, workspace);
                similar = find_similar_patches(
                    reference, anchor_x, anchor_y, anchor_frame, parameters,
                    flow_provider, workspace,
                    [&](int x_low, int x_high, int y_low, int y_high,
                        int candidate_frame, auto distance_limit_fn,
                        auto record_candidate_fn) {
                        scan_final_spatial_distances(
                            reference, x_low, x_high, y_low, y_high,
                            candidate_frame, parameters, workspace,
                            distance_limit_fn, record_candidate_fn);
                    });
            }

            workspace.output_sample_major_ = false;
            reset_aggregation_weights(workspace, similar);
            process_group(similar);
            finalize_aggregation_weights(workspace, similar);
            if (workspace.output_sample_major_) {
                write_sample_contributions_coupled(workspace, anchor_frame,
                                                   similar, contributions);
            } else {
                write_group_contributions(workspace, anchor_frame, similar,
                                          contributions);
            }
            ++stats.groups;
            apply_frame_paste_mask(workspace, anchor_frame, similar);
        }
    }

    return stats;
}

} // namespace

void validate_stage_parameters(StageParameters parameters) {
    require(std::isfinite(parameters.sigma) && parameters.sigma > 0.0F,
            "sigma must be finite and positive");
    require(parameters.patch_size > 0, "block_size must be positive");
    require(parameters.patch_time > 0, "patch_time must be positive");
    require(parameters.search_window > 0 && parameters.search_window % 2 == 1,
            "bm_range must describe a positive odd search window");
    require(parameters.search_bwd >= 0 && parameters.search_fwd >= 0,
            "search_bwd and search_fwd must be non-negative");
    require(parameters.similar > 0, "group_size must be positive");
    require(parameters.rank >= 0, "rank must be non-negative");
    require(std::isfinite(parameters.similar_cap_factor) &&
                (parameters.similar_cap_factor == 0.0F ||
                 parameters.similar_cap_factor >= 1.0F),
            "cap_factor must be zero or at least 1.0");
    require(std::isfinite(parameters.model_cap_factor) &&
                (parameters.model_cap_factor == 0.0F ||
                 parameters.model_cap_factor >= 1.0F),
            "model_cap_factor must be zero or at least 1.0");
    require(std::isfinite(parameters.beta) && parameters.beta > 0.0F,
            "beta must be finite and positive");
    require(std::isfinite(parameters.tau) && parameters.tau >= 0.0F,
            "tau must be finite and non-negative");
    require(std::isfinite(parameters.variance_threshold),
            "variance_threshold must be finite");
    require(std::isfinite(parameters.sigma_basic) &&
                parameters.sigma_basic >= 0.0F,
            "sigma_basic must be finite and non-negative");
    require(std::isfinite(parameters.gamma) && parameters.gamma > 0.0F,
            "gamma must be finite and positive");
    require(std::isfinite(parameters.weight_alpha) &&
                parameters.weight_alpha >= 0.0F,
            "weight_alpha must be finite and non-negative");
    require(std::isfinite(parameters.weight_beta) &&
                parameters.weight_beta >= 0.0F,
            "weight_beta must be finite and non-negative");
    require(std::isfinite(parameters.weight_gamma) &&
                parameters.weight_gamma >= 0.0F,
            "weight_gamma must be finite and non-negative");
    require(std::isfinite(parameters.weight_epsilon) &&
                parameters.weight_epsilon > 0.0F,
            "weight_epsilon must be finite and positive");
    require(std::isfinite(parameters.membership_noise_floor) &&
                parameters.membership_noise_floor > 0.0F,
            "membership_noise_floor must be finite and positive");
    require(parameters.proc_step >= 0,
            "block_step must be non-negative; use 0 for auto");
}

void validate_stage_configuration(VideoGeometry geometry,
                                  StageParameters parameters) {
    validate_stage_parameters(parameters);
    require(geometry.valid(), "video geometry must be non-empty");
    (void)geometry.checked_sample_count();
    require(geometry.first_frame >= 0, "first frame must be non-negative");
    const int local_frame_end =
        common::checked_add_int(geometry.frames, geometry.first_frame,
                                "local frame window overflows int");
    require(geometry.source_frames() >= local_frame_end,
            "local frame window must fit inside the source frame count");
    require(parameters.patch_size <= geometry.width &&
                parameters.patch_size <= geometry.height,
            "block_size must fit inside the frame");
    require(parameters.patch_time <= geometry.source_frames(),
            "patch_time must fit inside the clip");
    const int estimator_dim =
        checked_estimator_dim(parameters, geometry.channels);
    const int max_similar = checked_retained_group_count(parameters);
    (void)checked_group_capacity(geometry.channels,
                                 checked_patch_dim(parameters), max_similar);
    require(parameters.rank <= estimator_dim,
            "rank must not exceed the effective patch dimension");
    require(
        parameters.similar >=
            std::min(estimator_dim, effective_rank(parameters, estimator_dim)),
        "group_size must cover the effective VNLB rank");
}

FrameRange input_frame_range_for_anchor(VideoGeometry geometry,
                                        StageParameters parameters,
                                        int anchor_frame) {
    int temporal_low = 0;
    int temporal_high = 0;
    compute_temporal_range(geometry, parameters, anchor_frame, temporal_low,
                           temporal_high);
    return FrameRange{temporal_low, temporal_high + parameters.patch_time - 1};
}

template <Stage stage>
void StageWorkspace::prepare(VideoGeometry geometry,
                             StageParameters parameters) {
    validate_stage_configuration(geometry, parameters);

    constexpr Stage current_stage = stage;
    const bool geometry_changed = !geometry.same_shape(geometry_);
    const bool buffers_changed =
        !same_processing_shape(parameters, parameters_) ||
        current_stage != stage_;
    geometry_ = geometry;
    parameters_ = parameters;
    stage_ = current_stage;
    if (!geometry_changed && !buffers_changed) {
        return;
    }

    patch_area_ = checked_patch_area(parameters);
    patch_dim_ = checked_patch_dim(parameters);
    estimator_dim_ = checked_estimator_dim(parameters, geometry.channels);
    max_similar_ = checked_retained_group_count(parameters);
    const std::size_t group_capacity =
        checked_group_capacity(geometry.channels, patch_dim_, max_similar_);
    group_noisy_.resize(group_capacity);
    group_basic_.resize(group_capacity);
    mean_noisy_.resize(static_cast<std::size_t>(estimator_dim_));
    mean_basic_.resize(static_cast<std::size_t>(estimator_dim_));
    eigenvalues_.resize(static_cast<std::size_t>(estimator_dim_));
    filter_coefficients_.resize(static_cast<std::size_t>(estimator_dim_));
    reference_patch_.resize(
        static_cast<std::size_t>(geometry.channels * patch_dim_));
    aggregation_log_patch_weights_.resize(
        static_cast<std::size_t>(max_similar_));
    aggregation_patch_weights_.resize(static_cast<std::size_t>(max_similar_));
    aggregation_scores_.resize(static_cast<std::size_t>(max_similar_));
    prepare_aggregation_window(*this);
}

template void StageWorkspace::prepare<Stage::Basic>(VideoGeometry,
                                                    StageParameters);
template void StageWorkspace::prepare<Stage::Final>(VideoGeometry,
                                                    StageParameters);

ProcessStats process_basic_anchor_no_flow(
    ConstVideoView noisy, int anchor_frame, StageParameters parameters,
    const flow::SameLocationProvider& flow_provider,
    aggregate::ContributionStackView contributions, StageWorkspace& workspace) {
    return process_anchor_impl<Stage::Basic>(
        noisy, ConstVideoView{}, anchor_frame, parameters, flow_provider,
        contributions, workspace, [&](int similar) {
            if (parameters.couple_channels && noisy.geometry().channels > 1) {
                gather_basic_samples_coupled(noisy, workspace, similar);
                transform_vnlb_coupled_samples<Stage::Basic>(workspace,
                                                             similar);
            } else {
                gather_basic_group(noisy, workspace, similar);
                transform_vnlb_group<Stage::Basic>(workspace, similar);
            }
        });
}

ProcessStats process_final_anchor_no_flow(
    ConstVideoView noisy, ConstVideoView basic, int anchor_frame,
    StageParameters parameters, const flow::SameLocationProvider& flow_provider,
    aggregate::ContributionStackView contributions, StageWorkspace& workspace) {
    return process_anchor_impl<Stage::Final>(
        noisy, basic, anchor_frame, parameters, flow_provider, contributions,
        workspace, [&](int similar) {
            if (parameters.couple_channels && noisy.geometry().channels > 1) {
                gather_final_samples_coupled(noisy, basic, workspace, similar);
                transform_vnlb_coupled_samples<Stage::Final>(workspace,
                                                             similar);
            } else {
                gather_final_group(noisy, basic, workspace, similar);
                transform_vnlb_group<Stage::Final>(workspace, similar);
            }
        });
}

ProcessStats process_basic_anchor_mvtools(
    ConstVideoView noisy, int anchor_frame, StageParameters parameters,
    const flow::MVToolsFlowProvider& flow_provider,
    aggregate::ContributionStackView contributions, StageWorkspace& workspace) {
    return process_anchor_impl<Stage::Basic>(
        noisy, ConstVideoView{}, anchor_frame, parameters, flow_provider,
        contributions, workspace, [&](int similar) {
            if (parameters.couple_channels && noisy.geometry().channels > 1) {
                gather_basic_samples_coupled(noisy, workspace, similar);
                transform_vnlb_coupled_samples<Stage::Basic>(workspace,
                                                             similar);
            } else {
                gather_basic_group(noisy, workspace, similar);
                transform_vnlb_group<Stage::Basic>(workspace, similar);
            }
        });
}

ProcessStats process_final_anchor_mvtools(
    ConstVideoView noisy, ConstVideoView basic, int anchor_frame,
    StageParameters parameters, const flow::MVToolsFlowProvider& flow_provider,
    aggregate::ContributionStackView contributions, StageWorkspace& workspace) {
    return process_anchor_impl<Stage::Final>(
        noisy, basic, anchor_frame, parameters, flow_provider, contributions,
        workspace, [&](int similar) {
            if (parameters.couple_channels && noisy.geometry().channels > 1) {
                gather_final_samples_coupled(noisy, basic, workspace, similar);
                transform_vnlb_coupled_samples<Stage::Final>(workspace,
                                                             similar);
            } else {
                gather_final_group(noisy, basic, workspace, similar);
                transform_vnlb_group<Stage::Final>(workspace, similar);
            }
        });
}

} // namespace vnlb::core
