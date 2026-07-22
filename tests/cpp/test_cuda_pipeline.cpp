#include "cuda/pipeline.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr int skipped = 77;
constexpr int width = 4;
constexpr int height = 4;
constexpr int patch_size = 2;
constexpr int max_chunk_groups = 2;

void check_cuda(cudaError_t status, std::string_view operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

void require_close(float actual, float expected, float tolerance,
                   std::string_view label) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(std::string(label) +
                                 " mismatch: actual=" + std::to_string(actual) +
                                 " expected=" + std::to_string(expected));
    }
}

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

    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

  private:
    cudaStream_t stream_ = nullptr;
};

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
    [[nodiscard]] std::size_t size() const noexcept { return count_; }

    void upload(std::span<const T> source, cudaStream_t stream) {
        require(source.size() <= count_, "device upload exceeds allocation");
        check_cuda(cudaMemcpyAsync(data_, source.data(), source.size_bytes(),
                                   cudaMemcpyHostToDevice, stream),
                   "cudaMemcpyAsync host to device");
    }

    void download(std::span<T> destination, cudaStream_t stream) const {
        require(destination.size() <= count_,
                "device download exceeds allocation");
        check_cuda(cudaMemcpyAsync(destination.data(), data_,
                                   destination.size_bytes(),
                                   cudaMemcpyDeviceToHost, stream),
                   "cudaMemcpyAsync device to host");
    }

  private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

struct ExpectedResult {
    std::vector<float> packed;
    std::vector<float> normalized;
};

std::size_t frame_index(int channels, int channel, int y, int x) {
    (void)channels;
    return ((static_cast<std::size_t>(channel) * height + y) * width) + x;
}

ExpectedResult make_reference(int channels, std::span<const float> fallback,
                              std::span<const float> filtered_source,
                              std::span<const vnlbcu::PatchOrigin> anchors) {
    const std::size_t plane_values = static_cast<std::size_t>(width) * height;
    const std::size_t channel_packed_values = 2 * plane_values;
    ExpectedResult result{
        .packed = std::vector<float>(
            static_cast<std::size_t>(channels) * channel_packed_values, 0.0F),
        .normalized = std::vector<float>(
            static_cast<std::size_t>(channels) * plane_values, 0.0F),
    };
    std::vector<float> weights(plane_values, 0.0F);
    std::vector<float> numerators(
        static_cast<std::size_t>(channels) * plane_values, 0.0F);

    for (const vnlbcu::PatchOrigin anchor : anchors) {
        for (int py = 0; py < patch_size; ++py) {
            for (int px = 0; px < patch_size; ++px) {
                const int x = anchor.x + px;
                const int y = anchor.y + py;
                const std::size_t pixel =
                    static_cast<std::size_t>(y) * width + x;
                weights[pixel] += 1.0F;
                for (int channel = 0; channel < channels; ++channel) {
                    numerators[static_cast<std::size_t>(channel) *
                                   plane_values +
                               pixel] +=
                        filtered_source[frame_index(channels, channel, y, x)];
                }
            }
        }
    }

    for (int channel = 0; channel < channels; ++channel) {
        const std::size_t plane_base =
            static_cast<std::size_t>(channel) * plane_values;
        const std::size_t packed_base =
            static_cast<std::size_t>(channel) * channel_packed_values;
        for (std::size_t pixel = 0; pixel < plane_values; ++pixel) {
            const float numerator = numerators[plane_base + pixel];
            const float weight = weights[pixel];
            result.packed[packed_base + pixel] = numerator;
            result.packed[packed_base + plane_values + pixel] = weight;
            result.normalized[plane_base + pixel] =
                weight > 0.0F ? numerator / weight
                              : fallback[plane_base + pixel];
        }
    }
    return result;
}

vnlbcu::DeviceVideoView make_video_view(const float* data,
                                        const float* const* frame_data,
                                        int channels) {
    return vnlbcu::DeviceVideoView{
        .data = data,
        .frame_data = frame_data,
        .width = width,
        .height = height,
        .channels = channels,
        .frames = 1,
        .first_frame = 0,
        .source_frames = 1,
        .row_stride = width,
        .channel_stride = width * height,
        .frame_stride = frame_data == nullptr ? width * height * channels : 0,
    };
}

vnlbcu::StagePipelineShape make_shape(vnlbcu::Stage stage, int channels) {
    return vnlbcu::StagePipelineShape{
        .stage = stage,
        .max_groups = max_chunk_groups,
        .width = width,
        .height = height,
        .channels = channels,
        .source_frames = 1,
        .patch_size = patch_size,
        .patch_time = 1,
        .search_window = 1,
        .search_bwd = 0,
        .search_fwd = 0,
        .requested_similar = 1,
        .retained_stride = 1,
        .basis_similar = 1,
        .rank = 0,
        .contribution_slots = 1,
        .model_cap_factor = 1.0F,
        .couple_channels = true,
    };
}

vnlbcu::StagePipelineParameters make_parameters(vnlbcu::Stage stage,
                                                bool flat_areas) {
    return vnlbcu::StagePipelineParameters{
        .match = vnlbcu::MatchParameters{.tau = 0.0F},
        .filter =
            vnlbcu::FilterParameters{
                .stage = stage,
                .sigma = 0.1F,
                .sigma_basic = stage == vnlbcu::Stage::Final ? 0.05F : 0.0F,
                .beta = 1.0F,
                .variance_threshold = 1.1F,
                .weight_alpha = 0.0F,
                .weight_beta = 0.0F,
                .weight_epsilon = 1.0e-6F,
                .membership_noise_floor = 0.25F,
                .detect_equal_groups = false,
            },
        .flat_areas = flat_areas,
        .flat_gamma = 0.95F,
    };
}

void run_pipeline_case(vnlbcu::Stage stage, int channels, bool flat_areas,
                       bool pointer_table = false) {
    const std::size_t video_values =
        static_cast<std::size_t>(channels) * width * height;
    std::vector<float> noisy(video_values);
    std::vector<float> basic(video_values);
    for (int channel = 0; channel < channels; ++channel) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const std::size_t index = frame_index(channels, channel, y, x);
                if (flat_areas) {
                    noisy[index] = 0.75F - 0.25F * channel;
                    basic[index] = 0.25F - 0.125F * channel;
                } else {
                    noisy[index] =
                        0.01F * static_cast<float>(1 + x + width * y +
                                                   channel * width * height);
                    basic[index] = 0.8F * noisy[index];
                }
            }
        }
    }

    const std::vector<vnlbcu::PatchOrigin> anchors{
        vnlbcu::PatchOrigin{.x = 0, .y = 0, .frame = 0},
        vnlbcu::PatchOrigin{.x = 1, .y = 0, .frame = 0},
        vnlbcu::PatchOrigin{.x = 0, .y = 1, .frame = 0},
    };
    const std::span<const float> filtered_reference =
        stage == vnlbcu::Stage::Final && flat_areas
            ? std::span<const float>(basic)
            : std::span<const float>(noisy);
    const ExpectedResult expected = make_reference(
        channels, std::span<const float>(noisy), filtered_reference,
        std::span<const vnlbcu::PatchOrigin>(anchors));

    Stream stream;
    DeviceArray<float> device_noisy(video_values);
    DeviceArray<float> device_basic(video_values);
    DeviceArray<const float*> device_noisy_frame_data(pointer_table ? 1U : 0U);
    DeviceArray<const float*> device_basic_frame_data(pointer_table ? 1U : 0U);
    DeviceArray<vnlbcu::PatchOrigin> device_anchors(max_chunk_groups);
    DeviceArray<int> device_solver_info(anchors.size());
    device_noisy.upload(std::span<const float>(noisy), stream.get());
    device_basic.upload(std::span<const float>(basic), stream.get());
    if (pointer_table) {
        const std::vector<const float*> noisy_frame_data{device_noisy.data()};
        const std::vector<const float*> basic_frame_data{device_basic.data()};
        device_noisy_frame_data.upload(
            std::span<const float* const>(noisy_frame_data), stream.get());
        device_basic_frame_data.upload(
            std::span<const float* const>(basic_frame_data), stream.get());
    }

    const vnlbcu::StagePipelineShape shape = make_shape(stage, channels);
    const vnlbcu::StagePipelineParameters parameters =
        make_parameters(stage, flat_areas);
    vnlbcu::StagePipeline pipeline;
    // gamma=0 gives a rectangular window, making the hand-written reference
    // exact while still exercising the device window upload and scatter path.
    pipeline.reserve(shape, 0.0F);
    require(pipeline.numerator_values() ==
                static_cast<std::size_t>(channels) * width * height,
            "unexpected pipeline numerator size");
    require(pipeline.weight_values() ==
                static_cast<std::size_t>(width) * height,
            "unexpected pipeline weight size");

    DeviceArray<float> device_numerators(pipeline.numerator_values());
    DeviceArray<float> device_weights(pipeline.weight_values());
    DeviceArray<float> device_packed(pipeline.packed_values());
    DeviceArray<float> device_normalized(video_values);
    pipeline.clear_contributions(device_numerators.data(),
                                 device_weights.data(), stream.get());

    const vnlbcu::DeviceVideoView noisy_view =
        make_video_view(pointer_table ? nullptr : device_noisy.data(),
                        device_noisy_frame_data.data(), channels);
    const vnlbcu::DeviceVideoView basic_view =
        make_video_view(pointer_table ? nullptr : device_basic.data(),
                        device_basic_frame_data.data(), channels);

    device_anchors.upload(
        std::span<const vnlbcu::PatchOrigin>(anchors.data(), max_chunk_groups),
        stream.get());
    pipeline.enqueue(shape, parameters,
                     vnlbcu::DeviceStageBatch{
                         .noisy = noisy_view,
                         .basic = stage == vnlbcu::Stage::Final
                                      ? basic_view
                                      : vnlbcu::DeviceVideoView{},
                         .anchors = device_anchors.data(),
                         .search_centers = nullptr,
                         .groups = max_chunk_groups,
                         .anchor_frame = 0,
                         .contribution_numerators = device_numerators.data(),
                         .contribution_weights = device_weights.data(),
                         .solver_info = device_solver_info.data(),
                     },
                     stream.get());

    // Reuse every chunk-local device buffer.  Stream ordering guarantees that
    // the first scatter has consumed it before this one-element tail upload.
    device_anchors.upload(std::span<const vnlbcu::PatchOrigin>(
                              anchors.data() + max_chunk_groups, 1),
                          stream.get());
    pipeline.enqueue(
        shape, parameters,
        vnlbcu::DeviceStageBatch{
            .noisy = noisy_view,
            .basic = stage == vnlbcu::Stage::Final ? basic_view
                                                   : vnlbcu::DeviceVideoView{},
            .anchors = device_anchors.data(),
            .search_centers = nullptr,
            .groups = 1,
            .anchor_frame = 0,
            .contribution_numerators = device_numerators.data(),
            .contribution_weights = device_weights.data(),
            .solver_info = device_solver_info.data() + max_chunk_groups,
        },
        stream.get());

    pipeline.pack_contributions(device_numerators.data(), device_weights.data(),
                                device_packed.data(), stream.get());
    pipeline.normalize_contributions(device_numerators.data(),
                                     device_weights.data(), 0, noisy_view, 0,
                                     vnlbcu::DeviceMutableFrameView{
                                         .data = device_normalized.data(),
                                         .width = width,
                                         .height = height,
                                         .channels = channels,
                                         .row_stride = width,
                                         .channel_stride = width * height,
                                     },
                                     stream.get());

    std::vector<float> actual_packed(pipeline.packed_values());
    std::vector<float> actual_normalized(video_values);
    std::vector<int> solver_info(anchors.size(), -1);
    device_packed.download(std::span<float>(actual_packed), stream.get());
    device_normalized.download(std::span<float>(actual_normalized),
                               stream.get());
    device_solver_info.download(std::span<int>(solver_info), stream.get());
    check_cuda(cudaStreamSynchronize(stream.get()), "cudaStreamSynchronize");

    for (const int info : solver_info) {
        require(info == 0, "pipeline cuSOLVER status is non-zero");
    }
    require(actual_packed.size() == expected.packed.size(),
            "packed result size mismatch");
    for (std::size_t index = 0; index < actual_packed.size(); ++index) {
        require_close(actual_packed[index], expected.packed[index], 2.0e-5F,
                      "packed contribution");
    }
    for (std::size_t index = 0; index < actual_normalized.size(); ++index) {
        require_close(actual_normalized[index], expected.normalized[index],
                      2.0e-5F, "normalized contribution");
    }
}

void test_explicit_fast_path_constraints() {
    vnlbcu::StagePipeline pipeline;
    vnlbcu::StagePipelineShape shape = make_shape(vnlbcu::Stage::Basic, 1);
    shape.couple_channels = false;
    bool rejected = false;
    try {
        pipeline.reserve(shape, 0.0F);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "uncoupled pipeline shape was not rejected");

    shape = make_shape(vnlbcu::Stage::Basic, 1);
    shape.model_cap_factor = 2.0F;
    rejected = false;
    try {
        pipeline.reserve(shape, 0.0F);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected,
            "non-unit model_cap_factor pipeline shape was not rejected");

    shape = make_shape(vnlbcu::Stage::Basic, 1);
    shape.width = 50;
    shape.height = 50;
    shape.search_window = 47; // 2209 candidates, above the fused limit.
    rejected = false;
    try {
        pipeline.reserve(shape, 0.0F);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "oversized matcher candidate set was not rejected");
}

void test_device_anchor_frame_update() {
    vnlbcu::StagePipeline pipeline;
    const vnlbcu::StagePipelineShape shape =
        make_shape(vnlbcu::Stage::Basic, 1);
    pipeline.reserve(shape, 0.0F);

    std::vector<vnlbcu::PatchOrigin> anchors{
        {.x = 1, .y = 2, .frame = -1},
        {.x = 3, .y = 4, .frame = -1},
        {.x = 5, .y = 6, .frame = -1},
    };
    DeviceArray<vnlbcu::PatchOrigin> device_anchors(anchors.size());
    Stream stream;
    device_anchors.upload(std::span<const vnlbcu::PatchOrigin>(anchors),
                          stream.get());
    pipeline.set_anchor_frame(device_anchors.data(),
                              static_cast<int>(anchors.size()), 0,
                              stream.get());
    device_anchors.download(std::span<vnlbcu::PatchOrigin>(anchors),
                            stream.get());
    check_cuda(cudaStreamSynchronize(stream.get()), "cudaStreamSynchronize");
    require(anchors[0].x == 1 && anchors[0].y == 2 && anchors[0].frame == 0 &&
                anchors[1].x == 3 && anchors[1].y == 4 &&
                anchors[1].frame == 0 && anchors[2].x == 5 &&
                anchors[2].y == 6 && anchors[2].frame == 0,
            "device anchor-frame update changed spatial coordinates");
}

} // namespace

int main() {
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        device_count == 0) {
        std::cout << "CUDA device unavailable; skipping vnlbcu pipeline test\n";
        return skipped;
    }
    check_cuda(status, "cudaGetDeviceCount");

    try {
        run_pipeline_case(vnlbcu::Stage::Basic, 1, false);
        // Two coupled channels make flat detection average channel variances,
        // and pack must duplicate the shared weight plane for both channels.
        run_pipeline_case(vnlbcu::Stage::Final, 2, true, true);
        test_explicit_fast_path_constraints();
        test_device_anchor_frame_update();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
