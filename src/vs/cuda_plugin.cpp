#include "cuda/aggregation.hpp"
#include "cuda/frame_cache.hpp"
#include "cuda/pipeline.hpp"
#include "vnlb_version.hpp"

#include <VSHelper4.h>
#include <VapourSynth4.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr const char* kPluginId = "com.yuygfgg.vnlbcu";
constexpr const char* kPluginNamespace = "vnlbcu";
constexpr float kEightBitSampleScale = 255.0F;
constexpr float kEightBitDistanceScale =
    kEightBitSampleScale * kEightBitSampleScale;
constexpr int kDefaultGrayBasicGroupSize = 16;
constexpr int kDefaultColorBasicGroupSize = 16;
constexpr int kDefaultFinalGroupSize = 16;
constexpr int kDefaultRank = 15;
constexpr int kDefaultChunkGroups = 768;
constexpr int kMaximumStreams = 32;
constexpr int kMaximumModelSamples = 128;
constexpr std::size_t kDeviceMemoryMargin = std::size_t{256} << 20U;
constexpr std::size_t kPinnedUploadBudgetPerWorker = std::size_t{64} << 20U;

std::atomic<std::uint64_t> g_source_id{1};

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

[[nodiscard]] int checked_int(long long value, const char* label) {
    if (value < 0 || value > std::numeric_limits<int>::max()) {
        throw std::length_error(label);
    }
    return static_cast<int>(value);
}

[[nodiscard]] std::size_t checked_product(std::size_t left, std::size_t right,
                                          const char* label) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::length_error(label);
    }
    return left * right;
}

[[nodiscard]] std::size_t checked_sum(std::size_t left, std::size_t right,
                                      const char* label) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        throw std::length_error(label);
    }
    return left + right;
}

[[nodiscard]] std::uint64_t next_source_id() {
    const std::uint64_t id =
        g_source_id.fetch_add(1, std::memory_order_relaxed);
    if (id == 0) {
        throw std::overflow_error("CUDA frame-cache source id overflowed");
    }
    return id;
}

struct FrameRange {
    int first = 0;
    int last = -1;

    [[nodiscard]] bool empty() const noexcept { return last < first; }
    [[nodiscard]] int count() const {
        return empty() ? 0
                       : checked_int(static_cast<long long>(last) - first + 1,
                                     "frame range length overflows int");
    }
};

class FrameGuard final {
  public:
    FrameGuard() noexcept = default;
    FrameGuard(const VSFrame* frame, const VSAPI* vsapi) noexcept
        : frame_(frame), vsapi_(vsapi) {}
    ~FrameGuard() { reset(); }

    FrameGuard(const FrameGuard&) = delete;
    FrameGuard& operator=(const FrameGuard&) = delete;

    FrameGuard(FrameGuard&& other) noexcept
        : frame_(std::exchange(other.frame_, nullptr)), vsapi_(other.vsapi_) {}

    FrameGuard& operator=(FrameGuard&& other) noexcept {
        if (this != &other) {
            reset();
            frame_ = std::exchange(other.frame_, nullptr);
            vsapi_ = other.vsapi_;
        }
        return *this;
    }

    void reset() noexcept {
        if (frame_ != nullptr) {
            vsapi_->freeFrame(frame_);
            frame_ = nullptr;
        }
    }

    [[nodiscard]] const VSFrame* get() const noexcept { return frame_; }

    [[nodiscard]] const VSFrame* release() noexcept {
        return std::exchange(frame_, nullptr);
    }

  private:
    const VSFrame* frame_ = nullptr;
    const VSAPI* vsapi_ = nullptr;
};

class FrameListGuard final {
  public:
    FrameListGuard(std::vector<const VSFrame*>& frames,
                   const VSAPI* vsapi) noexcept
        : frames_(frames), vsapi_(vsapi) {}
    ~FrameListGuard() {
        for (const VSFrame* frame : frames_) {
            if (frame != nullptr) {
                vsapi_->freeFrame(frame);
            }
        }
        frames_.clear();
    }

    FrameListGuard(const FrameListGuard&) = delete;
    FrameListGuard& operator=(const FrameListGuard&) = delete;

  private:
    std::vector<const VSFrame*>& frames_;
    const VSAPI* vsapi_ = nullptr;
};

template <typename T> class DeviceBuffer final {
  public:
    DeviceBuffer() noexcept = default;
    explicit DeviceBuffer(std::size_t count) { allocate(count); }
    ~DeviceBuffer() {
        if (data_ != nullptr) {
            (void)cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void allocate(std::size_t count) {
        if (count == 0) {
            return;
        }
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_),
                              checked_product(count, sizeof(T),
                                              "CUDA buffer size overflows")),
                   "cudaMalloc(plugin workspace)");
        count_ = count;
    }

    [[nodiscard]] T* data() noexcept { return data_; }
    [[nodiscard]] const T* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return count_; }

  private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

template <typename T> class PinnedBuffer final {
  public:
    PinnedBuffer() noexcept = default;
    explicit PinnedBuffer(std::size_t count) { allocate(count); }
    ~PinnedBuffer() {
        if (data_ != nullptr) {
            (void)cudaFreeHost(data_);
        }
    }

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    void allocate(std::size_t count) {
        if (count == 0) {
            return;
        }
        check_cuda(
            cudaHostAlloc(reinterpret_cast<void**>(&data_),
                          checked_product(count, sizeof(T),
                                          "CUDA pinned buffer size overflows"),
                          cudaHostAllocPortable),
            "cudaHostAlloc(plugin staging)");
        count_ = count;
    }

    [[nodiscard]] T* data() noexcept { return data_; }
    [[nodiscard]] const T* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return count_; }

  private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

struct Parameters {
    vnlbcu::Stage stage = vnlbcu::Stage::Basic;
    float sigma = 1.0F;
    int patch_size = 10;
    int patch_time = 2;
    int search_window = 27;
    int search_bwd = 1;
    int search_fwd = 1;
    int similar = kDefaultGrayBasicGroupSize;
    int rank = kDefaultRank;
    float similar_cap_factor = 4.0F;
    float model_cap_factor = 1.0F;
    float beta = 1.0F;
    float tau = 0.0F;
    float variance_threshold = 3.7F;
    float sigma_basic = 0.0F;
    float flat_gamma = 0.95F;
    float weight_alpha = 0.75F;
    float weight_beta = 0.35F;
    float weight_gamma = 1.0F;
    float weight_epsilon = 1.0e-6F;
    float membership_noise_floor = 0.25F;
    int proc_step = 0;
    bool flat_areas = false;
    bool couple_channels = true;
    int device = 0;
    int num_streams = 1;
    int chunk_groups = kDefaultChunkGroups;
};

struct DerivedGeometry {
    int channels = 0;
    int plane_values = 0;
    int frame_values = 0;
    int valid_anchor_frames = 0;
    int temporal_count = 0;
    int cached_frames = 0;
    int candidates = 0;
    int retained_stride = 0;
    int basis_similar = 0;
    int contribution_slots = 0;
    int max_contributors = 0;
    int max_source_window = 0;
    int total_groups = 0;
    int max_chunk_groups = 0;
    std::size_t numerator_values = 0;
    std::size_t weight_values = 0;
    std::size_t contribution_bytes = 0;
    std::size_t frame_bytes = 0;
};

[[nodiscard]] bool map_has_key(const VSMap* in, const VSAPI* vsapi,
                               const char* key) noexcept {
    return vsapi->mapNumElements(in, key) > 0;
}

void require_single_value(const VSMap* in, const VSAPI* vsapi,
                          const char* key) {
    const int count = vsapi->mapNumElements(in, key);
    if (count != 1) {
        throw std::invalid_argument(std::string(key) +
                                    " must contain exactly one value");
    }
}

[[nodiscard]] int map_get_optional_int(const VSMap* in, const VSAPI* vsapi,
                                       const char* key, int fallback) {
    if (!map_has_key(in, vsapi, key)) {
        return fallback;
    }
    require_single_value(in, vsapi, key);
    int error = peSuccess;
    const std::int64_t value = vsapi->mapGetInt(in, key, 0, &error);
    if (error != peSuccess) {
        throw std::invalid_argument(std::string(key) + " must be an integer");
    }
    if (value < std::numeric_limits<int>::min() ||
        value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(std::string(key) +
                                    " is outside the supported int range");
    }
    return static_cast<int>(value);
}

[[nodiscard]] float map_get_optional_float(const VSMap* in, const VSAPI* vsapi,
                                           const char* key, float fallback) {
    if (!map_has_key(in, vsapi, key)) {
        return fallback;
    }
    require_single_value(in, vsapi, key);
    int error = peSuccess;
    const float value = vsapi->mapGetFloatSaturated(in, key, 0, &error);
    if (error != peSuccess) {
        throw std::invalid_argument(std::string(key) + " must be a float");
    }
    return value;
}

[[nodiscard]] float map_get_required_float(const VSMap* in, const VSAPI* vsapi,
                                           const char* key) {
    if (!map_has_key(in, vsapi, key)) {
        throw std::invalid_argument(std::string("missing required argument: ") +
                                    key);
    }
    return map_get_optional_float(in, vsapi, key, 0.0F);
}

[[nodiscard]] VSNode* map_get_required_node(const VSMap* in, const VSAPI* vsapi,
                                            const char* key) {
    if (!map_has_key(in, vsapi, key)) {
        throw std::invalid_argument(std::string(key) + " must be a video node");
    }
    require_single_value(in, vsapi, key);
    int error = peSuccess;
    VSNode* node = vsapi->mapGetNode(in, key, 0, &error);
    if (error != peSuccess || node == nullptr) {
        throw std::invalid_argument(std::string(key) + " must be a video node");
    }
    return node;
}

[[nodiscard]] bool is_supported_float_format(const VSVideoInfo* info) noexcept {
    if (info == nullptr || info->width <= 0 || info->height <= 0 ||
        info->numFrames <= 0 || !vsh::isConstantVideoFormat(info) ||
        info->format.sampleType != stFloat ||
        info->format.bitsPerSample != 32) {
        return false;
    }
    if (info->format.colorFamily == cfGray) {
        return info->format.numPlanes == 1;
    }
    return info->format.colorFamily == cfYUV && info->format.numPlanes == 3 &&
           info->format.subSamplingW == 0 && info->format.subSamplingH == 0;
}

[[nodiscard]] Parameters parse_parameters(const VSMap* in, const VSAPI* vsapi,
                                          vnlbcu::Stage stage,
                                          int default_streams,
                                          const VSVideoInfo& vi) {
    Parameters result{};
    result.stage = stage;
    result.num_streams = default_streams;
    const float sigma_eight_bit = map_get_required_float(in, vsapi, "sigma");
    result.sigma = sigma_eight_bit / kEightBitSampleScale;
    const bool gray = vi.format.numPlanes == 1;
    result.patch_size = gray ? 10 : 7;
    result.patch_time = vi.numFrames > 1 ? 2 : 1;
    result.search_window = 27;
    result.similar =
        stage == vnlbcu::Stage::Basic
            ? (gray ? kDefaultGrayBasicGroupSize : kDefaultColorBasicGroupSize)
            : kDefaultFinalGroupSize;
    result.rank = kDefaultRank;
    result.variance_threshold =
        gray ? (stage == vnlbcu::Stage::Basic
                    ? 3.7F
                    : std::max(0.0F, 1.87F - (0.028F * sigma_eight_bit)))
             : (stage == vnlbcu::Stage::Basic
                    ? 2.7F
                    : std::max(0.2F,
                               1.7F + ((1.2F - 1.7F) *
                                       (sigma_eight_bit - 10.0F) / 10.0F)));
    if (stage == vnlbcu::Stage::Final) {
        result.tau = 400.0F / kEightBitDistanceScale;
        result.flat_areas = true;
        result.weight_alpha = 1.0F;
        result.weight_beta = 0.5F;
        result.weight_gamma = 1.0F;
    }

    result.patch_size =
        map_get_optional_int(in, vsapi, "block_size", result.patch_size);
    result.patch_time =
        map_get_optional_int(in, vsapi, "patch_time", result.patch_time);
    const int default_temporal_radius =
        vi.numFrames > result.patch_time ? 1 : 0;
    result.search_bwd = default_temporal_radius;
    result.search_fwd = default_temporal_radius;

    if (map_has_key(in, vsapi, "radius")) {
        const int radius = map_get_optional_int(in, vsapi, "radius", 0);
        if (radius < 0) {
            throw std::invalid_argument("radius must be non-negative");
        }
        result.search_bwd = radius;
        result.search_fwd = radius;
    }
    result.search_bwd =
        map_get_optional_int(in, vsapi, "search_bwd", result.search_bwd);
    result.search_fwd =
        map_get_optional_int(in, vsapi, "search_fwd", result.search_fwd);

    if (map_has_key(in, vsapi, "bm_range")) {
        const int radius = map_get_optional_int(in, vsapi, "bm_range", 0);
        if (radius < 0 || radius > (std::numeric_limits<int>::max() - 1) / 2) {
            throw std::invalid_argument("bm_range must be non-negative");
        }
        result.search_window = (radius * 2) + 1;
    }

    result.similar =
        map_get_optional_int(in, vsapi, "group_size", result.similar);
    result.rank = map_get_optional_int(in, vsapi, "rank", result.rank);
    result.similar_cap_factor = map_get_optional_float(
        in, vsapi, "cap_factor", result.similar_cap_factor);
    result.model_cap_factor = map_get_optional_float(
        in, vsapi, "model_cap_factor", result.model_cap_factor);
    result.beta = map_get_optional_float(in, vsapi, "beta", result.beta);
    if (map_has_key(in, vsapi, "tau")) {
        result.tau = map_get_optional_float(in, vsapi, "tau", 0.0F) /
                     kEightBitDistanceScale;
    }
    result.variance_threshold = map_get_optional_float(
        in, vsapi, "variance_threshold", result.variance_threshold);
    result.proc_step =
        map_get_optional_int(in, vsapi, "block_step", result.proc_step);

    if (stage == vnlbcu::Stage::Final) {
        if (map_has_key(in, vsapi, "sigma_basic")) {
            result.sigma_basic =
                map_get_optional_float(in, vsapi, "sigma_basic", 0.0F) /
                kEightBitSampleScale;
        }
        result.flat_gamma =
            map_get_optional_float(in, vsapi, "gamma", result.flat_gamma);
        result.flat_areas =
            map_get_optional_int(in, vsapi, "flat_areas",
                                 result.flat_areas ? 1 : 0) != 0;
    }

    result.weight_alpha =
        map_get_optional_float(in, vsapi, "weight_alpha", result.weight_alpha);
    result.weight_beta =
        map_get_optional_float(in, vsapi, "weight_beta", result.weight_beta);
    result.weight_gamma =
        map_get_optional_float(in, vsapi, "weight_gamma", result.weight_gamma);
    result.weight_epsilon = map_get_optional_float(in, vsapi, "weight_epsilon",
                                                   result.weight_epsilon);
    result.membership_noise_floor = map_get_optional_float(
        in, vsapi, "membership_noise_floor", result.membership_noise_floor);
    result.couple_channels = map_get_optional_int(in, vsapi, "chroma", 1) != 0;
    result.device = map_get_optional_int(in, vsapi, "device_id", result.device);
    result.num_streams =
        map_get_optional_int(in, vsapi, "num_streams", result.num_streams);
    result.chunk_groups =
        map_get_optional_int(in, vsapi, "chunk_size", result.chunk_groups);

    // cuSOLVER's batched Jacobi solver has a severe matrix-size cliff above
    // 32 model samples.  The practical defaults stay on its fast path and use
    // the largest non-zero PCA rank possible for 16 centered samples.  Omitted
    // values still degrade for smaller explicit geometries or candidate sets;
    // explicit group_size/rank values are never silently changed.
    const bool default_group_size = !map_has_key(in, vsapi, "group_size");
    const bool default_rank = !map_has_key(in, vsapi, "rank");
    if (default_group_size && result.patch_size > 0 && result.patch_time > 0) {
        constexpr long long int_limit = std::numeric_limits<int>::max();
        const long long patch = result.patch_size;
        const long long patch_time = result.patch_time;
        const long long channels = vi.format.numPlanes;
        if (patch <= int_limit / patch &&
            patch * patch <= int_limit / patch_time &&
            patch * patch * patch_time <= int_limit / channels) {
            const int sample_dim =
                static_cast<int>(patch * patch * patch_time * channels);
            if (sample_dim > 1) {
                result.similar = std::min(result.similar, sample_dim - 1);
            }
        }
    }
    int default_candidates = 0;
    if ((default_group_size || default_rank) && result.patch_size > 0 &&
        result.patch_size <= vi.width && result.patch_size <= vi.height &&
        result.patch_time > 0 && result.patch_time <= vi.numFrames &&
        result.search_window > 0 && result.search_bwd >= 0 &&
        result.search_fwd >= 0) {
        const long long temporal_requested =
            static_cast<long long>(result.search_bwd) + result.search_fwd + 1;
        const int temporal_origins = vi.numFrames - result.patch_time + 1;
        const int candidate_width =
            std::min(result.search_window, vi.width - result.patch_size + 1);
        const int candidate_height =
            std::min(result.search_window, vi.height - result.patch_size + 1);
        const long long temporal_count =
            std::min<long long>(temporal_requested, temporal_origins);
        const long long spatial_candidates =
            static_cast<long long>(candidate_width) * candidate_height;
        const long long int_limit = std::numeric_limits<int>::max();
        if (temporal_count > 0 && spatial_candidates > 0 &&
            spatial_candidates <= int_limit / temporal_count) {
            default_candidates =
                static_cast<int>(spatial_candidates * temporal_count);
        }
    }
    if (default_group_size && default_candidates > 0) {
        result.similar = std::min(result.similar, default_candidates);
    }
    if (default_rank && default_candidates > 0) {
        result.rank =
            std::min(result.rank, std::min(result.similar, default_candidates));
    }
    return result;
}

void require_finite_nonnegative(float value, const char* label) {
    if (!std::isfinite(value) || value < 0.0F) {
        throw std::invalid_argument(std::string(label) +
                                    " must be finite and non-negative");
    }
}

void validate_parameters(const Parameters& parameters, const VSVideoInfo& vi) {
    if (!std::isfinite(parameters.sigma) || parameters.sigma <= 0.0F) {
        throw std::invalid_argument("sigma must be finite and positive");
    }
    if (parameters.patch_size <= 0 || parameters.patch_size > vi.width ||
        parameters.patch_size > vi.height) {
        throw std::invalid_argument("block_size must fit inside the frame");
    }
    if (parameters.patch_time <= 0 || parameters.patch_time > vi.numFrames) {
        throw std::invalid_argument("patch_time must fit inside the clip");
    }
    if (parameters.search_window <= 0 || (parameters.search_window & 1) == 0) {
        throw std::invalid_argument(
            "bm_range must describe a positive odd search window");
    }
    if (parameters.search_bwd < 0 || parameters.search_fwd < 0) {
        throw std::invalid_argument(
            "search_bwd and search_fwd must be non-negative");
    }
    (void)checked_int(static_cast<long long>(parameters.search_bwd) +
                          parameters.search_fwd + 1,
                      "temporal search span overflows int");
    if (parameters.similar <= 0) {
        throw std::invalid_argument("group_size must be positive");
    }
    if (parameters.rank < 0) {
        throw std::invalid_argument("rank must be non-negative");
    }
    if (!std::isfinite(parameters.similar_cap_factor) ||
        (parameters.similar_cap_factor != 0.0F &&
         parameters.similar_cap_factor < 1.0F)) {
        throw std::invalid_argument("cap_factor must be zero or at least 1.0");
    }
    if (!std::isfinite(parameters.model_cap_factor) ||
        parameters.model_cap_factor != 1.0F) {
        throw std::invalid_argument(
            "the CUDA backend currently requires model_cap_factor=1");
    }
    if (!std::isfinite(parameters.beta) || parameters.beta <= 0.0F) {
        throw std::invalid_argument("beta must be finite and positive");
    }
    require_finite_nonnegative(parameters.tau, "tau");
    if (!std::isfinite(parameters.variance_threshold)) {
        throw std::invalid_argument("variance_threshold must be finite");
    }
    require_finite_nonnegative(parameters.sigma_basic, "sigma_basic");
    if (!std::isfinite(parameters.flat_gamma) ||
        parameters.flat_gamma <= 0.0F) {
        throw std::invalid_argument("gamma must be finite and positive");
    }
    require_finite_nonnegative(parameters.weight_alpha, "weight_alpha");
    require_finite_nonnegative(parameters.weight_beta, "weight_beta");
    require_finite_nonnegative(parameters.weight_gamma, "weight_gamma");
    if (!std::isfinite(parameters.weight_epsilon) ||
        parameters.weight_epsilon <= 0.0F) {
        throw std::invalid_argument(
            "weight_epsilon must be finite and positive");
    }
    if (!std::isfinite(parameters.membership_noise_floor) ||
        parameters.membership_noise_floor <= 0.0F) {
        throw std::invalid_argument(
            "membership_noise_floor must be finite and positive");
    }
    if (parameters.proc_step < 0) {
        throw std::invalid_argument(
            "block_step must be non-negative; use 0 for auto");
    }
    if (!parameters.couple_channels) {
        throw std::invalid_argument(
            "the CUDA backend currently requires chroma=1 (coupled channels)");
    }
    if (parameters.device < 0) {
        throw std::invalid_argument("device_id must be non-negative");
    }
    if (parameters.num_streams < 1 ||
        parameters.num_streams > kMaximumStreams) {
        throw std::invalid_argument("num_streams must be in the range 1..32");
    }
    if (parameters.chunk_groups <= 0) {
        throw std::invalid_argument("chunk_size must be positive");
    }
}

template <typename Function>
void for_each_axis_position(int maximum, int step, int phase,
                            Function&& function) {
    function(0);
    if (maximum == 0) {
        return;
    }

    const int residue = phase % step;
    int position = residue == 0 ? step : residue;
    while (position < maximum) {
        function(position);
        if (position > maximum - step) {
            break;
        }
        position += step;
    }
    function(maximum);
}

[[nodiscard]] std::size_t anchor_count_for_phase(int width, int height,
                                                 int patch_size, int step,
                                                 int phase_y) {
    const int max_x = width - patch_size;
    const int max_y = height - patch_size;
    std::size_t result = 0;
    for_each_axis_position(max_y, step, phase_y, [&](int y) {
        const int phase_x = y == max_y ? 0 : phase_y + (y / step);
        for_each_axis_position(max_x, step, phase_x, [&](int) { ++result; });
    });
    return result;
}

[[nodiscard]] int build_anchor_grid(vnlbcu::PatchOrigin* destination,
                                    std::size_t capacity, int width, int height,
                                    int patch_size, int step,
                                    int valid_anchor_frames, int anchor_frame) {
    if (anchor_frame < 0 || anchor_frame >= valid_anchor_frames) {
        throw std::out_of_range("CUDA anchor frame is out of range");
    }
    const int max_x = width - patch_size;
    const int max_y = height - patch_size;
    const int phase_y =
        anchor_frame == valid_anchor_frames - 1 ? 0 : anchor_frame % step;
    std::size_t count = 0;
    for_each_axis_position(max_y, step, phase_y, [&](int y) {
        const int phase_x = y == max_y ? 0 : phase_y + (y / step);
        for_each_axis_position(max_x, step, phase_x, [&](int x) {
            if (count >= capacity) {
                throw std::logic_error(
                    "CUDA staggered anchor grid exceeds its reservation");
            }
            destination[count++] = vnlbcu::PatchOrigin{
                .x = x,
                .y = y,
                .frame = anchor_frame,
            };
        });
    });
    return checked_int(static_cast<long long>(count),
                       "anchor grid contains too many groups");
}

[[nodiscard]] DerivedGeometry derive_geometry(const Parameters& parameters,
                                              const VSVideoInfo& vi) {
    DerivedGeometry result{};
    result.channels = vi.format.numPlanes;
    result.plane_values =
        checked_int(static_cast<long long>(vi.width) * vi.height,
                    "frame plane sample count overflows int");
    result.frame_values = checked_int(
        static_cast<long long>(result.plane_values) * result.channels,
        "frame sample count overflows int");
    result.frame_bytes =
        checked_product(static_cast<std::size_t>(result.frame_values),
                        sizeof(float), "frame byte count overflows size_t");
    result.valid_anchor_frames = checked_int(
        static_cast<long long>(vi.numFrames) - parameters.patch_time + 1,
        "valid anchor frame count overflows int");

    const int requested_temporal =
        checked_int(static_cast<long long>(parameters.search_bwd) +
                        parameters.search_fwd + 1,
                    "temporal candidate count overflows int");
    result.temporal_count =
        std::min(requested_temporal, result.valid_anchor_frames);
    result.cached_frames =
        checked_int(static_cast<long long>(result.temporal_count) +
                        parameters.patch_time - 1,
                    "CUDA source window size overflows int");

    const int x_origins = vi.width - parameters.patch_size + 1;
    const int y_origins = vi.height - parameters.patch_size + 1;
    const int candidate_width = std::min(parameters.search_window, x_origins);
    const int candidate_height = std::min(parameters.search_window, y_origins);
    result.candidates =
        checked_int(static_cast<long long>(candidate_width) * candidate_height *
                        result.temporal_count,
                    "CUDA candidate count overflows int");
    result.basis_similar = std::min(parameters.similar, result.candidates);
    if (result.basis_similar > kMaximumModelSamples) {
        throw std::invalid_argument(
            "the CUDA dual-PCA filter supports at most 128 model samples");
    }
    if (parameters.rank > result.basis_similar) {
        throw std::invalid_argument(
            "rank must not exceed min(group_size, candidate_count)");
    }
    const int sample_dim = checked_int(
        static_cast<long long>(parameters.patch_size) * parameters.patch_size *
            parameters.patch_time * result.channels,
        "CUDA sample dimension overflows int");
    if (result.basis_similar >= sample_dim) {
        throw std::invalid_argument(
            "the CUDA backend currently requires the dual-PCA path "
            "(min(group_size, candidate_count) must be smaller than the "
            "coupled patch dimension)");
    }

    if (parameters.similar_cap_factor == 0.0F) {
        result.retained_stride = result.candidates;
    } else {
        const double requested =
            std::ceil(static_cast<double>(parameters.similar) *
                      static_cast<double>(parameters.similar_cap_factor));
        result.retained_stride =
            requested >= static_cast<double>(result.candidates)
                ? result.candidates
                : static_cast<int>(requested);
    }
    result.retained_stride =
        std::max(result.retained_stride, result.basis_similar);

    result.contribution_slots =
        checked_int(static_cast<long long>(parameters.search_bwd) +
                        parameters.search_fwd + parameters.patch_time,
                    "contribution slot count overflows int");
    result.max_contributors =
        std::min(result.valid_anchor_frames, result.contribution_slots);
    const long long source_span =
        2LL * (static_cast<long long>(parameters.search_bwd) +
               parameters.search_fwd + parameters.patch_time - 1) +
        1;
    result.max_source_window =
        checked_int(std::min(static_cast<long long>(vi.numFrames), source_span),
                    "source cache window overflows int");

    const int step = parameters.proc_step == 0
                         ? std::max(1, parameters.patch_size / 2)
                         : parameters.proc_step;
    std::size_t max_group_count = anchor_count_for_phase(
        vi.width, vi.height, parameters.patch_size, step, 0);
    const int last_nonfinal_phase =
        std::min(step - 1, std::max(0, result.valid_anchor_frames - 2));
    for (int phase = 1; phase <= last_nonfinal_phase; ++phase) {
        max_group_count = std::max(max_group_count,
                                   anchor_count_for_phase(vi.width, vi.height,
                                                          parameters.patch_size,
                                                          step, phase));
    }
    result.total_groups = checked_int(static_cast<long long>(max_group_count),
                                      "anchor grid contains too many groups");
    result.max_chunk_groups =
        std::min(result.total_groups, parameters.chunk_groups);

    const std::size_t plane = static_cast<std::size_t>(result.plane_values);
    result.numerator_values = checked_product(
        checked_product(static_cast<std::size_t>(result.channels),
                        static_cast<std::size_t>(result.contribution_slots),
                        "contribution numerator plane count overflows"),
        plane, "contribution numerator size overflows");
    result.weight_values =
        checked_product(static_cast<std::size_t>(result.contribution_slots),
                        plane, "contribution weight size overflows");
    result.contribution_bytes = checked_product(
        checked_sum(result.numerator_values, result.weight_values,
                    "contribution size overflows"),
        sizeof(float), "contribution byte count overflows");
    return result;
}

[[nodiscard]] FrameRange contributing_anchors(const Parameters& parameters,
                                              const DerivedGeometry& geometry,
                                              int output_frame) {
    const long long first = static_cast<long long>(output_frame) -
                            parameters.search_fwd - (parameters.patch_time - 1);
    const long long last =
        static_cast<long long>(output_frame) + parameters.search_bwd;
    return FrameRange{
        std::max(0, checked_int(std::max(0LL, first),
                                "contribution anchor range overflows int")),
        std::min(geometry.valid_anchor_frames - 1,
                 checked_int(std::max(0LL, last),
                             "contribution anchor range overflows int")),
    };
}

[[nodiscard]] FrameRange anchor_source_range(const Parameters& parameters,
                                             const DerivedGeometry& geometry,
                                             int anchor_frame) {
    const int max_origin = geometry.valid_anchor_frames - 1;
    const long long search_start =
        static_cast<long long>(anchor_frame) - parameters.search_bwd;
    const long long search_end =
        static_cast<long long>(anchor_frame) + parameters.search_fwd;
    const long long shift =
        std::min(0LL, search_start) + std::max(0LL, search_end - max_origin);
    const long long low = std::max(0LL, search_start - shift);
    const long long high =
        std::min(static_cast<long long>(max_origin), search_end - shift);
    return FrameRange{
        checked_int(low, "source frame range overflows int"),
        checked_int(high + parameters.patch_time - 1,
                    "source frame range overflows int"),
    };
}

[[nodiscard]] FrameRange required_source_range(const Parameters& parameters,
                                               const DerivedGeometry& geometry,
                                               int output_frame) {
    const FrameRange anchors =
        contributing_anchors(parameters, geometry, output_frame);
    FrameRange result{output_frame, output_frame};
    for (int anchor = anchors.first; anchor <= anchors.last; ++anchor) {
        const FrameRange source =
            anchor_source_range(parameters, geometry, anchor);
        result.first = std::min(result.first, source.first);
        result.last = std::max(result.last, source.last);
    }
    return result;
}

[[nodiscard]] vnlbcu::StagePipelineShape
make_pipeline_shape(const Parameters& parameters,
                    const DerivedGeometry& geometry, const VSVideoInfo& vi) {
    return vnlbcu::StagePipelineShape{
        .stage = parameters.stage,
        .max_groups = geometry.max_chunk_groups,
        .width = vi.width,
        .height = vi.height,
        .channels = geometry.channels,
        .source_frames = vi.numFrames,
        .patch_size = parameters.patch_size,
        .patch_time = parameters.patch_time,
        .search_window = parameters.search_window,
        .search_bwd = parameters.search_bwd,
        .search_fwd = parameters.search_fwd,
        .requested_similar = parameters.similar,
        .retained_stride = geometry.retained_stride,
        .basis_similar = geometry.basis_similar,
        .rank = parameters.rank,
        .contribution_slots = geometry.contribution_slots,
        .model_cap_factor = parameters.model_cap_factor,
        .couple_channels = true,
    };
}

[[nodiscard]] vnlbcu::StagePipelineParameters
make_pipeline_parameters(const Parameters& parameters) {
    return vnlbcu::StagePipelineParameters{
        .match = vnlbcu::MatchParameters{.tau = parameters.tau},
        .filter =
            vnlbcu::FilterParameters{
                .stage = parameters.stage,
                .sigma = parameters.sigma,
                .sigma_basic = parameters.sigma_basic,
                .beta = parameters.beta,
                .variance_threshold = parameters.variance_threshold,
                .weight_alpha = parameters.weight_alpha,
                .weight_beta = parameters.weight_beta,
                .weight_epsilon = parameters.weight_epsilon,
                .membership_noise_floor = parameters.membership_noise_floor,
                .detect_equal_groups = true,
            },
        .flat_areas = parameters.flat_areas,
        .flat_gamma = parameters.flat_gamma,
        .paste_mask = true,
    };
}

[[nodiscard]] vnlbcu::AggregationShape
make_aggregation_shape(const Parameters& parameters,
                       const DerivedGeometry& geometry, const VSVideoInfo& vi) {
    return vnlbcu::AggregationShape{
        .max_groups = geometry.max_chunk_groups,
        .width = vi.width,
        .height = vi.height,
        .channels = geometry.channels,
        .slots = geometry.contribution_slots,
        .retained_stride = geometry.retained_stride,
        .patch_size = parameters.patch_size,
        .patch_time = parameters.patch_time,
        .search_window = parameters.search_window,
        .search_bwd = parameters.search_bwd,
    };
}

struct WorkerConfig {
    int device = 0;
    vnlbcu::Stage stage = vnlbcu::Stage::Basic;
    vnlbcu::StagePipelineShape pipeline_shape{};
    vnlbcu::AggregationShape aggregation_shape{};
    float window_gamma = 0.0F;
    int frame_values = 0;
    int cached_frames = 0;
    int upload_slots = 0;
    int total_groups = 0;
    int max_contributors = 0;
    int max_source_window = 0;
    int width = 0;
    int height = 0;
    int patch_size = 0;
    int proc_step = 0;
    int valid_anchor_frames = 0;
    bool use_pointer_tables = false;
};

class Worker final {
  public:
    explicit Worker(const WorkerConfig& config)
        : device_(config.device), stage_(config.stage),
          use_pointer_tables_(config.use_pointer_tables), width_(config.width),
          height_(config.height), patch_size_(config.patch_size),
          proc_step_(config.proc_step),
          valid_anchor_frames_(config.valid_anchor_frames),
          total_groups_(config.total_groups), pipeline_(config.device),
          normalizer_(config.device),
          noisy_contiguous_(
              config.use_pointer_tables
                  ? 0
                  : checked_product(
                        static_cast<std::size_t>(config.frame_values),
                        static_cast<std::size_t>(config.max_source_window),
                        "CUDA noisy contiguous window size overflows")),
          basic_contiguous_(
              config.use_pointer_tables || config.stage != vnlbcu::Stage::Final
                  ? 0
                  : checked_product(
                        static_cast<std::size_t>(config.frame_values),
                        static_cast<std::size_t>(config.max_source_window),
                        "CUDA basic contiguous window size overflows")),
          noisy_frame_tables_(
              config.use_pointer_tables
                  ? checked_product(
                        static_cast<std::size_t>(config.cached_frames),
                        static_cast<std::size_t>(config.max_contributors),
                        "CUDA noisy frame-table size overflows")
                  : 0),
          basic_frame_tables_(
              config.use_pointer_tables && config.stage == vnlbcu::Stage::Final
                  ? checked_product(
                        static_cast<std::size_t>(config.cached_frames),
                        static_cast<std::size_t>(config.max_contributors),
                        "CUDA basic frame-table size overflows")
                  : 0),
          output_(static_cast<std::size_t>(config.frame_values)),
          anchors_(static_cast<std::size_t>(config.total_groups)),
          solver_info_(
              checked_product(static_cast<std::size_t>(config.total_groups),
                              static_cast<std::size_t>(config.max_contributors),
                              "CUDA solver status size overflows")),
          contribution_sources_(
              static_cast<std::size_t>(config.max_contributors)),
          host_output_(static_cast<std::size_t>(config.frame_values)),
          host_upload_(
              checked_product(static_cast<std::size_t>(config.frame_values),
                              static_cast<std::size_t>(config.upload_slots),
                              "pinned upload staging size overflows")),
          host_noisy_frame_tables_(
              config.use_pointer_tables
                  ? checked_product(
                        static_cast<std::size_t>(config.cached_frames),
                        static_cast<std::size_t>(config.max_contributors),
                        "pinned noisy frame-table size overflows")
                  : 0),
          host_basic_frame_tables_(
              config.use_pointer_tables && config.stage == vnlbcu::Stage::Final
                  ? checked_product(
                        static_cast<std::size_t>(config.cached_frames),
                        static_cast<std::size_t>(config.max_contributors),
                        "pinned basic frame-table size overflows")
                  : 0),
          host_anchors_(
              checked_product(static_cast<std::size_t>(config.total_groups),
                              static_cast<std::size_t>(config.max_contributors),
                              "pinned anchor staging size overflows")),
          host_solver_info_(
              checked_product(static_cast<std::size_t>(config.total_groups),
                              static_cast<std::size_t>(config.max_contributors),
                              "pinned solver status size overflows")),
          host_contribution_sources_(
              static_cast<std::size_t>(config.max_contributors)) {
        check_cuda(cudaSetDevice(device_), "cudaSetDevice(plugin worker)");
        check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                   "cudaStreamCreateWithFlags(plugin worker)");
        try {
            upload_events_.reserve(
                static_cast<std::size_t>(config.upload_slots));
            upload_event_recorded_.reserve(
                static_cast<std::size_t>(config.upload_slots));
            for (int index = 0; index < config.upload_slots; ++index) {
                cudaEvent_t event = nullptr;
                check_cuda(
                    cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
                    "cudaEventCreateWithFlags(upload staging)");
                upload_events_.push_back(event);
                upload_event_recorded_.push_back(false);
            }
            pipeline_.reserve(config.pipeline_shape, config.window_gamma);
            normalizer_.reserve(config.aggregation_shape, config.window_gamma);
            contribution_keys.reserve(
                static_cast<std::size_t>(config.max_contributors));
            noisy_keys.reserve(
                static_cast<std::size_t>(config.max_source_window));
            basic_keys.reserve(
                static_cast<std::size_t>(config.max_source_window));
            clip_frames.reserve(
                static_cast<std::size_t>(config.max_source_window));
            ref_frames.reserve(
                static_cast<std::size_t>(config.max_source_window));
        } catch (...) {
            for (cudaEvent_t event : upload_events_) {
                (void)cudaEventDestroy(event);
            }
            upload_events_.clear();
            (void)cudaStreamDestroy(stream_);
            stream_ = nullptr;
            throw;
        }
    }

    ~Worker() {
        (void)cudaSetDevice(device_);
        if (stream_ != nullptr) {
            (void)cudaStreamSynchronize(stream_);
            for (cudaEvent_t event : upload_events_) {
                (void)cudaEventDestroy(event);
            }
            (void)cudaStreamDestroy(stream_);
        }
    }

    Worker(const Worker&) = delete;
    Worker& operator=(const Worker&) = delete;

    void synchronize_noexcept() noexcept {
        (void)cudaSetDevice(device_);
        if (stream_ != nullptr) {
            (void)cudaStreamSynchronize(stream_);
        }
    }

    int device_ = 0;
    vnlbcu::Stage stage_ = vnlbcu::Stage::Basic;
    bool use_pointer_tables_ = false;
    int width_ = 0;
    int height_ = 0;
    int patch_size_ = 0;
    int proc_step_ = 0;
    int valid_anchor_frames_ = 0;
    int total_groups_ = 0;
    cudaStream_t stream_ = nullptr;
    vnlbcu::StagePipeline pipeline_;
    vnlbcu::Aggregator normalizer_;
    DeviceBuffer<float> noisy_contiguous_;
    DeviceBuffer<float> basic_contiguous_;
    DeviceBuffer<const float*> noisy_frame_tables_;
    DeviceBuffer<const float*> basic_frame_tables_;
    DeviceBuffer<float> output_;
    DeviceBuffer<vnlbcu::PatchOrigin> anchors_;
    DeviceBuffer<int> solver_info_;
    DeviceBuffer<vnlbcu::DeviceContributionSource> contribution_sources_;
    PinnedBuffer<float> host_output_;
    PinnedBuffer<float> host_upload_;
    PinnedBuffer<const float*> host_noisy_frame_tables_;
    PinnedBuffer<const float*> host_basic_frame_tables_;
    PinnedBuffer<vnlbcu::PatchOrigin> host_anchors_;
    PinnedBuffer<int> host_solver_info_;
    PinnedBuffer<vnlbcu::DeviceContributionSource> host_contribution_sources_;
    std::vector<cudaEvent_t> upload_events_;
    std::vector<bool> upload_event_recorded_;
    std::size_t upload_cursor_ = 0;
    std::vector<vnlbcu::FrameCacheKey> contribution_keys;
    std::vector<vnlbcu::FrameCacheKey> noisy_keys;
    std::vector<vnlbcu::FrameCacheKey> basic_keys;
    std::vector<const VSFrame*> clip_frames;
    std::vector<const VSFrame*> ref_frames;
};

class WorkerPool final {
  public:
    class Lease final {
      public:
        Lease() noexcept = default;
        Lease(WorkerPool* pool, std::size_t index) noexcept
            : pool_(pool), index_(index) {}
        ~Lease() { release(); }

        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;

        Lease(Lease&& other) noexcept
            : pool_(std::exchange(other.pool_, nullptr)), index_(other.index_) {
        }

        Lease& operator=(Lease&& other) noexcept {
            if (this != &other) {
                release();
                pool_ = std::exchange(other.pool_, nullptr);
                index_ = other.index_;
            }
            return *this;
        }

        [[nodiscard]] Worker& get() const noexcept {
            return *pool_->workers_[index_];
        }

      private:
        void release() noexcept {
            if (pool_ != nullptr) {
                pool_->release(index_);
                pool_ = nullptr;
            }
        }

        WorkerPool* pool_ = nullptr;
        std::size_t index_ = 0;
    };

    WorkerPool(int count, const WorkerConfig& config) {
        workers_.reserve(static_cast<std::size_t>(count));
        available_.reserve(static_cast<std::size_t>(count));
        for (int index = 0; index < count; ++index) {
            workers_.push_back(std::make_unique<Worker>(config));
            available_.push_back(true);
        }
    }

    WorkerPool(const WorkerPool&) = delete;
    WorkerPool& operator=(const WorkerPool&) = delete;

    [[nodiscard]] Lease acquire() {
        std::unique_lock lock(mutex_);
        ready_.wait(lock, [&]() noexcept {
            return std::find(available_.begin(), available_.end(), true) !=
                   available_.end();
        });
        const auto found =
            std::find(available_.begin(), available_.end(), true);
        const std::size_t index =
            static_cast<std::size_t>(found - available_.begin());
        available_[index] = false;
        return Lease(this, index);
    }

  private:
    friend class Lease;

    void release(std::size_t index) noexcept {
        {
            std::lock_guard lock(mutex_);
            available_[index] = true;
        }
        ready_.notify_one();
    }

    std::mutex mutex_;
    std::condition_variable ready_;
    std::vector<std::unique_ptr<Worker>> workers_;
    std::vector<bool> available_;
};

struct CudaFilterData {
    VSNode* node = nullptr;
    VSNode* ref_node = nullptr;
    VSVideoInfo vi{};
    VSVideoInfo ref_vi{};
    VSVideoInfo out_vi{};
    Parameters parameters{};
    DerivedGeometry geometry{};
    vnlbcu::StagePipelineShape pipeline_shape{};
    vnlbcu::StagePipelineParameters pipeline_parameters{};
    vnlbcu::AggregationShape aggregation_shape{};
    std::uint64_t noisy_source_id = 0;
    std::uint64_t basic_source_id = 0;
    std::uint64_t contribution_source_id = 0;
    std::unique_ptr<vnlbcu::FrameCache> noisy_cache;
    std::unique_ptr<vnlbcu::FrameCache> basic_cache;
    std::unique_ptr<vnlbcu::FrameCache> contribution_cache;
    std::unique_ptr<WorkerPool> workers;
};

[[nodiscard]] const VSFrame* get_frame_filter_checked(int frame, VSNode* node,
                                                      VSFrameContext* frame_ctx,
                                                      const VSAPI* vsapi,
                                                      const char* label) {
    const VSFrame* result = vsapi->getFrameFilter(frame, node, frame_ctx);
    if (result == nullptr) {
        throw std::runtime_error(std::string("failed to fetch ") + label +
                                 " frame");
    }
    return result;
}

[[nodiscard]] VSFrame* new_video_frame_checked(const VSVideoFormat* format,
                                               int width, int height,
                                               const VSFrame* prop_source,
                                               VSCore* core,
                                               const VSAPI* vsapi) {
    VSFrame* result =
        vsapi->newVideoFrame(format, width, height, prop_source, core);
    if (result == nullptr) {
        throw std::bad_alloc();
    }
    return result;
}

void request_frames(const CudaFilterData* data, int output_frame,
                    VSFrameContext* frame_ctx, const VSAPI* vsapi) {
    const FrameRange range =
        required_source_range(data->parameters, data->geometry, output_frame);
    for (int frame = range.first; frame <= range.last; ++frame) {
        vsapi->requestFrameFilter(frame, data->node, frame_ctx);
        if (data->ref_node != nullptr) {
            vsapi->requestFrameFilter(frame, data->ref_node, frame_ctx);
        }
    }
}

void fetch_frames(Worker& worker, const CudaFilterData* data,
                  FrameRange requested, VSFrameContext* frame_ctx,
                  const VSAPI* vsapi) {
    worker.clip_frames.clear();
    worker.ref_frames.clear();
    const int count = requested.count();
    if (count > data->geometry.max_source_window) {
        throw std::logic_error(
            "requested source frame window exceeds its reservation");
    }
    for (int frame = requested.first; frame <= requested.last; ++frame) {
        worker.clip_frames.push_back(get_frame_filter_checked(
            frame, data->node, frame_ctx, vsapi, "clip"));
        if (data->ref_node != nullptr) {
            worker.ref_frames.push_back(get_frame_filter_checked(
                frame, data->ref_node, frame_ctx, vsapi, "ref"));
        }
    }
}

void build_contiguous_keys(std::vector<vnlbcu::FrameCacheKey>& keys,
                           std::uint64_t source_id, FrameRange range) {
    keys.clear();
    keys.reserve(static_cast<std::size_t>(range.count()));
    for (int frame = range.first; frame <= range.last; ++frame) {
        keys.push_back(vnlbcu::FrameCacheKey{
            .source_id = source_id,
            .frame = frame,
            .generation = 0,
        });
    }
}

void upload_source_misses(Worker& worker,
                          const vnlbcu::FrameCache::Window& window,
                          std::span<const VSFrame* const> frames,
                          const VSAPI* vsapi, int width, int height,
                          int channels) {
    const std::span<const vnlbcu::CachedDeviceFrame> cached = window.frames();
    if (cached.size() != frames.size()) {
        throw std::logic_error("source cache window/frame list mismatch");
    }
    const std::size_t plane_values = checked_product(
        static_cast<std::size_t>(width), static_cast<std::size_t>(height),
        "source plane size overflows");
    const std::size_t plane_bytes = checked_product(
        plane_values, sizeof(float), "source plane byte count overflows");
    const std::size_t expected_bytes =
        checked_product(plane_bytes, static_cast<std::size_t>(channels),
                        "source frame byte count overflows");

    for (std::size_t index = 0; index < cached.size(); ++index) {
        if (!cached[index].needs_upload) {
            continue;
        }
        if (cached[index].bytes != expected_bytes) {
            throw std::logic_error("source cache slot has an invalid size");
        }
        if (worker.upload_events_.empty()) {
            throw std::logic_error("pinned upload staging is not allocated");
        }
        const std::size_t staging_index =
            worker.upload_cursor_ % worker.upload_events_.size();
        if (worker.upload_event_recorded_[staging_index]) {
            check_cuda(
                cudaEventSynchronize(worker.upload_events_[staging_index]),
                "cudaEventSynchronize(upload staging reuse)");
        }
        float* staging =
            worker.host_upload_.data() +
            staging_index * plane_values * static_cast<std::size_t>(channels);
        for (int channel = 0; channel < channels; ++channel) {
            const ptrdiff_t source_stride =
                vsapi->getStride(frames[index], channel);
            if (source_stride <
                static_cast<ptrdiff_t>(static_cast<std::size_t>(width) *
                                       sizeof(float))) {
                throw std::runtime_error("unsupported input frame stride");
            }
            const std::uint8_t* source =
                vsapi->getReadPtr(frames[index], channel);
            float* staging_plane =
                staging + static_cast<std::size_t>(channel) * plane_values;
            const std::size_t row_bytes =
                static_cast<std::size_t>(width) * sizeof(float);
            for (int y = 0; y < height; ++y) {
                std::memcpy(staging_plane + static_cast<std::size_t>(y) *
                                                static_cast<std::size_t>(width),
                            source +
                                static_cast<std::size_t>(y) *
                                    static_cast<std::size_t>(source_stride),
                            row_bytes);
            }
        }
        check_cuda(cudaMemcpyAsync(cached[index].data, staging, expected_bytes,
                                   cudaMemcpyHostToDevice, worker.stream_),
                   "cudaMemcpyAsync(source frame upload)");
        check_cuda(cudaEventRecord(worker.upload_events_[staging_index],
                                   worker.stream_),
                   "cudaEventRecord(source frame upload)");
        worker.upload_event_recorded_[staging_index] = true;
        ++worker.upload_cursor_;
    }
}

[[nodiscard]] const vnlbcu::CachedDeviceFrame&
cached_frame_at(const vnlbcu::FrameCache::Window& window,
                FrameRange window_range, int frame) {
    if (frame < window_range.first || frame > window_range.last) {
        throw std::logic_error("cached frame lookup is outside the window");
    }
    const std::size_t index =
        static_cast<std::size_t>(frame - window_range.first);
    const std::span<const vnlbcu::CachedDeviceFrame> frames = window.frames();
    if (index >= frames.size() || frames[index].key.frame != frame) {
        throw std::logic_error("source cache returned frames out of order");
    }
    return frames[index];
}

void assemble_contiguous_window(Worker& worker,
                                const vnlbcu::FrameCache::Window& cache_window,
                                FrameRange cache_range,
                                FrameRange assembly_range,
                                const CudaFilterData* data, bool basic) {
    const int frame_count = assembly_range.count();
    if (frame_count <= 0 || frame_count > data->geometry.max_source_window) {
        throw std::logic_error("invalid CUDA source assembly window");
    }
    float* destination = basic ? worker.basic_contiguous_.data()
                               : worker.noisy_contiguous_.data();
    if (destination == nullptr) {
        throw std::logic_error(
            "CUDA contiguous source window is not allocated");
    }
    for (int frame = assembly_range.first; frame <= assembly_range.last;
         ++frame) {
        const vnlbcu::CachedDeviceFrame& cached =
            cached_frame_at(cache_window, cache_range, frame);
        const std::size_t local =
            static_cast<std::size_t>(frame - assembly_range.first);
        check_cuda(cudaMemcpyAsync(destination +
                                       local * static_cast<std::size_t>(
                                                   data->geometry.frame_values),
                                   cached.data, data->geometry.frame_bytes,
                                   cudaMemcpyDeviceToDevice, worker.stream_),
                   "cudaMemcpyAsync(assemble contiguous source window)");
    }
}

[[nodiscard]] vnlbcu::DeviceVideoView
contiguous_video_view(Worker& worker, FrameRange assembly_range,
                      FrameRange source_range, const CudaFilterData* data,
                      bool basic) {
    const int frame_count = source_range.count();
    if (frame_count != data->geometry.cached_frames ||
        source_range.first < assembly_range.first ||
        source_range.last > assembly_range.last) {
        throw std::logic_error(
            "anchor source window does not fit its contiguous cache view");
    }
    float* base = basic ? worker.basic_contiguous_.data()
                        : worker.noisy_contiguous_.data();
    const std::size_t frame_offset = checked_product(
        static_cast<std::size_t>(source_range.first - assembly_range.first),
        static_cast<std::size_t>(data->geometry.frame_values),
        "contiguous source frame offset overflows");
    return vnlbcu::DeviceVideoView{
        .data = base + frame_offset,
        .frame_data = nullptr,
        .width = data->vi.width,
        .height = data->vi.height,
        .channels = data->geometry.channels,
        .frames = frame_count,
        .first_frame = source_range.first,
        .source_frames = data->vi.numFrames,
        .row_stride = data->vi.width,
        .channel_stride = data->geometry.plane_values,
        .frame_stride = data->geometry.frame_values,
    };
}

void stage_frame_table(Worker& worker,
                       const vnlbcu::FrameCache::Window& cache_window,
                       FrameRange cache_range, FrameRange source_range,
                       const CudaFilterData* data, int missing_index,
                       bool basic) {
    const int frame_count = source_range.count();
    if (frame_count != data->geometry.cached_frames) {
        throw std::logic_error(
            "anchor source window has an unexpected frame count");
    }
    const std::size_t table_base =
        checked_product(static_cast<std::size_t>(missing_index),
                        static_cast<std::size_t>(data->geometry.cached_frames),
                        "source frame-table offset overflows");
    const float** host_table = basic ? worker.host_basic_frame_tables_.data()
                                     : worker.host_noisy_frame_tables_.data();
    for (int frame = source_range.first; frame <= source_range.last; ++frame) {
        const vnlbcu::CachedDeviceFrame& cached =
            cached_frame_at(cache_window, cache_range, frame);
        const std::size_t local =
            static_cast<std::size_t>(frame - source_range.first);
        host_table[table_base + local] = static_cast<const float*>(cached.data);
    }
}

void upload_frame_tables(Worker& worker, const CudaFilterData* data,
                         int missing_count, bool basic) {
    if (missing_count <= 0) {
        return;
    }
    const std::size_t pointer_count =
        checked_product(static_cast<std::size_t>(missing_count),
                        static_cast<std::size_t>(data->geometry.cached_frames),
                        "source frame-table size overflows");
    const float* const* host_table =
        basic ? worker.host_basic_frame_tables_.data()
              : worker.host_noisy_frame_tables_.data();
    const float** device_table = basic ? worker.basic_frame_tables_.data()
                                       : worker.noisy_frame_tables_.data();
    check_cuda(
        cudaMemcpyAsync(device_table, host_table,
                        checked_product(pointer_count, sizeof(const float*),
                                        "source frame-table copy overflows"),
                        cudaMemcpyHostToDevice, worker.stream_),
        "cudaMemcpyAsync(source frame table)");
}

[[nodiscard]] vnlbcu::DeviceVideoView
frame_table_video_view(Worker& worker, FrameRange source_range,
                       const CudaFilterData* data, int missing_index,
                       bool basic) {
    const int frame_count = source_range.count();
    if (frame_count != data->geometry.cached_frames) {
        throw std::logic_error(
            "anchor source window has an unexpected frame count");
    }
    const std::size_t table_base =
        checked_product(static_cast<std::size_t>(missing_index),
                        static_cast<std::size_t>(data->geometry.cached_frames),
                        "source frame-table offset overflows");
    const float* const* device_table = basic
                                           ? worker.basic_frame_tables_.data()
                                           : worker.noisy_frame_tables_.data();
    return vnlbcu::DeviceVideoView{
        .data = nullptr,
        .frame_data = device_table + table_base,
        .width = data->vi.width,
        .height = data->vi.height,
        .channels = data->geometry.channels,
        .frames = frame_count,
        .first_frame = source_range.first,
        .source_frames = data->vi.numFrames,
        .row_stride = data->vi.width,
        .channel_stride = data->geometry.plane_values,
        .frame_stride = 0,
    };
}

[[nodiscard]] vnlbcu::DeviceContributionView
contribution_view(const CudaFilterData* data,
                  const vnlbcu::CachedDeviceFrame& cached) {
    if (cached.bytes != data->geometry.contribution_bytes) {
        throw std::logic_error("contribution cache slot has an invalid size");
    }
    auto* numerators = static_cast<float*>(cached.data);
    float* weights = numerators + data->geometry.numerator_values;
    return vnlbcu::make_contiguous_contribution_view(data->aggregation_shape,
                                                     numerators, weights);
}

void compute_contribution(Worker& worker, const CudaFilterData* data,
                          int missing_index, int anchor_frame,
                          const vnlbcu::CachedDeviceFrame& destination,
                          FrameRange assembly_range) {
    const FrameRange source_range =
        anchor_source_range(data->parameters, data->geometry, anchor_frame);
    const vnlbcu::DeviceVideoView noisy =
        worker.use_pointer_tables_
            ? frame_table_video_view(worker, source_range, data, missing_index,
                                     false)
            : contiguous_video_view(worker, assembly_range, source_range, data,
                                    false);
    vnlbcu::DeviceVideoView basic{};
    if (data->parameters.stage == vnlbcu::Stage::Final) {
        basic = worker.use_pointer_tables_
                    ? frame_table_video_view(worker, source_range, data,
                                             missing_index, true)
                    : contiguous_video_view(worker, assembly_range,
                                            source_range, data, true);
    }

    const vnlbcu::DeviceContributionView contributions =
        contribution_view(data, destination);
    worker.pipeline_.clear_contributions(contributions.numerators,
                                         contributions.weights, worker.stream_);

    const std::size_t solver_base =
        checked_product(static_cast<std::size_t>(missing_index),
                        static_cast<std::size_t>(data->geometry.total_groups),
                        "solver status offset overflows");
    const std::size_t host_anchor_base =
        checked_product(static_cast<std::size_t>(missing_index),
                        static_cast<std::size_t>(worker.total_groups_),
                        "pinned anchor offset overflows");
    vnlbcu::PatchOrigin* const host_anchors =
        worker.host_anchors_.data() + host_anchor_base;
    const int groups = build_anchor_grid(
        host_anchors, static_cast<std::size_t>(worker.total_groups_),
        worker.width_, worker.height_, worker.patch_size_, worker.proc_step_,
        worker.valid_anchor_frames_, anchor_frame);
    check_cuda(
        cudaMemsetAsync(
            worker.solver_info_.data() + solver_base, 0,
            checked_product(static_cast<std::size_t>(worker.total_groups_),
                            sizeof(int), "solver status clear size overflows"),
            worker.stream_),
        "cudaMemsetAsync(cuSOLVER status)");
    check_cuda(
        cudaMemcpyAsync(worker.anchors_.data(), host_anchors,
                        checked_product(static_cast<std::size_t>(groups),
                                        sizeof(vnlbcu::PatchOrigin),
                                        "anchor grid byte count overflows"),
                        cudaMemcpyHostToDevice, worker.stream_),
        "cudaMemcpyAsync(staggered anchor grid)");

    for (int first_group = 0; first_group < groups;
         first_group += data->geometry.max_chunk_groups) {
        const int chunk_groups =
            std::min(data->geometry.max_chunk_groups, groups - first_group);
        worker.pipeline_.enqueue(
            data->pipeline_shape, data->pipeline_parameters,
            vnlbcu::DeviceStageBatch{
                .noisy = noisy,
                .basic = basic,
                .anchors = worker.anchors_.data() + first_group,
                .search_centers = nullptr,
                .groups = chunk_groups,
                .anchor_frame = anchor_frame,
                .contribution_numerators = contributions.numerators,
                .contribution_weights = contributions.weights,
                .solver_info = worker.solver_info_.data() + solver_base +
                               static_cast<std::size_t>(first_group),
            },
            worker.stream_);
    }
}

void copy_output_to_frame(const Worker& worker, VSFrame* output,
                          const CudaFilterData* data, const VSAPI* vsapi) {
    const float* source = worker.host_output_.data();
    for (int channel = 0; channel < data->geometry.channels; ++channel) {
        const ptrdiff_t byte_stride = vsapi->getStride(output, channel);
        if (byte_stride <
            static_cast<ptrdiff_t>(static_cast<std::size_t>(data->vi.width) *
                                   sizeof(float))) {
            throw std::runtime_error("unsupported output frame stride");
        }
        auto* destination =
            reinterpret_cast<float*>(vsapi->getWritePtr(output, channel));
        const float* plane =
            source + static_cast<std::size_t>(channel) *
                         static_cast<std::size_t>(data->geometry.plane_values);
        const std::size_t row_bytes =
            static_cast<std::size_t>(data->vi.width) * sizeof(float);
        for (int y = 0; y < data->vi.height; ++y) {
            std::memcpy(reinterpret_cast<std::byte*>(destination) +
                            static_cast<std::size_t>(y) *
                                static_cast<std::size_t>(byte_stride),
                        plane + static_cast<std::size_t>(y) *
                                    static_cast<std::size_t>(data->vi.width),
                        row_bytes);
        }
    }
}

[[nodiscard]] const VSFrame* render_frame(int output_frame,
                                          CudaFilterData* data,
                                          VSFrameContext* frame_ctx,
                                          VSCore* core, const VSAPI* vsapi) {
    WorkerPool::Lease worker_lease = data->workers->acquire();
    Worker& worker = worker_lease.get();
    check_cuda(cudaSetDevice(data->parameters.device),
               "cudaSetDevice(render frame)");
    worker.upload_cursor_ = 0;

    const FrameRange requested =
        required_source_range(data->parameters, data->geometry, output_frame);
    FrameListGuard clip_guard(worker.clip_frames, vsapi);
    FrameListGuard ref_guard(worker.ref_frames, vsapi);
    fetch_frames(worker, data, requested, frame_ctx, vsapi);

    const VSFrame* property_source =
        worker.clip_frames[static_cast<std::size_t>(output_frame -
                                                    requested.first)];
    VSFrame* output = new_video_frame_checked(
        &data->out_vi.format, data->out_vi.width, data->out_vi.height,
        property_source, core, vsapi);
    FrameGuard output_guard(output, vsapi);

    vnlbcu::FrameCache::Window contribution_window;
    vnlbcu::FrameCache::Window noisy_window;
    vnlbcu::FrameCache::Window basic_window;
    try {
        // Every render acquires caches in the same order: source clip, Final
        // reference clip, then contribution buffers.  FrameCache claims each
        // individual window atomically, and this global ordering prevents a
        // worker holding one cache class while waiting on the reverse order.
        const FrameRange source_cache_range = requested;
        if (source_cache_range.count() > data->geometry.max_source_window) {
            throw std::logic_error(
                "dynamic source window exceeds its cache reservation");
        }
        build_contiguous_keys(worker.noisy_keys, data->noisy_source_id,
                              source_cache_range);
        noisy_window = data->noisy_cache->acquire(worker.noisy_keys);
        upload_source_misses(worker, noisy_window, worker.clip_frames, vsapi,
                             data->vi.width, data->vi.height,
                             data->geometry.channels);
        if (noisy_window.needs_upload()) {
            noisy_window.publish(worker.stream_);
        }
        noisy_window.wait_ready(worker.stream_);

        if (data->parameters.stage == vnlbcu::Stage::Final) {
            build_contiguous_keys(worker.basic_keys, data->basic_source_id,
                                  source_cache_range);
            basic_window = data->basic_cache->acquire(worker.basic_keys);
            upload_source_misses(worker, basic_window, worker.ref_frames, vsapi,
                                 data->vi.width, data->vi.height,
                                 data->geometry.channels);
            if (basic_window.needs_upload()) {
                basic_window.publish(worker.stream_);
            }
            basic_window.wait_ready(worker.stream_);
        }

        const FrameRange anchors = contributing_anchors(
            data->parameters, data->geometry, output_frame);
        build_contiguous_keys(worker.contribution_keys,
                              data->contribution_source_id, anchors);
        contribution_window =
            data->contribution_cache->acquire(worker.contribution_keys);
        contribution_window.wait_hits(worker.stream_);

        int missing_count = 0;
        FrameRange assembly_range{data->vi.numFrames, -1};
        const std::span<const vnlbcu::CachedDeviceFrame> contribution_frames =
            contribution_window.frames();
        for (const vnlbcu::CachedDeviceFrame& cached : contribution_frames) {
            if (cached.needs_upload) {
                const FrameRange source = anchor_source_range(
                    data->parameters, data->geometry, cached.key.frame);
                assembly_range.first =
                    std::min(assembly_range.first, source.first);
                assembly_range.last =
                    std::max(assembly_range.last, source.last);
                if (worker.use_pointer_tables_) {
                    stage_frame_table(worker, noisy_window, source_cache_range,
                                      source, data, missing_count, false);
                    if (data->parameters.stage == vnlbcu::Stage::Final) {
                        stage_frame_table(worker, basic_window,
                                          source_cache_range, source, data,
                                          missing_count, true);
                    }
                }
                ++missing_count;
            }
        }
        if (worker.use_pointer_tables_) {
            upload_frame_tables(worker, data, missing_count, false);
            if (data->parameters.stage == vnlbcu::Stage::Final) {
                upload_frame_tables(worker, data, missing_count, true);
            }
        } else if (missing_count > 0) {
            assemble_contiguous_window(worker, noisy_window, source_cache_range,
                                       assembly_range, data, false);
            if (data->parameters.stage == vnlbcu::Stage::Final) {
                assemble_contiguous_window(worker, basic_window,
                                           source_cache_range, assembly_range,
                                           data, true);
            }
        }

        int missing_index = 0;
        for (const vnlbcu::CachedDeviceFrame& cached : contribution_frames) {
            if (!cached.needs_upload) {
                continue;
            }
            compute_contribution(worker, data, missing_index, cached.key.frame,
                                 cached, assembly_range);
            ++missing_index;
        }
        if (missing_index != missing_count) {
            throw std::logic_error("contribution miss accounting mismatch");
        }

        const std::size_t solver_count = checked_product(
            static_cast<std::size_t>(missing_count),
            static_cast<std::size_t>(data->geometry.total_groups),
            "solver status count overflows");
        if (solver_count > 0) {
            check_cuda(
                cudaMemcpyAsync(worker.host_solver_info_.data(),
                                worker.solver_info_.data(),
                                checked_product(solver_count, sizeof(int),
                                                "solver status copy overflows"),
                                cudaMemcpyDeviceToHost, worker.stream_),
                "cudaMemcpyAsync(cuSOLVER status)");
        }

        int source_index = 0;
        for (const vnlbcu::CachedDeviceFrame& cached : contribution_frames) {
            const int slot =
                checked_int(static_cast<long long>(output_frame) -
                                cached.key.frame + data->parameters.search_bwd,
                            "contribution slot index overflows int");
            worker.host_contribution_sources_
                .data()[static_cast<std::size_t>(source_index)] =
                vnlbcu::make_contribution_source(
                    contribution_view(data, cached), slot);
            ++source_index;
        }
        if (source_index != anchors.count()) {
            throw std::logic_error("contribution source accounting mismatch");
        }
        check_cuda(
            cudaMemcpyAsync(
                worker.contribution_sources_.data(),
                worker.host_contribution_sources_.data(),
                checked_product(static_cast<std::size_t>(source_index),
                                sizeof(vnlbcu::DeviceContributionSource),
                                "contribution descriptor copy overflows"),
                cudaMemcpyHostToDevice, worker.stream_),
            "cudaMemcpyAsync(contribution descriptors)");

        const vnlbcu::CachedDeviceFrame& fallback =
            cached_frame_at(noisy_window, source_cache_range, output_frame);
        const vnlbcu::DeviceVideoView fallback_view{
            .data = static_cast<const float*>(fallback.data),
            .width = data->vi.width,
            .height = data->vi.height,
            .channels = data->geometry.channels,
            .frames = 1,
            .first_frame = output_frame,
            .source_frames = data->vi.numFrames,
            .row_stride = data->vi.width,
            .channel_stride = data->geometry.plane_values,
            .frame_stride = data->geometry.frame_values,
        };
        const vnlbcu::DeviceMutableFrameView output_view{
            .data = worker.output_.data(),
            .width = data->vi.width,
            .height = data->vi.height,
            .channels = data->geometry.channels,
            .row_stride = data->vi.width,
            .channel_stride = data->geometry.plane_values,
        };
        worker.normalizer_.enqueue_normalize_many(
            worker.contribution_sources_.data(), source_index, fallback_view,
            output_frame, output_view, worker.stream_);
        check_cuda(cudaMemcpyAsync(worker.host_output_.data(),
                                   worker.output_.data(),
                                   data->geometry.frame_bytes,
                                   cudaMemcpyDeviceToHost, worker.stream_),
                   "cudaMemcpyAsync(final output)");
        check_cuda(cudaStreamSynchronize(worker.stream_),
                   "cudaStreamSynchronize(render frame)");

        for (std::size_t index = 0; index < solver_count; ++index) {
            const int status = worker.host_solver_info_.data()[index];
            if (status != 0) {
                const std::size_t anchor_index =
                    index /
                    static_cast<std::size_t>(data->geometry.total_groups);
                const std::size_t group_index =
                    index %
                    static_cast<std::size_t>(data->geometry.total_groups);
                std::ostringstream message;
                message << "cuSOLVER failed for generated anchor "
                        << anchor_index << ", group " << group_index
                        << " (info=" << status << ')';
                throw std::runtime_error(message.str());
            }
        }

        if (contribution_window.needs_upload()) {
            contribution_window.publish(worker.stream_);
        }
        contribution_window.release();
        basic_window.release();
        noisy_window.release();
        copy_output_to_frame(worker, output, data, vsapi);
        return output_guard.release();
    } catch (...) {
        worker.synchronize_noexcept();
        contribution_window.abandon();
        contribution_window.release();
        basic_window.release();
        noisy_window.release();
        throw;
    }
}

void initialize_cuda(CudaFilterData* data) {
    check_cuda(cudaSetDevice(data->parameters.device),
               "cudaSetDevice(plugin initialization)");

    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
    (void)total_bytes;

    const std::size_t clip_count =
        data->parameters.stage == vnlbcu::Stage::Final ? 2U : 1U;
    const std::size_t minimum_source_slots =
        static_cast<std::size_t>(data->geometry.max_source_window);
    const std::size_t desired_source_slots = std::min(
        static_cast<std::size_t>(data->vi.numFrames),
        checked_sum(minimum_source_slots,
                    static_cast<std::size_t>(data->parameters.num_streams - 1),
                    "source frame cache size overflows"));
    const std::size_t minimum_contribution_slots =
        static_cast<std::size_t>(data->geometry.max_contributors);
    const std::size_t minimum_source_cache_bytes = checked_product(
        checked_product(minimum_source_slots, data->geometry.frame_bytes,
                        "minimum source frame cache bytes overflow"),
        clip_count, "source frame cache bytes overflow");
    const std::size_t minimum_contribution_bytes = checked_product(
        minimum_contribution_slots, data->geometry.contribution_bytes,
        "minimum contribution cache bytes overflow");
    const std::size_t contiguous_worker_bytes = checked_product(
        checked_product(
            checked_product(
                static_cast<std::size_t>(data->parameters.num_streams),
                static_cast<std::size_t>(data->geometry.max_source_window),
                "contiguous worker bytes overflow"),
            data->geometry.frame_bytes, "contiguous worker bytes overflow"),
        clip_count, "contiguous worker bytes overflow");
    const std::size_t known_contiguous_bytes = checked_sum(
        checked_sum(minimum_source_cache_bytes, minimum_contribution_bytes,
                    "CUDA minimum working set overflows"),
        checked_sum(contiguous_worker_bytes, kDeviceMemoryMargin,
                    "CUDA minimum working set overflows"),
        "CUDA minimum working set overflows");
    const bool benchmark_prefers_pointer_tables =
        data->parameters.stage == vnlbcu::Stage::Basic &&
        data->parameters.patch_time == 1;
    // Matcher microbenchmarks favor contiguous windows outside Basic/Pt1,
    // but a copied union can dominate VRAM for large frames, Pt or radii.
    // Fall back to the zero-copy frame table before allocating workers when
    // the known minimum working set would not leave a safety margin.
    const bool use_pointer_tables =
        benchmark_prefers_pointer_tables || known_contiguous_bytes > free_bytes;

    const std::size_t upload_slots_by_budget = std::max<std::size_t>(
        1U, kPinnedUploadBudgetPerWorker /
                std::max<std::size_t>(data->geometry.frame_bytes, 1U));
    const int upload_slots = static_cast<int>(std::min<std::size_t>(
        static_cast<std::size_t>(data->geometry.max_source_window),
        upload_slots_by_budget));
    const WorkerConfig worker_config{
        .device = data->parameters.device,
        .stage = data->parameters.stage,
        .pipeline_shape = data->pipeline_shape,
        .aggregation_shape = data->aggregation_shape,
        .window_gamma = data->parameters.weight_gamma,
        .frame_values = data->geometry.frame_values,
        .cached_frames = data->geometry.cached_frames,
        .upload_slots = upload_slots,
        .total_groups = data->geometry.total_groups,
        .max_contributors = data->geometry.max_contributors,
        .max_source_window = data->geometry.max_source_window,
        .width = data->vi.width,
        .height = data->vi.height,
        .patch_size = data->parameters.patch_size,
        .proc_step = data->parameters.proc_step == 0
                         ? std::max(1, data->parameters.patch_size / 2)
                         : data->parameters.proc_step,
        .valid_anchor_frames = data->geometry.valid_anchor_frames,
        .use_pointer_tables = use_pointer_tables,
    };
    data->workers = std::make_unique<WorkerPool>(data->parameters.num_streams,
                                                 worker_config);

    const auto allocate_source_caches = [data](std::size_t slots) {
        auto noisy = std::make_unique<vnlbcu::FrameCache>(
            slots, data->geometry.frame_bytes, data->parameters.device);
        std::unique_ptr<vnlbcu::FrameCache> basic;
        if (data->parameters.stage == vnlbcu::Stage::Final) {
            basic = std::make_unique<vnlbcu::FrameCache>(
                slots, data->geometry.frame_bytes, data->parameters.device);
        }
        data->noisy_cache = std::move(noisy);
        data->basic_cache = std::move(basic);
    };
    try {
        allocate_source_caches(desired_source_slots);
    } catch (...) {
        if (desired_source_slots == minimum_source_slots) {
            throw;
        }
        // A desired-capacity allocation can fail after the free-memory
        // preflight because each slot is a separate cudaMalloc and other
        // filters may allocate concurrently.  Rebuild the complete source
        // cache set at the minimum capacity so Final never leaves the noisy
        // cache oversized at the expense of the required basic cache.
        allocate_source_caches(minimum_source_slots);
    }

    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
    (void)total_bytes;
    const std::size_t minimum_slots = minimum_contribution_slots;
    const std::size_t desired_slots = std::min(
        static_cast<std::size_t>(data->geometry.valid_anchor_frames),
        checked_sum(
            minimum_slots,
            checked_sum(
                static_cast<std::size_t>(data->parameters.num_streams),
                static_cast<std::size_t>(data->parameters.search_bwd) +
                    static_cast<std::size_t>(data->parameters.search_fwd),
                "contribution cache target overflows"),
            "contribution cache target overflows"));
    const std::size_t budget = free_bytes > kDeviceMemoryMargin
                                   ? free_bytes - kDeviceMemoryMargin
                                   : free_bytes;
    std::size_t affordable =
        budget / std::max<std::size_t>(data->geometry.contribution_bytes, 1U);
    if (affordable < minimum_slots &&
        free_bytes /
                std::max<std::size_t>(data->geometry.contribution_bytes, 1U) >=
            minimum_slots) {
        affordable = minimum_slots;
    }
    const std::size_t cache_slots = std::min(desired_slots, affordable);
    if (cache_slots < minimum_slots) {
        const std::size_t required =
            checked_product(minimum_slots, data->geometry.contribution_bytes,
                            "minimum contribution cache size overflows");
        std::ostringstream message;
        message << "insufficient device memory for the fused contribution "
                   "cache (need at least "
                << (required >> 20U) << " MiB for " << minimum_slots
                << " slots, have " << (free_bytes >> 20U) << " MiB free)";
        throw std::runtime_error(message.str());
    }
    try {
        data->contribution_cache = std::make_unique<vnlbcu::FrameCache>(
            cache_slots, data->geometry.contribution_bytes,
            data->parameters.device);
    } catch (...) {
        if (cache_slots == minimum_slots) {
            throw;
        }
        data->contribution_cache = std::make_unique<vnlbcu::FrameCache>(
            minimum_slots, data->geometry.contribution_bytes,
            data->parameters.device);
    }
}

void VS_CC CudaFilterFree(void* instance_data, [[maybe_unused]] VSCore* core,
                          const VSAPI* vsapi) noexcept {
    auto data = std::unique_ptr<CudaFilterData>(
        static_cast<CudaFilterData*>(instance_data));
    data->workers.reset();
    data->contribution_cache.reset();
    data->basic_cache.reset();
    data->noisy_cache.reset();
    if (data->ref_node != nullptr) {
        vsapi->freeNode(data->ref_node);
    }
    if (data->node != nullptr) {
        vsapi->freeNode(data->node);
    }
}

const VSFrame* VS_CC CudaFilterGetFrame(int n, int activation_reason,
                                        void* instance_data,
                                        [[maybe_unused]] void** frame_data,
                                        VSFrameContext* frame_ctx, VSCore* core,
                                        const VSAPI* vsapi) {
    auto* data = static_cast<CudaFilterData*>(instance_data);
    try {
        if (activation_reason == arInitial) {
            request_frames(data, n, frame_ctx, vsapi);
        } else if (activation_reason == arAllFramesReady) {
            return render_frame(n, data, frame_ctx, core, vsapi);
        }
    } catch (const std::exception& error) {
        const std::string message = std::string("VNLB CUDA: ") + error.what();
        vsapi->setFilterError(message.c_str(), frame_ctx);
    }
    return nullptr;
}

template <vnlbcu::Stage stage>
[[nodiscard]] constexpr const char* filter_name() noexcept {
    if constexpr (stage == vnlbcu::Stage::Basic) {
        return "Basic";
    }
    return "Final";
}

template <vnlbcu::Stage stage>
void create_filter(const VSMap* in, VSMap* out, VSCore* core,
                   const VSAPI* vsapi) {
    auto data = std::make_unique<CudaFilterData>();
    try {
        data->node = map_get_required_node(in, vsapi, "clip");
        const VSVideoInfo* vi = vsapi->getVideoInfo(data->node);
        if (!is_supported_float_format(vi)) {
            throw std::invalid_argument(
                "only constant GrayS and YUV444PS clips are supported");
        }
        data->vi = *vi;
        data->out_vi = *vi;

        if constexpr (stage == vnlbcu::Stage::Final) {
            data->ref_node = map_get_required_node(in, vsapi, "ref");
            const VSVideoInfo* ref_vi = vsapi->getVideoInfo(data->ref_node);
            if (!is_supported_float_format(ref_vi) ||
                !vsh::isSameVideoFormat(&ref_vi->format, &vi->format) ||
                ref_vi->width != vi->width || ref_vi->height != vi->height ||
                ref_vi->numFrames != vi->numFrames) {
                throw std::invalid_argument(
                    "ref clip properties must match input clip");
            }
            data->ref_vi = *ref_vi;
        }

        constexpr int default_streams = 1;
        data->parameters =
            parse_parameters(in, vsapi, stage, default_streams, data->vi);
        validate_parameters(data->parameters, data->vi);
        data->geometry = derive_geometry(data->parameters, data->vi);
        data->pipeline_shape =
            make_pipeline_shape(data->parameters, data->geometry, data->vi);
        data->pipeline_parameters = make_pipeline_parameters(data->parameters);
        data->aggregation_shape =
            make_aggregation_shape(data->parameters, data->geometry, data->vi);
        data->noisy_source_id = next_source_id();
        data->basic_source_id = next_source_id();
        data->contribution_source_id = next_source_id();
        initialize_cuda(data.get());

        std::vector<VSFilterDependency> dependencies;
        dependencies.reserve(stage == vnlbcu::Stage::Final ? 2U : 1U);
        const bool strict_spatial = data->parameters.patch_time == 1 &&
                                    data->parameters.search_bwd == 0 &&
                                    data->parameters.search_fwd == 0;
        const int request_pattern =
            strict_spatial ? rpStrictSpatial : rpGeneral;
        dependencies.push_back(VSFilterDependency{data->node, request_pattern});
        if constexpr (stage == vnlbcu::Stage::Final) {
            dependencies.push_back(
                VSFilterDependency{data->ref_node, request_pattern});
        }
        CudaFilterData* const instance = data.release();
        vsapi->createVideoFilter(
            out, filter_name<stage>(), &instance->out_vi, CudaFilterGetFrame,
            CudaFilterFree, fmParallel, dependencies.data(),
            static_cast<int>(dependencies.size()), instance, core);
    } catch (const std::exception& error) {
        if (data != nullptr) {
            CudaFilterFree(data.release(), core, vsapi);
        }
        const std::string message = std::string("VNLB CUDA: ") + error.what();
        vsapi->mapSetError(out, message.c_str());
    }
}

void VS_CC BasicCreate(const VSMap* in, VSMap* out,
                       [[maybe_unused]] void* user_data, VSCore* core,
                       const VSAPI* vsapi) {
    create_filter<vnlbcu::Stage::Basic>(in, out, core, vsapi);
}

void VS_CC FinalCreate(const VSMap* in, VSMap* out,
                       [[maybe_unused]] void* user_data, VSCore* core,
                       const VSAPI* vsapi) {
    create_filter<vnlbcu::Stage::Final>(in, out, core, vsapi);
}

} // namespace

VS_EXTERNAL_API(void)
VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    vspapi->configPlugin(
        kPluginId, kPluginNamespace, "VapourSynth VNLB CUDA",
        VS_MAKE_VERSION(VNLB_VERSION_MAJOR, VNLB_VERSION_MINOR),
        VAPOURSYNTH_API_VERSION, 0, plugin);

    constexpr const char* basic_args =
        "clip:vnode;sigma:float;block_size:int:opt;block_step:int:opt;"
        "group_size:int:opt;bm_range:int:opt;patch_time:int:opt;"
        "radius:int:opt;search_bwd:int:opt;search_fwd:int:opt;"
        "rank:int:opt;cap_factor:float:opt;model_cap_factor:float:opt;"
        "beta:float:opt;tau:float:opt;variance_threshold:float:opt;"
        "weight_alpha:float:opt;weight_beta:float:opt;"
        "weight_gamma:float:opt;weight_epsilon:float:opt;"
        "membership_noise_floor:float:opt;chroma:int:opt;"
        "device_id:int:opt;num_streams:int:opt;chunk_size:int:opt;";
    vspapi->registerFunction("Basic", basic_args, "clip:vnode;", BasicCreate,
                             nullptr, plugin);

    constexpr const char* final_args =
        "clip:vnode;ref:vnode;sigma:float;block_size:int:opt;"
        "block_step:int:opt;group_size:int:opt;bm_range:int:opt;"
        "patch_time:int:opt;radius:int:opt;search_bwd:int:opt;"
        "search_fwd:int:opt;rank:int:opt;cap_factor:float:opt;"
        "model_cap_factor:float:opt;beta:float:opt;tau:float:opt;"
        "variance_threshold:float:opt;weight_alpha:float:opt;"
        "weight_beta:float:opt;weight_gamma:float:opt;"
        "weight_epsilon:float:opt;membership_noise_floor:float:opt;"
        "chroma:int:opt;device_id:int:opt;num_streams:int:opt;"
        "chunk_size:int:opt;"
        "sigma_basic:float:opt;gamma:float:opt;flat_areas:int:opt;";
    vspapi->registerFunction("Final", final_args, "clip:vnode;", FinalCreate,
                             nullptr, plugin);
}
