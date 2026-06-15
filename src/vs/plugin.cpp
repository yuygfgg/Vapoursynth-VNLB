#include "aggregate/aggregate.hpp"
#include "common/arithmetic.hpp"
#include "common/compiler.hpp"
#include "core/core.hpp"
#include "flow/flow.hpp"
#include "vnlb_version.hpp"

#include <VSHelper4.h>
#include <VapourSynth4.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using vnlb::aggregate::ContributionPlaneView;
using vnlb::aggregate::ContributionStackView;
using vnlb::core::ConstPlaneView;
using vnlb::core::ConstVideoView;
using vnlb::core::FrameRange;
using vnlb::core::Stage;
using vnlb::core::StageParameters;
using vnlb::core::StageWorkspace;
using vnlb::core::VideoGeometry;

constexpr const char* kPluginId = "com.yuygfgg.vnlb";
constexpr const char* kPluginNamespace = "vnlb";
constexpr float kEightBitSampleScale = 255.0F;
constexpr float kEightBitDistanceScale =
    kEightBitSampleScale * kEightBitSampleScale;

struct CachedMVToolsGrid {
    int frame = -1;
    vnlb::flow::MVToolsVectorGrid grid;
};

struct VNLBThreadData {
    StageWorkspace workspace;
    std::vector<const VSFrame*> clip_frames;
    std::vector<const VSFrame*> ref_frames;
    std::vector<const VSFrame*> mvfw_frames;
    std::vector<const VSFrame*> mvbw_frames;
    std::vector<ConstPlaneView> clip_planes;
    std::vector<ConstPlaneView> ref_planes;
    std::vector<ContributionPlaneView> contribution_planes;
    std::vector<CachedMVToolsGrid> mvfw_cache;
    std::vector<CachedMVToolsGrid> mvbw_cache;
    std::vector<const vnlb::flow::MVToolsVectorGrid*> mvfw_grids;
    std::vector<const vnlb::flow::MVToolsVectorGrid*> mvbw_grids;
};

struct VNLBData {
    VSNode* node = nullptr;
    VSNode* ref_node = nullptr;
    VSNode* mvfw_node = nullptr;
    VSNode* mvbw_node = nullptr;
    VSVideoInfo vi{};
    VSVideoInfo ref_vi{};
    VSVideoInfo mvfw_vi{};
    VSVideoInfo mvbw_vi{};
    VSVideoInfo out_vi{};
    StageParameters parameters{};
    vnlb::flow::MVToolsAnalysisData mvfw_analysis{};
    vnlb::flow::MVToolsAnalysisData mvbw_analysis{};
    int slot_count = 0;
    mutable std::mutex buffer_lock;
    mutable std::unordered_map<std::thread::id, std::unique_ptr<VNLBThreadData>>
        buffer;
};

struct VAggregateData {
    VSNode* node = nullptr;
    VSNode* src_node = nullptr;
    VSVideoInfo vi{};
    VSVideoInfo src_vi{};
    VSVideoInfo out_vi{};
    int search_bwd = 1;
    int search_fwd = 1;
    int patch_time = 1;
    int slot_count = 1;
};

struct FrameListGuard {
    std::vector<const VSFrame*>& frames;
    const VSAPI* vsapi = nullptr;

    FrameListGuard(std::vector<const VSFrame*>& owned_frames,
                   const VSAPI* api) noexcept
        : frames(owned_frames), vsapi(api) {}
    FrameListGuard(const FrameListGuard&) = delete;
    FrameListGuard& operator=(const FrameListGuard&) = delete;
    FrameListGuard(FrameListGuard&&) = delete;
    FrameListGuard& operator=(FrameListGuard&&) = delete;

    ~FrameListGuard() {
        for (const VSFrame* frame : frames) {
            if (frame != nullptr) {
                vsapi->freeFrame(frame);
            }
        }
        frames.clear();
    }
};

class FrameGuard {
  public:
    FrameGuard(const VSFrame* frame, const VSAPI* vsapi) noexcept
        : frame_(frame), vsapi_(vsapi) {}
    FrameGuard(const FrameGuard&) = delete;
    FrameGuard& operator=(const FrameGuard&) = delete;
    FrameGuard(FrameGuard&&) = delete;
    FrameGuard& operator=(FrameGuard&&) = delete;

    ~FrameGuard() {
        if (frame_ != nullptr) {
            vsapi_->freeFrame(frame_);
        }
    }

    [[nodiscard]] const VSFrame* get() const noexcept { return frame_; }

    [[nodiscard]] const VSFrame* release() noexcept {
        const VSFrame* frame = frame_;
        frame_ = nullptr;
        return frame;
    }

  private:
    const VSFrame* frame_ = nullptr;
    const VSAPI* vsapi_ = nullptr;
};

struct TemporalParameters {
    int patch_time = 1;
    int search_bwd = 1;
    int search_fwd = 1;
};

[[nodiscard]] bool is_supported_float_format(const VSVideoInfo* info) noexcept {
    if (info == nullptr || info->width <= 0 || info->height <= 0 ||
        !vsh::isConstantVideoFormat(info) ||
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

[[nodiscard]] int plane_count(const VSVideoInfo& info) noexcept {
    return info.format.numPlanes;
}

[[nodiscard]] int stride_in_floats(const VSFrame* frame, int plane,
                                   const VSAPI* vsapi) {
    const ptrdiff_t stride = vsapi->getStride(frame, plane);
    if (stride <= 0 ||
        stride / static_cast<ptrdiff_t>(sizeof(float)) >
            static_cast<ptrdiff_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("unsupported frame stride");
    }
    return static_cast<int>(stride / static_cast<ptrdiff_t>(sizeof(float)));
}

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
                                               const VSFrame* prop_src,
                                               VSCore* core,
                                               const VSAPI* vsapi) {
    VSFrame* frame =
        vsapi->newVideoFrame(format, width, height, prop_src, core);
    if (frame == nullptr) {
        throw std::bad_alloc();
    }
    return frame;
}

[[nodiscard]] VideoGeometry make_geometry(const VSVideoInfo& info,
                                          int first_frame = 0,
                                          int local_frames = 0) {
    const int frames = local_frames > 0 ? local_frames : info.numFrames;
    return VideoGeometry{info.width,        info.height, frames,
                         plane_count(info), first_frame, info.numFrames};
}

[[nodiscard]] int map_get_optional_int(const VSMap* in, const VSAPI* vsapi,
                                       const char* key, int fallback) {
    int error = peSuccess;
    const int value = vsapi->mapGetIntSaturated(in, key, 0, &error);
    return error == peSuccess ? value : fallback;
}

[[nodiscard]] bool map_has_key(const VSMap* in, const VSAPI* vsapi,
                               const char* key) noexcept {
    return vsapi->mapNumElements(in, key) > 0;
}

[[nodiscard]] TemporalParameters
parse_temporal_parameters(const VSMap* in, const VSAPI* vsapi,
                          TemporalParameters fallback) {
    TemporalParameters result = fallback;
    result.patch_time =
        map_get_optional_int(in, vsapi, "patch_time", result.patch_time);

    const int radius = map_get_optional_int(in, vsapi, "radius", -1);
    if (radius >= 0) {
        result.search_bwd = radius;
        result.search_fwd = radius;
    } else if (map_has_key(in, vsapi, "radius")) {
        throw std::invalid_argument("radius must be non-negative");
    }

    result.search_bwd =
        map_get_optional_int(in, vsapi, "search_bwd", result.search_bwd);
    result.search_fwd =
        map_get_optional_int(in, vsapi, "search_fwd", result.search_fwd);
    if (result.patch_time <= 0 || result.search_bwd < 0 ||
        result.search_fwd < 0) {
        throw std::invalid_argument(
            "patch_time must be positive and search_bwd/search_fwd must be "
            "non-negative");
    }
    return result;
}

[[nodiscard]] int checked_slot_count(int search_bwd, int search_fwd,
                                     int patch_time) {
    return vnlb::common::checked_add_int(
        vnlb::common::checked_add_int(search_bwd, search_fwd,
                                      "slot count overflows int"),
        patch_time, "slot count overflows int");
}

[[nodiscard]] int checked_contribution_height(int source_height,
                                              int slot_count) {
    return vnlb::common::checked_mul_int(
        source_height,
        vnlb::common::checked_mul_int(2, slot_count,
                                      "contribution height overflows int"),
        "contribution height overflows int");
}

[[nodiscard]] float map_get_optional_float(const VSMap* in, const VSAPI* vsapi,
                                           const char* key, float fallback) {
    int error = peSuccess;
    const float value = vsapi->mapGetFloatSaturated(in, key, 0, &error);
    return error == peSuccess ? value : fallback;
}

[[nodiscard]] float map_get_required_float(const VSMap* in, const VSAPI* vsapi,
                                           const char* key) {
    int error = peSuccess;
    const float value = vsapi->mapGetFloatSaturated(in, key, 0, &error);
    if (error != peSuccess) {
        throw std::invalid_argument(std::string("missing required argument: ") +
                                    key);
    }
    return value;
}

[[nodiscard]] float map_get_required_sigma_8bit(const VSMap* in,
                                                const VSAPI* vsapi,
                                                const char* key) {
    return map_get_required_float(in, vsapi, key) / kEightBitSampleScale;
}

[[nodiscard]] float map_get_optional_sigma_8bit(const VSMap* in,
                                                const VSAPI* vsapi,
                                                const char* key,
                                                float fallback) {
    int error = peSuccess;
    const float value = vsapi->mapGetFloatSaturated(in, key, 0, &error);
    return error == peSuccess ? value / kEightBitSampleScale : fallback;
}

[[nodiscard]] float map_get_optional_tau_8bit(const VSMap* in,
                                              const VSAPI* vsapi,
                                              const char* key, float fallback) {
    int error = peSuccess;
    const float value = vsapi->mapGetFloatSaturated(in, key, 0, &error);
    return error == peSuccess ? value / kEightBitDistanceScale : fallback;
}

[[nodiscard]] VSNode* map_get_optional_node(const VSMap* in, const VSAPI* vsapi,
                                            const char* key) {
    int error = peSuccess;
    VSNode* node = vsapi->mapGetNode(in, key, 0, &error);
    if (error == peSuccess) {
        return node;
    }
    if (error == peUnset) {
        return nullptr;
    }
    throw std::invalid_argument(std::string(key) + " must be a video node");
}

template <Stage stage>
[[nodiscard]] StageParameters parse_stage_parameters(const VSMap* in,
                                                     const VSAPI* vsapi) {
    StageParameters parameters{};
    parameters.sigma = map_get_required_sigma_8bit(in, vsapi, "sigma");
    if constexpr (stage == Stage::Final) {
        parameters.tau = 400.0F / kEightBitDistanceScale;
        parameters.variance_threshold = 1.7F;
        parameters.flat_areas = true;
        parameters.weight_alpha = 1.0F;
        parameters.weight_beta = 0.5F;
        parameters.weight_gamma = 1.0F;
    }

    parameters.patch_size =
        map_get_optional_int(in, vsapi, "block_size", parameters.patch_size);
    const TemporalParameters temporal = parse_temporal_parameters(
        in, vsapi,
        TemporalParameters{parameters.patch_time, parameters.search_bwd,
                           parameters.search_fwd});
    parameters.patch_time = temporal.patch_time;
    parameters.search_bwd = temporal.search_bwd;
    parameters.search_fwd = temporal.search_fwd;
    if (map_has_key(in, vsapi, "bm_range")) {
        const int bm_range = map_get_optional_int(in, vsapi, "bm_range", 0);
        if (bm_range < 0 ||
            bm_range > (std::numeric_limits<int>::max() - 1) / 2) {
            throw std::invalid_argument("bm_range must be non-negative");
        }
        parameters.search_window = (bm_range * 2) + 1;
    }
    parameters.similar =
        map_get_optional_int(in, vsapi, "group_size", parameters.similar);
    parameters.rank = map_get_optional_int(in, vsapi, "rank", parameters.rank);
    parameters.similar_cap_factor = map_get_optional_float(
        in, vsapi, "cap_factor", parameters.similar_cap_factor);
    parameters.model_cap_factor = map_get_optional_float(
        in, vsapi, "model_cap_factor", parameters.model_cap_factor);
    parameters.beta =
        map_get_optional_float(in, vsapi, "beta", parameters.beta);
    parameters.tau =
        map_get_optional_tau_8bit(in, vsapi, "tau", parameters.tau);
    parameters.variance_threshold = map_get_optional_float(
        in, vsapi, "variance_threshold", parameters.variance_threshold);
    parameters.proc_step =
        map_get_optional_int(in, vsapi, "block_step", parameters.proc_step);
    if constexpr (stage == Stage::Final) {
        parameters.sigma_basic = map_get_optional_sigma_8bit(
            in, vsapi, "sigma_basic", parameters.sigma_basic);
        parameters.gamma =
            map_get_optional_float(in, vsapi, "gamma", parameters.gamma);
        parameters.flat_areas =
            map_get_optional_int(in, vsapi, "flat_areas",
                                 parameters.flat_areas ? 1 : 0) != 0;
    }
    parameters.weight_alpha = map_get_optional_float(in, vsapi, "weight_alpha",
                                                     parameters.weight_alpha);
    parameters.weight_beta = map_get_optional_float(in, vsapi, "weight_beta",
                                                    parameters.weight_beta);
    parameters.weight_gamma = map_get_optional_float(in, vsapi, "weight_gamma",
                                                     parameters.weight_gamma);
    parameters.weight_epsilon = map_get_optional_float(
        in, vsapi, "weight_epsilon", parameters.weight_epsilon);
    parameters.membership_noise_floor = map_get_optional_float(
        in, vsapi, "membership_noise_floor", parameters.membership_noise_floor);
    parameters.couple_channels =
        map_get_optional_int(in, vsapi, "chroma",
                             parameters.couple_channels ? 1 : 0) != 0;

    return parameters;
}

[[nodiscard]] int slot_count(const StageParameters& parameters) {
    return checked_slot_count(parameters.search_bwd, parameters.search_fwd,
                              parameters.patch_time);
}

[[nodiscard]] std::span<const std::byte>
map_get_binary_prop(const VSMap* props, const VSAPI* vsapi, const char* key) {
    int error = peSuccess;
    const char* data = vsapi->mapGetData(props, key, 0, &error);
    if (error != peSuccess || data == nullptr) {
        throw std::invalid_argument(std::string("missing frame property: ") +
                                    key);
    }
    const int size = vsapi->mapGetDataSize(props, key, 0, &error);
    if (error != peSuccess || size < 0) {
        throw std::invalid_argument(std::string("invalid frame property: ") +
                                    key);
    }
    return std::span<const std::byte>(reinterpret_cast<const std::byte*>(data),
                                      static_cast<std::size_t>(size));
}

[[nodiscard]] bool map_get_optional_bool_prop(const VSMap* props,
                                              const VSAPI* vsapi,
                                              const char* key) {
    int error = peSuccess;
    const int value = vsapi->mapGetIntSaturated(props, key, 0, &error);
    return error == peSuccess && value != 0;
}

[[nodiscard]] vnlb::flow::MVToolsAnalysisData
read_mvtools_analysis_from_frame(const VSFrame* frame, const VSAPI* vsapi) {
    const VSMap* props = vsapi->getFramePropertiesRO(frame);
    return vnlb::flow::parse_mvtools_analysis(
        map_get_binary_prop(props, vsapi, "MVTools_MVAnalysisData"));
}

[[nodiscard]] vnlb::flow::MVToolsAnalysisData
read_mvtools_analysis_from_node(VSNode* node, const char* argument_name,
                                const VSAPI* vsapi) {
    char error_message[1024]{};
    const VSFrame* frame = vsapi->getFrame(
        0, node, error_message, static_cast<int>(sizeof(error_message)));
    if (frame == nullptr) {
        throw std::invalid_argument(std::string("failed to read ") +
                                    argument_name +
                                    " analysis frame: " + error_message);
    }
    FrameGuard frame_guard{frame, vsapi};

    return read_mvtools_analysis_from_frame(frame_guard.get(), vsapi);
}

void validate_mvtools_clip(const VSVideoInfo& source_vi,
                           const VSVideoInfo& vector_vi,
                           const vnlb::flow::MVToolsAnalysisData& analysis,
                           const char* argument_name, bool expected_backwards) {
    if (vector_vi.numFrames != source_vi.numFrames) {
        throw std::invalid_argument(std::string(argument_name) +
                                    " frame count must match clip");
    }
    if (analysis.width != source_vi.width ||
        analysis.height != source_vi.height) {
        throw std::invalid_argument(std::string(argument_name) +
                                    " analysis dimensions must match clip");
    }
    if (analysis.delta_frame != 1) {
        throw std::invalid_argument(std::string(argument_name) +
                                    " must use MVTools delta=1 vectors");
    }
    if (analysis.backwards != expected_backwards) {
        throw std::invalid_argument(
            std::string(argument_name) +
            (expected_backwards ? " must be generated with isb=True"
                                : " must be generated with isb=False"));
    }
}

[[nodiscard]] bool has_scene_change(const VSFrame* frame, const VSAPI* vsapi,
                                    bool backwards) {
    const VSMap* props = vsapi->getFramePropertiesRO(frame);
    if (backwards) {
        return map_get_optional_bool_prop(props, vsapi, "_SceneChangeNext");
    }
    return map_get_optional_bool_prop(props, vsapi, "_SceneChangePrev") ||
           map_get_optional_bool_prop(props, vsapi, "Scenechange");
}

void prune_mvtools_cache(std::vector<CachedMVToolsGrid>& cache, int first_frame,
                         int last_frame) {
    cache.erase(std::remove_if(cache.begin(), cache.end(),
                               [&](const CachedMVToolsGrid& entry) noexcept {
                                   return entry.frame < first_frame ||
                                          entry.frame > last_frame;
                               }),
                cache.end());
}

[[nodiscard]] const vnlb::flow::MVToolsVectorGrid*
find_cached_mvtools_grid(std::vector<CachedMVToolsGrid>& cache, int frame) {
    auto existing =
        std::find_if(cache.begin(), cache.end(),
                     [frame](const CachedMVToolsGrid& entry) noexcept {
                         return entry.frame == frame;
                     });
    return existing == cache.end() ? nullptr : &existing->grid;
}

void cache_mvtools_grid(const VSFrame* vector_frame,
                        const VSFrame* source_frame, int frame,
                        vnlb::flow::MVToolsAnalysisData analysis,
                        std::vector<CachedMVToolsGrid>& cache,
                        const VSAPI* vsapi) {
    if (find_cached_mvtools_grid(cache, frame) != nullptr) {
        return;
    }

    vnlb::flow::MVToolsVectorGrid grid;
    const VSMap* props = vsapi->getFramePropertiesRO(vector_frame);
    vnlb::flow::parse_mvtools_vectors(
        map_get_binary_prop(props, vsapi, "MVTools_vectors"), analysis, grid);
    if (has_scene_change(source_frame, vsapi, analysis.backwards)) {
        grid.valid = false;
    }

    cache.push_back(CachedMVToolsGrid{frame, std::move(grid)});
}

void build_mvtools_grid_window(
    const std::vector<const VSFrame*>& vector_frames,
    const std::vector<const VSFrame*>& source_frames, int first_frame,
    vnlb::flow::MVToolsAnalysisData analysis,
    std::vector<CachedMVToolsGrid>& cache,
    std::vector<const vnlb::flow::MVToolsVectorGrid*>& grids,
    const VSAPI* vsapi) {
    if (vector_frames.size() != source_frames.size()) {
        throw std::invalid_argument(
            "MVTools vector frame window does not match source frame window");
    }

    if (vector_frames.empty()) {
        grids.clear();
        return;
    }

    const int last_frame = vnlb::common::checked_add_int(
        first_frame, static_cast<int>(vector_frames.size() - 1),
        "MVTools vector frame window overflows int");
    prune_mvtools_cache(cache, first_frame, last_frame);
    cache.reserve(cache.size() + vector_frames.size());
    for (std::size_t index = 0; index < vector_frames.size(); ++index) {
        const int current_frame = vnlb::common::checked_add_int(
            first_frame, static_cast<int>(index),
            "MVTools vector frame window overflows int");
        cache_mvtools_grid(vector_frames[index], source_frames[index],
                           current_frame, analysis, cache, vsapi);
    }

    grids.resize(vector_frames.size());
    for (std::size_t index = 0; index < vector_frames.size(); ++index) {
        const int current_frame = vnlb::common::checked_add_int(
            first_frame, static_cast<int>(index),
            "MVTools vector frame window overflows int");
        grids[index] = find_cached_mvtools_grid(cache, current_frame);
    }
}

[[nodiscard]] VNLBThreadData& thread_data_for(VNLBData* data) {
    const std::thread::id id = std::this_thread::get_id();
    std::lock_guard lock(data->buffer_lock);
    auto [it, inserted] = data->buffer.try_emplace(id);
    if (inserted) {
        it->second = std::make_unique<VNLBThreadData>();
    }
    return *it->second;
}

[[nodiscard]] FrameRange required_stage_frames(const VNLBData* data,
                                               int frame) {
    if (frame > data->vi.numFrames - data->parameters.patch_time) {
        return FrameRange{0, -1};
    }
    return vnlb::core::input_frame_range_for_anchor(make_geometry(data->vi),
                                                    data->parameters, frame);
}

template <Stage stage, bool has_mvfw, bool has_mvbw>
void request_stage_frames(const VNLBData* data, int frame,
                          VSFrameContext* frame_ctx, const VSAPI* vsapi) {
    const FrameRange range = required_stage_frames(data, frame);
    for (int needed = range.first; needed <= range.last; ++needed) {
        vsapi->requestFrameFilter(needed, data->node, frame_ctx);
        if constexpr (stage == Stage::Final) {
            vsapi->requestFrameFilter(needed, data->ref_node, frame_ctx);
        }
        if constexpr (has_mvfw) {
            vsapi->requestFrameFilter(needed, data->mvfw_node, frame_ctx);
        }
        if constexpr (has_mvbw) {
            vsapi->requestFrameFilter(needed, data->mvbw_node, frame_ctx);
        }
    }
}

void append_frame_planes(const VSFrame* frame, int channels,
                         std::vector<ConstPlaneView>& planes,
                         const VSAPI* vsapi) {
    for (int plane = 0; plane < channels; ++plane) {
        const float* VNLB_RESTRICT plane_data =
            reinterpret_cast<const float*>(vsapi->getReadPtr(frame, plane));
        planes.push_back(ConstPlaneView{
            plane_data,
            stride_in_floats(frame, plane, vsapi),
        });
    }
}

template <Stage stage, bool has_mvfw, bool has_mvbw>
[[nodiscard]] const VSFrame*
render_stage_frame(int frame, VNLBData* data, VSFrameContext* frame_ctx,
                   VSCore* core, const VSAPI* vsapi) {
    VNLBThreadData& thread_data = thread_data_for(data);
    const VideoGeometry geometry = make_geometry(data->vi);
    const auto layout = vnlb::aggregate::make_contribution_layout(
        geometry, data->parameters.search_bwd, data->parameters.search_fwd,
        data->parameters.patch_time);
    VSFrame* out =
        new_video_frame_checked(&data->out_vi.format, data->out_vi.width,
                                data->out_vi.height, nullptr, core, vsapi);
    FrameGuard out_guard{out, vsapi};
    thread_data.contribution_planes.clear();
    thread_data.contribution_planes.reserve(
        static_cast<std::size_t>(layout.channels));
    for (int channel = 0; channel < layout.channels; ++channel) {
        float* VNLB_RESTRICT plane_data =
            reinterpret_cast<float*>(vsapi->getWritePtr(out, channel));
        thread_data.contribution_planes.push_back(ContributionPlaneView{
            plane_data,
            stride_in_floats(out, channel, vsapi),
        });
    }
    const ContributionStackView contributions(
        std::span<ContributionPlaneView>(thread_data.contribution_planes),
        layout);

    const FrameRange range = required_stage_frames(data, frame);
    if (range.last >= range.first) {
        const int channels = plane_count(data->vi);
        const int local_frames = range.last - range.first + 1;
        thread_data.clip_frames.clear();
        thread_data.ref_frames.clear();
        thread_data.mvfw_frames.clear();
        thread_data.mvbw_frames.clear();
        thread_data.clip_planes.clear();
        thread_data.ref_planes.clear();
        thread_data.clip_frames.reserve(static_cast<std::size_t>(local_frames));
        thread_data.ref_frames.reserve(static_cast<std::size_t>(local_frames));
        thread_data.mvfw_frames.reserve(static_cast<std::size_t>(local_frames));
        thread_data.mvbw_frames.reserve(static_cast<std::size_t>(local_frames));
        const std::size_t local_plane_count = vnlb::common::checked_mul_size(
            static_cast<std::size_t>(local_frames),
            static_cast<std::size_t>(channels),
            "local frame plane count overflows size_t");
        thread_data.clip_planes.reserve(local_plane_count);
        thread_data.ref_planes.reserve(local_plane_count);

        FrameListGuard clip_guard{thread_data.clip_frames, vsapi};
        FrameListGuard ref_guard{thread_data.ref_frames, vsapi};
        FrameListGuard mvfw_guard{thread_data.mvfw_frames, vsapi};
        FrameListGuard mvbw_guard{thread_data.mvbw_frames, vsapi};
        for (int needed = range.first; needed <= range.last; ++needed) {
            const VSFrame* clip_frame = get_frame_filter_checked(
                needed, data->node, frame_ctx, vsapi, "clip");
            thread_data.clip_frames.push_back(clip_frame);
            append_frame_planes(clip_frame, channels, thread_data.clip_planes,
                                vsapi);

            if constexpr (stage == Stage::Final) {
                const VSFrame* ref_frame = get_frame_filter_checked(
                    needed, data->ref_node, frame_ctx, vsapi, "ref");
                thread_data.ref_frames.push_back(ref_frame);
                append_frame_planes(ref_frame, channels, thread_data.ref_planes,
                                    vsapi);
            }

            if constexpr (has_mvfw) {
                thread_data.mvfw_frames.push_back(get_frame_filter_checked(
                    needed, data->mvfw_node, frame_ctx, vsapi, "mvfw"));
            }
            if constexpr (has_mvbw) {
                thread_data.mvbw_frames.push_back(get_frame_filter_checked(
                    needed, data->mvbw_node, frame_ctx, vsapi, "mvbw"));
            }
        }

        const VideoGeometry local_geometry =
            make_geometry(data->vi, range.first, local_frames);
        const ConstVideoView clip_view(
            std::span<const ConstPlaneView>(thread_data.clip_planes),
            local_geometry);

        if constexpr (has_mvfw || has_mvbw) {
            if constexpr (has_mvfw) {
                build_mvtools_grid_window(
                    thread_data.mvfw_frames, thread_data.clip_frames,
                    range.first, data->mvfw_analysis, thread_data.mvfw_cache,
                    thread_data.mvfw_grids, vsapi);
            } else {
                thread_data.mvfw_grids.clear();
            }
            if constexpr (has_mvbw) {
                build_mvtools_grid_window(
                    thread_data.mvbw_frames, thread_data.clip_frames,
                    range.first, data->mvbw_analysis, thread_data.mvbw_cache,
                    thread_data.mvbw_grids, vsapi);
            } else {
                thread_data.mvbw_grids.clear();
            }

            const vnlb::flow::MVToolsFlowProvider flow_provider(
                std::span<const vnlb::flow::MVToolsVectorGrid* const>(
                    thread_data.mvfw_grids.data(),
                    thread_data.mvfw_grids.size()),
                range.first,
                std::span<const vnlb::flow::MVToolsVectorGrid* const>(
                    thread_data.mvbw_grids.data(),
                    thread_data.mvbw_grids.size()),
                range.first);
            if constexpr (stage == Stage::Basic) {
                vnlb::core::process_basic_anchor_mvtools(
                    clip_view, frame, data->parameters, flow_provider,
                    contributions, thread_data.workspace);
            } else {
                const ConstVideoView ref_view(
                    std::span<const ConstPlaneView>(thread_data.ref_planes),
                    local_geometry);
                vnlb::core::process_final_anchor_mvtools(
                    clip_view, ref_view, frame, data->parameters, flow_provider,
                    contributions, thread_data.workspace);
            }
        } else {
            const vnlb::flow::SameLocationProvider flow_provider;
            if constexpr (stage == Stage::Basic) {
                vnlb::core::process_basic_anchor_no_flow(
                    clip_view, frame, data->parameters, flow_provider,
                    contributions, thread_data.workspace);
            } else {
                const ConstVideoView ref_view(
                    std::span<const ConstPlaneView>(thread_data.ref_planes),
                    local_geometry);
                vnlb::core::process_final_anchor_no_flow(
                    clip_view, ref_view, frame, data->parameters, flow_provider,
                    contributions, thread_data.workspace);
            }
        }
    } else {
        vnlb::aggregate::clear_contributions(contributions);
    }
    return out_guard.release();
}

template <Stage stage, bool has_mvfw, bool has_mvbw>
const VSFrame* VS_CC VNLBGetFrame(int n, int activationReason,
                                  void* instanceData,
                                  [[maybe_unused]] void** frameData,
                                  VSFrameContext* frameCtx, VSCore* core,
                                  const VSAPI* vsapi) {
    auto* data = static_cast<VNLBData*>(instanceData);
    try {
        if (activationReason == arInitial) {
            const FrameRange range = required_stage_frames(data, n);
            if (range.last < range.first) {
                return render_stage_frame<stage, has_mvfw, has_mvbw>(
                    n, data, frameCtx, core, vsapi);
            }
            request_stage_frames<stage, has_mvfw, has_mvbw>(data, n, frameCtx,
                                                            vsapi);
        } else if (activationReason == arAllFramesReady) {
            return render_stage_frame<stage, has_mvfw, has_mvbw>(
                n, data, frameCtx, core, vsapi);
        }
    } catch (const std::exception& error) {
        const std::string message = std::string("VNLB: ") + error.what();
        vsapi->setFilterError(message.c_str(), frameCtx);
    }
    return nullptr;
}

void VS_CC VNLBFree(void* instanceData, [[maybe_unused]] VSCore* core,
                    const VSAPI* vsapi) noexcept {
    auto data = std::unique_ptr<VNLBData>(static_cast<VNLBData*>(instanceData));
    if (data->mvbw_node != nullptr) {
        vsapi->freeNode(data->mvbw_node);
    }
    if (data->mvfw_node != nullptr) {
        vsapi->freeNode(data->mvfw_node);
    }
    if (data->ref_node != nullptr) {
        vsapi->freeNode(data->ref_node);
    }
    if (data->node != nullptr) {
        vsapi->freeNode(data->node);
    }
}

template <Stage stage>
[[nodiscard]] constexpr const char* vnlb_filter_name() noexcept {
    if constexpr (stage == Stage::Basic) {
        return "VNLB Basic";
    }
    return "VNLB Final";
}

template <Stage stage, bool has_mvfw, bool has_mvbw>
void create_bound_vnlb_filter(VSMap* out, VSCore* core, const VSAPI* vsapi,
                              std::unique_ptr<VNLBData>& data,
                              const std::array<VSFilterDependency, 4>& deps,
                              int dep_count) {
    vsapi->createVideoFilter(out, vnlb_filter_name<stage>(), &data->out_vi,
                             VNLBGetFrame<stage, has_mvfw, has_mvbw>, VNLBFree,
                             fmParallel, deps.data(), dep_count, data.release(),
                             core);
}

template <Stage stage>
void create_bound_vnlb_filter(VSMap* out, VSCore* core, const VSAPI* vsapi,
                              std::unique_ptr<VNLBData>& data,
                              const std::array<VSFilterDependency, 4>& deps,
                              int dep_count) {
    const bool has_mvfw = data->mvfw_node != nullptr;
    const bool has_mvbw = data->mvbw_node != nullptr;
    if (has_mvfw) {
        if (has_mvbw) {
            create_bound_vnlb_filter<stage, true, true>(out, core, vsapi, data,
                                                        deps, dep_count);
            return;
        }
        create_bound_vnlb_filter<stage, true, false>(out, core, vsapi, data,
                                                     deps, dep_count);
        return;
    }
    if (has_mvbw) {
        create_bound_vnlb_filter<stage, false, true>(out, core, vsapi, data,
                                                     deps, dep_count);
        return;
    }
    create_bound_vnlb_filter<stage, false, false>(out, core, vsapi, data, deps,
                                                  dep_count);
}

template <Stage stage>
void VNLBCreate(const VSMap* in, VSMap* out, VSCore* core, const VSAPI* vsapi) {
    auto data = std::make_unique<VNLBData>();
    try {
        int error = peSuccess;
        data->node = vsapi->mapGetNode(in, "clip", 0, &error);
        if (error != peSuccess) {
            throw std::invalid_argument("clip must be a video node");
        }
        const VSVideoInfo* vi = vsapi->getVideoInfo(data->node);
        if (!is_supported_float_format(vi)) {
            throw std::invalid_argument(
                "only constant GrayS and YUV444PS clips are supported");
        }

        if constexpr (stage == Stage::Final) {
            data->ref_node = vsapi->mapGetNode(in, "ref", 0, &error);
            if (error != peSuccess) {
                throw std::invalid_argument("ref must be a video node");
            }
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

        data->mvfw_node = map_get_optional_node(in, vsapi, "mvfw");
        if (data->mvfw_node != nullptr) {
            const VSVideoInfo* mvfw_vi = vsapi->getVideoInfo(data->mvfw_node);
            data->mvfw_analysis =
                read_mvtools_analysis_from_node(data->mvfw_node, "mvfw", vsapi);
            validate_mvtools_clip(*vi, *mvfw_vi, data->mvfw_analysis, "mvfw",
                                  false);
            data->mvfw_vi = *mvfw_vi;
        }

        data->mvbw_node = map_get_optional_node(in, vsapi, "mvbw");
        if (data->mvbw_node != nullptr) {
            const VSVideoInfo* mvbw_vi = vsapi->getVideoInfo(data->mvbw_node);
            data->mvbw_analysis =
                read_mvtools_analysis_from_node(data->mvbw_node, "mvbw", vsapi);
            validate_mvtools_clip(*vi, *mvbw_vi, data->mvbw_analysis, "mvbw",
                                  true);
            data->mvbw_vi = *mvbw_vi;
        }

        data->parameters = parse_stage_parameters<stage>(in, vsapi);
        vnlb::core::validate_stage_configuration(make_geometry(*vi),
                                                 data->parameters);
        data->slot_count = slot_count(data->parameters);
        data->vi = *vi;
        data->out_vi = *vi;
        data->out_vi.height =
            checked_contribution_height(data->out_vi.height, data->slot_count);

        VSCoreInfo core_info;
        vsapi->getCoreInfo(core, &core_info);
        data->buffer.reserve(static_cast<std::size_t>(core_info.numThreads));

        std::array<VSFilterDependency, 4> deps{};
        int dep_count = 0;
        deps[static_cast<std::size_t>(dep_count++)] = {data->node, rpGeneral};
        if constexpr (stage == Stage::Final) {
            deps[static_cast<std::size_t>(dep_count++)] = {data->ref_node,
                                                           rpGeneral};
        }
        if (data->mvfw_node != nullptr) {
            deps[static_cast<std::size_t>(dep_count++)] = {data->mvfw_node,
                                                           rpGeneral};
        }
        if (data->mvbw_node != nullptr) {
            deps[static_cast<std::size_t>(dep_count++)] = {data->mvbw_node,
                                                           rpGeneral};
        }
        create_bound_vnlb_filter<stage>(out, core, vsapi, data, deps,
                                        dep_count);
    } catch (const std::exception& error) {
        if (data) {
            VNLBFree(data.release(), core, vsapi);
        }
        const std::string message = std::string("VNLB: ") + error.what();
        vsapi->mapSetError(out, message.c_str());
    }
}

[[nodiscard]] int aggregate_first_anchor(const VAggregateData* data,
                                         int frame) {
    const int forward_span =
        vnlb::common::checked_add_int(data->search_fwd, data->patch_time - 1,
                                      "aggregate temporal span overflows int");
    const int first = vnlb::common::checked_add_int(
        frame, -forward_span, "aggregate anchor frame overflows int");
    return std::max(0, first);
}

[[nodiscard]] int aggregate_last_anchor(const VAggregateData* data, int frame) {
    const int last = vnlb::common::checked_add_int(
        frame, data->search_bwd, "aggregate anchor frame overflows int");
    return std::min(data->src_vi.numFrames - 1, last);
}

void request_aggregate_frames(const VAggregateData* data, int frame,
                              VSFrameContext* frameCtx, const VSAPI* vsapi) {
    vsapi->requestFrameFilter(frame, data->src_node, frameCtx);
    const int first_anchor = aggregate_first_anchor(data, frame);
    const int last_anchor = aggregate_last_anchor(data, frame);
    for (int anchor = first_anchor; anchor <= last_anchor; ++anchor) {
        vsapi->requestFrameFilter(anchor, data->node, frameCtx);
    }
}

[[nodiscard]] const VSFrame* render_aggregate_frame(int frame,
                                                    VAggregateData* data,
                                                    VSFrameContext* frameCtx,
                                                    VSCore* core,
                                                    const VSAPI* vsapi) {
    const VSFrame* src_frame =
        get_frame_filter_checked(frame, data->src_node, frameCtx, vsapi, "src");
    FrameGuard src_guard{src_frame, vsapi};
    VSFrame* dst_frame =
        new_video_frame_checked(&data->out_vi.format, data->out_vi.width,
                                data->out_vi.height, src_frame, core, vsapi);
    FrameGuard dst_guard{dst_frame, vsapi};

    const int first_anchor = aggregate_first_anchor(data, frame);
    const int last_anchor = aggregate_last_anchor(data, frame);
    std::vector<const VSFrame*> frames;
    frames.reserve(static_cast<std::size_t>(last_anchor - first_anchor + 1));
    FrameListGuard frame_guard{frames, vsapi};
    for (int anchor = first_anchor; anchor <= last_anchor; ++anchor) {
        frames.push_back(get_frame_filter_checked(anchor, data->node, frameCtx,
                                                  vsapi, "contribution"));
    }

    const int channels = plane_count(data->src_vi);
    for (int plane = 0; plane < channels; ++plane) {
        const int width = vsapi->getFrameWidth(src_frame, plane);
        const int height = vsapi->getFrameHeight(src_frame, plane);
        const float* VNLB_RESTRICT src =
            reinterpret_cast<const float*>(vsapi->getReadPtr(src_frame, plane));
        const int src_stride = stride_in_floats(src_frame, plane, vsapi);
        float* VNLB_RESTRICT dst =
            reinterpret_cast<float*>(vsapi->getWritePtr(dst_frame, plane));
        const int dst_stride = stride_in_floats(dst_frame, plane, vsapi);

        std::vector<const float*> contribution_ptrs(frames.size());
        std::vector<int> contribution_strides(frames.size());
        std::vector<std::ptrdiff_t> numerator_base_rows(frames.size());
        for (std::size_t index = 0; index < frames.size(); ++index) {
            const int anchor = first_anchor + static_cast<int>(index);
            const int slot = vnlb::common::checked_add_int(
                vnlb::common::checked_add_int(
                    frame, -anchor, "aggregate slot index overflows int"),
                data->search_bwd, "aggregate slot index overflows int");
            numerator_base_rows[index] =
                static_cast<std::ptrdiff_t>(slot) * 2 * height;
            const float* VNLB_RESTRICT contribution =
                reinterpret_cast<const float*>(
                    vsapi->getReadPtr(frames[index], plane));
            contribution_ptrs[index] = contribution;
            contribution_strides[index] =
                stride_in_floats(frames[index], plane, vsapi);
        }

        for (int y = 0; y < height; ++y) {
            const float* VNLB_RESTRICT src_row =
                src + (static_cast<std::ptrdiff_t>(y) * src_stride);
            float* VNLB_RESTRICT dst_row =
                dst + (static_cast<std::ptrdiff_t>(y) * dst_stride);
            for (int x = 0; x < width; ++x) {
                float numerator = 0.0F;
                float weight = 0.0F;
                for (std::size_t index = 0; index < frames.size(); ++index) {
                    const std::ptrdiff_t base_row = numerator_base_rows[index];
                    const float* VNLB_RESTRICT contribution_ptr =
                        contribution_ptrs[index];
                    const int contribution_stride = contribution_strides[index];
                    numerator += contribution_ptr[((base_row + y) *
                                                   contribution_stride) +
                                                  x];
                    weight += contribution_ptr[((base_row + height + y) *
                                                contribution_stride) +
                                               x];
                }
                dst_row[x] = weight > 0.0F ? numerator / weight : src_row[x];
            }
        }
    }

    return dst_guard.release();
}

const VSFrame* VS_CC VAggregateGetFrame(int n, int activationReason,
                                        void* instanceData,
                                        [[maybe_unused]] void** frameData,
                                        VSFrameContext* frameCtx, VSCore* core,
                                        const VSAPI* vsapi) {
    auto* data = static_cast<VAggregateData*>(instanceData);
    try {
        if (activationReason == arInitial) {
            request_aggregate_frames(data, n, frameCtx, vsapi);
        } else if (activationReason == arAllFramesReady) {
            return render_aggregate_frame(n, data, frameCtx, core, vsapi);
        }
    } catch (const std::exception& error) {
        const std::string message = std::string("VNLB: ") + error.what();
        vsapi->setFilterError(message.c_str(), frameCtx);
    }
    return nullptr;
}

void VS_CC VAggregateFree(void* instanceData, [[maybe_unused]] VSCore* core,
                          const VSAPI* vsapi) noexcept {
    auto data = std::unique_ptr<VAggregateData>(
        static_cast<VAggregateData*>(instanceData));
    if (data->src_node != nullptr) {
        vsapi->freeNode(data->src_node);
    }
    if (data->node != nullptr) {
        vsapi->freeNode(data->node);
    }
}

void VS_CC VAggregateCreate(const VSMap* in, VSMap* out,
                            [[maybe_unused]] void* userData, VSCore* core,
                            const VSAPI* vsapi) {
    auto data = std::make_unique<VAggregateData>();
    try {
        int error = peSuccess;
        data->node = vsapi->mapGetNode(in, "clip", 0, &error);
        if (error != peSuccess) {
            throw std::invalid_argument("clip must be a contribution node");
        }
        data->src_node = vsapi->mapGetNode(in, "src", 0, &error);
        if (error != peSuccess) {
            throw std::invalid_argument("src must be a video node");
        }

        const VSVideoInfo* vi = vsapi->getVideoInfo(data->node);
        const VSVideoInfo* src_vi = vsapi->getVideoInfo(data->src_node);
        if (!is_supported_float_format(vi) ||
            !is_supported_float_format(src_vi) ||
            !vsh::isSameVideoFormat(&vi->format, &src_vi->format)) {
            throw std::invalid_argument(
                "Aggregate requires matching constant GrayS or YUV444PS clips");
        }

        const TemporalParameters temporal = parse_temporal_parameters(
            in, vsapi,
            TemporalParameters{data->patch_time, data->search_bwd,
                               data->search_fwd});
        data->patch_time = temporal.patch_time;
        data->search_bwd = temporal.search_bwd;
        data->search_fwd = temporal.search_fwd;
        data->slot_count = checked_slot_count(
            data->search_bwd, data->search_fwd, data->patch_time);
        if (vi->width != src_vi->width ||
            vi->height !=
                checked_contribution_height(src_vi->height, data->slot_count) ||
            vi->numFrames != src_vi->numFrames) {
            throw std::invalid_argument(
                "contribution stack layout does not match src and parameters");
        }

        data->vi = *vi;
        data->src_vi = *src_vi;
        data->out_vi = *src_vi;

        const std::array<VSFilterDependency, 2> deps = {{
            {data->node, rpGeneral},
            {data->src_node, rpStrictSpatial},
        }};
        vsapi->createVideoFilter(out, "VAggregate", &data->out_vi,
                                 VAggregateGetFrame, VAggregateFree, fmParallel,
                                 deps.data(), static_cast<int>(deps.size()),
                                 data.release(), core);
    } catch (const std::exception& error) {
        if (data) {
            VAggregateFree(data.release(), core, vsapi);
        }
        const std::string message = std::string("VNLB: ") + error.what();
        vsapi->mapSetError(out, message.c_str());
    }
}

void VS_CC BasicCreate(const VSMap* in, VSMap* out,
                       [[maybe_unused]] void* userData, VSCore* core,
                       const VSAPI* vsapi) {
    VNLBCreate<Stage::Basic>(in, out, core, vsapi);
}

void VS_CC FinalCreate(const VSMap* in, VSMap* out,
                       [[maybe_unused]] void* userData, VSCore* core,
                       const VSAPI* vsapi) {
    VNLBCreate<Stage::Final>(in, out, core, vsapi);
}

} // namespace

VS_EXTERNAL_API(void)
VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    vspapi->configPlugin(
        kPluginId, kPluginNamespace, "VapourSynth VNLB",
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
        "mvfw:vnode:opt;mvbw:vnode:opt;";
    vspapi->registerFunction("Basic", basic_args, "clip:vnode;", BasicCreate,
                             nullptr, plugin);

    constexpr const char* final_args =
        "clip:vnode;ref:vnode;sigma:float;block_size:int:opt;"
        "block_step:int:opt;group_size:int:opt;bm_range:int:opt;"
        "patch_time:int:opt;radius:int:opt;search_bwd:int:opt;"
        "search_fwd:int:opt;rank:int:opt;cap_factor:float:opt;"
        "model_cap_factor:float:opt;beta:float:opt;tau:float:opt;"
        "variance_threshold:float:opt;sigma_basic:float:opt;"
        "gamma:float:opt;flat_areas:int:opt;weight_alpha:float:opt;"
        "weight_beta:float:opt;weight_gamma:float:opt;"
        "weight_epsilon:float:opt;membership_noise_floor:float:opt;"
        "chroma:int:opt;mvfw:vnode:opt;mvbw:vnode:opt;";
    vspapi->registerFunction("Final", final_args, "clip:vnode;", FinalCreate,
                             nullptr, plugin);

    constexpr const char* aggregate_args =
        "clip:vnode;src:vnode;patch_time:int:opt;radius:int:opt;"
        "search_bwd:int:opt;search_fwd:int:opt;";
    vspapi->registerFunction("Aggregate", aggregate_args, "clip:vnode;",
                             VAggregateCreate, nullptr, plugin);
}
