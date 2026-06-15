#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace vnlb::flow {

struct SearchCenter {
    int x = 0;
    int y = 0;
};

struct SearchCenterRequest {
    int anchor_x = 0;
    int anchor_y = 0;
    int anchor_frame = 0;
    int candidate_frame = 0;
    int patch_size = 1;
    int frame_width = 0;
    int frame_height = 0;
};

class SameLocationProvider {
  public:
    [[nodiscard]] constexpr SearchCenter
    center_for(SearchCenterRequest request) const noexcept {
        return SearchCenter{request.anchor_x, request.anchor_y};
    }
};

struct MVToolsAnalysisData {
    int block_size_x = 0;
    int block_size_y = 0;
    int pel = 1;
    int level_count = 0;
    int delta_frame = 1;
    bool backwards = false;
    int width = 0;
    int height = 0;
    int overlap_x = 0;
    int overlap_y = 0;
    int block_count_x = 0;
    int block_count_y = 0;
    int bits_per_sample = 0;
    int chroma_ratio_x = 0;
    int chroma_ratio_y = 0;
    int padding_x = 0;
    int padding_y = 0;
};

struct MVToolsVector {
    int x = 0;
    int y = 0;
    std::uint64_t sad = 0;
};

class MVToolsVectorGrid {
  public:
    MVToolsAnalysisData analysis{};
    bool valid = false;
    std::vector<MVToolsVector> vectors;

    [[nodiscard]] bool usable() const noexcept {
        return valid && analysis.pel > 0 && analysis.block_size_x > 0 &&
               analysis.block_size_y > 0 && analysis.block_count_x > 0 &&
               analysis.block_count_y > 0 && !vectors.empty();
    }

    [[nodiscard]] MVToolsVector nearest_vector(int x, int y) const noexcept {
        if (!usable()) {
            return MVToolsVector{};
        }

        const int step_x = std::max(1, analysis.block_size_x -
                                           std::max(0, analysis.overlap_x));
        const int step_y = std::max(1, analysis.block_size_y -
                                           std::max(0, analysis.overlap_y));
        const int block_x = nearest_block_index(x, analysis.block_size_x / 2,
                                                step_x, analysis.block_count_x);
        const int block_y = nearest_block_index(y, analysis.block_size_y / 2,
                                                step_y, analysis.block_count_y);
        const std::size_t index =
            static_cast<std::size_t>(block_y) *
                static_cast<std::size_t>(analysis.block_count_x) +
            static_cast<std::size_t>(block_x);
        if (index >= vectors.size()) {
            return MVToolsVector{};
        }
        return vectors[index];
    }

  private:
    [[nodiscard]] static int nearest_block_index(int coordinate,
                                                 int first_center, int step,
                                                 int count) noexcept {
        if (count <= 1) {
            return 0;
        }

        const int shifted = coordinate - first_center;
        int index = 0;
        if (shifted >= 0) {
            index = (shifted + step / 2) / step;
        } else {
            index = -((-shifted + step / 2) / step);
        }
        return std::clamp(index, 0, count - 1);
    }
};

namespace detail {

[[nodiscard]] inline std::uint32_t read_u32(std::span<const std::byte> data,
                                            std::size_t offset) {
    if (offset + sizeof(std::uint32_t) > data.size()) {
        throw std::invalid_argument("truncated MVTools data");
    }

    std::uint32_t value = 0;
    auto* bytes = reinterpret_cast<unsigned char*>(&value);
    for (std::size_t byte = 0; byte < sizeof(std::uint32_t); ++byte) {
        bytes[byte] = static_cast<unsigned char>(data[offset + byte]);
    }
    if constexpr (std::endian::native == std::endian::big) {
        value = ((value & 0x000000FFU) << 24) | ((value & 0x0000FF00U) << 8) |
                ((value & 0x00FF0000U) >> 8) | ((value & 0xFF000000U) >> 24);
    }
    return value;
}

[[nodiscard]] inline int read_i32(std::span<const std::byte> data,
                                  std::size_t offset) {
    return static_cast<int>(
        std::bit_cast<std::int32_t>(read_u32(data, offset)));
}

[[nodiscard]] inline std::uint64_t read_u64(std::span<const std::byte> data,
                                            std::size_t offset) {
    if (offset + sizeof(std::uint64_t) > data.size()) {
        throw std::invalid_argument("truncated MVTools data");
    }

    std::uint64_t value = 0;
    auto* bytes = reinterpret_cast<unsigned char*>(&value);
    for (std::size_t byte = 0; byte < sizeof(std::uint64_t); ++byte) {
        bytes[byte] = static_cast<unsigned char>(data[offset + byte]);
    }
    if constexpr (std::endian::native == std::endian::big) {
        value = ((value & 0x00000000000000FFULL) << 56) |
                ((value & 0x000000000000FF00ULL) << 40) |
                ((value & 0x0000000000FF0000ULL) << 24) |
                ((value & 0x00000000FF000000ULL) << 8) |
                ((value & 0x000000FF00000000ULL) >> 8) |
                ((value & 0x0000FF0000000000ULL) >> 24) |
                ((value & 0x00FF000000000000ULL) >> 40) |
                ((value & 0xFF00000000000000ULL) >> 56);
    }
    return value;
}

[[nodiscard]] inline int checked_positive_u32(std::uint32_t value,
                                              const char* field) {
    if (value == 0 ||
        value > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(field);
    }
    return static_cast<int>(value);
}

[[nodiscard]] inline int checked_nonnegative_u32(std::uint32_t value,
                                                 const char* field) {
    if (value > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(field);
    }
    return static_cast<int>(value);
}

[[nodiscard]] inline int round_subpixel_to_pixel(int value, int pel) noexcept {
    if (pel <= 1) {
        return value;
    }
    if (value >= 0) {
        return (value + pel / 2) / pel;
    }
    return -((-value + pel / 2) / pel);
}

} // namespace detail

[[nodiscard]] inline MVToolsAnalysisData
parse_mvtools_analysis(std::span<const std::byte> data) {
    if (data.size() != 21 * sizeof(std::uint32_t)) {
        throw std::invalid_argument(
            "MVTools_MVAnalysisData must contain 21 32-bit fields");
    }

    MVToolsAnalysisData analysis{};
    analysis.block_size_x = detail::checked_positive_u32(
        detail::read_u32(data, 2 * sizeof(std::uint32_t)),
        "invalid MVTools horizontal block size");
    analysis.block_size_y = detail::checked_positive_u32(
        detail::read_u32(data, 3 * sizeof(std::uint32_t)),
        "invalid MVTools vertical block size");
    analysis.pel = detail::checked_positive_u32(
        detail::read_u32(data, 4 * sizeof(std::uint32_t)),
        "invalid MVTools pel");
    analysis.level_count = detail::checked_positive_u32(
        detail::read_u32(data, 5 * sizeof(std::uint32_t)),
        "invalid MVTools level count");
    analysis.delta_frame = detail::read_i32(data, 6 * sizeof(std::uint32_t));
    analysis.backwards = detail::read_u32(data, 7 * sizeof(std::uint32_t)) != 0;
    analysis.width = detail::checked_positive_u32(
        detail::read_u32(data, 10 * sizeof(std::uint32_t)),
        "invalid MVTools width");
    analysis.height = detail::checked_positive_u32(
        detail::read_u32(data, 11 * sizeof(std::uint32_t)),
        "invalid MVTools height");
    analysis.overlap_x = detail::checked_nonnegative_u32(
        detail::read_u32(data, 12 * sizeof(std::uint32_t)),
        "invalid MVTools horizontal overlap");
    analysis.overlap_y = detail::checked_nonnegative_u32(
        detail::read_u32(data, 13 * sizeof(std::uint32_t)),
        "invalid MVTools vertical overlap");
    analysis.block_count_x = detail::checked_positive_u32(
        detail::read_u32(data, 14 * sizeof(std::uint32_t)),
        "invalid MVTools horizontal block count");
    analysis.block_count_y = detail::checked_positive_u32(
        detail::read_u32(data, 15 * sizeof(std::uint32_t)),
        "invalid MVTools vertical block count");
    analysis.bits_per_sample = detail::checked_positive_u32(
        detail::read_u32(data, 16 * sizeof(std::uint32_t)),
        "invalid MVTools bit depth");
    analysis.chroma_ratio_y = detail::checked_positive_u32(
        detail::read_u32(data, 17 * sizeof(std::uint32_t)),
        "invalid MVTools vertical chroma ratio");
    analysis.chroma_ratio_x = detail::checked_positive_u32(
        detail::read_u32(data, 18 * sizeof(std::uint32_t)),
        "invalid MVTools horizontal chroma ratio");
    analysis.padding_x = detail::checked_nonnegative_u32(
        detail::read_u32(data, 19 * sizeof(std::uint32_t)),
        "invalid MVTools horizontal padding");
    analysis.padding_y = detail::checked_nonnegative_u32(
        detail::read_u32(data, 20 * sizeof(std::uint32_t)),
        "invalid MVTools vertical padding");

    if (analysis.overlap_x >= analysis.block_size_x ||
        analysis.overlap_y >= analysis.block_size_y) {
        throw std::invalid_argument(
            "MVTools overlap must be smaller than block size");
    }
    return analysis;
}

inline void parse_mvtools_vectors(std::span<const std::byte> data,
                                  MVToolsAnalysisData analysis,
                                  MVToolsVectorGrid& grid) {
    if (data.size() < 2 * sizeof(std::uint32_t)) {
        throw std::invalid_argument("MVTools_vectors is too small");
    }

    const std::uint32_t declared_size = detail::read_u32(data, 0);
    if (declared_size != data.size()) {
        throw std::invalid_argument("MVTools_vectors size header is invalid");
    }

    grid.analysis = analysis;
    grid.valid = detail::read_u32(data, sizeof(std::uint32_t)) == 1;
    grid.vectors.clear();
    if (!grid.valid) {
        return;
    }

    std::size_t position = 2 * sizeof(std::uint32_t);
    std::size_t finest_begin = 0;
    std::size_t finest_end = 0;
    while (position < data.size()) {
        const std::size_t level_begin = position;
        if (level_begin + sizeof(std::uint32_t) > data.size()) {
            throw std::invalid_argument("truncated MVTools level header");
        }
        const std::uint32_t level_size = detail::read_u32(data, level_begin);
        if (level_size < sizeof(std::uint32_t) ||
            level_begin + level_size > data.size()) {
            throw std::invalid_argument("invalid MVTools level size");
        }
        finest_begin = level_begin + sizeof(std::uint32_t);
        finest_end = level_begin + level_size;
        position = finest_end;
    }

    const std::size_t finest_size = finest_end - finest_begin;
    if (finest_size % 16 != 0) {
        throw std::invalid_argument(
            "MVTools finest level has invalid vector size");
    }
    const std::size_t vector_count = finest_size / 16;
    const std::size_t expected_count =
        static_cast<std::size_t>(analysis.block_count_x) *
        static_cast<std::size_t>(analysis.block_count_y);
    if (vector_count != expected_count) {
        throw std::invalid_argument(
            "MVTools finest level vector count does not match analysis data");
    }

    grid.vectors.resize(vector_count);
    std::size_t offset = finest_begin;
    for (MVToolsVector& vector : grid.vectors) {
        vector.x = detail::read_i32(data, offset);
        vector.y = detail::read_i32(data, offset + sizeof(std::uint32_t));
        vector.sad = detail::read_u64(data, offset + 2 * sizeof(std::uint32_t));
        offset += 16;
    }
}

class MVToolsFlowProvider {
  public:
    constexpr MVToolsFlowProvider() noexcept = default;
    constexpr MVToolsFlowProvider(std::span<const MVToolsVectorGrid> previous,
                                  int previous_first_frame,
                                  std::span<const MVToolsVectorGrid> next,
                                  int next_first_frame) noexcept
        : previous_(previous), next_(next),
          previous_first_frame_(previous_first_frame),
          next_first_frame_(next_first_frame) {}
    constexpr MVToolsFlowProvider(
        std::span<const MVToolsVectorGrid* const> previous,
        int previous_first_frame,
        std::span<const MVToolsVectorGrid* const> next,
        int next_first_frame) noexcept
        : previous_ptrs_(previous), next_ptrs_(next),
          previous_first_frame_(previous_first_frame),
          next_first_frame_(next_first_frame) {}

    [[nodiscard]] SearchCenter
    center_for(SearchCenterRequest request) const noexcept {
        SearchCenter center{request.anchor_x, request.anchor_y};
        if (request.candidate_frame == request.anchor_frame) {
            return center;
        }

        const auto clamp_center = [&]() noexcept {
            if (request.frame_width > 0) {
                center.x = std::clamp(center.x, 0, request.frame_width - 1);
            }
            if (request.frame_height > 0) {
                center.y = std::clamp(center.y, 0, request.frame_height - 1);
            }
        };

        const int patch_center = std::max(0, request.patch_size / 2);
        if (request.candidate_frame < request.anchor_frame) {
            for (int frame = request.anchor_frame;
                 frame > request.candidate_frame; --frame) {
                const MVToolsVectorGrid* grid = previous_grid(frame);
                if (grid == nullptr || !grid->usable()) {
                    continue;
                }
                const MVToolsVector vector = grid->nearest_vector(
                    center.x + patch_center, center.y + patch_center);
                center.x += detail::round_subpixel_to_pixel(vector.x,
                                                            grid->analysis.pel);
                center.y += detail::round_subpixel_to_pixel(vector.y,
                                                            grid->analysis.pel);
                clamp_center();
            }
        } else {
            for (int frame = request.anchor_frame;
                 frame < request.candidate_frame; ++frame) {
                const MVToolsVectorGrid* grid = next_grid(frame);
                if (grid == nullptr || !grid->usable()) {
                    continue;
                }
                const MVToolsVector vector = grid->nearest_vector(
                    center.x + patch_center, center.y + patch_center);
                center.x += detail::round_subpixel_to_pixel(vector.x,
                                                            grid->analysis.pel);
                center.y += detail::round_subpixel_to_pixel(vector.y,
                                                            grid->analysis.pel);
                clamp_center();
            }
        }
        return center;
    }

  private:
    [[nodiscard]] const MVToolsVectorGrid*
    previous_grid(int frame) const noexcept {
        const int local = frame - previous_first_frame_;
        if (!previous_ptrs_.empty()) {
            if (local < 0 || local >= static_cast<int>(previous_ptrs_.size())) {
                return nullptr;
            }
            return previous_ptrs_[static_cast<std::size_t>(local)];
        }
        if (local < 0 || local >= static_cast<int>(previous_.size())) {
            return nullptr;
        }
        return &previous_[static_cast<std::size_t>(local)];
    }

    [[nodiscard]] const MVToolsVectorGrid* next_grid(int frame) const noexcept {
        const int local = frame - next_first_frame_;
        if (!next_ptrs_.empty()) {
            if (local < 0 || local >= static_cast<int>(next_ptrs_.size())) {
                return nullptr;
            }
            return next_ptrs_[static_cast<std::size_t>(local)];
        }
        if (local < 0 || local >= static_cast<int>(next_.size())) {
            return nullptr;
        }
        return &next_[static_cast<std::size_t>(local)];
    }

    std::span<const MVToolsVectorGrid> previous_{};
    std::span<const MVToolsVectorGrid> next_{};
    std::span<const MVToolsVectorGrid* const> previous_ptrs_{};
    std::span<const MVToolsVectorGrid* const> next_ptrs_{};
    int previous_first_frame_ = 0;
    int next_first_frame_ = 0;
};

} // namespace vnlb::flow
