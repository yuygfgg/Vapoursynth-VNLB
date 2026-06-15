#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace vnlb::common {

[[nodiscard]] constexpr std::size_t
checked_mul_size(std::size_t lhs, std::size_t rhs,
                 const char* message = "size calculation overflow") {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        throw std::length_error(message);
    }
    return lhs * rhs;
}

[[nodiscard]] constexpr int
checked_add_int(int lhs, int rhs,
                const char* message = "integer calculation overflow") {
    const auto result =
        static_cast<long long>(lhs) + static_cast<long long>(rhs);
    if (result > static_cast<long long>(std::numeric_limits<int>::max()) ||
        result < static_cast<long long>(std::numeric_limits<int>::min())) {
        throw std::length_error(message);
    }
    return static_cast<int>(result);
}

[[nodiscard]] constexpr int
checked_mul_int(int lhs, int rhs,
                const char* message = "integer calculation overflow") {
    const auto result =
        static_cast<long long>(lhs) * static_cast<long long>(rhs);
    if (result > static_cast<long long>(std::numeric_limits<int>::max()) ||
        result < static_cast<long long>(std::numeric_limits<int>::min())) {
        throw std::length_error(message);
    }
    return static_cast<int>(result);
}

} // namespace vnlb::common
