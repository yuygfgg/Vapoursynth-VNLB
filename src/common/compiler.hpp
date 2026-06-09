#pragma once

#if defined(VS_RESTRICT)
#define VNLB_RESTRICT VS_RESTRICT
#elif defined(__clang__) || defined(__GNUC__)
#define VNLB_RESTRICT __restrict__
#else
#define VNLB_RESTRICT
#endif
