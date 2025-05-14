#pragma once

#include <xrtailor/core/Scalar.hpp>
#include <vector_types.h>

namespace XRTailor {
#ifdef XRTAILOR_USE_DOUBLE
__device__ constexpr double SCALAR_MAX = DBL_MAX;
#else
__device__ constexpr float SCALAR_MAX = FLT_MAX;
#endif
}  // namespace XRTailor