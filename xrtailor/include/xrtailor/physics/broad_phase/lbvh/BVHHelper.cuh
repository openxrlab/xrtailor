#pragma once

#include <vector_types.h>
#include <cuda_runtime.h>

#include <xrtailor/core/Scalar.hpp>

namespace XRTailor {

class Bounds;
class Primitive;

struct distance_calculator {
  __host__ __device__ Scalar operator()(Vector3 p, const Primitive& f, Vector3& qs, Scalar& _u,
                                        Scalar& _v, Scalar& _w) const noexcept;
};

__device__ Scalar Infinity() noexcept;

// metrics defined in
// Nearest Neighbor Queries (1995) ACS-SIGMOD
// - Nick Roussopoulos, Stephen Kelley FredericVincent
__host__ __device__ Scalar MinDist(const Bounds& lhs, const Vector3& rhs) noexcept;

__host__ __device__ Scalar MinMaxDist(const Bounds& lhs, const Vector3& rhs) noexcept;

__device__ Scalar AtomicMin(Scalar* address, Scalar val);

__device__ Scalar AtomicMax(Scalar* address, Scalar val);
}  // namespace XRTailor