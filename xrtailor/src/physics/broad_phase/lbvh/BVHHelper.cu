#include <xrtailor/physics/broad_phase/lbvh/BVHHelper.cuh>

#include <xrtailor/physics/broad_phase/Bounds.cuh>
#include <xrtailor/physics/broad_phase/Primitive.cuh>

#include "math_constants.h"

namespace XRTailor {
__host__ __device__ Scalar distance_calculator::operator()(Vector3 p, const Primitive& f,
                                                           Vector3& qs, Scalar& _u, Scalar& _v,
                                                           Scalar& _w) const noexcept {
  const Vector3 a = Vector3(f.v1.x, f.v1.y, f.v1.z);
  const Vector3 b = Vector3(f.v2.x, f.v2.y, f.v2.z);
  const Vector3 c = Vector3(f.v3.x, f.v3.y, f.v3.z);

  Vector3 e1 = b - a;
  Vector3 e2 = c - a;
  Vector3 e3 = c - b;

  // check if p is outside vertex region a
  Vector3 v1 = p - a;
  Scalar d1 = glm::dot(e1, v1), d2 = glm::dot(e2, v1);

  if (d1 <= 0 && d2 <= 0) {
    qs = a;

    _u = 1;
    _v = 0;
    _w = 0;
    return glm::length(p - qs);
  }

  // check if p is outside vertex region b
  Vector3 v2 = p - b;
  Scalar d3 = glm::dot(e1, v2), d4 = glm::dot(e2, v2);
  if (d3 >= 0 && d4 <= d3) {
    qs = b;

    _u = 0;
    _v = 1;
    _w = 0;
    return glm::length(p - qs);
  }

  // check if p is in edge region e1, if so return projection of p onto e1
  Scalar vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    Scalar v = d1 / (d1 - d3);
    qs = a + v * e1;

    _u = 1 - v;
    _v = v;
    _w = 0;
    return glm::length(p - qs);
  }

  // check if p in vertex region outside c
  Vector3 v3 = p - c;
  Scalar d5 = glm::dot(e1, v3), d6 = glm::dot(e2, v3);
  if (d6 >= 0 && d5 <= d6) {
    qs = c;

    _u = 0;
    _v = 0;
    _w = 1;
    return glm::length(p - qs);
  }

  // check if p is in edge region e2, if so return projection of p onto e2
  Scalar vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    Scalar w = d2 / (d2 - d6);
    qs = a + w * e2;

    _u = 1 - w;
    _v = 0;
    _w = w;
    return glm::length(p - qs);
  }

  // check if p is in edge region e3, if so return projection of p onto e3
  Scalar va = d3 * d6 - d5 * d4;
  if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
    Scalar w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    qs = b + w * e3;

    _u = 0;
    _v = 1 - w;
    _w = w;
    return glm::length(p - qs);
  }

  // p inside face region. Compute point through its barycentric coordinates (u,v,w)
  Scalar d = 1 / (va + vb + vc);
  Scalar v = vb * d;
  Scalar w = vc * d;
  qs = a + e1 * v + e2 * w;

  _u = 1 - v - w;
  _v = v;
  _w = w;

  return glm::length(p - qs);
}

__device__ Scalar Infinity() noexcept {
#ifdef XRTAILOR_USE_DOUBLE
  return CUDART_INF;
#else
  return CUDART_INF_F;
#endif  // XRTAILOR_USE_DOUBLE
}

__host__ __device__ Scalar MinDist(const Bounds& lhs, const Vector3& rhs) noexcept {
#ifdef XRTAILOR_USE_DOUBLE
  const Scalar dx = ::fmin(lhs.upper.x, ::fmax(lhs.lower.x, rhs.x)) - rhs.x;
  const Scalar dy = ::fmin(lhs.upper.y, ::fmax(lhs.lower.y, rhs.y)) - rhs.y;
  const Scalar dz = ::fmin(lhs.upper.z, ::fmax(lhs.lower.z, rhs.z)) - rhs.z;
  return dx * dx + dy * dy + dz * dz;
#else
  const Scalar dx = ::fminf(lhs.upper.x, ::fmaxf(lhs.lower.x, rhs.x)) - rhs.x;
  const Scalar dy = ::fminf(lhs.upper.y, ::fmaxf(lhs.lower.y, rhs.y)) - rhs.y;
  const Scalar dz = ::fminf(lhs.upper.z, ::fmaxf(lhs.lower.z, rhs.z)) - rhs.z;
  return dx * dx + dy * dy + dz * dz;
#endif  // XRTAILOR_USE_DOUBLE
}

__host__ __device__ Scalar MinMaxDist(const Bounds& lhs, const Vector3& rhs) noexcept {
  Vector3 rm_sq((lhs.lower.x - rhs.x) * (lhs.lower.x - rhs.x),
                (lhs.lower.y - rhs.y) * (lhs.lower.y - rhs.y),
                (lhs.lower.z - rhs.z) * (lhs.lower.z - rhs.z));
  Vector3 rM_sq((lhs.upper.x - rhs.x) * (lhs.upper.x - rhs.x),
                (lhs.upper.y - rhs.y) * (lhs.upper.y - rhs.y),
                (lhs.upper.z - rhs.z) * (lhs.upper.z - rhs.z));

  if ((lhs.upper.x + lhs.lower.x) * static_cast<Scalar>(0.5) < rhs.x) {
    thrust::swap(rm_sq.x, rM_sq.x);
  }
  if ((lhs.upper.y + lhs.lower.y) * static_cast<Scalar>(0.5) < rhs.y) {
    thrust::swap(rm_sq.y, rM_sq.y);
  }
  if ((lhs.upper.z + lhs.lower.z) * static_cast<Scalar>(0.5) < rhs.z) {
    thrust::swap(rm_sq.z, rM_sq.z);
  }

  const Scalar dx = rm_sq.x + rM_sq.y + rM_sq.z;
  const Scalar dy = rM_sq.x + rm_sq.y + rM_sq.z;
  const Scalar dz = rM_sq.x + rM_sq.y + rm_sq.z;

  return MathFunctions::min(dx, MathFunctions::min(dy, dz));
}

__device__ float AtomicMinFloat(float* address, float val) {
  int* address_as_ull = (int*)address;
  int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old =
        ::atomicCAS(address_as_ull, assumed, __float_as_int(::fminf(val, __int_as_float(assumed))));
  } while (assumed != old);

  return __int_as_float(old);
}

__device__ double AtomicMinDouble(double* address, double val) {
  unsigned long long* address_as_ull = (unsigned long long*)address;
  unsigned long long old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = ::atomicCAS(address_as_ull, assumed,
                      __double_as_longlong(::fmin(val, __longlong_as_double(assumed))));
  } while (assumed != old);

  return __longlong_as_double(old);
}

__device__ float AtomicMaxFloat(float* address, float val) {
  int* address_as_ull = (int*)address;
  int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old =
        ::atomicCAS(address_as_ull, assumed, __float_as_int(::fmaxf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}

__device__ double AtomicMaxDouble(double* address, double val) {
  unsigned long long* address_as_ull = (unsigned long long*)address;
  unsigned long long old = *address_as_ull, assumed;

  do {
    assumed = old;
    // Use fmax() for double precision and __double_as_longlong() and __longlong_as_double() for conversions
    old = ::atomicCAS(address_as_ull, assumed,
                      __double_as_longlong(::fmax(val, __longlong_as_double(assumed))));
  } while (assumed != old);

  return __longlong_as_double(old);
}

__device__ Scalar AtomicMin(Scalar* address, Scalar val) {
#ifdef XRTAILOR_USE_DOUBLE
  return AtomicMinDouble(address, val);
#else
  return AtomicMinFloat(address, val);
#endif  // XRTAILOR_USE_DOUBLE
}

__device__ Scalar AtomicMax(Scalar* address, Scalar val) {
#ifdef XRTAILOR_USE_DOUBLE
  return AtomicMaxDouble(address, val);
#else
  return AtomicMaxFloat(address, val);
#endif  // XRTAILOR_USE_DOUBLE
}

}  // namespace XRTailor