#pragma once

#include <xrtailor/core/DeviceHelper.cuh>
#include <xrtailor/memory/Pair.cuh>

namespace XRTailor {
namespace Untangling {

struct EFIntersection {
  PairEF pair;
  Vector3 p;
  Scalar s;
  Vector3 G;
};

struct EFIntersectionIsNull {
  __host__ __device__ bool operator()(EFIntersection intersection);
};

struct IntersectionWithGradient {
  int ev0_idx{-1};
  int ev1_idx{-1};
  int f_idx{-1};
  Vector3 p{static_cast<Scalar>(0)};
  Vector3 ev0, ev1;
  Vector3 v0, v1, v2;
  Vector3 G;
};

struct IntersectionWithGradientIsNull {
  __host__ __device__ bool operator()(IntersectionWithGradient intersection);
};

}  // namespace Untangling
}  // namespace XRTailor