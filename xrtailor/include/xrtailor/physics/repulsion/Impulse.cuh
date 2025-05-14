#pragma once

#include <xrtailor/core/DeviceHelper.cuh>
#include <xrtailor/memory/Node.cuh>

namespace XRTailor {

struct Impulse {
  Node* nodes[4];
  Vector3 corrs[4];
};

struct Quadrature {
  Node* nodes[4];
  Scalar bary[4];
};

struct QuadInd {
  int ids[4];
  int types[4];
};

struct QuadIsNull {
  __host__ __device__ bool operator()(Quadrature quad) const;
};

}  // namespace XRTailor