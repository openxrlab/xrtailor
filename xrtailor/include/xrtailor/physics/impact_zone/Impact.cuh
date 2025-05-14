#pragma once

#include <xrtailor/memory/Node.cuh>
#include <xrtailor/memory/Pair.cuh>

namespace XRTailor {

class Impact {
 public:
  Node* nodes[4];
  Scalar t, w[4];
  Vector3 n;
  PairFF pair;  // XXX: debug
  bool is_vf;   // XXX: debug
  __host__ __device__ Impact();
  ~Impact() = default;
  __host__ __device__ bool operator<(const Impact& impact) const;
};

class ImpactDebug {
 public:
  int indices[4];
  Scalar t, w[4];
  Vector3 n;
  bool is_vf;
  int f1_idx, f2_idx;
  __host__ __device__ ImpactDebug();
  ~ImpactDebug() = default;
};

}  // namespace XRTailor