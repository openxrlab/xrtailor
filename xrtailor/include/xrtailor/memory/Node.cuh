#pragma once

#include <xrtailor/physics/broad_phase/Bounds.cuh>

namespace XRTailor {

class Node {
 public:
  int index, min_index, color;
  Vector3 x0, x1, x, n, n1, v;
  Scalar area, inv_mass;
  bool is_free, removed, is_cloth;

  __host__ __device__ Node();

  __host__ __device__ Node(const Vector3& x, bool is_free, bool is_cloth);

  ~Node() = default;

  __host__ __device__ Bounds ComputeBounds(bool ccd) const;

  __host__ __device__ Vector3 Position(Scalar t) const;
};

}  // namespace XRTailor