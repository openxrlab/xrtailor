#pragma once

#include <vector_types.h>

#include <xrtailor/memory/Face.cuh>

namespace XRTailor {

class Proximity {
 public:
  __host__ __device__ Proximity();

  __host__ __device__ Proximity(Node* node, Face* face, Scalar stiffness);

  __host__ __device__ Proximity(Edge* edge0, Edge* edge1, Scalar stiffness);

  ~Proximity() = default;

  bool is_fv;
  Node* nodes[4];
  Scalar stiffness;
  Vector3 n;
};

struct ProximityIsNull {
  __host__ __device__ bool operator()(Proximity prox);
};

class VFProximity {
 public:
  __host__ __device__ VFProximity();

  ~VFProximity() = default;

  __host__ __device__ bool operator<(const VFProximity& p) const;

  Node* node;
  Face* face;
  Scalar d;
  Vector3 qs;
};

class EEProximity {
 public:
  __host__ __device__ EEProximity();

  ~EEProximity() = default;

  Edge* edge;
  Scalar dist;
  Vector3 p2;
  Vector3 q2;
  Scalar s;
  Scalar t;
  Vector3 fn;
};

class RTProximity {
 public:
  __host__ __device__ RTProximity();

  ~RTProximity() = default;

  Node* node;
  Face* face;
  Vector3 qc;
};

}  // namespace XRTailor