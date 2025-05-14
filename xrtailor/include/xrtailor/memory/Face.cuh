#pragma once

#include <cuda_runtime.h>

#include <xrtailor/memory/Edge.cuh>

namespace XRTailor {

class Face {
 public:
  __host__ __device__ Face();

  __host__ __device__ Face(const Node* v0, const Node* v1, const Node* v2);

  ~Face() = default;

  __host__ __device__ void SetEdge(const Edge* edge);

  __host__ __device__ void SetEdges(const Edge* edge0, const Edge* edge1, const Edge* edge2);

  __host__ __device__ bool IsFree() const;

  __host__ __device__ bool Contain(const Node* node) const;

  __host__ __device__ bool Contain(const Edge* edge) const;

  __host__ __device__ bool Adjacent(const Face* face) const;

  __host__ __device__ void Update();

  __host__ __device__ Bounds ComputeBounds(bool ccd) const;

  __host__ __device__ Node* OppositeNode(Node* node0, Node* node1) const;

  Node* nodes[3];
  Edge* edges[3];
  Vector3 n;
  uint r_tri;
  int type = -1;
  int index = -1;
};

}  // namespace XRTailor