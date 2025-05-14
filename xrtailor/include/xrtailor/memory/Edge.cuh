#pragma once

#include <xrtailor/memory/Node.cuh>

namespace XRTailor {

#define MAX_EF_ADJACENTS 4

class Face;

class Edge {
 public:  
  __host__ __device__ Edge();

  __host__ __device__ Edge(const Node* node0, const Node* node1);

  ~Edge() = default;

  __host__ __device__ void ReplaceNode(const Node* src_node, const Node* tgt_node);

  __host__ __device__ bool IsFree() const;
  
  __host__ __device__ Scalar Length() const;
  
  __host__ __device__ Bounds ComputeBounds(bool ccd) const;
  
  __host__ __device__ Vector3 ComputeNormal() const;

  Node* nodes[2];
  Face* adjacents[MAX_EF_ADJACENTS];
  int index;
};

}  // namespace XRTailor