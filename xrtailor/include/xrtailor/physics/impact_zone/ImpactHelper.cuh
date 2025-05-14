#pragma once

#include "device_types.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <xrtailor/memory/Pair.cuh>
#include <xrtailor/memory/Face.cuh>
#include <xrtailor/physics/impact_zone/Impact.cuh>

class BVH;

namespace XRTailor {

enum ImpactType { VertexFace, EdgeEdge };

struct ImpactDbg {
  int idx0;
  int idx1;
  int idx2;
  int idx3;

  int eval0;
  int eval1;
  int eval2;
  int eval3;

  Vector3 p0;
  Vector3 p1;
  Vector3 p2;
  Vector3 p3;

  Vector3 q0;
  Vector3 q1;
  Vector3 q2;
  Vector3 q3;

  int n_roots;
};

const int MAX_COLLISION_ITERATION = 100;

__host__ __device__ bool CheckImpact(ImpactType type, const Node* node0, const Node* node1,
                                     const Node* node2, const Node* node3, Impact& impact);

__host__ __device__ bool CheckVertexFaceImpact(const Node* node, const Face* face, Scalar thickness,
                                               Impact& impact);

__host__ __device__ bool CheckEdgeEdgeImpact(const Edge* edge0, const Edge* edge1, Scalar thickness,
                                             Impact& impact);

void CheckImpacts(const PairFF* pairs, int nPairs, Scalar thickness, Impact* impacts);

thrust::device_vector<Impact> FindImpacts(std::shared_ptr<BVH> cloth_bvh,
                                          std::shared_ptr<BVH> obstacle_bvh, Face** faces_cloth,
                                          Face** faces_obstacle, Scalar thickness, int frame_index,
                                          int iteration);

/**
    * @brief Setup nodes relatived to the impacts
*/
void InitializeImpactNodes(int n_impacts, const Impact* impacts, int deform);

void CollectRelativeImpacts(int n_impacts, const Impact* impacts, int deform, Node** nodes,
                            int* relative_impacts);

void SetImpactMinIndices(int n_nodes, const int* relative_impacts, Node** nodes);

void CheckIndependentImpacts(int n_impacts, const Impact* impacts, int deform,
                             Impact* independent_impacts);

__host__ __device__ Bounds FaceBounds(Node* node0, Node* node1, Node* node2, bool ccd);

}  // namespace XRTailor