#include <xrtailor/physics/impact_zone/ImpactHelper.cuh>

#include <iostream>
#include <fstream>
#include <iomanip>

#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/count.h>

#include <xrtailor/physics/impact_zone/Defs.hpp>
#include <xrtailor/math/MathFunctions.cuh>
#include <xrtailor/core/DeviceHelper.cuh>
#include <xrtailor/physics/broad_phase/lbvh/BVH.cuh>
#include <xrtailor/math/BasicPrimitiveTests.cuh>

namespace XRTailor {

__host__ __device__ Bounds FaceBounds(Node* node0, Node* node1, Node* node2, bool ccd) {
  Bounds ans;
  ans += node0->x;
  ans += node1->x;
  ans += node2->x;

  if (ccd) {
    ans += node0->x0;
    ans += node1->x0;
    ans += node2->x0;
  }

  return ans;
}

__host__ __device__ bool CheckImpact(ImpactType type, const Node* node0, const Node* node1,
                                     const Node* node2, const Node* node3, Impact& impact) {
  impact.nodes[0] = const_cast<Node*>(node0);
  impact.nodes[1] = const_cast<Node*>(node1);
  impact.nodes[2] = const_cast<Node*>(node2);
  impact.nodes[3] = const_cast<Node*>(node3);

  Vector3 x0 = node0->x0;
  Vector3 v0 = node0->x - x0;
  Vector3 x1 = node1->x0 - x0;
  Vector3 x2 = node2->x0 - x0;
  Vector3 x3 = node3->x0 - x0;
  Vector3 v1 = (node1->x - node1->x0) - v0;
  Vector3 v2 = (node2->x - node2->x0) - v0;
  Vector3 v3 = (node3->x - node3->x0) - v0;
  Scalar a0 = MathFunctions::Mixed(x1, x2, x3);
  Scalar a1 = MathFunctions::Mixed(v1, x2, x3) + MathFunctions::Mixed(x1, v2, x3) +
              MathFunctions::Mixed(x1, x2, v3);
  Scalar a2 = MathFunctions::Mixed(x1, v2, v3) + MathFunctions::Mixed(v1, x2, v3) +
              MathFunctions::Mixed(v1, v2, x3);
  Scalar a3 = MathFunctions::Mixed(v1, v2, v3);

  Scalar t[3];
  int n_solution = MathFunctions::SolveCubic(a3, a2, a1, a0, t);

  for (int i = 0; i < n_solution; i++) {
    if (t[i] < 0 || t[i] > 1)
      continue;
    impact.t = t[i];
    Vector3 x0 = node0->Position(t[i]);
    Vector3 x1 = node1->Position(t[i]);
    Vector3 x2 = node2->Position(t[i]);
    Vector3 x3 = node3->Position(t[i]);

    Vector3& n = impact.n;
    Scalar* w = impact.w;
    Scalar d;
    bool inside;
    if (type == VertexFace) {
      d = BasicPrimitiveTests::SignedVertexFaceDistance(x0, x1, x2, x3, n, w);
      inside = (MathFunctions::min(-w[1], -w[2], -w[3]) >= static_cast<Scalar>(-1e-6));
    } else {
      d = BasicPrimitiveTests::SignedEdgeEdgeDistance(x0, x1, x2, x3, n, w);
      inside = (MathFunctions::min(w[0], w[1], -w[2], -w[3]) >= static_cast<Scalar>(-1e-6));
    }

    if (glm::dot(n, w[1] * v1 + w[2] * v2 + w[3] * v3) > 0)
      n = -n;
    if (MathFunctions::abs(d) < static_cast<Scalar>(1e-6) && inside) {
      return true;
    }
  }
  return false;
}

bool CheckVertexFaceImpact(const Node* node, const Face* face, Scalar thickness, Impact& impact) {
  Node* node0 = face->nodes[0];
  Node* node1 = face->nodes[1];
  Node* node2 = face->nodes[2];

  if (node == node0 || node == node1 || node == node2)
    return false;
  if (!node->ComputeBounds(true).Overlap(face->ComputeBounds(true), thickness))
    return false;

  return CheckImpact(VertexFace, node, node0, node1, node2, impact);
}

bool CheckEdgeEdgeImpact(const Edge* edge0, const Edge* edge1, Scalar thickness, Impact& impact) {
  Node* node0 = edge0->nodes[0];
  Node* node1 = edge0->nodes[1];
  Node* node2 = edge1->nodes[0];
  Node* node3 = edge1->nodes[1];
  if (node0 == node2 || node0 == node3 || node1 == node2 || node1 == node3)
    return false;

  if (!edge0->ComputeBounds(true).Overlap(edge1->ComputeBounds(true), thickness))
    return false;

  return CheckImpact(EdgeEdge, node0, node1, node2, node3, impact);
}

__global__ void CheckImpacts_Kernel(int nPairs, const PairFF* pairs, Scalar thickness,
                                    Impact* impacts) {
  GET_CUDA_ID(idx, nPairs);

  const PairFF& pair = pairs[idx];

  Face* face0 = pair.first;
  Face* face1 = pair.second;

  int index = 15 * idx;

  for (int i = 0; i < 3; i++) {
    Impact& impact = impacts[index++];
    impact.is_vf = true;
    impact.nodes[0] = face0->nodes[i];
    impact.nodes[1] = face1->nodes[0];
    impact.nodes[2] = face1->nodes[1];
    impact.nodes[3] = face1->nodes[2];
    impact.w[0] = 0;
    impact.w[1] = 0;
    impact.w[2] = 0;
    impact.w[3] = 0;
    if (!CheckVertexFaceImpact(face0->nodes[i], face1, thickness, impact))
      impact.t = -1;
  }
  for (int i = 0; i < 3; i++) {
    Impact& impact = impacts[index++];
    impact.is_vf = true;
    impact.nodes[0] = face1->nodes[i];
    impact.nodes[1] = face0->nodes[0];
    impact.nodes[2] = face0->nodes[1];
    impact.nodes[3] = face0->nodes[2];
    impact.w[0] = 0;
    impact.w[1] = 0;
    impact.w[2] = 0;
    impact.w[3] = 0;
    if (!CheckVertexFaceImpact(face1->nodes[i], face0, thickness, impact))
      impact.t = -1;
  }
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      Impact& impact = impacts[index++];
      impact.is_vf = false;
      impact.nodes[0] = face0->edges[i]->nodes[0];
      impact.nodes[1] = face0->edges[i]->nodes[1];
      impact.nodes[2] = face1->edges[j]->nodes[0];
      impact.nodes[3] = face1->edges[j]->nodes[1];
      impact.w[0] = 0;
      impact.w[1] = 0;
      impact.w[2] = 0;
      impact.w[3] = 0;
      if (!CheckEdgeEdgeImpact(face0->edges[i], face1->edges[j], thickness, impact)) {
        impact.t = -1;
      }
    }
}

void CheckImpacts(const PairFF* pairs, int n_pairs, Scalar thickness, Impact* impacts) {
  CUDA_CALL(CheckImpacts_Kernel, n_pairs)
  (n_pairs, pairs, thickness, impacts);
  CUDA_CHECK_LAST();
}

__global__ void ClearColors_Kernel(Face** faces, int n_faces) {
  GET_CUDA_ID(i, n_faces);
  for (int j = 0; j < 3; j++)
    faces[i]->nodes[j]->color = 0;
}

__global__ void ClearObstacleColors_Kernel(Face** faces, int n_faces) {
  GET_CUDA_ID(i, n_faces);

  for (int j = 0; j < 3; j++)
    faces[i]->nodes[j]->color = 0;
}

__global__ void SetupColor_Kernel(PairFF* pairs, int n_pairs) {
  GET_CUDA_ID(i, n_pairs);

  Face* f1 = pairs[i].first;
  Face* f2 = pairs[i].second;

  bool flag = false;

  for (int j = 0; j < 3; j++) {
    if (!f1->nodes[j]->is_free || !f2->nodes[j]->is_free) {
      flag = true;
      break;
    }
  }

  if (!flag)
    return;

  for (int j = 0; j < 3; j++) {
    f1->nodes[j]->color = (f1->nodes[j]->is_free) ? 2 : 3;
    f2->nodes[j]->color = (f2->nodes[j]->is_free) ? 2 : 3;
  }
}

thrust::device_vector<Impact> FindImpacts(std::shared_ptr<BVH> cloth_bvh,
                                          std::shared_ptr<BVH> obstacle_bvh, Face** faces_cloth,
                                          Face** faces_obstacle, Scalar thickness, int frame_index,
                                          int iteration) {
  CUDA_CALL(ClearColors_Kernel, cloth_bvh->NumObjects())
  (faces_cloth, cloth_bvh->NumObjects());
  CUDA_CALL(ClearObstacleColors_Kernel, obstacle_bvh->NumObjects())
  (faces_obstacle, obstacle_bvh->NumObjects());
  CUDA_CHECK_LAST();

  thrust::device_vector<PairFF> pairs =
      std::move(Traverse(cloth_bvh, obstacle_bvh, faces_cloth, faces_obstacle, thickness));
  thrust::device_vector<PairFF> self_pairs =
      std::move(Traverse(cloth_bvh, cloth_bvh, faces_cloth, faces_cloth, thickness));
  pairs.insert(pairs.end(), self_pairs.begin(), self_pairs.end());

  int n_pairs = pairs.size();
  CUDA_CALL(SetupColor_Kernel, n_pairs)
  (pointer(pairs), n_pairs);
  CUDA_CHECK_LAST();

  thrust::device_vector<Impact> ans(15 * n_pairs);
  CheckImpacts(pointer(pairs), n_pairs, thickness, pointer(ans));

  ans.erase(thrust::remove_if(ans.begin(), ans.end(), IsNull()), ans.end());

  return ans;
}

__global__ void InitializeImpactNode_Kernel(int n_impacts, const Impact* impacts, int deform) {
  GET_CUDA_ID(idx, n_impacts);

  const Impact& impact = impacts[idx];

  for (int j = 0; j < 4; j++) {
    Node* node = impact.nodes[j];
    if (deform == 1 || node->is_free) {
      node->removed = false;
      node->min_index = n_impacts;
    }
  }
}

void InitializeImpactNodes(int n_impacts, const Impact* impacts, int deform) {
  CUDA_CALL(InitializeImpactNode_Kernel, n_impacts)
  (n_impacts, impacts, deform);
  CUDA_CHECK_LAST();
}

__global__ void CollectRelativeImpacts_Kernel(int n_impacts, const Impact* impacts, int deform,
                                              Node** nodes, int* relative_impacts) {
  GET_CUDA_ID(i, n_impacts);

  const Impact& impact = impacts[i];
  bool flag = true;  // whether the impact is available

  for (int j = 0; j < 4; j++) {
    Node* node = impact.nodes[j];
    if ((deform == 1 || node->is_free) && node->removed) {
      flag = false;
      break;
    }
  }

  for (int j = 0; j < 4; j++) {
    Node* node = impact.nodes[j];
    int index = 4 * i + j;
    if (flag && (deform == 1 || node->is_free)) {
      nodes[index] = node;
      relative_impacts[index] = i;
    } else {
      nodes[index] = nullptr;
      relative_impacts[index] = -1;
    }
  }
}

void CollectRelativeImpacts(int n_impacts, const Impact* impacts, int deform, Node** nodes,
                            int* relative_impacts) {
  CUDA_CALL(CollectRelativeImpacts_Kernel, n_impacts)
  (n_impacts, impacts, deform, nodes, relative_impacts);
  CUDA_CHECK_LAST();
}

__global__ void SetImpactMinIndices_Kernel(int n_nodes, const int* relative_impacts, Node** nodes) {
  GET_CUDA_ID(idx, n_nodes);

  nodes[idx]->min_index = relative_impacts[idx];
}

void SetImpactMinIndices(int n_nodes, const int* relative_impacts, Node** nodes) {
  CUDA_CALL(SetImpactMinIndices_Kernel, n_nodes)
  (n_nodes, relative_impacts, nodes);
  CUDA_CHECK_LAST();
}

__global__ void CheckIndependentImpacts_Kernel(int n_impacts, const Impact* impacts, int deform,
                                               Impact* independent_impacts) {
  GET_CUDA_ID(i, n_impacts);

  const Impact& impact = impacts[i];
  bool flag = true;

  for (int j = 0; j < 4; j++) {
    Node* node = impact.nodes[j];
    if ((deform == 1 || node->is_free) && (node->removed || node->min_index != i)) {
      flag = false;
      break;
    }
  }

  if (flag) {
    independent_impacts[i] = impact;
    for (int j = 0; j < 4; j++) {
      Node* node = impact.nodes[j];
      if (deform == 1 || node->is_free)
        node->removed = true;
    }
  }
}

void CheckIndependentImpacts(int n_impacts, const Impact* impacts, int deform,
                             Impact* independent_impacts) {
  CUDA_CALL(CheckIndependentImpacts_Kernel, n_impacts)
  (n_impacts, impacts, deform, independent_impacts);
  CUDA_CHECK_LAST();
}

}  // namespace XRTailor