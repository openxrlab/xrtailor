#include <xrtailor/physics/repulsion/PBDRepulsion.cuh>
#include <iomanip>
#include <thrust/unique.h>
#include <thrust/remove.h>
#include <xrtailor/physics/representative_triangle/RTriangle.cuh>
#include <xrtailor/physics/repulsion/Impulse.cuh>
#include <xrtailor/utils/ObjUtils.hpp>
#include <xrtailor/math/BasicPrimitiveTests.cuh>

namespace XRTailor {
PBDRepulsionSolver::PBDRepulsionSolver() {}

PBDRepulsionSolver::PBDRepulsionSolver(PhysicsMesh* cloth) {
  int n_nodes = cloth->NumNodes();

  deltas_.resize(n_nodes, Vector3(0));
  delta_counts_.resize(n_nodes, 0);
}

PBDRepulsionSolver::~PBDRepulsionSolver() {}

void PBDRepulsionSolver::UpdatePairs(PhysicsMesh* cloth, PhysicsMesh* obstacle, BVH* cloth_bvh,
                                     BVH* obstacle_bvh, Scalar thickness) {
  cloth_bvh->Update(pointer(cloth->faces), true);

  thrust::device_vector<PairFF> pairs = std::move(Traverse(
      cloth_bvh, obstacle_bvh, pointer(cloth->faces), pointer(obstacle->faces), thickness));
  thrust::device_vector<PairFF> self_pairs = std::move(
      Traverse(cloth_bvh, cloth_bvh, pointer(cloth->faces), pointer(cloth->faces), thickness));
  pairs.insert(pairs.end(), self_pairs.begin(), self_pairs.end());

  this->pairs_ = std::move(pairs);
}

__host__ __device__ void CheckVFProximity(Node* node, Face* face, Proximity& proximity) {
  Node* node0 = face->nodes[0];
  Node* node1 = face->nodes[1];
  Node* node2 = face->nodes[2];

  if (node == node0 || node == node1 || node == node2)
    return;

  proximity = Proximity(node, face, static_cast<Scalar>(1));
}

__host__ __device__ void CheckEEProximity(Edge* edge0, Edge* edge1, Proximity& proximity) {
  Node* node0 = edge0->nodes[0];
  Node* node1 = edge0->nodes[1];
  Node* node2 = edge1->nodes[0];
  Node* node3 = edge1->nodes[1];

  if (node0 == node2 || node0 == node3 || node1 == node2 || node1 == node3)
    return;

  proximity = Proximity(edge0, edge1, static_cast<Scalar>(1.0));
}

__global__ void EvaluateProximity_Kernel(PairFF* pairs, Proximity* vf_proximities,
                                         Proximity* ee_proximities, Quadrature* quads,
                                         int n_pairs) {
  GET_CUDA_ID(pid, n_pairs);

  Face* face0 = pairs[pid].first;
  Face* face1 = pairs[pid].second;

  const int vf_offset = pid * 3;
  for (int i = 0; i < 3; i++) {
    if (!RTriVertex(face0->r_tri, i))
      continue;
    CheckVFProximity(face0->nodes[i], face1, vf_proximities[vf_offset + i]);
  }

  int ee_offset = pid * 9;
  for (int i = 0; i < 3; i++) {
    if (!RTriEdge(face0->r_tri, i))
      continue;
    for (int j = 0; j < 3; j++) {
      if (!RTriEdge(face1->r_tri, j))
        continue;
      CheckEEProximity(face0->edges[i], face1->edges[j], ee_proximities[ee_offset + i * 3 + j]);

      {
        bool is_proximity = true;
        Node* node0 = face0->edges[i]->nodes[0];
        Node* node1 = face0->edges[i]->nodes[1];
        Node* node2 = face1->edges[j]->nodes[0];
        Node* node3 = face1->edges[j]->nodes[1];

        Node* nodes[4] = {node0, node1, node2, node3};

        if (node0 == node2 || node0 == node3 || node1 == node2 || node1 == node3)
          is_proximity = false;

        Scalar b[4];
        Vector3 n;
        BasicPrimitiveTests::SignedEdgeEdgeDistance(node0->x, node1->x, node2->x, node3->x, n, b);
        bool inside = (MathFunctions::min(b[0], b[1], -b[2], -b[3]) >= static_cast<Scalar>(1e-6) &&
                       BasicPrimitiveTests::InEdge(b[1], face0->edges[i], face0->edges[j]) &&
                       BasicPrimitiveTests::InEdge(-b[3], face0->edges[j], face0->edges[i]));
        if (!inside)
          is_proximity = false;

        if (is_proximity) {
          Quadrature& quad = quads[ee_offset + i * 3 + j];
          for (int t = 0; t < 4; t++) {
            quad.nodes[t] = nodes[t];
            quad.bary[t] = b[t];
          }
        }
      }
    }
  }
}

void PBDRepulsionSolver::UpdateProximities() {
  // evaluate quadratures
  int nPairs = this->pairs_.size();

  thrust::device_vector<Proximity> vf_proximities(nPairs * 3, Proximity());
  thrust::device_vector<Proximity> ee_proximities(nPairs * 9, Proximity());

  thrust::device_vector<Quadrature> quads(nPairs * 9, Quadrature());

  CUDA_CALL(EvaluateProximity_Kernel, nPairs)
  (pointer(pairs_), pointer(vf_proximities), pointer(ee_proximities), pointer(quads), nPairs);
  CUDA_CHECK_LAST();

  int old_vf_size = vf_proximities.size();

  vf_proximities.erase(
      thrust::remove_if(vf_proximities.begin(), vf_proximities.end(), ProximityIsNull()),
      vf_proximities.end());

  int new_vf_size = vf_proximities.size();

  int old_ee_size = ee_proximities.size();
  ee_proximities.erase(
      thrust::remove_if(ee_proximities.begin(), ee_proximities.end(), ProximityIsNull()),
      ee_proximities.end());
  int new_ee_size = ee_proximities.size();

  ee_proximities.insert(ee_proximities.end(), vf_proximities.begin(), vf_proximities.end());

  proximities_ = std::move(ee_proximities);

  quads.erase(thrust::remove_if(quads.begin(), quads.end(), QuadIsNull()), quads.end());

  quads_ = std::move(quads);
}

__host__ __device__ bool CheckEEImpulse(Proximity* proximity, Impulse* impulse) {
  Node** nodes = proximity->nodes;

  Scalar b[4];
  Vector3 n;
  BasicPrimitiveTests::SignedEdgeEdgeDistance(nodes[0]->x, nodes[1]->x, nodes[2]->x, nodes[3]->x, n,
                                              b);
  bool inside = (MathFunctions::min(b[0], b[1], -b[2], -b[3]) >= static_cast<Scalar>(1e-6));
  if (!inside)
    return false;

  Vector3 d(0);
  for (int i = 0; i < 4; i++) {
    d += b[i] * nodes[i]->x;
  }

  Scalar h = static_cast<Scalar>(1e-3);

  Scalar C = glm::dot(d, n) - h;

  if (C > static_cast<Scalar>(0.0))
    return false;
  // hack: grad is vec
  Scalar grads[4] = {b[0], b[1], b[2], b[3]};

  Scalar denom = static_cast<Scalar>(0.0);
  for (int i = 0; i < 4; i++) {
    denom += nodes[i]->inv_mass * b[i] * b[i];
  }

  if (denom < EPSILON)
    return false;

  Scalar s = C / denom;

  Scalar dps[4];
  for (int i = 0; i < 4; i++) {
    impulse->nodes[i] = nodes[i];
    impulse->corrs[i] = -s * nodes[i]->inv_mass * b[i] * n;
  }

  return true;
}

__host__ __device__ bool CheckVFImpulse(Proximity* proximity, Impulse* impulse) {
  Node** nodes = proximity->nodes;

  Vector3 fn;
  Scalar w[4];
  BasicPrimitiveTests::SignedVertexFaceDistance(nodes[0]->x, nodes[1]->x, nodes[2]->x, nodes[3]->x,
                                                fn, w);

  bool inside = (MathFunctions::min(-w[1], -w[2], -w[3]) >= static_cast<Scalar>(1e-6));

  if (!inside)
    return false;

  Scalar h = static_cast<Scalar>(1e-3);

  Vector3 n = MathFunctions::FaceNormal(nodes[1]->x, nodes[2]->x, nodes[3]->x);

  Scalar side = (glm::dot(n, proximity->n) > static_cast<Scalar>(0.0)) ? static_cast<Scalar>(1.0)
                                                                       : static_cast<Scalar>(-1.0);

  Vector3 q, p1, p2, p3;
  q = nodes[0]->x;
  p1 = nodes[1]->x;
  p2 = nodes[2]->x;
  p3 = nodes[3]->x;

  q = q - p1;
  p2 = p2 - p1;
  p3 = p3 - p1;
  p1 = Vector3(0);

  n = side * glm::cross(p2, p3);
  Scalar c23 = glm::length(n);
  n = glm::normalize(n);

  Scalar C = glm::dot(q, n) - h;
  if (C > static_cast<Scalar>(0.0))
    return false;

  Vector3 dcq, dcp1, dcp2, dcp3;
  dcq = n;
  dcp2 = (glm::cross(p3, q) + glm::cross(n, p3) * glm::dot(n, q)) / c23;
  dcp3 = -(glm::cross(p2, q) + glm::cross(n, p2) * glm::dot(n, q)) / c23;
  dcp1 = -dcq - dcp2 - dcp3;

  Scalar denom =
      nodes[0]->inv_mass * glm::dot(dcq, dcq) + nodes[1]->inv_mass * glm::dot(dcp1, dcp1) +
      nodes[2]->inv_mass * glm::dot(dcp2, dcp2) + nodes[3]->inv_mass * glm::dot(dcp3, dcp3);

  if (denom < EPSILON)
    return false;

  Vector3 dq, dp1, dp2, dp3;
  Scalar s = C / denom;
  dq = -nodes[0]->inv_mass * s * dcq;
  dp1 = -nodes[1]->inv_mass * s * dcp1;
  dp2 = -nodes[2]->inv_mass * s * dcp2;
  dp3 = -nodes[3]->inv_mass * s * dcp3;

  Vector3 dxs[4] = {dq, dp1, dp2, dp3};

  for (int i = 0; i < 4; i++) {
    impulse->nodes[i] = nodes[i];
    impulse->corrs[i] = dxs[i];
  }

  return true;
}

__device__ void AccumulateRepulsions(Vector3* deltas, int* delta_counts, Impulse& impulse) {
  for (int i = 0; i < 4; i++) {
    if (glm::length(impulse.corrs[i]) < EPSILON)
      continue;

    if (!impulse.nodes[i]->is_free)
      continue;

    const int& node_idx = impulse.nodes[i]->index;

    AtomicAdd(deltas, node_idx, impulse.corrs[i]);
    atomicAdd(&delta_counts[node_idx], 1);
  }
}

__global__ void EvaluatePBDRepulsion_Kernel(Proximity* proximities, Vector3* deltas,
                                            int* delta_counts, int n_proximity) {
  GET_CUDA_ID(pid, n_proximity);

  Proximity proximity = proximities[pid];

  Node** nodes = proximity.nodes;

  Impulse impulse;
  bool hasImpulse = false;
  if (proximity.is_fv) {
    hasImpulse = CheckVFImpulse(&proximity, &impulse);
  } else {
    hasImpulse = CheckEEImpulse(&proximity, &impulse);
  }

  if (hasImpulse)
    AccumulateRepulsions(deltas, delta_counts, impulse);
}

__global__ void ApplyPBDImpulse_Kernel(Node** nodes, Vector3* dx_deltas, int* dx_delta_counts,
                                       Scalar relaxation_rate, int n_nodes) {
  GET_CUDA_ID(i, n_nodes);
  Scalar count = static_cast<Scalar>(dx_delta_counts[i]);

  if (count <= EPSILON)
    return;

  Vector3 dx = dx_deltas[i] / count * relaxation_rate;

  nodes[i]->x = nodes[i]->x + dx;

  dx_deltas[i] = Vector3(0);
  dx_delta_counts[i] = 0;
}

void PBDRepulsionSolver::Solve(PhysicsMesh* cloth, Scalar thickness, Scalar relaxation_rate,
                               Scalar dt, int frame_index, int iter) {
  int n_nodes = cloth->NumNodes();
  int n_proximity = proximities_.size();

  CUDA_CALL(EvaluatePBDRepulsion_Kernel, n_proximity)
  (pointer(proximities_), pointer(deltas_), pointer(delta_counts_), n_proximity);
  CUDA_CHECK_LAST();

  CUDA_CALL(ApplyPBDImpulse_Kernel, n_nodes)
  (pointer(cloth->nodes), pointer(deltas_), pointer(delta_counts_), relaxation_rate, n_nodes);
  CUDA_CHECK_LAST();
}

__host__ __device__ bool EvaluateVFProximity(Node* node, Face* face, const Scalar& h,
                                             Impulse* impulse) {
  Node* node0 = face->nodes[0];
  Node* node1 = face->nodes[1];
  Node* node2 = face->nodes[2];

  if (node == node0 || node == node1 || node == node2)
    return false;

  Proximity proximity(node, face, static_cast<Scalar>(1.0));

  Node** nodes = proximity.nodes;

  Vector3 fn;
  Scalar w[4];
  BasicPrimitiveTests::SignedVertexFaceDistance(nodes[0]->x, nodes[1]->x, nodes[2]->x, nodes[3]->x,
                                                fn, w);

  bool inside = (MathFunctions::min(-w[1], -w[2], -w[3]) >= 1e-6f);

  if (!inside)
    return false;

  Vector3 n = MathFunctions::FaceNormal(nodes[1]->x, nodes[2]->x, nodes[3]->x);

  Scalar side = (glm::dot(n, proximity.n) > static_cast<Scalar>(0.0)) ? static_cast<Scalar>(1.0)
                                                                      : static_cast<Scalar>(-1.0);

  Vector3 q, p1, p2, p3;
  q = nodes[0]->x;
  p1 = nodes[1]->x;
  p2 = nodes[2]->x;
  p3 = nodes[3]->x;

  q = q - p1;
  p2 = p2 - p1;
  p3 = p3 - p1;
  p1 = Vector3(0);

  n = side * glm::cross(p2, p3);
  //if (face->is_free())
  //    n = side * glm::cross(p2, p3);
  //else
  //    n = glm::cross(p2, p3);

  Scalar c23 = glm::length(n);
  n = glm::normalize(n);

  Scalar C = glm::dot(q, n) - h;
  if (C > 0)
    return false;

  Vector3 dcq, dcp1, dcp2, dcp3;
  dcq = n;
  dcp2 = (glm::cross(p3, q) + glm::cross(n, p3) * glm::dot(n, q)) / c23;
  dcp3 = -(glm::cross(p2, q) + glm::cross(n, p2) * glm::dot(n, q)) / c23;
  dcp1 = -dcq - dcp2 - dcp3;

  Scalar denom =
      nodes[0]->inv_mass * glm::dot(dcq, dcq) + nodes[1]->inv_mass * glm::dot(dcp1, dcp1) +
      nodes[2]->inv_mass * glm::dot(dcp2, dcp2) + nodes[3]->inv_mass * glm::dot(dcp3, dcp3);

  if (denom < EPSILON)
    return false;

  Vector3 dq, dp1, dp2, dp3;
  Scalar s = C / denom;
  dq = -nodes[0]->inv_mass * s * dcq;
  dp1 = -nodes[1]->inv_mass * s * dcp1;
  dp2 = -nodes[2]->inv_mass * s * dcp2;
  dp3 = -nodes[3]->inv_mass * s * dcp3;

  Vector3 dxs[4] = {dq, dp1, dp2, dp3};

  for (int i = 0; i < 4; i++) {
    impulse->nodes[i] = nodes[i];
    impulse->corrs[i] = dxs[i];
  }

  return true;
}

__host__ __device__ bool EvaluateEEProximity(Edge* edge0, Edge* edge1, const Scalar& h,
                                             Impulse* impulse) {
  Node* node0 = edge0->nodes[0];
  Node* node1 = edge0->nodes[1];
  Node* node2 = edge1->nodes[0];
  Node* node3 = edge1->nodes[1];

  if (node0 == node2 || node0 == node3 || node1 == node2 || node1 == node3)
    return false;

  Proximity proximity(edge0, edge1, static_cast<Scalar>(1.0));

  Node** nodes = proximity.nodes;

  Scalar b[4];
  Vector3 n;
  BasicPrimitiveTests::SignedEdgeEdgeDistance(nodes[0]->x, nodes[1]->x, nodes[2]->x, nodes[3]->x, n,
                                              b);
  bool inside = (MathFunctions::min(b[0], b[1], -b[2], -b[3]) >= static_cast<Scalar>(1e-6) &&
                 BasicPrimitiveTests::InEdge(b[1], edge0, edge1) &&
                 BasicPrimitiveTests::InEdge(-b[3], edge1, edge0));
  if (!inside)
    return false;

  Vector3 d(0);
  for (int i = 0; i < 4; i++) {
    d += b[i] * nodes[i]->x;
  }

  n = proximity.n;

  Scalar C = glm::dot(d, n) - h;

  if (C > 0)
    return false;

  b[2] *= -1;
  b[3] *= -1;

  Vector3 grads[4] = {b[0] * n, b[1] * n, -b[2] * n, -b[3] * n};

  Scalar denom = 0;
  for (int i = 0; i < 4; i++) {
    denom += nodes[i]->inv_mass * b[i] * b[i];
  }

  if (denom < EPSILON)
    return false;

  Scalar s = C / denom;

  for (int i = 0; i < 4; i++) {
    impulse->nodes[i] = nodes[i];
    impulse->corrs[i] = -s * nodes[i]->inv_mass * grads[i];
  }

  return true;
}

__global__ void EvaluateRep_Kernel(PairFF* pairs, Vector3* deltas, int* delta_counts,
                                   Scalar thickness, int n_pairs) {
  GET_CUDA_ID(pid, n_pairs);

  Face* face0 = pairs[pid].first;
  Face* face1 = pairs[pid].second;

  for (int i = 0; i < 3; i++) {
    if (!RTriVertex(face0->r_tri, i))
      continue;

    Impulse impulse;
    if (!EvaluateVFProximity(face0->nodes[i], face1, thickness, &impulse))
      continue;

    AccumulateRepulsions(deltas, delta_counts, impulse);
  }

  for (int i = 0; i < 3; i++) {
    if (!RTriEdge(face0->r_tri, i))
      continue;
    for (int j = 0; j < 3; j++) {
      if (!RTriEdge(face1->r_tri, j))
        continue;

      Impulse impulse;
      if (!EvaluateEEProximity(face0->edges[i], face1->edges[j], thickness, &impulse))
        continue;
      AccumulateRepulsions(deltas, delta_counts, impulse);
    }
  }
}

void PBDRepulsionSolver::SolveNoProximity(PhysicsMesh* cloth, Scalar thickness,
                                          Scalar relaxation_rate, Scalar dt, int frame_index,
                                          int iter) {
  int n_nodes = cloth->NumNodes();
  int n_pairs = pairs_.size();

  CUDA_CALL(EvaluateRep_Kernel, n_pairs)
  (pointer(pairs_), pointer(deltas_), pointer(delta_counts_), thickness, n_pairs);
  CUDA_CHECK_LAST();

  CUDA_CALL(ApplyPBDImpulse_Kernel, n_nodes)
  (pointer(cloth->nodes), pointer(deltas_), pointer(delta_counts_), relaxation_rate, n_nodes);
  CUDA_CHECK_LAST();
}

__global__ void MapQuad_Kernel(Quadrature* quads, QuadInd* quad_inds, int n_quad) {
  GET_CUDA_ID(qid, n_quad);

  for (int i = 0; i < 4; i++) {
    Node* node = quads[qid].nodes[i];
    quad_inds[qid].ids[i] = node->index;
  }
}

thrust::host_vector<QuadInd> PBDRepulsionSolver::HostQuadIndices() {
  int n_quad = quads_.size();

  thrust::device_vector<QuadInd> d_quad_inds(n_quad);
  CUDA_CALL(MapQuad_Kernel, n_quad)
  (pointer(quads_), pointer(d_quad_inds), n_quad);
  CUDA_CHECK_LAST();

  thrust::host_vector<QuadInd> h_quad_inds = std::move(d_quad_inds);

  return h_quad_inds;
}

}  // namespace XRTailor