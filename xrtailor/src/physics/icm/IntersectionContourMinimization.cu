#include <xrtailor/physics/icm/IntersectionContourMinimization.cuh>
#include <thrust/remove.h>
#include <thrust/reduce.h>

#include <xrtailor/physics/representative_triangle/RTriangle.cuh>

namespace XRTailor {
namespace Untangling {

__host__ __device__ bool EFIntersectionIsNull::operator()(EFIntersection intersection) {
  return intersection.pair.first == nullptr;
}

__host__ __device__ bool IntersectionWithGradientIsNull::operator()(
    IntersectionWithGradient intersection) {
  return intersection.ev0_idx < 0;
}

ICM::ICM() {}

ICM::ICM(int n_nodes, int n_edges) {
  impulses_.resize(n_nodes, Vector3(0));
  gradients_.resize(n_edges, Vector3(0));

  deltas_.resize(n_nodes, Vector3(0));
  delta_counts_.resize(n_nodes, 0);
}

ICM::~ICM() {}

void ICM::UpdatePairs(PhysicsMesh* cloth, PhysicsMesh* obstacle, BVH* cloth_bvh,
                      BVH* obstacle_bvh) {
  pairs_ = std::move(
      Traverse(cloth_bvh, cloth_bvh, pointer(cloth->faces), pointer(cloth->faces), 1e-6f));
}

__device__ __inline__ Vector3 EvalGradient(Node* node0, Node* node1, Node* node2, Vector3 efn,
                                           Vector3 fn) {
  Vector3 E = glm::normalize(node1->x - node0->x);
  Vector3 N = fn;
  Vector3 M = efn;
  Vector3 R = glm::normalize(glm::cross(N, M));

  Vector3 in_plane_edge_normal = node2->x - node0->x;
  in_plane_edge_normal = in_plane_edge_normal - glm::dot(in_plane_edge_normal, E) * E;
  if (glm::dot(R, in_plane_edge_normal) < 0)
    R = -R;

  Scalar ER = glm::dot(E, R);
  Scalar EN = glm::dot(E, N);

  Scalar N_coeff;
  if (EN < EPSILON)
    N_coeff = static_cast<Scalar>(0);
  else
    N_coeff = static_cast<Scalar>(2) * ER / EN;

  return R - N_coeff * N;
}

__device__ __inline__ bool checkEFPair(Edge* edge, Face* face, Vector3& G,
                                       IntersectionWithGradient* intersection) {
  if (face->Contain(edge))
    return false;

  Node* node0 = edge->nodes[0];
  Node* node1 = edge->nodes[1];

  if (face->Contain(node0) || face->Contain(node1))
    return false;

  Node* fnode0 = face->nodes[0];
  Node* fnode1 = face->nodes[1];
  Node* fnode2 = face->nodes[2];

  Vector3 fn = glm::cross(fnode1->x - fnode0->x, fnode2->x - fnode0->x);
  if (glm::length(fn) < EPSILON)
    return false;
  fn = glm::normalize(fn);

  Scalar u, v, w = static_cast<Scalar>(-1);
  if (!MathFunctions::LineTriangleIntersects(node0->x, node1->x, fnode0->x, fnode1->x, fnode2->x, u,
                                             v, w))
    return false;

  Vector3 isect = u * fnode0->x + v * fnode1->x + w * fnode2->x;
  Vector3 e = node1->x - node0->x;
  Vector3 d = isect - node0->x;
  if (glm::dot(d, e) < 0)
    return false;

  Scalar t = glm::length(isect - node0->x) / glm::length(e);
  if (t >= static_cast<Scalar>(1))
    return false;

  int n_valid_adjacents = 0;
  for (int i = 0; i < MAX_EF_ADJACENTS; i++) {
    Face* nb_face = edge->adjacents[i];
    if (nb_face == nullptr)
      break;

    Vector3 efn =
        MathFunctions::FaceNormal(nb_face->nodes[0]->x, nb_face->nodes[1]->x, nb_face->nodes[2]->x);

    Node* node_opposite = nb_face->OppositeNode(node0, node1);

    Vector3 grad = EvalGradient(node0, node1, node_opposite, efn, fn);
    G += grad;
    n_valid_adjacents++;
  }

  if (n_valid_adjacents > 0)
    G = G / static_cast<Scalar>(n_valid_adjacents);

  intersection->f_idx = face->index;
  intersection->ev0_idx = node0->index;
  intersection->ev1_idx = node1->index;
  intersection->ev0 = node0->x;
  intersection->ev1 = node1->x;
  intersection->v0 = fnode0->x;
  intersection->v1 = fnode1->x;
  intersection->v2 = fnode2->x;
  intersection->p = isect;
  intersection->G = static_cast<Scalar>(0.01) * glm::normalize(G);

  return true;
}

__global__ void UpdateGradient_Kernel(PairFF* pairs, Vector3* gradients, Vector3* impulses,
                                      IntersectionWithGradient* intersections, int n_pair) {
  GET_CUDA_ID(pid, n_pair);

  Face* face0 = pairs[pid].first;
  Face* face1 = pairs[pid].second;

  if (face0 == face1)
    return;

  for (int i = 0; i < 3; i++) {
    if (!RTriEdge(face0->r_tri, i))
      continue;

    Vector3 G(0);

    if (!checkEFPair(face0->edges[i], face1, G, &intersections[pid * 3 + i]))
      continue;

    if (glm::length(G) < EPSILON)
      continue;
    for (int m = 0; m < 2; m++) {
      if (face0->edges[i]->nodes[m]->is_free)
        AtomicAdd(impulses, face0->edges[i]->nodes[m]->index, G);
    }

    AtomicAdd(gradients, face0->edges[i]->index, G);
    for (int n = 0; n < 3; n++) {
      if (face1->IsFree())
        AtomicAdd(gradients, face1->edges[n]->index, -G);
    }

    for (int n = 0; n < 3; n++) {
      if (face1->nodes[n]->is_free)
        AtomicAdd(impulses, face1->nodes[n]->index, -G);
    }
  }
}

void ICM::UpdateGradient(PhysicsMesh* cloth, PhysicsMesh* obstacle) {
  int nPair = pairs_.size();

  intersections_.resize(nPair * 3, IntersectionWithGradient());
  CUDA_CALL(UpdateGradient_Kernel, nPair)
  (pointer(pairs_), pointer(gradients_), pointer(impulses_), pointer(intersections_), nPair);
  CUDA_CHECK_LAST();
  intersections_.erase(thrust::remove_if(intersections_.begin(), intersections_.end(),
                                         IntersectionWithGradientIsNull()),
                       intersections_.end());
}

__device__ __inline__ bool CheckEFPairGIA(Edge* edge, Face* face, Vector3& G,
                                          EFIntersection* intersection) {
  if (face->Contain(edge))
    return false;

  Node* node0 = edge->nodes[0];
  Node* node1 = edge->nodes[1];

  if (face->Contain(node0) || face->Contain(node1))
    return false;

  Node* fnode0 = face->nodes[0];
  Node* fnode1 = face->nodes[1];
  Node* fnode2 = face->nodes[2];

  Vector3 fn = glm::cross(fnode1->x - fnode0->x, fnode2->x - fnode0->x);
  if (glm::length(fn) < EPSILON)
    return false;
  fn = glm::normalize(fn);

  Scalar u, v, w = static_cast<Scalar>(-1.0);
  if (!MathFunctions::LineTriangleIntersects(node0->x, node1->x, fnode0->x, fnode1->x, fnode2->x, u,
                                             v, w))
    return false;

  Vector3 isect = u * fnode0->x + v * fnode1->x + w * fnode2->x;
  Vector3 e = node1->x - node0->x;
  Vector3 d = isect - node0->x;
  if (glm::dot(d, e) < 0)
    return false;

  Scalar t = glm::length(isect - node0->x) / glm::length(e);
  if (t >= static_cast<Scalar>(1.0))
    return false;

  int n_valid_adjacents = 0;
  for (int i = 0; i < MAX_EF_ADJACENTS; i++) {
    Face* nb_face = edge->adjacents[i];
    if (nb_face == nullptr)
      break;

    Vector3 efn =
        MathFunctions::FaceNormal(nb_face->nodes[0]->x, nb_face->nodes[1]->x, nb_face->nodes[2]->x);

    Node* node_opposite = nb_face->OppositeNode(node0, node1);

    Vector3 grad = EvalGradient(node0, node1, node_opposite, efn, fn);
    G += grad;
    n_valid_adjacents++;
  }
  if (n_valid_adjacents > 0)
    G = G / static_cast<Scalar>(n_valid_adjacents);

  intersection->pair = PairEF(edge, face);
  intersection->p = isect;
  intersection->s = t;
  intersection->G = G;

  return true;
}

__global__ void UpdateGradientGIA_Kernel(PairFF* pairs, EFIntersection* intersections, int nPair) {
  GET_CUDA_ID(pid, nPair);

  Face* face0 = pairs[pid].first;
  Face* face1 = pairs[pid].second;

  if (face0 == face1)
    return;

  for (int i = 0; i < 3; i++) {
    if (!RTriEdge(face0->r_tri, i))
      continue;

    Vector3 G(0.0f);

    if (!CheckEFPairGIA(face0->edges[i], face1, G, &intersections[pid * 3 + i]))
      continue;
  }
}

__global__ void AccumulateGradientGIA_Kernel(EFIntersection* intersections,
                                             const EdgeState* e_states, const FaceState* f_states,
                                             Vector3* sum_gradients, int n_intersection) {
  GET_CUDA_ID(id, n_intersection);

  EFIntersection* intersection = &intersections[id];

  const Edge* edge = intersection->pair.first;
  const int& color = e_states[edge->index].color;

  if (color == GIA_DEFAULT_COLOR)
    return;

  AtomicAdd(sum_gradients, color, intersection->G);
}

__global__ void WriteBackGradient_Kernel(EFIntersection* intersections, const EdgeState* e_states,
                                         const FaceState* f_states, Vector3* sum_gradients,
                                         Vector3* impulses, int n_intersection) {
  GET_CUDA_ID(id, n_intersection);
  EFIntersection* intersection = &intersections[id];
  const Edge* edge = intersection->pair.first;
  const Face* face = intersection->pair.second;
  const int& color = e_states[edge->index].color;

  Vector3 G = (color == GIA_DEFAULT_COLOR) ? intersection->G : sum_gradients[color];

  if (glm::length(G) < EPSILON)
    return;
  for (int m = 0; m < 2; m++) {
    if (edge->nodes[m]->is_free)
      AtomicAdd(impulses, edge->nodes[m]->index, G);
  }

  for (int n = 0; n < 3; n++) {
    if (face->nodes[n]->is_free)
      AtomicAdd(impulses, face->nodes[n]->index, -G);
  }
}

bool ICM::UpdateGradientGIA(PhysicsMesh* cloth, PhysicsMesh* obstacle, const int& n_contours,
                            GlobalIntersectionAnalysis* gia) {
  int n_pair = pairs_.size();
  gia_intersections_.resize(n_pair * 3, EFIntersection());
  CUDA_CALL(UpdateGradientGIA_Kernel, n_pair)
  (pointer(pairs_), pointer(gia_intersections_), n_pair);
  CUDA_CHECK_LAST();
  gia_intersections_.erase(thrust::remove_if(gia_intersections_.begin(), gia_intersections_.end(),
                                             EFIntersectionIsNull()),
                           gia_intersections_.end());

  int n_intersection = gia_intersections_.size();
  if (n_intersection == 0)
    return false;

  const thrust::device_vector<EdgeState>& e_states = gia->EdgeStates();
  const thrust::device_vector<FaceState>& f_states = gia->FaceStates();

  thrust::device_vector<Vector3> sum_gradients(n_contours, Vector3(0));
  CUDA_CALL(AccumulateGradientGIA_Kernel, n_intersection)
  (pointer(gia_intersections_), pointer(e_states), pointer(f_states), pointer(sum_gradients),
   n_intersection);
  CUDA_CHECK_LAST();

  CUDA_CALL(WriteBackGradient_Kernel, n_intersection)
  (pointer(gia_intersections_), pointer(e_states), pointer(f_states), pointer(sum_gradients),
   pointer(impulses_), n_intersection);
  CUDA_CHECK_LAST();

  return true;
}

thrust::host_vector<IntersectionWithGradient> ICM::HostIntersections() {
  thrust::host_vector<IntersectionWithGradient> ans = std::move(intersections_);
  return ans;
}

__global__ void ApplyImpulse_Kernel(Vector3* impulses, Scalar h0, Scalar g0, Node** nodes,
                                    int n_node) {
  GET_CUDA_ID(id, n_node);

  Vector3 G = impulses[id];
  if (glm::length(G) < EPSILON)
    return;
  Vector3 n = glm::normalize(G);
  Scalar G_length = glm::length(G);
  Scalar denom = sqrt(G_length * G_length + g0 * g0);
  if (denom < EPSILON)
    return;

  Scalar H_length = h0 * G_length / denom;

  Vector3 corr = H_length * n;

  if (nodes[id]->inv_mass > EPSILON)
    nodes[id]->x += corr;
  impulses[id] = Vector3(0);
}

void ICM::ApplyImpulse(PhysicsMesh* cloth, Scalar h0, Scalar g0) {
  int n_node = cloth->NumNodes();
  CUDA_CALL(ApplyImpulse_Kernel, n_node)
  (pointer(impulses_), h0, g0, pointer(cloth->nodes), n_node);
  CUDA_CHECK_LAST();
}

__global__ void AccumulateGradient_Kernel(Vector3* gradients, Scalar h0, Scalar g0, Edge** edges,
                                          Vector3* deltas, int* delta_counts, int n_edge) {
  GET_CUDA_ID(eid, n_edge);

  Node* node0 = edges[eid]->nodes[0];
  Node* node1 = edges[eid]->nodes[1];

  Vector3 G = gradients[eid];
  if (glm::length(G) < EPSILON)
    return;
  Vector3 n = glm::normalize(G);
  Scalar G_length = glm::length(G);
  Scalar denom = sqrt(G_length * G_length + g0 * g0);
  if (denom < EPSILON)
    return;

  Scalar H_length = h0 * G_length / denom;

  Vector3 corr = H_length * n;

  int reorder = node0->index + node1->index;
  AtomicAdd(deltas, node0->index, corr, reorder);
  AtomicAdd(deltas, node1->index, corr, reorder);
  atomicAdd(&delta_counts[node0->index], 1);
  atomicAdd(&delta_counts[node1->index], 1);
}

__global__ void ApplyGradient_Kernel(Node** nodes, Vector3* deltas, int* delta_counts,
                                     Scalar relaxation_rate, int n_nodes) {
  GET_CUDA_ID(i, n_nodes);

  Scalar count = static_cast<Scalar>(delta_counts[i]);

  if (count <= 0)
    return;

  Vector3 impulse = deltas[i] / count * relaxation_rate;

  nodes[i]->x = nodes[i]->x + impulse;

  deltas[i] = Vector3(0);
  delta_counts[i] = 0;
}

void ICM::ApplyGradient(PhysicsMesh* cloth, Scalar h0, Scalar g0) {
  int n_edge = cloth->NumEdges();
  CUDA_CALL(AccumulateGradient_Kernel, n_edge)
  (pointer(gradients_), h0, g0, pointer(cloth->edges), pointer(deltas_), pointer(delta_counts_),
   n_edge);
  CUDA_CHECK_LAST();

  int n_node = cloth->NumNodes();
  CUDA_CALL(ApplyGradient_Kernel, n_node)
  (pointer(cloth->nodes), pointer(deltas_), pointer(delta_counts_), 1.0f, n_node);
  CUDA_CHECK_LAST();
}

void DetangleStep(PhysicsMesh* cloth, PhysicsMesh* obstacle, BVH* cloth_bvh, BVH* obstacle_bvh,
                  const Scalar& g0, const Scalar& h0) {

  ICM* icm_solver = new ICM(cloth->NumNodes(), cloth->NumEdges());

  for (int iter = 0; iter < 100; iter++) {
    if (iter % 10 == 0) {
      cloth_bvh->Update(pointer(cloth->faces), true);
      icm_solver->UpdatePairs(cloth, obstacle, cloth_bvh, obstacle_bvh);
    }
    icm_solver->UpdateGradient(cloth, obstacle);
    icm_solver->ApplyImpulse(cloth, h0, g0);
  }

  delete icm_solver;
}

}  // namespace Untangling
}  // namespace XRTailor