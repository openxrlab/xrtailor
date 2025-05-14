#include <xrtailor/physics/icm/GlobalIntersectionAnalysis.cuh>
#include <xrtailor/math/MathFunctions.cuh>
#include <thrust/remove.h>
#include <xrtailor/physics/representative_triangle/RTriangle.cuh>

//#define GIA_DEBUG

namespace XRTailor {
namespace Untangling {

GlobalIntersectionAnalysis::GlobalIntersectionAnalysis() : palette_(0) {}

GlobalIntersectionAnalysis::~GlobalIntersectionAnalysis() {}

__global__ void FloodFillColoring_Kernel(EFIntersection* intersection, int color,
                                         EdgeState* e_states, FaceState* f_states,
                                         IntersectionState* i_states) {
  GET_CUDA_ID(id, 1);

  Edge* edge = intersection[0].pair.first;
  Face* face = intersection[0].pair.second;
  e_states[edge->index].color = color;
  f_states[face->index].color = color;

  Face* stack[1024];
  Face** stack_ptr = stack;

  for (int i = 0; i < MAX_EF_ADJACENTS; i++) {
    if (edge->adjacents[i] == nullptr)
      break;
    *stack_ptr++ = edge->adjacents[i];
  }

  do {
    Face* f = *--stack_ptr;
    if (f_states[f->index].color == -1) {
      f_states[f->index].color = color;
    }

    for (int i = 0; i < 3; i++) {
      int eidx = f->edges[i]->index;
      if (e_states[eidx].color == -1) {
        e_states[eidx].color = color;
        if (e_states[eidx].active) {
          for (int j = 0; j < MAX_EF_ADJACENTS; j++) {
            if (f->edges[i]->adjacents[j] == nullptr)
              break;
            *stack_ptr++ = f->edges[i]->adjacents[j];
          }
        }
      }
    }
  } while (stack < stack_ptr);
}

__global__ void UpdateState_Kernel(EFIntersection* intersections, EdgeState* e_states,
                                   FaceState* f_states, IntersectionState* i_states,
                                   int n_intersection) {
  GET_CUDA_ID(id, n_intersection);

  EFIntersection* intersection = &intersections[id];

  Edge* e = intersection->pair.first;
  Face* f = intersection->pair.second;

  e_states[e->index].active = true;
  f_states[f->index].active = true;
}

__global__ void WriteBackColor_Kernel(EFIntersection* intersections, EdgeState* e_states,
                                      FaceState* f_states, IntersectionState* i_states,
                                      int n_intersection) {
  GET_CUDA_ID(id, n_intersection);

  if (i_states[id].color != -1)
    return;  // skip intersections that have already been colored

  EFIntersection* intersection = &intersections[id];

  Edge* e = intersection->pair.first;
  Face* f = intersection->pair.second;

  if (e_states[e->index].color != -1)
    i_states[id].color = e_states[e->index].color;
  else if (f_states[f->index].color != -1)
    i_states[id].color = f_states[f->index].color;

  //if (e_states[e->index].color != f_states[f->index].color)
  //{
  //	printf("EF not share the same color, should never happen\n");
  //}
}

int GlobalIntersectionAnalysis::FloodFillIntersectionIslands(std::shared_ptr<PhysicsMesh> cloth,
                                                             std::shared_ptr<BVH> cloth_bvh) {
#ifdef GIA_DEBUG
  checkCudaErrors(cudaDeviceSynchronize());
  printf("FloodFillIntersectionIslands\n");
#endif  // GIA_DEBUG

  intersections_ = std::move(FindIntersections(cloth, cloth_bvh));

  int n_edges = cloth->NumEdges();
  int n_faces = cloth->NumFaces();
  int n_intersections = intersections_.size();

#ifdef GIA_DEBUG
  checkCudaErrors(cudaDeviceSynchronize());
  printf("%d intersections found\n", nIntersections);
#endif  // GIA_DEBUG

  if (n_intersections == 0) {
#ifdef GIA_DEBUG
    checkCudaErrors(cudaDeviceSynchronize());
    printf("No intersections detected\n");
#endif  // GIA_DEBUG

    return 0;
  }

  edge_states_.resize(n_edges);
  face_states_.resize(n_faces);
  intersection_states_.resize(n_intersections);

  bool all_visited = false;

  while (!all_visited) {
    int color = palette_;

    if (color > 99) {
      printf("too many attempts, exit\n");
      break;
    }

#ifdef GIA_DEBUG
    checkCudaErrors(cudaDeviceSynchronize());
    printf("Flood fill with color %d\n", color);
    printf("  [Step1] Update state\n");
#endif  // GIA_DEBUG

    CUDA_CALL(UpdateState_Kernel, n_intersections)
    (pointer(intersections_), pointer(edge_states_), pointer(face_states_),
     pointer(intersection_states_), n_intersections);
    CUDA_CHECK_LAST();

    auto iter = thrust::find_if(
        intersection_states_.begin(), intersection_states_.end(),
        [] __device__(const IntersectionState& i_state) { return i_state.color == -1; });

    if (iter == intersection_states_.end())
      break;  // all intersections have been colored

    int pos = iter - intersection_states_.begin();

#ifdef GIA_DEBUG
    checkCudaErrors(cudaDeviceSynchronize());
    printf("  [Step2] Select intersection %d as entry point\n");
#endif  // GIA_DEBUG

#ifdef GIA_DEBUG
    checkCudaErrors(cudaDeviceSynchronize());
    printf("  [Step3] Flood fill coloring\n");
#endif  // GIA_DEBUG

    CUDA_CALL(FloodFillColoring_Kernel, 1)
    (pointer(intersections_, pos), color, pointer(edge_states_), pointer(face_states_),
     pointer(intersection_states_));
    CUDA_CHECK_LAST();

#ifdef GIA_DEBUG
    checkCudaErrors(cudaDeviceSynchronize());
    printf("  [Step4] Write back colors\n");
#endif  // GIA_DEBUG

    CUDA_CALL(WriteBackColor_Kernel, n_intersections)
    (pointer(intersections_), pointer(edge_states_), pointer(face_states_),
     pointer(intersection_states_), n_intersections);
    CUDA_CHECK_LAST();

    palette_++;
  }
#ifdef GIA_DEBUG
  checkCudaErrors(cudaDeviceSynchronize());
  printf("Finished coloring, found %d islands\n", palette_);
#endif  // GIA_DEBUG

  return palette_;
}

__host__ __device__ bool IsEFIntersection(Edge* edge, Face* face, Vector3& p, Scalar& t) {
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
                                             v, w) &&
      !MathFunctions::LineTriangleIntersects(node1->x, node0->x, fnode0->x, fnode1->x, fnode2->x, u,
                                             v, w))
    return false;

  p = u * fnode0->x + v * fnode1->x + w * fnode2->x;
  Vector3 e = node1->x - node0->x;
  Vector3 d = p - node0->x;
  if (glm::dot(d, e) < 0)
    return false;

  t = glm::length(p - node0->x) / glm::length(e);
  if (t >= static_cast<Scalar>(1.0))
    return false;

  return true;
}

__global__ void EvaluateIntersection_Kernel(PairFF* pairs, EFIntersection* intersections,
                                            int n_pair) {
  GET_CUDA_ID(pid, n_pair);

  Face* face0 = pairs[pid].first;
  Face* face1 = pairs[pid].second;

  if (face0 == face1)
    return;

  for (int i = 0; i < 3; i++) {
    if (!RTriEdge(face0->r_tri, i))
      continue;

    Edge* edge = face0->edges[i];
    Vector3 p;
    Scalar t;
    if (!IsEFIntersection(edge, face1, p, t))
      continue;

    EFIntersection* isec = &intersections[pid * 3 + i];
    isec->p = p;
    isec->s = t;
    isec->pair = PairEF(edge, face1);
  }
}

thrust::device_vector<EFIntersection> GlobalIntersectionAnalysis::FindIntersections(
    std::shared_ptr<PhysicsMesh> cloth, std::shared_ptr<BVH> cloth_bvh) {
  cloth_bvh->Update(pointer(cloth->faces), true);
  thrust::device_vector<PairFF> pairs = std::move(
      Traverse(cloth_bvh, cloth_bvh, pointer(cloth->faces), pointer(cloth->faces), 1e-6f));

  int nPair = pairs.size();
  thrust::device_vector<EFIntersection> intersections(nPair * 3, EFIntersection());

  CUDA_CALL(EvaluateIntersection_Kernel, nPair)
  (pointer(pairs), pointer(intersections), nPair);
  CUDA_CHECK_LAST();

  intersections.erase(
      thrust::remove_if(intersections.begin(), intersections.end(), EFIntersectionIsNull()),
      intersections.end());

  return intersections;
}

thrust::host_vector<EFIntersection> GlobalIntersectionAnalysis::HostIntersections() {
  return std::move(intersections_);
}

thrust::host_vector<IntersectionState> GlobalIntersectionAnalysis::HostIntersectionStates() {
  return std::move(intersection_states_);
}

thrust::device_vector<EdgeState> GlobalIntersectionAnalysis::EdgeStates() {
  return edge_states_;
}

thrust::device_vector<FaceState> GlobalIntersectionAnalysis::FaceStates() {
  return face_states_;
}

}  // namespace Untangling
}  // namespace XRTailor