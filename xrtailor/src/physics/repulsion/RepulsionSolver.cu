#include <xrtailor/physics/repulsion/RepulsionSolver.cuh>
#include <xrtailor/math/BasicPrimitiveTests.cuh>

#include <thrust/remove.h>
#include <thrust/unique.h>

namespace XRTailor {

RepulsionSolver::RepulsionSolver() {}

RepulsionSolver::~RepulsionSolver() {}

void RepulsionSolver::CheckProximity(BVH* cloth_bvh, BVH* obstacle_bvh, Face** cloth_faces,
                                     Face** obstacle_faces) {
  Scalar repulsion_thickness = static_cast<Scalar>(1e-4);
  pairs_ = std::move(Traverse(cloth_bvh, obstacle_bvh, cloth_faces, obstacle_faces,
                             static_cast<Scalar>(2.0) * repulsion_thickness));
}

__device__ void GenerateVFProximity(Node* node, Face* face, VFProximity& proximity) {
  auto pred = node->x;

  Scalar u, v, w;
  Scalar d = BasicPrimitiveTests::NearestPoint(node->x, face->nodes[0]->x, face->nodes[1]->x,
                                               face->nodes[2]->x, proximity.qs, u, v, w);

  proximity.node = node;
  proximity.face = face;
  proximity.d = d;
}

__global__ void GenerateProximity_Kernel(PairFF* pairs, VFProximity* vf_proximities,
                                         EEProximity* ee_proximities, RTProximity* rt_proximities,
                                         int n_pairs) {
  GET_CUDA_ID(i, n_pairs);

  const PairFF& pair = pairs[i];
  Face* f1 = pair.first;
  Face* f2 = pair.second;

  if (f1->IsFree() & !f2->IsFree()) {
    for (int j = 0; j < 3; j++) {
      if (!RTriVertex(f1->r_tri, j))
        continue;

      GenerateVFProximity(f1->nodes[j], f2, vf_proximities[i * 3 + j]);
    }
  }
}

struct is_duplicate_vf_proximity {
  __host__ __device__ bool operator()(const VFProximity& lhs, const VFProximity& rhs) const {
    return lhs.node == rhs.node;
  }
};

void RepulsionSolver::GenerateRepulsiveConstraints() {
  int n_pairs = pairs_.size();

  vf_proximities_.resize(n_pairs * 3, VFProximity());

  CUDA_CALL(GenerateProximity_Kernel, n_pairs)
  (pointer(pairs_), pointer(vf_proximities_), pointer(ee_proximities_), pointer(rt_proximities_),
   n_pairs);

  CUDA_CHECK_LAST();

  vf_proximities_.erase(thrust::remove_if(vf_proximities_.begin(), vf_proximities_.end(), IsNull()),
                       vf_proximities_.end());

  thrust::sort(vf_proximities_.begin(), vf_proximities_.end());

  vf_proximities_.erase(
      thrust::unique(vf_proximities_.begin(), vf_proximities_.end(), is_duplicate_vf_proximity()),
      vf_proximities_.end());
}

__global__ void solveVFProximity_Kernel(VFProximity* proximities, int n_proximities) {
  GET_CUDA_ID(i, n_proximities);

  VFProximity& proximity = proximities[i];
  Node* node = proximity.node;
  Face* face = proximity.face;

  Vector3 ns = MathFunctions::FaceNormal(face->nodes[0]->x, face->nodes[1]->x, face->nodes[2]->x);

  Scalar lambda = glm::dot(ns, node->x - proximity.qs) - static_cast<Scalar>(1e-3);
  if (lambda < static_cast<Scalar>(0.0)) {
    Vector3 corr = -lambda * ns;
    for (int j = 0; j < 3; j++)
      atomicAdd(&(node->x[j]), corr[j]);
  }
}

void RepulsionSolver::Solve() {
  int nVFProximities = vf_proximities_.size();
  CUDA_CALL(solveVFProximity_Kernel, nVFProximities)
  (pointer(vf_proximities_), nVFProximities);
  CUDA_CHECK_LAST();
}

}  // namespace XRTailor