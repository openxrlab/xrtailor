#include <xrtailor/physics/repulsion/ImminentRepulsion.cuh>

#include <iomanip>

#include <thrust/unique.h>
#include <thrust/remove.h>

#include <xrtailor/memory/Pair.cuh>
#include <xrtailor/math/BasicPrimitiveTests.cuh>
#include <xrtailor/physics/representative_triangle/RTriangle.cuh>
#include <xrtailor/utils/ObjUtils.hpp>

namespace XRTailor {
ImminentRepulsion::ImminentRepulsion() {}

void ImminentRepulsion::InitializeDeltas(int n_nodes) {
  deltas_.resize(n_nodes, Vector3(0));
  delta_counts_.resize(n_nodes, 0);
}

void ImminentRepulsion::UpdateProximity(PhysicsMesh* cloth, PhysicsMesh* obstacle, BVH* cloth_bvh,
                                        BVH* obstacle_bvh, Scalar thickness) {
  thrust::device_vector<PairFF> self_pairs = std::move(Traverse(cloth_bvh, cloth_bvh, pointer(cloth->faces), pointer(cloth->faces), 1e-6f));
  thrust::device_vector<PairFF> pairs = std::move(Traverse(cloth_bvh, obstacle_bvh, pointer(cloth->faces), pointer(obstacle->faces), 1e-6f));
  pairs.insert(pairs.end(), self_pairs.begin(), self_pairs.end());

  this->pairs_ = std::move(pairs);
}

__global__ void EvaluateVFQuadrature_Kernel(PairFF* pairs, Quadrature* quads, int n_pairs) {
  GET_CUDA_ID(pid, n_pairs);

  Face* face0 = pairs[pid].first;
  Face* face1 = pairs[pid].second;

  if (face0 == face1)
    return;

  int offset = pid * 3;

  for (int i = 0; i < 3; i++) {
    if (!RTriVertex(face0->r_tri, i))
      continue;

    CheckImminentVFQuadrature(face0->nodes[i], face1, quads[offset + i]);
  }
}

__global__ void EvaluateEEQuadrature_Kernel(PairFF* pairs, Quadrature* quads, int n_pairs) {
  GET_CUDA_ID(pid, n_pairs);

  Face* face0 = pairs[pid].first;
  Face* face1 = pairs[pid].second;

  if (face0 == face1)
    return;

  int offset = pid * 9;

  for (int i = 0; i < 3; i++) {
    if (!RTriEdge(face0->r_tri, i))
      continue;
    for (int j = 0; j < 3; j++) {
      if (!RTriEdge(face1->r_tri, j))
        continue;

      CheckImminentEEQuadrature(face0->edges[i], face1->edges[j], quads[offset + i * 3 + j]);
    }
  }
}

void ImminentRepulsion::Generate() {
  // evaluate quadratures
  int n_pairs = this->pairs_.size();

  thrust::device_vector<Quadrature> vf_quads(n_pairs * 3, Quadrature());

  CUDA_CALL(EvaluateVFQuadrature_Kernel, n_pairs)
  (pointer(pairs_), pointer(vf_quads), n_pairs);
  CUDA_CHECK_LAST();

  int oldVFSize = vf_quads.size();

  vf_quads.erase(thrust::remove_if(vf_quads.begin(), vf_quads.end(), QuadIsNull()), vf_quads.end());

  // EE
  thrust::device_vector<Quadrature> ee_quads(n_pairs * 9, Quadrature());

  CUDA_CALL(EvaluateEEQuadrature_Kernel, n_pairs)
  (pointer(pairs_), pointer(ee_quads), n_pairs);
  CUDA_CHECK_LAST();

  int old_ee_size = ee_quads.size();

  ee_quads.erase(thrust::remove_if(ee_quads.begin(), ee_quads.end(), QuadIsNull()), ee_quads.end());

  int new_ee_size = ee_quads.size();
  ee_quads.insert(ee_quads.end(), vf_quads.begin(), vf_quads.end());

  quads_ = std::move(ee_quads);

}

__global__ void ComputeImminentImpulse_Kernel(Quadrature* quads, Vector3* deltas, int* delta_counts,
                                              Scalar imminent_thickness, bool add_repulsion,
                                              Scalar dt, int num_dcd_collisions) {
  GET_CUDA_ID(cid, num_dcd_collisions);

  Quadrature& quad = quads[cid];

  if (quad.nodes[0] == nullptr)
    return;

  Impulse impulse;

  if (!EvaluateImminentImpulse(quad, impulse, imminent_thickness, add_repulsion, dt))
    return;

  AccumulateImpulses(deltas, delta_counts, impulse);
}

__global__ void ApplyImminentImpulse_Kernel(Node** nodes, Vector3* dx_deltas, int* dx_delta_counts,
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

void ApplyImminentImpulse(Node** nodes, thrust::device_vector<Vector3>& dx_deltas,
                          thrust::device_vector<int>& dx_delta_counts, Scalar relaxation_rate) {
  int n_nodes = dx_deltas.size();

  CUDA_CALL(ApplyImminentImpulse_Kernel, n_nodes)
  (nodes, pointer(dx_deltas), pointer(dx_delta_counts), relaxation_rate, n_nodes);
  CUDA_CHECK_LAST();
}

__global__ void UpdateVelocity_Kernel(Node** nodes, Scalar dt, int n_nodes) {
  GET_CUDA_ID(i, n_nodes);

  nodes[i]->v = (nodes[i]->x - nodes[i]->x0) / dt;
}

void ImminentRepulsion::Solve(PhysicsMesh* cloth, Scalar imminent_thickness,
                              bool add_repulsion_force, Scalar relaxation_rate, Scalar dt,
                              int frame_index, int iter) {
  int n_nodes = cloth->NumNodes();
  CUDA_CALL(UpdateVelocity_Kernel, n_nodes)
  (pointer(cloth->nodes), dt, n_nodes);
  CUDA_CHECK_LAST();

  // evaluate imminent impulses
  int n_quads = quads_.size();
  CUDA_CALL(ComputeImminentImpulse_Kernel, n_quads)
  (pointer(quads_), pointer(deltas_), pointer(delta_counts_), imminent_thickness,
   add_repulsion_force, dt, n_quads);
  CUDA_CHECK_LAST();

  ApplyImminentImpulse(pointer(cloth->nodes), deltas_, delta_counts_, relaxation_rate);
}

__global__ void MapQuadIndex_Kernel(Quadrature* quads, QuadInd* quad_inds, int n_quad) {
  GET_CUDA_ID(qid, n_quad);

  for (int i = 0; i < 4; i++) {
    Node* node = quads[qid].nodes[i];
    quad_inds[qid].ids[i] = node->index;
  }
}

thrust::host_vector<QuadInd> ImminentRepulsion::HostQuadIndices() {
  int n_quad = quads_.size();

  thrust::device_vector<QuadInd> d_quad_inds(n_quad);
  CUDA_CALL(MapQuadIndex_Kernel, n_quad)
  (pointer(quads_), pointer(d_quad_inds), n_quad);
  CUDA_CHECK_LAST();

  thrust::host_vector<QuadInd> h_quad_inds = std::move(d_quad_inds);

  return h_quad_inds;
}

}  // namespace XRTailor