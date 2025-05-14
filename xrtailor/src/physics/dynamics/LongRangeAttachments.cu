#include <xrtailor/physics/dynamics/LongRangeAttachments.cuh>
#include <xrtailor/utils/Timer.hpp>

namespace XRTailor {
__global__ void SolveGeodesicLRA_Kernel(Node** nodes, CONST(unsigned int*) geodesic_src_indices,
                                        CONST(unsigned int*) geodesic_tgt_indices,
                                        CONST(Scalar*) geodesic_rest_length,
                                        const unsigned int num_constraints,
                                        Scalar long_range_stretchiness) {
  GET_CUDA_ID(constraint_id, num_constraints);
  uint src_id = geodesic_src_indices[constraint_id];
  uint tgt_id = geodesic_tgt_indices[constraint_id];
  Scalar target_dist = geodesic_rest_length[constraint_id] * long_range_stretchiness;

  if (nodes[tgt_id]->inv_mass < EPSILON && target_dist > 0)
    return;

  Vector3 slot_pos = nodes[src_id]->x;
  Vector3 tgt_pos = nodes[tgt_id]->x;
  Vector3 diff = tgt_pos - slot_pos;
  Scalar dist = glm::length(diff);

  if (dist > target_dist) {
    nodes[tgt_id]->x += -diff + diff / dist * target_dist;
  }
}

void SolveGeodesicLRA(Node** nodes, CONST(unsigned int*) src_indices,
                      CONST(unsigned int*) tgt_indices, CONST(Scalar*) rest_length,
                      const unsigned int num_constraints, Scalar long_range_stretchiness) {
  CUDA_CALL(SolveGeodesicLRA_Kernel, num_constraints)
  (nodes, src_indices, tgt_indices, rest_length, num_constraints, long_range_stretchiness);
  CUDA_CHECK_LAST();
}

__global__ void SolveEuclideanLRA_Kernel(Vector3* predicted, Vector3* deltas, int* delta_counts,
                                         CONST(Scalar*) inv_mass, CONST(int*) attach_particle_ids,
                                         CONST(int*) attach_slot_ids,
                                         CONST(Vector3*) attach_slot_positions,
                                         CONST(Scalar*) attach_distances, const int num_constraints,
                                         Scalar long_range_stretchiness) {
  GET_CUDA_ID(id, num_constraints);

  uint pid = attach_particle_ids[id];

  Vector3 slot_pos = attach_slot_positions[attach_slot_ids[id]];
  Scalar target_dist = attach_distances[id] * long_range_stretchiness;
  if (inv_mass[pid] == 0 && target_dist > static_cast<Scalar>(0.0))
    return;

  Vector3 pred = predicted[pid];
  Vector3 diff = pred - slot_pos;
  Scalar dist = glm::length(diff);

  if (dist > target_dist) {
    Vector3 correction = -diff + diff / dist * target_dist;
    AtomicAdd(deltas, pid, correction, id);
    atomicAdd(&delta_counts[pid], 1);
  }
}

void SolveEuclideanLRA(Vector3* predicted, Vector3* deltas, int* delta_counts,
                       CONST(Scalar*) inv_mass, CONST(int*) attach_particle_ids,
                       CONST(int*) attach_slot_ids, CONST(Vector3*) attach_slot_positions,
                       CONST(Scalar*) attach_distances, const int num_constraints,
                       Scalar long_range_stretchiness) {
  ScopedTimerGPU timer("Solver_SolveAttach");
  CUDA_CALL(SolveEuclideanLRA_Kernel, num_constraints)
  (predicted, deltas, delta_counts, inv_mass, attach_particle_ids, attach_slot_ids, attach_slot_positions,
   attach_distances, num_constraints, long_range_stretchiness);
}

}  // namespace XRTailor