#pragma once

#include <xrtailor/core/Common.cuh>
#include <xrtailor/core/Common.hpp>
#include <xrtailor/core/DeviceHelper.cuh>
#include <xrtailor/math/MathFunctions.cuh>
#include <xrtailor/memory/Node.cuh>

namespace XRTailor {

void SolveGeodesicLRA(Node** nodes, CONST(uint*) src_indices,
                      CONST(uint*) tgt_indices, CONST(Scalar*) rest_length,
                      const uint num_constraints, Scalar long_range_stretchiness);

void SolveEuclideanLRA(Vector3* predicted, Vector3* deltas, int* delta_counts,
                       CONST(Scalar*) inv_mass, CONST(int*) attach_particle_ids,
                       CONST(int*) attach_slot_ids, CONST(Vector3*) attach_slot_positions,
                       CONST(Scalar*) attach_distances, const int num_constraints,
                       Scalar long_range_stretchiness);

}  // namespace XRTailor