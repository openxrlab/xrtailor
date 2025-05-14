#pragma once

#include <xrtailor/physics/repulsion/Impulse.cuh>

namespace XRTailor {

__device__ void AccumulateImpulses(Vector3* deltas, int* delta_counts, Impulse& impulse);

__host__ __device__ void CheckImminentVFQuadrature(Node* node, Face* face, Quadrature& quad);

__host__ __device__ void CheckImminentEEQuadrature(Edge* edge0, Edge* edge1, Quadrature& quad);

__device__ bool EvaluateImminentImpulse(Quadrature& quad, Impulse& impulse, const Scalar& thickness,
                                        const bool& add_repulsion, const Scalar& dt);

}  // namespace XRTailor