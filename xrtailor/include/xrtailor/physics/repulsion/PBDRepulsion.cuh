#pragma once

#include <xrtailor/physics/PhysicsMesh.cuh>
#include <xrtailor/physics/broad_phase/lbvh/BVH.cuh>
#include <xrtailor/core/DeviceHelper.cuh>
#include <xrtailor/physics/repulsion/Proximity.cuh>
#include <xrtailor/physics/repulsion/Impulse.cuh>

namespace XRTailor {

class PBDRepulsionSolver {
 public:
  PBDRepulsionSolver();

  PBDRepulsionSolver(PhysicsMesh* cloth);

  ~PBDRepulsionSolver();

  void UpdatePairs(PhysicsMesh* cloth, PhysicsMesh* obstacle, BVH* cloth_bvh, BVH* obstacle_bvh,
                   Scalar thickness);

  void UpdateProximities();

  void Solve(PhysicsMesh* cloth, Scalar thickness, Scalar relaxationRate, Scalar dt,
             int frame_index, int iter);

  void SolveNoProximity(PhysicsMesh* cloth, Scalar thickness, Scalar relaxationRate, Scalar dt,
                        int frame_index, int iter);

  thrust::host_vector<QuadInd> HostQuadIndices();

 private:
  thrust::device_vector<PairFF> pairs_;
  thrust::device_vector<Proximity> proximities_;

  thrust::device_vector<Vector3> deltas_;
  thrust::device_vector<int> delta_counts_;

  thrust::device_vector<Quadrature> quads_;
};

}  // namespace XRTailor