#pragma once

#include <xrtailor/physics/PhysicsMesh.cuh>
#include <xrtailor/physics/broad_phase/lbvh/BVH.cuh>
#include <xrtailor/core/DeviceHelper.cuh>
#include <xrtailor/physics/repulsion/ImminentRepulsionHelper.cuh>
#include <thrust/host_vector.h>

namespace XRTailor {

class ImminentRepulsion {
 public:
  ImminentRepulsion();

  void UpdateProximity(PhysicsMesh* cloth, PhysicsMesh* obstacle, BVH* cloth_bvh, BVH* obstacle_bvh,
                       Scalar thickness);

  void InitializeDeltas(int n_nodes);

  void Generate();

  void Solve(PhysicsMesh* cloth, Scalar imminent_thickness, bool add_repulsion_force,
             Scalar relaxationRate, Scalar dt, int frame_index, int iter);

  thrust::host_vector<QuadInd> HostQuadIndices();

 private:
  thrust::device_vector<Vector3> deltas_;
  thrust::device_vector<int> delta_counts_;
  thrust::device_vector<PairFF> pairs_;
  thrust::device_vector<Quadrature> quads_;
};

}  // namespace XRTailor