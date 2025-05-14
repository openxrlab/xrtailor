#pragma once

#include <xrtailor/physics/representative_triangle/RTriangle.cuh>
#include <xrtailor/utils/Timer.hpp>
#include <xrtailor/physics/broad_phase/lbvh/BVH.cuh>
#include <xrtailor/physics/repulsion/Proximity.cuh>
#include <xrtailor/memory/Edge.cuh>


namespace XRTailor {

class RepulsionSolver {
 public:
  RepulsionSolver();

  ~RepulsionSolver();

  void CheckProximity(BVH* cloth_bvh, BVH* obstacle_bvh, Face** cloth_faces, Face** obstacle_faces);

  void GenerateRepulsiveConstraints();

  void Solve();

 private:
  thrust::device_vector<PairFF> pairs_;

  thrust::device_vector<VFProximity> vf_proximities_;
  thrust::device_vector<EEProximity> ee_proximities_;
  thrust::device_vector<RTProximity> rt_proximities_;
};

}  // namespace XRTailor