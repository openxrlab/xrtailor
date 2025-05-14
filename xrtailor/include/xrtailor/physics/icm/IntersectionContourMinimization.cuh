#pragma once

#include <xrtailor/core/DeviceHelper.cuh>
#include <xrtailor/math/BasicPrimitiveTests.cuh>
#include <xrtailor/physics/broad_phase/lbvh/BVH.cuh>
#include <xrtailor/physics/icm/IntersectionContourMinimizationHelper.cuh>
#include <xrtailor/physics/icm/GlobalIntersectionAnalysis.cuh>

namespace XRTailor {
namespace Untangling {

class ICM {
 public:
  ICM();

  ICM(int n_nodes, int n_edges);

  ~ICM();

  void UpdatePairs(PhysicsMesh* cloth, PhysicsMesh* obstacle, BVH* cloth_bvh, BVH* obstacle_bvh);

  void UpdateGradient(PhysicsMesh* cloth, PhysicsMesh* obstacle);

  bool UpdateGradientGIA(PhysicsMesh* cloth, PhysicsMesh* obstacle, const int& n_contours,
                         GlobalIntersectionAnalysis* gia);

  void ApplyImpulse(PhysicsMesh* cloth, Scalar h0, Scalar g0);

  void ApplyGradient(PhysicsMesh* cloth, Scalar h0, Scalar g0);

  thrust::host_vector<IntersectionWithGradient> HostIntersections();

 private:
  thrust::device_vector<PairFF> pairs_;
  thrust::device_vector<Vector3> gradients_;
  thrust::device_vector<Vector3> deltas_;
  thrust::device_vector<int> delta_counts_;
  thrust::device_vector<Vector3> impulses_;

  thrust::device_vector<IntersectionWithGradient> intersections_;
  thrust::device_vector<EFIntersection> gia_intersections_;
};

void DetangleStep(PhysicsMesh* cloth, PhysicsMesh* obstacle, BVH* cloth_bvh, BVH* obstacle_bvh,
                  const Scalar& g0, const Scalar& h0);

}  // namespace Untangling
}  // namespace XRTailor