#pragma once

#include <xrtailor/core/DeviceHelper.cuh>
#include <xrtailor/physics/icm/IntersectionContourMinimizationHelper.cuh>
#include <xrtailor/physics/icm/GlobalIntersectionAnalysisHelper.cuh>
#include <xrtailor/physics/PhysicsMesh.cuh>
#include <xrtailor/physics/broad_phase/lbvh/BVH.cuh>

namespace XRTailor {
namespace Untangling {

class GlobalIntersectionAnalysis {
 public:
  GlobalIntersectionAnalysis();

  ~GlobalIntersectionAnalysis();

  int FloodFillIntersectionIslands(std::shared_ptr<PhysicsMesh> cloth,
                                   std::shared_ptr<BVH> cloth_bvh);

  thrust::device_vector<EdgeState> EdgeStates();

  thrust::device_vector<FaceState> FaceStates();

  thrust::host_vector<EFIntersection> HostIntersections();

  thrust::host_vector<IntersectionState> HostIntersectionStates();

 private:
  thrust::device_vector<EFIntersection> FindIntersections(std::shared_ptr<PhysicsMesh> cloth,
                                                          std::shared_ptr<BVH> cloth_bvh);
  thrust::device_vector<EdgeState> edge_states_;
  thrust::device_vector<FaceState> face_states_;
  thrust::device_vector<IntersectionState> intersection_states_;
  thrust::device_vector<EFIntersection> intersections_;
  int palette_;
};

}  // namespace Untangling
}  // namespace XRTailor