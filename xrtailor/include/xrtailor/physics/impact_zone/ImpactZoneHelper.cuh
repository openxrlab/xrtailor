#pragma once

#include <thrust/device_vector.h>

#include <xrtailor/memory/MemoryPool.cuh>
#include <xrtailor/core/DeviceHelper.cuh>
#include <xrtailor/physics/PhysicsMesh.cuh>
#include <xrtailor/physics/impact_zone/Impact.cuh>
#include <xrtailor/physics/impact_zone/ImpactZone.cuh>

namespace XRTailor {
namespace ImpactZoneOptimization {

struct ImpactZones {
  thrust::device_vector<int> colors;
  int n_colors;
  thrust::device_vector<int> impact_offsets;
  thrust::device_vector<int> node_offsets;
  thrust::device_vector<Impact> impacts;
  thrust::device_vector<int> active;
};

#define IZO_MAX_ZONES 10
#define IZO_MAX_NODES_PER_ZONE 10000
#define IZO_MAX_IMPACTS_PER_ZONE 2500

class ZoneAttribute {
 public:
  Scalar s;  // step length

  Scalar f;  // objective function
  Scalar O;
  Scalar C2 = 0;  // sum of norm2 constraints

  Scalar ft;  // advanced objective function
  Scalar Ot;
  Scalar C2t = 0;  // sum of norm2 constraints

  Scalar G2 = 0;       // sum of norm2 gradients
  Scalar lambda2 = 0;  // sum of norm2 Lagrange multipliers
  int impact_offset, node_offset;
  int n_impacts, n_nodes;
  bool converged, wolfe_condition_satisfied;
  int converge_reason = 0;
  int line_search_reason = 0;
  __host__ __device__ ZoneAttribute();
  ~ZoneAttribute() = default;
};

ImpactZones IsolateImpactZones(int deform, thrust::device_vector<Impact> independent_impacts,
                               std::shared_ptr<PhysicsMesh> cloth,
                               std::shared_ptr<PhysicsMesh> obstacle,
                               std::shared_ptr<MemoryPool> memory_pool, int frame_index,
                               int global_iter);

void AddImpacts(int deform, thrust::host_vector<ImpactZone>& zones,
                thrust::device_vector<Impact>& new_impacts, std::shared_ptr<PhysicsMesh> cloth,
                std::shared_ptr<PhysicsMesh> obstacle, std::shared_ptr<MemoryPool> memory_pool,
                int frame_index, int global_iter);

ImpactZones FlattenImpactZones(int n_impacts, thrust::host_vector<ImpactZone>& zones);

}  // namespace ImpactZoneOptimization
}  // namespace XRTailor