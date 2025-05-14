#pragma once

#include <thrust/device_vector.h>

#include <xrtailor/core/DeviceHelper.cuh>
#include <xrtailor/physics/impact_zone/Impact.cuh>

namespace XRTailor {
namespace ImpactZoneOptimization {

class ImpactZone {
 public:
  thrust::device_vector<int> node_indices;
  thrust::device_vector<Impact> impacts;
  bool active;
  bool nodes_sorted;
  bool is_enabled;

 public:
  ImpactZone();

  ~ImpactZone();

  void FlattenNodes();

  void SortNodes();

  bool Enabled() const;

  void Enable();

  void Disable();

  bool IsActive() const;

  void SetActive();

  void SetPassive();

  bool Merge(ImpactZone& zone);

  bool Intersects(ImpactZone& zone);
};

}  // namespace ImpactZoneOptimization
}  // namespace XRTailor