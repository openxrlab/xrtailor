#include <xrtailor/physics/impact_zone/ImpactZone.cuh>

#include <thrust/set_operations.h>
#include <thrust/iterator/discard_iterator.h>

namespace XRTailor {
namespace ImpactZoneOptimization {

ImpactZone::ImpactZone() : active(false), nodes_sorted(false), is_enabled(true) {}

ImpactZone::~ImpactZone() {}

__global__ void FlattenNode_Kernel(int n_impact, int* node_indices, Impact* impacts) {
  GET_CUDA_ID(id, n_impact);

  for (int i = 0; i < 4; i++) {
    node_indices[id * 4 + i] = impacts[id].nodes[i]->index;
  }
}

void ImpactZone::FlattenNodes() {
  int n_impact = impacts.size();
  node_indices.resize(n_impact * 4);
  CUDA_CALL(FlattenNode_Kernel, n_impact)
  (n_impact, pointer(node_indices), pointer(impacts));
  CUDA_CHECK_LAST();
}

void ImpactZone::SortNodes() {
  if (nodes_sorted)
    return;

  thrust::sort(node_indices.begin(), node_indices.end());
  nodes_sorted = true;
}

bool ImpactZone::Enabled() const {
  return is_enabled;
}
void ImpactZone::Enable() {
  is_enabled = true;
}
void ImpactZone::Disable() {
  is_enabled = false;
}

bool ImpactZone::IsActive() const {
  return active;
}

void ImpactZone::SetActive() {
  active = true;
}

void ImpactZone::SetPassive() {
  active = false;
}

bool ImpactZone::Merge(ImpactZone& zone) {
  if (!zone.Enabled())
    return false;

  if (!Intersects(zone))
    return false;

  impacts.insert(impacts.end(), zone.impacts.begin(), zone.impacts.end());
  node_indices.insert(node_indices.end(), zone.node_indices.begin(), zone.node_indices.end());
  active = true;
  nodes_sorted = false;
  this->Enable();
  zone.Disable();

  return true;
}

bool ImpactZone::Intersects(ImpactZone& zone) {
  SortNodes();
  zone.SortNodes();

  thrust::discard_iterator<> I_begin, I_end;
  I_end = thrust::set_intersection(node_indices.begin(), node_indices.end(),
                                   zone.node_indices.begin(), zone.node_indices.end(), I_begin);

  return ((I_end - I_begin) > 0);
}

}  // namespace ImpactZoneOptimization
}  // namespace XRTailor