#include <xrtailor/physics/impact_zone/ImpactZoneHelper.cuh>
#include <iomanip>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/unique.h>
#include <thrust/set_operations.h>
#include <thrust/remove.h>

namespace XRTailor {
namespace ImpactZoneOptimization {
//#define IMPACT_ZONE_ISOLATION_DEBUG
#define NODE_OUTSIDE_IMPACT -2
#define NODE_INSIDE_IMPACT -1
#define IMPACT_NOT_COLORED -1
//#define IMPACT_ZONE_ISOLATION_WRITE_LOG

__device__ int num_impacts_per_zone;

ZoneAttribute::ZoneAttribute() {}

__global__ void InitializeNodeColor_Kernel(CONST(Impact*) impacts, int* cloth_node_colors,
                                           int* obstacle_node_colors, int n_impact) {
  GET_CUDA_ID(id, n_impact);

  const Impact& impact = impacts[id];

  for (int i = 0; i < 4; i++) {
    const Node* node = impact.nodes[i];
    if (node->is_cloth) {
      cloth_node_colors[node->index] = NODE_INSIDE_IMPACT;
    } else {
      obstacle_node_colors[node->index] = NODE_INSIDE_IMPACT;
    }
  }
}

__global__ void FloodFillColoring_Kernel(CONST(Impact*) impact, const Face* const* cloth_faces,
                                         const Face* const* obstacle_faces, int color,
                                         int* cloth_node_colors, int* obstacle_node_colors,
                                         CONST(uint*) cloth_nb_vf, CONST(uint*) cloth_nb_vf_prefix,
                                         CONST(uint*) obstacle_nb_vf,
                                         CONST(uint*) obstacle_nb_vf_prefix) {
  GET_CUDA_ID(id, 1);

  const Impact& _impact = impact[0];

  Node* stack[1024];
  Node** stack_ptr = stack;

  for (int i = 0; i < 4; i++) {
    *stack_ptr++ = _impact.nodes[i];
  }

  do {
    Node* node = *--stack_ptr;

    if (node->is_cloth) {
      if (node->is_free) {
        uint start = cloth_nb_vf_prefix[node->index];
        uint end = cloth_nb_vf_prefix[node->index + 1];
        for (int i = start; i < end; i++) {
          const Face* f = cloth_faces[cloth_nb_vf[i]];
          for (int j = 0; j < 3; j++) {
            int& src_color = cloth_node_colors[f->nodes[j]->index];
            if (src_color == NODE_INSIDE_IMPACT) {
#ifdef IMPACT_ZONE_ISOLATION_DEBUG
              printf("Face %d (%d/%d/%d) %d\n", f->index, f->nodes[0]->index, f->nodes[1]->index,
                     f->nodes[2]->index, j);
#endif  // IMPACT_ZONE_ISOLATION_DEBUG

              // XXX: single thread, no conflict memory access
              src_color = color;
              *stack_ptr++ = f->nodes[j];
            }
          }
        }
      }
    } else {
      uint start = obstacle_nb_vf_prefix[node->index];
      uint end = obstacle_nb_vf_prefix[node->index + 1];
      for (int i = start; i < end; i++) {
        const Face* f = obstacle_faces[obstacle_nb_vf[i]];
        for (int j = 0; j < 3; j++) {
          int& src_color = obstacle_node_colors[f->nodes[j]->index];
          if (src_color == NODE_INSIDE_IMPACT) {
            src_color = color;
            *stack_ptr++ = f->nodes[j];
          }
        }
      }
    }

  } while (stack < stack_ptr);
}

__global__ void WriteBackNodeColor_Kernel(CONST(Impact*) impacts, int* impact_colors,
                                          CONST(int*) cloth_node_colors,
                                          CONST(int*) obstacle_node_colors, int n_impact) {
  GET_CUDA_ID(id, n_impact);

  if (impact_colors[id] != IMPACT_NOT_COLORED)
    return;

  const Impact& impact = impacts[id];
  for (int i = 0; i < 4; i++) {
    const Node* node = impact.nodes[i];
    const int& node_color =
        node->is_cloth ? cloth_node_colors[node->index] : obstacle_node_colors[node->index];
    if (node_color != NODE_OUTSIDE_IMPACT && node_color != NODE_INSIDE_IMPACT) {
      impact_colors[id] = node_color;
      atomicAdd(&num_impacts_per_zone, 1);
      return;
    }
  }
}

__global__ void EncodeImpactNode_Kernel(Impact* impacts, int* node_indices, int* impact_colors,
                                        int* node_colors, int n_impacts) {
  GET_CUDA_ID(id, n_impacts);

  Impact& impact = impacts[id];
  int c = impact_colors[id];
  for (int i = 0; i < 4; i++) {
    node_indices[id * 4 + i] = impact.nodes[i]->index;
    if (impact.nodes[i]->is_cloth && impact.nodes[i]->is_free)
      node_colors[impact.nodes[i]->index] = c;
  }
}

ImpactZones IsolateImpactZones(int deform, thrust::device_vector<Impact> independent_impacts,
                               std::shared_ptr<PhysicsMesh> cloth,
                               std::shared_ptr<PhysicsMesh> obstacle,
                               std::shared_ptr<MemoryPool> memory_pool, int frame_index,
                               int global_iter) {
  int n_impacts = independent_impacts.size();
  int n_cloth_nodes = cloth->NumNodes();
  int n_obstacle_nodes = obstacle->NumNodes();

  ImpactZones zones;

  if (n_impacts == 0)
    return zones;

  zones.n_colors = 0;
  zones.colors.resize(n_impacts, -1);
  zones.impact_offsets.push_back(0);

  thrust::device_vector<int> cloth_node_colors(n_cloth_nodes, NODE_OUTSIDE_IMPACT);
  thrust::device_vector<int> obstacle_node_colors(n_obstacle_nodes, NODE_OUTSIDE_IMPACT);
#ifdef IMPACT_ZONE_ISOLATION_DEBUG
  printf("Initialize impact nodes\n");
#endif  // IMPACT_ZONE_ISOLATION_DEBUG

  CUDA_CALL(InitializeNodeColor_Kernel, n_impacts)
  (pointer(independent_impacts), pointer(cloth_node_colors), pointer(obstacle_node_colors),
   n_impacts);
  CUDA_CHECK_LAST();
  int h_num_impacts_per_zone;
  while (true) {
    if (zones.n_colors > 99) {
      printf("too many attempts, exit\n");
      break;
    }

    auto iter =
        thrust::find_if(zones.colors.begin(), zones.colors.end(),
                        [] __device__(const int& color) { return color == IMPACT_NOT_COLORED; });

    if (iter == zones.colors.end()) {
#ifdef IMPACT_ZONE_ISOLATION_DEBUG
      printf("All impacts have been colored, total colors: %d\n", zones_.n_colors);
#endif        // IMPACT_ZONE_ISOLATION_DEBUG
      break;  // all impacts have been colored
    }

    int pos = iter - zones.colors.begin();  // select candidate impact

#ifdef IMPACT_ZONE_ISOLATION_DEBUG
    printf("[color %d] Flood fill coloring using impact %d\n", zones_.n_colors, pos);
#endif  // IMPACT_ZONE_ISOLATION_DEBUG

    CUDA_CALL(FloodFillColoring_Kernel, 1)
    (pointer(independent_impacts, pos), pointer(cloth->faces), pointer(obstacle->faces),
     zones.n_colors, pointer(cloth_node_colors), pointer(obstacle_node_colors),
     pointer(memory_pool->cloth_nb_vf), pointer(memory_pool->cloth_nb_vf_prefix),
     pointer(memory_pool->obstacle_nb_vf), pointer(memory_pool->obstacle_nb_vf_prefix));
    CUDA_CHECK_LAST();

#ifdef IMPACT_ZONE_ISOLATION_DEBUG
    printf("[color %d] Write back\n", zones_.n_colors);
#endif  // IMPACT_ZONE_ISOLATION_DEBUG

    h_num_impacts_per_zone = 0;
    checkCudaErrors(cudaMemcpyToSymbol(num_impacts_per_zone, &h_num_impacts_per_zone, sizeof(int)));

    CUDA_CALL(WriteBackNodeColor_Kernel, n_impacts)
    (pointer(independent_impacts), pointer(zones.colors), pointer(cloth_node_colors),
     pointer(obstacle_node_colors), n_impacts);
    CUDA_CHECK_LAST();

    checkCudaErrors(
        cudaMemcpyFromSymbol(&h_num_impacts_per_zone, num_impacts_per_zone, sizeof(int)));
    zones.impact_offsets.push_back(h_num_impacts_per_zone);
#ifdef IMPACT_ZONE_ISOLATION_DEBUG
    printf("color %d, num_impacts_per_zone: %d\n", zones_.n_colors, h_num_impacts_per_zone);
#endif  // IMPACT_ZONE_ISOLATION_DEBUG

    zones.n_colors++;
  }

  thrust::device_vector<int> impact_indices(n_impacts);
  thrust::sequence(impact_indices.begin(), impact_indices.end());
  thrust::stable_sort_by_key(zones.colors.begin(), zones.colors.end(), impact_indices.begin());

  thrust::device_vector<Impact> sorted_impacts(n_impacts);
  thrust::gather(impact_indices.begin(), impact_indices.end(), independent_impacts.begin(),
                 sorted_impacts.begin());
  independent_impacts.swap(sorted_impacts);

  thrust::inclusive_scan(zones.impact_offsets.begin(), zones.impact_offsets.end(),
                         zones.impact_offsets.begin());

  return zones;
}

thrust::host_vector<ImpactZone> IsolateImpacts(int deform, thrust::device_vector<Impact> impacts,
                                               std::shared_ptr<PhysicsMesh> cloth,
                                               std::shared_ptr<PhysicsMesh> obstacle,
                                               std::shared_ptr<MemoryPool> memory_pool,
                                               int frame_index, int global_iter) {
  int n_impacts = impacts.size();
  int n_cloth_nodes = cloth->NumNodes();
  int n_obstacle_nodes = obstacle->NumNodes();

  thrust::host_vector<ImpactZone> zones;

  if (n_impacts == 0)
    return zones;

  int n_colors = 0;
  thrust::device_vector<int> impact_colors(n_impacts, -1);
  thrust::host_vector<int> h_impact_offsets;
  h_impact_offsets.push_back(0);

  thrust::device_vector<int> cloth_node_colors(n_cloth_nodes, NODE_OUTSIDE_IMPACT);
  thrust::device_vector<int> obstacle_node_colors(n_obstacle_nodes, NODE_OUTSIDE_IMPACT);
#ifdef IMPACT_ZONE_ISOLATION_DEBUG
  printf("Initialize impact nodes\n");
#endif  // IMPACT_ZONE_ISOLATION_DEBUG

  CUDA_CALL(InitializeNodeColor_Kernel, n_impacts)
  (pointer(impacts), pointer(cloth_node_colors), pointer(obstacle_node_colors), n_impacts);
  CUDA_CHECK_LAST();
  //checkCudaErrors(cudaDeviceSynchronize());
  int h_num_impacts_per_zone;
  while (true) {
    if (n_colors > 99) {
      printf("too many attempts, exit\n");
      break;
    }

    auto iter =
        thrust::find_if(impact_colors.begin(), impact_colors.end(),
                        [] __device__(const int& color) { return color == IMPACT_NOT_COLORED; });

    if (iter == impact_colors.end()) {
#ifdef IMPACT_ZONE_ISOLATION_DEBUG
      printf("All impacts have been colored, total colors: %d\n", zones_.size());
#endif        // IMPACT_ZONE_ISOLATION_DEBUG
      break;  // all impacts have been colored
    }

    int pos = iter - impact_colors.begin();  // select candidate impact

#ifdef IMPACT_ZONE_ISOLATION_DEBUG
    printf("[color %d] Flood fill coloring using impact %d\n", zones_.size(), pos);
#endif  // IMPACT_ZONE_ISOLATION_DEBUG

    CUDA_CALL(FloodFillColoring_Kernel, 1)
    (pointer(impacts, pos), pointer(cloth->faces), pointer(obstacle->faces), n_colors,
     pointer(cloth_node_colors), pointer(obstacle_node_colors), pointer(memory_pool->cloth_nb_vf),
     pointer(memory_pool->cloth_nb_vf_prefix), pointer(memory_pool->obstacle_nb_vf),
     pointer(memory_pool->obstacle_nb_vf_prefix));
    CUDA_CHECK_LAST();

#ifdef IMPACT_ZONE_ISOLATION_DEBUG
    checkCudaErrors(cudaDeviceSynchronize());
    printf("[color %d] Write back\n", zones_.size());
#endif  // IMPACT_ZONE_ISOLATION_DEBUG

    h_num_impacts_per_zone = 0;
    checkCudaErrors(cudaMemcpyToSymbol(num_impacts_per_zone, &h_num_impacts_per_zone, sizeof(int)));

    CUDA_CALL(WriteBackNodeColor_Kernel, n_impacts)
    (pointer(impacts), pointer(impact_colors), pointer(cloth_node_colors),
     pointer(obstacle_node_colors), n_impacts);
    CUDA_CHECK_LAST();

    checkCudaErrors(
        cudaMemcpyFromSymbol(&h_num_impacts_per_zone, num_impacts_per_zone, sizeof(int)));
    h_impact_offsets.push_back(h_num_impacts_per_zone);

#ifdef IMPACT_ZONE_ISOLATION_DEBUG
    printf("color %d, num_impacts_per_zone: %d\n", zones_.size(), h_num_impacts_per_zone);
#endif  // IMPACT_ZONE_ISOLATION_DEBUG

    n_colors++;
  }

  thrust::device_vector<int> impact_indices(n_impacts);
  thrust::sequence(impact_indices.begin(), impact_indices.end());
  thrust::stable_sort_by_key(impact_colors.begin(), impact_colors.end(), impact_indices.begin());

  thrust::device_vector<Impact> sorted_impacts(n_impacts);
  thrust::gather(impact_indices.begin(), impact_indices.end(), impacts.begin(),
                 sorted_impacts.begin());
  impacts.swap(sorted_impacts);

  thrust::inclusive_scan(h_impact_offsets.begin(), h_impact_offsets.end(),
                         h_impact_offsets.begin());

  for (int i = 0; i < n_colors; i++) {
    int lhs = h_impact_offsets[i];
    int rhs = h_impact_offsets[i + 1];
    ImpactZone z;
    z.impacts.insert(z.impacts.end(), impacts.begin() + lhs, impacts.begin() + rhs);
    z.active = true;
    z.FlattenNodes();
    zones.push_back(z);
  }

  return zones;
}

__global__ void FlattenImpactZone_Kernel(int n_impacts, int offset, CONST(Impact*) local_impacts,
                                         Impact* global_impacts, int local_color, int* colors) {
  GET_CUDA_ID(id, n_impacts);

  global_impacts[id + offset] = local_impacts[id];
  colors[id + offset] = local_color;
}

void AddImpacts(int deform, thrust::host_vector<ImpactZone>& zones,
                thrust::device_vector<Impact>& new_impacts, std::shared_ptr<PhysicsMesh> cloth,
                std::shared_ptr<PhysicsMesh> obstacle, std::shared_ptr<MemoryPool> memory_pool,
                int frame_index, int global_iter) {
  thrust::host_vector<ImpactZone> new_zones =
      IsolateImpacts(deform, new_impacts, cloth, obstacle, memory_pool, frame_index, global_iter);

  int n_zone = zones.size();
  int n_new_zone = new_zones.size();

  for (int i = 0; i < n_zone; i++) {
    zones[i].SetPassive();
  }

  // merge new zones
  for (int i = 0; i < n_zone; i++) {
    ImpactZone& zone = zones[i];
    for (int j = 0; j < n_new_zone; j++) {
      bool res = zone.Merge(new_zones[j]);
    }
  }

  // remaining new zones not merged to any existing zones
  for (int i = 0; i < n_new_zone; i++) {
    ImpactZone& new_zone = new_zones[i];
    if (new_zone.Enabled()) {
      zones.push_back(new_zone);
    }
  }
  n_zone = zones.size();
  for (int i = 0; i < n_zone; i++) {
    ImpactZone& zone = zones[i];
    for (int j = i + 1; j < n_zone; j++) {
      bool res = zone.Merge(zones[j]);
    }
  }

  zones.erase(thrust::remove_if(zones.begin(), zones.end(),
                                [](const ImpactZone& z) { return !z.Enabled(); }),
              zones.end());
}

ImpactZones FlattenImpactZones(int n_impacts, thrust::host_vector<ImpactZone>& zones) {
  int n_zone = zones.size();
  ImpactZones merged_zone;
  merged_zone.impact_offsets.push_back(0);
  int offset = 0;
  int n_colors = 0;
  for (int i = 0; i < n_zone; i++) {
    const ImpactZone& z = zones[i];

    if (!z.IsActive())
      continue;

    int n_local_impact = z.impacts.size();
    merged_zone.impact_offsets.push_back(n_local_impact);
    merged_zone.impacts.insert(merged_zone.impacts.end(), z.impacts.begin(), z.impacts.end());
    merged_zone.colors.insert(merged_zone.colors.end(), n_local_impact, n_colors);
    merged_zone.active.push_back(z.active);
    CUDA_CALL(FlattenImpactZone_Kernel, n_local_impact)
    (n_local_impact, offset, pointer(z.impacts), pointer(merged_zone.impacts), n_colors,
     pointer(merged_zone.colors));
    CUDA_CHECK_LAST();
    offset += n_local_impact;
    n_colors++;
  }
  merged_zone.n_colors = n_colors;

  thrust::inclusive_scan(merged_zone.impact_offsets.begin(), merged_zone.impact_offsets.end(),
                         merged_zone.impact_offsets.begin());

  return merged_zone;
}

}  // namespace ImpactZoneOptimization
}  // namespace XRTailor
