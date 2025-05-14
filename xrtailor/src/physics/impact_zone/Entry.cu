#include <xrtailor/physics/impact_zone/Entry.cuh>

#include <iomanip>

#include <xrtailor/utils/Timer.hpp>

#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/unique.h>
#include "math_constants.h"

#include <xrtailor/utils/FileSystemUtils.hpp>
#include <xrtailor/physics/broad_phase/lbvh/BVH.cuh>
#include <xrtailor/core/DeviceHelper.cuh>
#include <xrtailor/physics/impact_zone/ImpactZoneOptimizer.cuh>
#include <xrtailor/physics/impact_zone/ParallelImpactZoneOptimizer.cuh>
#include <xrtailor/physics/impact_zone/ImpactZoneHelper.cuh>
#include <xrtailor/physics/impact_zone/ImpactHelper.cuh>

namespace XRTailor {
namespace ImpactZoneOptimization {
thrust::device_vector<Impact> independentImpacts(const thrust::device_vector<Impact>& impacts,
                                                 int deform) {
  thrust::device_vector<Impact> sorted = impacts;

  // sort the impacts by the TOI by the ascending order
  thrust::sort(sorted.begin(), sorted.end());

  int n_impacts = sorted.size();
  const Impact* sorted_pointer = pointer(sorted);
  int n_nodes = 4 * n_impacts;  // number of nodes related to the impacts
  thrust::device_vector<Impact> ans(n_impacts);
  Impact* ans_pointer = pointer(ans);
  InitializeImpactNodes(n_impacts, sorted_pointer, deform);
  CUDA_CHECK_LAST();
  int num, new_num = n_impacts;
  do {
    num = new_num;
    thrust::device_vector<Node*> nodes(n_nodes);
    thrust::device_vector<int> relative_impacts(n_nodes);
    // determine which node belongs to which impact
    CollectRelativeImpacts(n_impacts, sorted_pointer, deform, pointer(nodes),
                           pointer(relative_impacts));
    CUDA_CHECK_LAST();
    nodes.erase(thrust::remove(nodes.begin(), nodes.end(), nullptr), nodes.end());
    relative_impacts.erase(thrust::remove(relative_impacts.begin(), relative_impacts.end(), -1),
                          relative_impacts.end());

    // sort the node pointers by their memory address
    thrust::sort_by_key(nodes.begin(), nodes.end(), relative_impacts.begin());
    thrust::device_vector<Node*> output_nodes(n_nodes);
    thrust::device_vector<int> output_relative_impacts(n_nodes);
    auto iter = thrust::reduce_by_key(nodes.begin(), nodes.end(), relative_impacts.begin(),
                                      output_nodes.begin(), output_relative_impacts.begin(),
                                      thrust::equal_to<Node*>(), thrust::minimum<int>());
    SetImpactMinIndices(iter.first - output_nodes.begin(), pointer(output_relative_impacts),
                        pointer(output_nodes));

    CheckIndependentImpacts(n_impacts, sorted_pointer, deform, ans_pointer);
    new_num = thrust::count_if(ans.begin(), ans.end(), IsNull());
  } while (num > new_num);

  ans.erase(thrust::remove_if(ans.begin(), ans.end(), IsNull()), ans.end());
  CUDA_CHECK_LAST();

  return ans;
}

__global__ void ClearColors_Kernel(Node** nodes, int n_nodes) {
  GET_CUDA_ID(i, n_nodes);

  nodes[i]->color = 0;
}

__global__ void ClearObstacleColors_Kernel(Node** nodes, int n_nodes) {
  GET_CUDA_ID(i, n_nodes);

  nodes[i]->color = 0;
}

__global__ void SetupColor_Kernel(Node** cloth_nodes, Node** obstacle_nodes, Impact* impacts,
                                  int n_impacts) {
  GET_CUDA_ID(i, n_impacts);

  int idx0 = impacts[i].nodes[0]->index;
  int idx1 = impacts[i].nodes[1]->index;
  int idx2 = impacts[i].nodes[2]->index;
  int idx3 = impacts[i].nodes[3]->index;

  if (impacts[i].nodes[0]->is_cloth)
    cloth_nodes[idx0]->color = 1;
  else
    obstacle_nodes[idx0]->color = 4;

  if (impacts[i].nodes[1]->is_cloth)
    cloth_nodes[idx1]->color = 1;
  else
    obstacle_nodes[idx1]->color = 4;

  if (impacts[i].nodes[2]->is_cloth)
    cloth_nodes[idx2]->color = 1;
  else
    obstacle_nodes[idx2]->color = 4;

  if (impacts[i].nodes[3]->is_cloth)
    cloth_nodes[idx3]->color = 1;
  else
    obstacle_nodes[idx3]->color = 4;
}

__global__ void SetupNode_Kernel(Node** nodes, int n_nodes) {
  GET_CUDA_ID(i, n_nodes);

  nodes[i]->x1 = nodes[i]->x;
}

__global__ void SetupObstacleNode_Kernel(Node** nodes, Scalar dt, int n_nodes) {
  GET_CUDA_ID(i, n_nodes);

  nodes[i]->x1 = nodes[i]->x;
  nodes[i]->is_free = false;
  nodes[i]->v = (nodes[i]->x - nodes[i]->x0) / dt;
  nodes[i]->removed = false;
}

__global__ void FillNodeColor_Kernel(int n_impact, Node** nodes, int* colors, Impact* impacts,
                                     int* impact_colors) {
  GET_CUDA_ID(id, n_impact);

  const Impact& impact = impacts[id];
  int color = impact_colors[id];
  for (int i = 0; i < 4; i++) {
    colors[impact.nodes[i]->index] = color;
  }
}

void CollisionStep(std::shared_ptr<PhysicsMesh> cloth, std::shared_ptr<PhysicsMesh> obstacle,
                   std::shared_ptr<MemoryPool> memory_pool, int frame_index, Scalar dt,
                   const Scalar& init_obstacle_mass, const Scalar& collision_thickness) {
  CUDA_CALL(SetupNode_Kernel, cloth->NumNodes())
  (pointer(cloth->nodes), cloth->NumNodes());
  CUDA_CALL(SetupObstacleNode_Kernel, obstacle->NumNodes())
  (pointer(obstacle->nodes), dt, obstacle->NumNodes());
  CUDA_CHECK_LAST();

  std::shared_ptr<BVH> cloth_bvh = std::make_shared<BVH>();
  std::shared_ptr<BVH> obstacle_bvh = std::make_shared<BVH>();

  cloth_bvh->UpdateData(pointer(cloth->faces), 0.0f, true, cloth->NumFaces());
  cloth_bvh->Construct(0.0f);
  obstacle_bvh->UpdateData(pointer(obstacle->faces), 0.0f, true, obstacle->NumFaces());
  obstacle_bvh->Construct(0.0f);

  thrust::host_vector<uint> h_indices = cloth->HostIndices();
  thrust::host_vector<unsigned int> h_obstacle_indices = obstacle->HostIndices();
  int deform;
  Scalar obstacle_mass = init_obstacle_mass;
  bool success = false;

  thrust::device_vector<Impact> impacts;

#ifdef IZO_PARALLEL_OPTIMIZATION
  thrust::host_vector<ImpactZoneOptimization::ImpactZone> impact_zones;
#endif  // IZO_PARALLEL_OPTIMIZATION

  int n_zones = 0;
  for (deform = 0; deform < 2; deform++) {
    impacts.clear();
#ifdef IZO_PARALLEL_OPTIMIZATION
    impact_zones.clear();
#endif  // IZO_PARALLEL_OPTIMIZATION
    for (int i = 0; i < MAX_COLLISION_ITERATION; i++) {
      thrust::device_vector<Impact> new_impacts;
      new_impacts =
          std::move(FindImpacts(cloth_bvh, obstacle_bvh, pointer(cloth->faces),
                                pointer(obstacle->faces), collision_thickness, frame_index, i));

      if (new_impacts.empty()) {
        success = true;
#ifdef COLLISION_STEP_WRITE_LOG
        ofs << "Global: Collision converged in " << i << " iterations\n";
#endif  // COLLISION_STEP_WRITE_LOG
        break;
      }

      CUDA_CALL(ClearColors_Kernel, cloth->NumNodes())
      (pointer(cloth->nodes), cloth->NumNodes());
      CUDA_CALL(ClearObstacleColors_Kernel, obstacle->NumNodes())
      (pointer(obstacle->nodes), obstacle->NumNodes());
      CUDA_CHECK_LAST();

      new_impacts = std::move(independentImpacts(new_impacts, deform));
      impacts.insert(impacts.end(), new_impacts.begin(), new_impacts.end());
#ifdef COLLISION_STEP_WRITE_LOG
      ofs << "[iter " << i << "] new independent impacts: " << new_impacts.size()
          << ", total impacts to solve:" << impacts.size() << std::endl;
#endif  // COLLISION_STEP_WRITE_LOG

      std::cout << "[frame" << frame_index << ", iter" << i
                << "] new independent impacts: " << new_impacts.size()
                << ", total impacts to solve:" << impacts.size() << std::endl;

#ifdef IZO_PARALLEL_OPTIMIZATION
      Timer::StartTimerGPU("IZ_Isolation");
      ImpactZoneOptimization::AddImpacts(deform, impact_zones, new_impacts, cloth, obstacle,
                                          memory_pool, frame_index, i);
      ImpactZoneOptimization::ImpactZones merged_zone =
          ImpactZoneOptimization::FlattenImpactZones(impacts.size(), impact_zones);

      Timer::EndTimerGPU("IZ_Isolation");
      double t2 = Timer::GetTimerGPU("IZ_Isolation");
#endif  // IZO_PARALLEL_OPTIMIZATION

      CUDA_CALL(SetupColor_Kernel, impacts.size())
      (pointer(cloth->nodes), pointer(obstacle->nodes), pointer(impacts), impacts.size());
      CUDA_CHECK_LAST();

#ifdef IZO_PARALLEL_OPTIMIZATION
      ImpactZoneOptimization::ParallelImpactZoneOptimizer* optimizator =
          new ImpactZoneOptimization::ParallelImpactZoneOptimizer(collision_thickness, deform,
                                                                  obstacle_mass, merged_zone);
      optimizator->Solve(frame_index, i);
      delete optimizator;
#else
      ImpactZoneOptimizer* optimizator =
          new ImpactZoneOptimizer(impacts, collision_thickness, deform, obstacle_mass_);
      optimizator->Solve(frame_index, i);
      delete optimizator;
#endif  // IZO_PARALLEL_OPTIMIZATION

      cloth_bvh->Update(pointer(cloth->faces), true);
      if (deform == 1) {
        obstacle_bvh->Update(pointer(obstacle->faces), true);
        obstacle_mass *= 0.5f;
      }

      if (i >= 99) {
        std::cout << "Collision step failed." << std::endl;
#ifdef COLLISION_STEP_WRITE_LOG
        ofs << "Collision step failed." << std::endl;
        ofs.close();
#endif  // COLLISION_STEP_WRITE_LOG
      }
    }
    if (success) {
      break;
    }
  }

  if (!success) {
    std::cout << "Collision step failed." << std::endl;
  }
}
}  // namespace ImpactZoneOptimization
}  // namespace XRTailor