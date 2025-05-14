#pragma once

#include <xrtailor/core/Common.cuh>
#include <xrtailor/core/DeviceHelper.cuh>
#include <xrtailor/memory/Node.cuh>

namespace XRTailor {

struct HashParams {
  uint num_objects;
  uint max_num_neighbors;
  Scalar cell_spacing;
  Scalar cell_spacing2;
  int table_size;
  Scalar particle_diameter;
  Scalar particle_diameter2;
};

class SpatialHashGPU {
 public:
  SpatialHashGPU(Scalar particle_diameter, int max_num_objects, Scalar hash_cell_size,
                 int max_num_neighbors);

  // particles that are initially close won't generate collision in the future
  void SetInitialPositions(const Node* const* nodes, int n_nodes);

  void Hash(Node** nodes, int n_nodes);

  thrust::device_vector<uint> neighbors;
  thrust::device_vector<Vector3> initial_positions;
  thrust::device_vector<uint> particle_hash;
  thrust::device_vector<uint> particle_index;
  thrust::device_vector<uint> cell_start;
  thrust::device_vector<uint> cell_end;
 private:
  HashParams h_params_;
};

}  // namespace XRTailor