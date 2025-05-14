#include <xrtailor/utils/Timer.hpp>

#include <xrtailor/physics/broad_phase/spatial_hash/SpatialHashGPU.cuh>
#include <xrtailor/core/DeviceHelper.cuh>
#include <cub/device/device_radix_sort.cuh>

namespace XRTailor {

__device__ __inline__ int ComputeIntCoord(Scalar value, Scalar cell_spacing) {
  return static_cast<int>(floor(value / cell_spacing));
}

__device__ __inline__ int HashCoords(int x, int y, int z, int table_size) {
  int h = (x * 92837111) ^ (y * 689287499) ^ (z * 283923481);  // fantasy function
  return abs(h % table_size);
}

__device__ __inline__ int HashPosition(Vector3 position, Scalar cell_spacing, int table_size) {
  int x = ComputeIntCoord(position.x, cell_spacing);
  int y = ComputeIntCoord(position.y, cell_spacing);
  int z = ComputeIntCoord(position.z, cell_spacing);

  int h = HashCoords(x, y, z, table_size);
  return h;
}

__global__ void FindCellStart_Kernel(uint* cell_start, uint* cell_end, uint* particle_hash,
                                     int num_objects) {
  extern __shared__ uint shared_hash[];

  GET_CUDA_ID_NO_RETURN(id, num_objects);

  uint hash = particle_hash[id];
  shared_hash[threadIdx.x + 1] = hash;
  if (id > 0 && threadIdx.x == 0) {
    shared_hash[0] = particle_hash[id - 1];
  }
  __syncthreads();

  if (id >= num_objects)
    return;

  if (id == 0 || hash != shared_hash[threadIdx.x]) {
    cell_start[hash] = id;

    if (id > 0) {
      cell_end[shared_hash[threadIdx.x]] = id;
    }
  }

  if (id == num_objects - 1) {
    cell_end[hash] = id + 1;
  }
}

__global__ void CacheNeighbors_Kernel(uint* neighbors, CONST(uint*) particle_index,
                                      CONST(uint*) cell_start, CONST(uint*) cell_end, Node** nodes,
                                      CONST(Vector3*) original_positions, HashParams params) {
  GET_CUDA_ID(thread_id, params.num_objects);
  uint id = particle_index[thread_id];

  Vector3 position = nodes[id]->x;
  Vector3 original_pos = original_positions[id];
  int ix = ComputeIntCoord(position.x, params.cell_spacing);
  int iy = ComputeIntCoord(position.y, params.cell_spacing);
  int iz = ComputeIntCoord(position.z, params.cell_spacing);

  int neighbor_index = id;
  for (int x = ix - 1; x <= ix + 1; x++) {
    for (int y = iy - 1; y <= iy + 1; y++) {
      for (int z = iz - 1; z <= iz + 1; z++) {
        int h = HashCoords(x, y, z, params.table_size);
        int start = cell_start[h];
        if (start == 0xffffffff)
          continue;

        int end = min(cell_end[h], start + params.max_num_neighbors);

        for (int i = start; i < end; i++) {
          uint neighbor = particle_index[i];
          // ignore collision when particles are initially close
          if (neighbor != id &&
              (MathFunctions::Length2(position - nodes[neighbor]->x) <
               static_cast<Scalar>(params.cell_spacing2)) &&
              (MathFunctions::Length2(original_pos - original_positions[neighbor]) >
               static_cast<Scalar>(params.particle_diameter2))) {
            neighbors[neighbor_index] = neighbor;
            neighbor_index += params.num_objects;
            if (neighbor_index >= params.num_objects * params.max_num_neighbors)
              return;
          }
        }
      }
    }
  }
  if (neighbor_index < params.num_objects * params.max_num_neighbors) {
    neighbors[neighbor_index] = 0xffffffff;
  }
}

// cub::sort outperform thrust (roughly half time)
void Sort(uint* d_keys_in, uint* d_values_in, int num_items, int max_bit) {
  static void* d_temp_storage = NULL;
  static size_t temp_storage_bytes = 0;

  // Determine temporary device storage requirements
  size_t new_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs(NULL, new_storage_bytes, d_keys_in, d_keys_in, d_values_in,
                                  d_values_in, num_items);

  if (temp_storage_bytes != new_storage_bytes) {
    temp_storage_bytes = new_storage_bytes;
    checkCudaErrors(cudaMalloc((void**)&d_temp_storage, temp_storage_bytes));
  }

  // Run sorting operation
  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_in,
                                  d_values_in, d_values_in, num_items, 0, max_bit);
}

__global__ void ComputeParticleHash_Kernel(uint* particle_hash, uint* particle_index, Node** nodes,
                                           Scalar cell_spacing, int table_size, int n_nodes) {
  GET_CUDA_ID(id, n_nodes);
  particle_hash[id] = HashPosition(nodes[id]->x, cell_spacing, table_size);
  particle_index[id] = id;
}

void HashObjects(uint* particle_hash, uint* particle_index, uint* cell_start, uint* cell_end,
                 uint* neighbors, Node** nodes, CONST(Vector3*) original_positions,
                 HashParams h_params) {
  {
    ScopedTimerGPU timer("Solver_HashParticle");

    CUDA_CALL(ComputeParticleHash_Kernel, h_params.num_objects)
    (particle_hash, particle_index, nodes, h_params.cell_spacing, h_params.table_size,
     h_params.num_objects);
  }

  {
    ScopedTimerGPU timer("Solver_HashSort");
    int maxBit = static_cast<int>(ceil(log2(h_params.table_size)));
    Sort(particle_hash, particle_index, h_params.num_objects, maxBit);
  }

  {
    ScopedTimerGPU timer("Solver_HashBuildCell");
    cudaMemsetAsync(cell_start, 0xffffffff, sizeof(uint) * h_params.table_size, 0);
    uint num_blocks, num_threads;
    ComputeGridSize(h_params.num_objects, num_blocks, num_threads);
    uint smem_size = sizeof(uint) * (num_threads + 1);
    CUDA_CALL_V(FindCellStart_Kernel, num_blocks, num_threads, smem_size)
    (cell_start, cell_end, particle_hash, h_params.num_objects);
  }
  {
    ScopedTimerGPU timer("Solver_HashCache");
    CUDA_CALL(CacheNeighbors_Kernel, h_params.num_objects)
    (neighbors, particle_index, cell_start, cell_end, nodes, original_positions, h_params);
  }
  CUDA_CHECK_LAST();
}

SpatialHashGPU::SpatialHashGPU(Scalar particle_diameter, int max_num_objects, Scalar hash_cell_size,
                               int max_num_neighbors) {
  h_params_.cell_spacing = particle_diameter * hash_cell_size;
  h_params_.cell_spacing2 = h_params_.cell_spacing * h_params_.cell_spacing;
  h_params_.table_size = 2 * max_num_objects;
  h_params_.max_num_neighbors = max_num_neighbors;
  h_params_.particle_diameter = particle_diameter;
  h_params_.particle_diameter2 = particle_diameter * particle_diameter;

  neighbors.resize(max_num_objects * max_num_neighbors);
  particle_hash.resize(max_num_objects);
  particle_index.resize(max_num_objects);
  cell_start.resize(h_params_.table_size);
  cell_end.resize(h_params_.table_size);
  checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void SetInitialPositions_Kernel(const Node* const* nodes, Vector3* initial_positions,
                                           int n_nodes) {
  GET_CUDA_ID(i, n_nodes);
  initial_positions[i] = nodes[i]->x0;
}

void SpatialHashGPU::SetInitialPositions(const Node* const* nodes, int n_nodes) {
  initial_positions.resize(n_nodes);
  CUDA_CALL(SetInitialPositions_Kernel, n_nodes)
  (nodes, pointer(initial_positions), n_nodes);
  CUDA_CHECK_LAST();
}

void SpatialHashGPU::Hash(Node** nodes, int n_nodes) {

  h_params_.num_objects = n_nodes;

  HashObjects(pointer(particle_hash), pointer(particle_index), pointer(cell_start),
              pointer(cell_end), pointer(neighbors), nodes, pointer(initial_positions), h_params_);
}

}  // namespace XRTailor
