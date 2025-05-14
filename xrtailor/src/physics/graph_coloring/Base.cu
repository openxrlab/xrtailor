#include <xrtailor/physics/graph_coloring/Base.cuh>
#include <xrtailor/utils/Timer.hpp>
#include <xrtailor/core/DeviceHelper.cuh>

//#define PGS_DEBUG
//#define PGS_DEBUG_DETAILED

namespace XRTailor {
namespace GraphColoring {
__device__ bool has_uncolored;

__global__ __inline__ void Initialization_Kernel(uint* is_colored, Palette* palettes,
                                                 AdjItem* adj_table, Scalar shrinking,
                                                 int node_size) {
  GET_CUDA_ID(idx, node_size);

  is_colored[idx] = 0u;

  palettes[idx].valid_size =
      static_cast<int>(static_cast<Scalar>(adj_table[idx].valid_size) / shrinking) + 1;
#ifdef PGS_DEBUG_DETAILED
  printf("%d palette size: %d = %d / %.2f\n", idx, palettes[idx].valid_size,
         adj_table[idx].valid_size, shrinking);
#endif  // PGS_DEBUG_DETAILED
  for (uint i = 0u; i < palettes[idx].valid_size; ++i) {
    palettes[idx].colors[i] = 1u;
  }
}

__global__ __inline__ void SetupRandState_Kernel(curandState* rand_state, int seed, int node_size) {
  GET_CUDA_ID(idx, node_size);

  curand_init(seed, idx, 0, &rand_state[idx]);
}

__global__ __inline__ void TentativeColoring_Kernel(curandState* rand_state, uint* is_colored,
                                                    Palette* palettes, AdjItem* adj_table,
                                                    uint* colors, Scalar shrinking,
                                                    uint node_size) {
  GET_CUDA_ID(idx, node_size);

  if (is_colored[idx] == 1u)
    return;
  // select a color between [0, \delta_{v} / s]
  uint rand_color =
      static_cast<int>((curand_uniform(&rand_state[idx]) *
                        (palettes[idx].valid_size - 1 + static_cast<Scalar>(.999999))));
  uint loop_color = rand_color;
  uint prev_color = colors[idx];
  bool visited = false;
  while (true) {
    if (palettes[idx].colors[loop_color] == 1)  // color is available on the palette
    {
      colors[idx] = loop_color;
#ifdef PGS_DEBUG_DETAILED
      printf("%d rand_idx: %d, palette size: %d, rand color: %d\n", idx, rand_color,
             palettes[idx].valid_size, loop_color);
#endif  // PGS_DEBUG_DETAILED
      break;
    } else  // not available, try next color
    {
      loop_color = (loop_color + 1) % palettes[idx].valid_size;
#ifdef PGS_DEBUG_DETAILED
      printf("%d selected color %d is not available, next...\n", idx,
             palettes[idx].colors[loop_color]);
#endif  // PGS_DEBUG_DETAILED
    }
  }
}

__global__ __inline__ void ConflictResolution_Kernel(uint* is_colored, Palette* palettes,
                                                     AdjItem* adj_table, uint* colors,
                                                     uint node_size) {
  GET_CUDA_ID(idx, node_size);

  if (is_colored[idx] == 1u)
    return;

  bool is_conflict = false;
  bool is_highest_index = true;
  for (auto i = 0u; i < adj_table[idx].valid_size; i++) {
    uint neighbor_idx = adj_table[idx].adjs[i];
    uint neighbor_color = colors[neighbor_idx];
    if (neighbor_color == colors[idx]) {
      is_conflict = true;
#ifdef PGS_DEBUG_DETAILED
      printf("node %d color %d conflicts with neighbor %d, neighbor is colored: %d\n", idx,
             colors[idx], neighbor_idx, is_colored[neighbor_idx]);
#endif  // PGS_DEBUG_DETAILED
    }
    if ((!is_colored[neighbor_idx]) && (idx < neighbor_idx)) {
#ifdef PGS_DEBUG_DETAILED
      printf("node %d marked not as highest index due to it is less than %d\n", idx, neighbor_idx);
#endif  // PGS_DEBUG_DETAILED
      is_highest_index = false;
    }
  }
  if (!is_highest_index && is_conflict)
    return;

#ifdef PGS_DEBUG_DETAILED
  if (is_conflict && is_highest_index)
    printf("node %d is the highest index, accept as Hungarian heuristic\n", idx);
  else
    printf("node %d color %d does not has any conflicts, accept\n", idx, colors[idx]);
#endif  // PGS_DEBUG_DETAILED

  // attended color does not belong to its vertex neighbor colors
  for (uint i = 0u; i < adj_table[idx].valid_size; i++) {
#ifdef PGS_DEBUG_DETAILED
    printf("node: %d, removing color %d from adj %d\n", idx, colors[idx], adj_table[idx].adjs[i]);
#endif  // PGS_DEBUG_DETAILED \
    //atomicCAS(palettes[adj_table[idx].adjs[i]].colors + colors[idx], 1, 0);
    palettes[adj_table[idx].adjs[i]].colors[colors[idx]] = 0u;
  }
  is_colored[idx] = 1u;
}

__global__ __inline__ void FeedTheHungry_Kernel(uint* is_colored, Palette* palettes,
                                                uint node_size) {
  GET_CUDA_ID(idx, node_size);

  if (is_colored[idx] == 1u)
    return;

  uint count = 0u;
  for (auto i = 0u; i < palettes[idx].valid_size; i++) {
    if (palettes[idx].colors[i] == 1u)
      count += 1u;
  }
  if (count == 0u) {
    palettes[idx].colors[palettes[idx].valid_size] = 1u;
    palettes[idx].valid_size += 1u;
#ifdef PGS_DEBUG_DETAILED
    printf("palette %d is hungry, feed from: %d to %d\n", idx, palettes[idx].valid_size - 1,
           palettes[idx].valid_size);
#endif  // PGS_DEBUG_DETAILED
  }
}

__global__ void ValidateColored_Kernel(uint* is_colored, uint node_size) {
  GET_CUDA_ID(idx, node_size);

  if (!is_colored[idx])
    has_uncolored = true;
}

uint GraphColoring::ColorUsed() {
  std::vector<int> color_counts;
  color_counts.resize(MAX_DEGREE);
  for (uint i = 0u; i < node_size; i++) {
    color_counts[colors[i]] += 1u;
  }
  uint color_used = 0u;
  for (uint i = 0u; i < MAX_DEGREE; i++) {
    if (color_counts[i] != 0u) {
      color_used += 1u;
    }
  }
  return color_used;
}

void GraphColoring::Paint(const uint& seed, Scalar shrinking) {
  n_steps = 0;
  n_colors = 0;
  CUDA_CALL(Initialization_Kernel, node_size)
  (pointer(is_colored), pointer(palettes), pointer(adj_table), shrinking, node_size);

#ifdef PGS_DEBUG
  printf("initialization using seed %d\n", seed);
#endif  // PGS_DEBUG

  CUDA_CALL(SetupRandState_Kernel, node_size)
  (pointer(rand_state), seed, node_size);

#ifdef PGS_DEBUG
  printf("setup rand state\n");
  checkCudaErrors(cudaDeviceSynchronize());
#endif  // PGS_DEBUG

  bool all_colored = false;
  while (!all_colored) {
#ifdef PGS_DEBUG
    printf("------------------------------------\n", n_steps);
#endif  // PGS_DEBUG
    CUDA_CALL(TentativeColoring_Kernel, node_size)
    (pointer(rand_state), pointer(is_colored), pointer(palettes), pointer(adj_table),
     pointer(colors), shrinking, node_size);
#ifdef PGS_DEBUG
    checkCudaErrors(cudaDeviceSynchronize());
    printf("=>step%d: tentative coloring\n", n_steps);
#endif  // PGS_DEBUG

#ifdef PGS_DEBUG_DETAILED
    printf("Palettes:\n");
    for (auto i = 0u; i < node_size; i++) {
      auto palette = palettes[i];
      printf(" - palette %d: ", i);
      for (auto j = 0u; j < palette.valid_size; j++) {
        printf("%d(%d) ", j, palette.colors[j]);
      }
      printf("\n");
    }
#endif  // PGS_DEBUG_DETAILED

    CUDA_CALL(ConflictResolution_Kernel, node_size)
    (pointer(is_colored), pointer(palettes), pointer(adj_table), pointer(colors), node_size);
#ifdef PGS_DEBUG
    checkCudaErrors(cudaDeviceSynchronize());
    printf("=>step%d: conflict resolution\n", n_steps);
#endif  // PGS_DEBUG

#ifdef PGS_DEBUG_DETAILED
    printf("colors: ");
    for (auto i = 0u; i < node_size; i++) {
      printf("%d ", colors[i]);
    }
    printf("\nPalettes:\n");
    for (auto i = 0u; i < node_size; i++) {
      auto palette = palettes[i];
      printf(" - palette %d: ", i);
      for (auto j = 0u; j < palette.valid_size; j++) {
        printf("%d(%d) ", j, palette.colors[j]);
      }
      printf("\n");
    }
#endif  // PGS_DEBUG_DETAILED

    CUDA_CALL(FeedTheHungry_Kernel, node_size)
    (pointer(is_colored), pointer(palettes), node_size);
#ifdef PGS_DEBUG
    //checkCudaErrors(cudaDeviceSynchronize());
    printf("=>step%d: feed the hungry\n", n_steps);
#endif  // PGS_DEBUG

#ifdef PGS_DEBUG
    std::vector<int> color_counts;
    color_counts.resize(MAX_DEGREE);
    for (uint i = 0u; i < node_size; i++) {
      color_counts[colors[i]] += 1u;
    }
    uint color_used = 0u;
    for (uint i = 0u; i < MAX_DEGREE; i++) {
      if (color_counts[i] != 0u) {
        color_used += 1u;
      }
    }
    printf("=>step%d: %d colors used\n", n_steps, color_used);
    for (uint i = 0u; i < MAX_DEGREE; i++) {
      printf("  - color %d: %d nodes\t\t", i, color_counts[i]);
      if (i > 0 && ((i + 1) % 4 == 0u || i == (MAX_DEGREE - 1))) {
        printf("\n");
      }
    }
    uint colored = 0;
    uint uncolored = 0;
    for (auto i = 0u; i < node_size; i++) {
      if (is_colored[i] == 0u)
        uncolored++;
      else
        colored++;
    }
    printf("nodes: %d, colored: %d, uncolored: %d\n", node_size, colored, uncolored);

#endif  // PGS_DEBUG
    all_colored = true;
    bool h_has_uncolored = false;
    checkCudaErrors(cudaMemcpyToSymbol(has_uncolored, &h_has_uncolored, sizeof(bool)));
    CUDA_CALL(ValidateColored_Kernel, node_size)
    (pointer(is_colored), node_size);
    CUDA_CHECK_LAST();
    checkCudaErrors(cudaMemcpyFromSymbol(&h_has_uncolored, has_uncolored, sizeof(bool)));
    if (h_has_uncolored)
      all_colored = false;

#ifdef PGS_DEBUG_DETAILED
    printf("color: ");
    for (auto i = 0u; i < node_size; i++) {
      printf("%d(%d) ", i, colors[i]);
    }
    printf("\n");

    printf("is_colored: ");
    for (auto i = 0u; i < node_size; i++) {
      printf("%d(%d) ", i, is_colored[i]);
    }
    printf("\n");
#endif  // PGS_DEBUG_DETAILED

    //if (step > 999)
    //{
    //	printf("max iteration reached, exit loop\n");
    //	break;
    //}
    n_steps++;
  }

#ifdef PGS_DEBUG_DETAILED
  for (auto i = 0u; i < node_size; i++) {
    printf("node: %d, color: %d\n", i, colors[i]);
  }
  //std::vector<int> color_counts;
  //color_counts.resize(MAX_DEGREE);
  //for (uint i = 0u; i < node_size; i++)
  //{
  //	color_counts[colors[i]] += 1u;
  //}
  //for (uint i = 0u; i < MAX_DEGREE; i++)
  //{
  //	printf("  - color %d: %d nodes\t\t", i, color_counts[i]);
  //	if (i > 0 && ((i + 1) % 4 == 0u || i == (MAX_DEGREE - 1)))
  //	{
  //		printf("\n");
  //	}
  //}
#endif  // PGS_DEBUG_DETAILED
}

}  // namespace GraphColoring
}  // namespace XRTailor