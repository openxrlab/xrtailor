#pragma once

#include <vector>
#include <set>

#include "curand.h"
#include "curand_kernel.h"

#include <xrtailor/core/DeviceHelper.cuh>
#include <xrtailor/memory/Edge.cuh>
#include <xrtailor/memory/Face.cuh>
#include <xrtailor/physics/graph_coloring/GraphColoringHelper.cuh>

namespace XRTailor {
namespace GraphColoring {

class GraphColoring {
 public:
  GraphColoring() {
    node_size = 0u;
    n_colors = 0u;
    n_steps = 0u;

    max_degree = 0u;
    min_degree = 9999u;
    avg_degree = 0.0f;
  };

  void Paint(const uint& seed, Scalar shrinking = 1.05f);

  uint ColorUsed();

  ~GraphColoring(){};

 public:
  thrust::device_vector<uint> colors;
  thrust::device_vector<uint> is_colored;
  thrust::device_vector<Palette> palettes;
  thrust::device_vector<curandState> rand_state;
  thrust::device_vector<AdjItem> adj_table;
  thrust::device_vector<AdjItem> adj_shared;

  uint node_size;
  uint n_steps;
  uint n_colors;
  uint max_degree;
  uint min_degree;
  Scalar avg_degree;
};

}  // namespace GraphColoring
}  // namespace XRTailor