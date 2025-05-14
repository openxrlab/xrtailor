#pragma once

#include <xrtailor/core/DeviceHelper.cuh>
#include <xrtailor/physics/graph_coloring/Defs.cuh>

namespace XRTailor {
namespace GraphColoring {

struct AdjItem {
  static const uint max_size = MAX_DEGREE;
  uint valid_size;
  uint adjs[max_size];
};

struct Palette {
  static const uint max_size = MAX_DEGREE;
  uint valid_size;
  uint colors[max_size];
};

void BuildBendAdjacencyTable(CONST(uint*) bend_indices, AdjItem* adj_cons, uint num_constraints);

}  // namespace GraphColoring
}  // namespace XRTailor