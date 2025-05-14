#pragma once

#include <xrtailor/physics/graph_coloring/Base.cuh>

namespace XRTailor {
namespace GraphColoring {

class BendColoring : public GraphColoring {
 public:
  BendColoring(){};

  void BuildAdjacencyTable(CONST(uint*) bend_indices, uint num_constraints);
};

}  // namespace GraphColoring
}  // namespace XRTailor