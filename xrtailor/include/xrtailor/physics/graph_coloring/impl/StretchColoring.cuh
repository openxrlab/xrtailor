#pragma once

#include <xrtailor/physics/graph_coloring/Base.cuh>

namespace XRTailor {
namespace GraphColoring {

class StretchColoring : public GraphColoring {
 public:
  StretchColoring(){};

  void BuildAdjacencyTable(const Edge* const* edges, uint num_constraints);
};

}  // namespace GraphColoring
}  // namespace XRTailor