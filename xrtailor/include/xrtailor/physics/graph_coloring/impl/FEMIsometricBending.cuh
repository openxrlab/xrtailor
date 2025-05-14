#pragma once

#include <xrtailor/physics/graph_coloring/Base.cuh>

namespace XRTailor {
namespace GraphColoring {

class FEMIsometricBendingColoring : public GraphColoring {
 public:
  FEMIsometricBendingColoring(){};

  void BuildAdjacencyTable(CONST(uint*) bend_indices, uint n_constraints);
};

}  // namespace GraphColoring
}  // namespace XRTailor