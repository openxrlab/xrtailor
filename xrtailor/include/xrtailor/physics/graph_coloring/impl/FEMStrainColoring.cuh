#pragma once

#include <xrtailor/physics/graph_coloring/Base.cuh>

namespace XRTailor {
namespace GraphColoring {

class FEMStrainColoring : public GraphColoring {
 public:
  FEMStrainColoring(){};

  void BuildAdjacencyTable(const Face* const* indices, uint n_constraints);
};

}  // namespace GraphColoring
}  // namespace XRTailor