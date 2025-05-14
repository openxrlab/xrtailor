#pragma once

#include <xrtailor/physics/graph_coloring/Base.cuh>

namespace XRTailor {
namespace GraphColoring {

class MeshColoring : public GraphColoring {
 public:
  MeshColoring(){};

  void BuildAdjacencyTable(CONST(int*) stretch_indices, uint n_cons, uint n_verts);

  void BuildAdjacencyTableDummy();

  void AssignColors();

  uint n_verts;

  std::vector<std::set<uint>> h_nb_vv;

  thrust::device_vector<Vector3> verts_colors;

  std::vector<Vector3> color_map = {
      Vector3(0.94, 0.64, 1.00), Vector3(0.00, 0.46, 0.86), Vector3(0.60, 0.25, 0.00),
      Vector3(0.10, 0.10, 0.10), Vector3(0.17, 0.81, 0.28), Vector3(1.00, 0.80, 0.60),
      Vector3(1.00, 0.00, 0.06), Vector3(1.00, 0.64, 0.02), Vector3(0.76, 0.00, 0.53),
      Vector3(0.37, 0.95, 0.95), Vector3(0.00, 0.60, 0.56), Vector3(0.88, 1.00, 0.40),
      Vector3(0.45, 0.04, 1.00), Vector3(1.00, 0.31, 0.02), Vector3(0.00, 0.36, 0.19),
      Vector3(0.50, 0.50, 0.50)};
};

}  // namespace GraphColoring
}  // namespace XRTailor