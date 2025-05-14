#pragma once

#include <vector>
#include <string>
#include <unordered_map>

#include <xrtailor/runtime/mesh/Mesh.hpp>

namespace XRTailor {
class Mesh;

class Geodesic {
 public:
  typedef std::vector<uint> PathType;
  // stores all paths
  // row 'i' represents a path from a source vertex to i-th vertex
  typedef std::vector<PathType> MapType;

  Geodesic(Mesh* _mesh) : mesh_(_mesh) {}

  void ComputeGeodesicDistance(const uint& source_index);

  std::vector<uint> sources;
  std::vector<std::vector<Scalar>> distances;

  std::vector<std::vector<uint>> vec_previous;
  std::vector<MapType> maps;

 private:
  Mesh* mesh_;
};
}  // namespace XRTailor