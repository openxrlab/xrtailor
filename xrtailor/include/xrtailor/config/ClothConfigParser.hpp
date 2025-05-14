#pragma once

#include <string>
#include <vector>

#include "xrtailor/config/Base.hpp"
#include "xrtailor/core/Scalar.hpp"

namespace XRTailor {
class ClothConfigParser {
 public:
  ClothConfigParser();

  bool LoadFromJson(const std::string& path);

  std::string style;
  float mass;
  std::vector<uint> attached_indices;

  std::vector<BindingParam> binding_uv_island;
  std::vector<BindingParam> binding_boundary;
  std::vector<BindingParam> binding_neighbor;
  std::vector<BindingParam> binding_nonmanifold_edges;
};
}  // namespace XRTailor