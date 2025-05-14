#pragma once

#include "glm/glm.hpp"

namespace XRTailor {

// Axis Aligned Bounding Box
class AABB {
 public:
  AABB();

  AABB(const glm::vec3& _lower, const glm::vec3& _upper);

  glm::vec3 lower;
  glm::vec3 upper;
  glm::vec3 extent;
};
}  // namespace XRTailor