#pragma once

#include "glm/glm.hpp"

namespace XRTailor {
class Arrow {
 public:
  Arrow();

  Arrow(const glm::vec3& _start, const glm::vec3& _end);

  glm::vec3 start;
  glm::vec3 end;
  glm::vec3 lhs;
  glm::vec3 rhs;

  float arrow_head_size;
};
}  // namespace XRTailor