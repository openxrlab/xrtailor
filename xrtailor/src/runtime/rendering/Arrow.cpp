#include <xrtailor/runtime/rendering/Arrow.hpp>

namespace XRTailor {
Arrow::Arrow() = default;

Arrow::Arrow(const glm::vec3& _start, const glm::vec3& _end) : start(_start), end(_end) {
  // Compute the vector along the arrow direction
  glm::vec3 v = glm::normalize(end - start);

  float length = glm::length(v);

  arrow_head_size = 0.01f * length;

  // Compute two perpendicular vectors to v
  glm::vec3 v_perp1 = glm::vec3(-v.y, v.x, 0.0f);
  glm::vec3 v_perp2 = glm::vec3(v.y, -v.x, 0.0f);

  glm::vec3 v1 = 0.1f * glm::normalize(v + v_perp1);
  glm::vec3 v2 = 0.1f * glm::normalize(v + v_perp2);

  this->lhs = end - arrow_head_size * v1;
  this->rhs = end - arrow_head_size * v2;
}
}  // namespace XRTailor