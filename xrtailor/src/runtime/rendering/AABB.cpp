#include <xrtailor/runtime/rendering/AABB.hpp>

namespace XRTailor {
AABB::AABB() = default;

AABB::AABB(const glm::vec3& _lower, const glm::vec3& _upper) : lower(_lower), upper(_upper) {
  extent = _upper - _lower;
}
}  // namespace XRTailor