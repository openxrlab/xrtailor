#include <xrtailor/runtime/scene/Transform.hpp>

#include <xrtailor/utils/Helper.hpp>

namespace XRTailor {

Transform::Transform(Actor* _actor) {
  actor_ = _actor;
}

glm::mat4 Transform::matrix() {
  glm::mat4 result = glm::mat4(1.0f);
  result = glm::translate(result, position);
  result = Helper::RotateWithDegree(result, rotation);
  result = glm::scale(result, scale);
  return result;
}

Actor* Transform::GetActor() {
  return actor_;
}

void Transform::Reset() {
  position = glm::vec3(0.0f);
  rotation = glm::vec3(0.0f);
  scale = glm::vec3(1.0f);
}

}  // namespace XRTailor