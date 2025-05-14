#include <xrtailor/runtime/scene/Camera.hpp>

#include <xrtailor/core/Global.hpp>
#include <xrtailor/utils/Helper.hpp>

#include <xrtailor/runtime/engine/GameInstance.hpp>

namespace XRTailor {

Camera::Camera() {
  name = __func__;
  Global::camera = this;
}

void Camera::Start() {
  LOG_TRACE("Starting {0}'s component: {1}", actor->name, name);
}

glm::vec3 Camera::Position() const {
  return actor->transform->position;
}

glm::vec3 Camera::Front() const {
  const glm::vec3 k_front = glm::vec3(0.0f, 0.0f, -1.0f);
  return Helper::RotateWithDegree(k_front, actor->transform->rotation);
}

glm::vec3 Camera::Up() const {
  const glm::vec3 k_up = glm::vec3(0.0f, 1.0f, 0.0f);
  return Helper::RotateWithDegree(k_up, actor->transform->rotation);
}

glm::mat4 Camera::View() const {
  auto trans = actor->transform;
  auto rotation = trans->rotation;
  auto result = glm::lookAt(Position(), Position() + Front(), Up());
  return result;
}

glm::mat4 Camera::Projection() const {
  auto size = Global::game->WindowSize();
  auto screen_aspect = static_cast<float>(size.x) / static_cast<float>(size.y);
  return glm::perspective(glm::radians(zoom), screen_aspect, 0.01f, 100.0f);
}

}  // namespace XRTailor