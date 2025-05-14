#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <xrtailor/runtime/scene/Component.hpp>
#include <xrtailor/utils/Logger.hpp>

namespace XRTailor {
class Camera : public Component {
 public:
  float zoom = 45.0f;

  Camera();

  void Start();

  glm::vec3 Position() const;

  glm::vec3 Front() const;

  glm::vec3 Up() const;

  glm::mat4 View() const;

  glm::mat4 Projection() const;
};
}  // namespace XRTailor