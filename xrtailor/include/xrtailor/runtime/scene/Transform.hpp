#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace XRTailor {
class Actor;

class Transform {
 public:
  Transform(Actor* _actor);

  glm::mat4 matrix();

  Actor* GetActor();

  void Reset();

  glm::vec3 position = glm::vec3(0.0f);
  glm::vec3 rotation = glm::vec3(0.0f);
  glm::vec3 scale = glm::vec3(1.0f);

 private:
  Actor* actor_;
};
}  // namespace XRTailor