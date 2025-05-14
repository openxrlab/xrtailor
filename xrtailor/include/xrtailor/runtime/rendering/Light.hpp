#pragma once

#include <xrtailor/runtime/scene/Component.hpp>
#include <xrtailor/core/Global.hpp>

namespace XRTailor {
enum class LightType {
  Point,
  Directional,
  SpotLight,
};

class Light : public Component {
 public:
  Light(LightType _type = LightType::SpotLight) {
    Global::lights.push_back(this);
    name = __func__;
    type = _type;
  }

  ~Light() {
    auto& lights = Global::lights;
    auto position = std::find(lights.begin(), lights.end(), this);
    if (position != lights.end()) {
      lights.erase(position);
    }
  }

  glm::vec4 position() {
    if (type == LightType::Point || type == LightType::SpotLight) {
      return glm::vec4(transform()->position, 1.0);
    } else {
      return glm::vec4(transform()->position, 0.0);
    }
  }

  LightType type = LightType::Point;
  glm::vec3 color = glm::vec3(1.3f);
  float ambient = 0.15f;
  float inner_cutoff = 40.0f;
  float outer_cutoff = 50.0f;

  float constant = 1.0f;
  float linear = 0.09f;
  float quadratic = 0.032f;
};
}  // namespace XRTailor