#pragma once

#include <memory>
#include <string>

#include <xrtailor/runtime/scene/Transform.hpp>

#define SET_COMPONENT_NAME name = __func__

namespace XRTailor {
class Actor;

class Component {
 public:
  virtual void Start() {}

  virtual void Update() {}

  virtual void FixedUpdate() {}

  virtual void OnDestroy() {}

  std::string name = "Component";

  Actor* actor = nullptr;

  std::shared_ptr<Transform> transform();

  bool enabled = true;
};
}  // namespace XRTailor