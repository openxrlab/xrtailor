#include <xrtailor/runtime/scene/Component.hpp>
#include <xrtailor/runtime/scene/Actor.hpp>

namespace XRTailor {

std::shared_ptr<Transform> Component::transform() {
  if (actor != nullptr) {
    return actor->transform;
  }

  return std::make_shared<Transform>(Transform(nullptr));
}

}  // namespace XRTailor
