#if defined(_WIN64) || defined(WIN32) || defined(_WIN32)
#include <Windows.h>
#endif

#include <xrtailor/physics/sdf/Collider.hpp>
#include <xrtailor/core/Global.hpp>
#include <xrtailor/utils/Timer.hpp>
#include <xrtailor/utils/Logger.hpp>

namespace XRTailor {

Collider::Collider(int _type)
{
  name = __func__;
  type = _type;
}

void Collider::Start() {
  std::string type_str;
  switch (type) {
    case 0:
      type_str = "Sphere";
      break;
    case 1:
      type_str = "Plane";
      break;
    case 2:
      type_str = "Cube";
      break;
    default:
      break;
  }
  LOG_TRACE("Starting {0}'s component: {1}", actor->name, type_str + name);
  last_pos = actor->transform->position;

  cur_transform = actor->transform->matrix();
  last_transform = cur_transform;
}

void Collider::FixedUpdate() {
  Vector3 cur_pos = actor->transform->position;
  velocity = (cur_pos - last_pos) / static_cast<Scalar>(Timer::FixedDeltaTime());
  last_pos = actor->transform->position;

  last_transform = cur_transform;
  cur_transform = actor->transform->matrix();
}

Vector3 Collider::ComputeSDF(Vector3 position) {
  if (type == 1)
  {
    return ComputePlaneSDF(position);
  }
  if (type == 0)
  {
    return ComputeSphereSDF(position);
  }
}

Vector3 Collider::ComputePlaneSDF(Vector3 position) {
  if (position.y < Global::sim_params.collision_margin) {
    return Vector3(0, Global::sim_params.collision_margin - position.y, 0);
  }
  return Vector3(0);
}

Vector3 Collider::ComputeSphereSDF(Vector3 position) {
  Vector3 actor_pos = actor->transform->position;
  Scalar radius = actor->transform->scale.x + Global::sim_params.collision_margin;

  Vector3 diff = position - actor_pos;
  Scalar distance = glm::length(diff);
  if (distance < radius) {
    auto direction = diff / distance;
    return (radius - distance) * direction;
  }
  return Vector3(0);
}

}  // namespace XRTailor