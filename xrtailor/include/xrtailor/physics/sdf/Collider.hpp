#pragma once

#include <iostream>
#include <string>

#include <xrtailor/config/Base.hpp>
#include <xrtailor/runtime/scene/Actor.hpp>
#include <xrtailor/core/Scalar.hpp>

namespace XRTailor {

class Collider : public Component {
 public:
  int type = 0;
  Vector3 last_pos;
  Vector3 velocity;
  Mat4 cur_transform;
  Mat4 last_transform;

  Collider(int _type);

  void Start() override;

  void FixedUpdate() override;

  virtual Vector3 ComputeSDF(Vector3 position);

  virtual Vector3 ComputePlaneSDF(Vector3 position);

  virtual Vector3 ComputeSphereSDF(Vector3 position);
};

}  // namespace XRTailor