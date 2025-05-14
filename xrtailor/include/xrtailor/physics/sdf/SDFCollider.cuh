#pragma once

#include <xrtailor/core/DeviceHelper.cuh>

namespace XRTailor {

struct SDFCollider {
  int type;

  Vector3 position;
  Vector3 scale;

  Scalar delta_time;
  Mat3 cur_transform;
  Mat4 inv_cur_transform;
  Mat4 last_transform;

  __host__ __device__ SDFCollider()
      : type(0),
        position(static_cast<Scalar>(0)),
        scale(static_cast<Scalar>(1)),
        delta_time(static_cast<Scalar>(0)),
        cur_transform(static_cast<Scalar>(1)),
        inv_cur_transform(static_cast<Scalar>(1)),
        last_transform(static_cast<Scalar>(1)) {};

  ~SDFCollider() = default;

  __host__ __device__ Scalar sgn(Scalar value) const {
    return (value > static_cast<Scalar>(0))
               ? static_cast<Scalar>(1)
               : (value < static_cast<Scalar>(0) ? static_cast<Scalar>(-1)
                                                 : static_cast<Scalar>(0));
  }

  __host__ __device__ Vector3 ComputeSDF(const Vector3 target_position,
                                         const Scalar collision_margin) const {
    if (type == 1) {
      Scalar offset = target_position.y - (position.y + collision_margin);
      if (offset < 0) {
        return Vector3(static_cast<Scalar>(0), -offset, static_cast<Scalar>(0));
      }
    } else if (type == 0) {
      Scalar radius = scale.x + collision_margin;
      auto diff = target_position - position;
      Scalar distance = glm::length(diff);
      Scalar offset = distance - radius;
      if (offset < 0) {
        Vector3 direction = diff / distance;
        return -offset * direction;
      }
    } else if (type == 2) {
      Vector3 correction = Vector3(0);
      Vector3 local_pos = inv_cur_transform * Vector4(target_position, static_cast<Scalar>(1.0));
      Vector3 cube_size = Vector3(0.5, 0.5, 0.5) + collision_margin / scale;
      Vector3 offset = glm::abs(local_pos) - cube_size;

      Scalar max_val = MathFunctions::max(offset.x, MathFunctions::max(offset.y, offset.z));
      Scalar min_val = MathFunctions::min(offset.x, MathFunctions::min(offset.y, offset.z));
      Scalar mid_val = offset.x + offset.y + offset.z - max_val - min_val;
      Scalar scalar = static_cast<Scalar>(1);

      if (max_val < static_cast<Scalar>(0)) {
        // make cube corner round to avoid particle vibration
        Scalar margin = static_cast<Scalar>(0.03);
        if (mid_val > -margin)
          scalar = static_cast<Scalar>(0.2);
        if (min_val > -margin) {
          Vector3 mask;
          mask.x = offset.x < 0 ? sgn(local_pos.x) : 0;
          mask.y = offset.y < 0 ? sgn(local_pos.y) : 0;
          mask.z = offset.z < 0 ? sgn(local_pos.z) : 0;

          Vector3 vec = offset + Vector3(margin);
          Scalar len = glm::length(vec);
          if (len < margin)
            correction = mask * glm::normalize(vec) * (margin - len);
        } else if (offset.x == max_val) {
          correction = Vector3(MathFunctions::CopySign(-offset.x, local_pos.x), 0, 0);
        } else if (offset.y == max_val) {
          correction = Vector3(0, MathFunctions::CopySign(-offset.y, local_pos.y), 0);
        } else if (offset.z == max_val) {
          correction = Vector3(0, 0, MathFunctions::CopySign(-offset.z, local_pos.z));
        }
      }
      return cur_transform * scalar * correction;
    }
    return Vector3(0);
  }

  __host__ __device__ Vector3 VelocityAt(const Vector3 target_position) {
    Vector4 last_pos =
        last_transform * inv_cur_transform * Vector4(target_position, static_cast<Scalar>(1));
    Vector3 vel = (target_position - Vector3(last_pos)) / delta_time;
    return vel;
  }
};

}  // namespace XRTailor