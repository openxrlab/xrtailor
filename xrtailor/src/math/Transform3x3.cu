#include <xrtailor/math/Transform3x3.cuh>

#include <cmath>
#include <limits>

namespace XRTailor {
__host__ __device__ Transform3x3::Transform3x3() {
  translation_ = Vector3(0);
  scale_ = Vector3(1);
  rotation_ = Mat3x3(1);
}

__host__ __device__ Transform3x3::Transform3x3(const Vector3& t, const Mat3x3& M,
                                               const Vector3& s) {
  translation_ = t;
  scale_ = s;
  rotation_ = M;
}

__host__ __device__ Transform3x3::Transform3x3(const Transform3x3& t) {
  rotation_ = t.rotation_;
  translation_ = t.translation_;
  scale_ = t.scale_;
}

__host__ __device__ const Vector3 Transform3x3::operator*(const Vector3& vec) const {
  Vector3 scaled = Vector3(vec.x * scale_.x, vec.y * scale_.y, vec.z * scale_.z);

  return rotation_ * scaled + translation_;
}

}  // namespace XRTailor