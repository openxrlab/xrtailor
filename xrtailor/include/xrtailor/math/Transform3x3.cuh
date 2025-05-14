#pragma once

#include <xrtailor/core/Scalar.cuh>

namespace XRTailor {

class Transform3x3 {
 public:
  __host__ __device__ Transform3x3();

  __host__ __device__ Transform3x3(const Vector3& t, const Mat3x3& M,
                                   const Vector3& s = Vector3(1));

  __host__ __device__ Transform3x3(const Transform3x3&);

  ~Transform3x3() = default;

  __host__ __device__ static unsigned int Rows() { return 3; }

  __host__ __device__ static unsigned int Cols() { return 3; }

  __host__ __device__ inline Mat3x3& rotation() { return rotation_; }

  __host__ __device__ inline const Mat3x3 rotation() const { return rotation_; }

  __host__ __device__ inline Vector3& translation() { return translation_; }

  __host__ __device__ inline Vector3 translation() const { return translation_; }

  __host__ __device__ inline Vector3& scale() { return scale_; }

  __host__ __device__ inline Vector3 scale() const { return scale_; }

  __host__ __device__ const Vector3 operator*(const Vector3&) const;

 protected:
  Vector3 translation_;
  Vector3 scale_;
  Mat3x3 rotation_;
};

}  // namespace XRTailor
