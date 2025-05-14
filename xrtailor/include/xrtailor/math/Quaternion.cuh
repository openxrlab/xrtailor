#pragma once

#include <xrtailor/core/Scalar.cuh>

namespace XRTailor {
/*
* Quaternion is defined for float or double, all functions are taking radians as parameters
*/
class Quaternion {
 public:
  __host__ __device__ Quaternion();

  __host__ __device__ Quaternion(Scalar x, Scalar y, Scalar z, Scalar w);

  __host__ __device__
  Quaternion(Scalar rot, const Vector3& axis);  //init from the rotation axis and angle(in radian)

  __host__ __device__ Quaternion(const Vector3 u0, const Vector3 u1);  // u0 --[rot]--> u1

  __host__ __device__ Quaternion(const Quaternion&);

  ~Quaternion() = default;

  __host__ __device__ explicit Quaternion(const Mat3x3&);  //init from a 3x3 matrix

  __host__ __device__ explicit Quaternion(const Mat4x4&);  //init from a 4x4 matrix

  // yaw (Z), pitch (Y), roll (X);
  __host__ __device__ explicit Quaternion(const Scalar yaw, const Scalar pitch, const Scalar roll);

  /* Assignment operators */
  __host__ __device__ Quaternion& operator=(const Quaternion&);

  __host__ __device__ Quaternion& operator+=(const Quaternion&);

  __host__ __device__ Quaternion& operator-=(const Quaternion&);

  /* Special functions */
  __host__ __device__ Scalar Norm() const;

  __host__ __device__ Scalar NormSquared() const;

  __host__ __device__ Quaternion& Normalize();

  __host__ __device__ Quaternion Inverse() const;

  __host__ __device__ Scalar
  Angle() const;  // return the angle between this quat and the identity quaternion.

  __host__ __device__ Scalar
  Angle(const Quaternion&) const;  // return the angle between this and the argument

  __host__ __device__ Quaternion Conjugate() const;  // return the conjugate

  /**
	* @brief Rotate a vector by the quaternion,
	*		  guarantee the quaternion is normalized before rotating the vector
	*
	* @return v' where (0, v') is calculate by q(0, v)q^{*}.
	*/
  __host__ __device__ Vector3 Rotate(const Vector3& v) const;

  __host__ __device__ void ToRotationAxis(Scalar& rot, Vector3& axis) const;

  __host__ __device__ void ToEulerAngle(Scalar& yaw, Scalar& pitch, Scalar& roll) const;

  __host__ __device__ Mat3x3 ToMatrix3x3() const;  //return 3x3 matrix format

  __host__ __device__ Mat4x4 ToMatrix4x4() const;  //return 4x4 matrix with a identity transform.

  /* Operator overloading */
  __host__ __device__ Quaternion operator-(const Quaternion&) const;

  __host__ __device__ Quaternion operator-(void) const;

  __host__ __device__ Quaternion operator+(const Quaternion&) const;

  __host__ __device__ Quaternion operator*(const Quaternion&) const;

  __host__ __device__ Quaternion operator*(const Scalar&) const;

  __host__ __device__ Quaternion operator/(const Scalar&) const;

  __host__ __device__ bool operator==(const Quaternion&) const;

  __host__ __device__ bool operator!=(const Quaternion&) const;

  __host__ __device__ operator int() const;

  __host__ __device__ Scalar Dot(const Quaternion&) const;

  __host__ __device__ Quaternion Identity();

  __host__ __device__ Quaternion FromEulerAngles(const Scalar& yaw, const Scalar& pitch,
                                                 const Scalar& roll);

 public:
  Scalar x, y, z, w;
};

__host__ __device__ inline Quaternion operator*(Scalar scale, const Quaternion& quad) {
  return quad * scale;
}

}  // namespace XRTailor