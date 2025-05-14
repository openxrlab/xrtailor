#pragma once

#include <xrtailor/core/Scalar.hpp>

namespace XRTailor {
namespace Helper {

Mat4 RotateWithDegree(Mat4 result, const Vector3& rotation);

Vector3 RotateWithDegree(Vector3 result, const Vector3& rotation);

Scalar Random(Scalar min = 0, Scalar max = 1);

Vector3 RandomUnitVector();

template <class T>
T Lerp(T value1, T value2, float a) {
  a = std::min(std::max(a, 0.0f), 1.0f);
  return a * value2 + (1 - a) * value1;
}

}  // namespace Helper
}  // namespace XRTailor