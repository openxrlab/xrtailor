#include <xrtailor/utils/Helper.hpp>
#include <glm/ext/matrix_transform.hpp>

namespace XRTailor {
namespace Helper {

Mat4 RotateWithDegree(Mat4 result, const Vector3& rotation) {
  result = glm::rotate(result, glm::radians(rotation.y), Vector3(0, 1, 0));
  result = glm::rotate(result, glm::radians(rotation.z), Vector3(0, 0, 1));
  result = glm::rotate(result, glm::radians(rotation.x), Vector3(1, 0, 0));

  return result;
}

Vector3 RotateWithDegree(Vector3 result, const Vector3& rotation) {
  Mat4 rotation_matrix(1);
  rotation_matrix = RotateWithDegree(rotation_matrix, rotation);
  result = rotation_matrix * Vector4(result, 0.0f);
  return glm::normalize(result);
}

Scalar Random(Scalar min, Scalar max) {
  Scalar zero_to_one = static_cast<Scalar>(rand()) / RAND_MAX;
  return min + zero_to_one * (max - min);
}

Vector3 RandomUnitVector() {
  const Scalar pi = 3.1415926535;
  Scalar phi = Random(0, pi * static_cast<Scalar>(2));
  Scalar theta = Random(0, pi * static_cast<Scalar>(2));

  Scalar cos_theta = cos(theta);
  Scalar sin_theta = sin(theta);

  Scalar cos_phi = cos(phi);
  Scalar sin_phi = sin(phi);

  return Vector3(cos_theta * sin_phi, cos_phi, sin_theta * sin_phi);
}

}  // namespace Helper
}  // namespace XRTailor