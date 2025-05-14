#include <xrtailor/math/Quaternion.cuh>

namespace XRTailor {

__host__ __device__ Quaternion::Quaternion() : w(1), x(0), y(0), z(0) {}

__host__ __device__ Quaternion::Quaternion(Scalar _x, Scalar _y, Scalar _z, Scalar _w)
    : w(_w), x(_x), y(_y), z(_z) {}

__host__ __device__ Quaternion::Quaternion(Scalar rot, const Vector3& axis) {
  const Scalar a = rot * 0.5;
  const Scalar s = glm::sin(a);
  w = glm::cos(a);
  x = axis[0] * s;
  y = axis[1] * s;
  z = axis[2] * s;
}

__host__ __device__ Quaternion::Quaternion(const Vector3 u0, const Vector3 u1) {
  Vector3 c = glm::cross(u0, u1);
  x = c[0];
  y = c[1];
  z = c[2];
  w = glm::dot(u0, u1) + glm::length(u0) * glm::length(u1);
  Normalize();
}

__host__ __device__ Quaternion::Quaternion(const Quaternion& quat)
    : w(quat.w), x(quat.x), y(quat.y), z(quat.z) {}

__host__ __device__ Quaternion::Quaternion(const Scalar yaw, const Scalar pitch,
                                           const Scalar roll) {
  Scalar cy = glm::cos(Scalar(yaw * 0.5));
  Scalar sy = glm::sin(Scalar(yaw * 0.5));
  Scalar cp = glm::cos(Scalar(pitch * 0.5));
  Scalar sp = glm::sin(Scalar(pitch * 0.5));
  Scalar cr = glm::cos(Scalar(roll * 0.5));
  Scalar sr = glm::sin(Scalar(roll * 0.5));

  w = cr * cp * cy + sr * sp * sy;
  x = sr * cp * cy - cr * sp * sy;
  y = cr * sp * cy + sr * cp * sy;
  z = cr * cp * sy - sr * sp * cy;
}

__host__ __device__ Quaternion& Quaternion::operator=(const Quaternion& quat) {
  w = quat.w;
  x = quat.x;
  y = quat.y;
  z = quat.z;
  return *this;
}

__host__ __device__ Quaternion& Quaternion::operator+=(const Quaternion& quat) {
  w += quat.w;
  x += quat.x;
  y += quat.y;
  z += quat.z;
  return *this;
}

__host__ __device__ Quaternion& Quaternion::operator-=(const Quaternion& quat) {
  w -= quat.w;
  x -= quat.x;
  y -= quat.y;
  z -= quat.z;
  return *this;
}

__host__ __device__ Quaternion Quaternion::operator-(const Quaternion& quat) const {
  return Quaternion(x - quat.x, y - quat.y, z - quat.z, w - quat.w);
}

__host__ __device__ Quaternion Quaternion::operator-(void) const {
  return Quaternion(-x, -y, -z, -w);
}

__host__ __device__ Quaternion Quaternion::operator+(const Quaternion& quat) const {
  return Quaternion(x + quat.x, y + quat.y, z + quat.z, w + quat.w);
}

__host__ __device__ Quaternion Quaternion::operator*(const Quaternion& q) const {
  Quaternion result;

  result.w = -x * q.x - y * q.y - z * q.z + w * q.w;

  result.x = x * q.w + y * q.z - z * q.y + w * q.x;
  result.y = -x * q.z + y * q.w + z * q.x + w * q.y;
  result.z = x * q.y - y * q.x + z * q.w + w * q.z;

  return result;
}

__host__ __device__ Quaternion Quaternion::operator*(const Scalar& scale) const {
  return Quaternion(x * scale, y * scale, z * scale, w * scale);
}

__host__ __device__ Quaternion Quaternion::operator/(const Scalar& scale) const {
  return Quaternion(x / scale, y / scale, z / scale, w / scale);
}

__host__ __device__ bool Quaternion::operator==(const Quaternion& quat) const {
  if (w == quat.w && x == quat.x && y == quat.y && z == quat.z)
    return true;
  return false;
}

__host__ __device__ bool Quaternion::operator!=(const Quaternion& quat) const {
  if (*this == quat)
    return false;
  return true;
}

__host__ __device__ Quaternion::operator int() const {
  return static_cast<int>(this->Norm());
}

__host__ __device__ Scalar Quaternion::Norm() const {
  Scalar result = w * w + x * x + y * y + z * z;
  result = glm::sqrt(result);

  return result;
}

__host__ __device__ Scalar Quaternion::NormSquared() const {
  return w * w + x * x + y * y + z * z;
}

__host__ __device__ Quaternion& Quaternion::Normalize() {
  Scalar d = Norm();
  // Set the rotation along the x-axis
  if (d < 0.00001) {
    z = Scalar(1.0);
    x = y = w = Scalar(0.0);
    return *this;
  }
  d = 1 / d;
  x *= d;
  y *= d;
  z *= d;
  w *= d;
  return *this;
}

__host__ __device__ Quaternion Quaternion::Inverse() const {
  return Conjugate() / NormSquared();
}

__host__ __device__ void Quaternion::ToEulerAngle(Scalar& yaw, Scalar& pitch, Scalar& roll) const {
  // roll (x-axis rotation)
  Scalar sinr_cosp = 2 * (w * x + y * z);
  Scalar cosr_cosp = 1 - 2 * (x * x + y * y);
  roll = atan2(sinr_cosp, cosr_cosp);

  // pitch (y-axis rotation)
  Scalar sinp = 2 * (w * y - z * x);
  if (glm::abs(sinp) >= 1) {
    pitch = sinp > 0 ? (M_PI / 2) : -(M_PI / 2);  // use 90 degrees if out of range
  } else
    pitch = glm::asin(sinp);

  // yaw (z-axis rotation)
  Scalar siny_cosp = 2 * (w * z + x * y);
  Scalar cosy_cosp = 1 - 2 * (y * y + z * z);
  yaw = atan2(siny_cosp, cosy_cosp);
}

__host__ __device__ Scalar Quaternion::Angle() const {
  return glm::acos(w) * static_cast<Scalar>(2);
}

__host__ __device__ Scalar Quaternion::Angle(const Quaternion& quat) const {
  Scalar dot_product = Dot(quat);

  dot_product = glm::clamp(dot_product, static_cast<Scalar>(-1), static_cast<Scalar>(1));
  return glm::acos(dot_product) * static_cast<Scalar>(2);
}

__host__ __device__ Scalar Quaternion::Dot(const Quaternion& quat) const {
  return w * quat.w + x * quat.x + y * quat.y + z * quat.z;
}

__host__ __device__ Quaternion Quaternion::Identity() {
  return Quaternion(0, 0, 0, 1);
}

__host__ __device__ Quaternion Quaternion::FromEulerAngles(const Scalar& yaw, const Scalar& pitch,
                                                           const Scalar& roll) {
  Scalar cr = glm::cos(roll * static_cast<Scalar>(0.5));
  Scalar sr = glm::sin(roll * static_cast<Scalar>(0.5));
  Scalar cp = glm::cos(pitch * static_cast<Scalar>(0.5));
  Scalar sp = glm::sin(pitch * static_cast<Scalar>(0.5));
  Scalar cy = glm::cos(yaw * static_cast<Scalar>(0.5));
  Scalar sy = glm::sin(yaw * static_cast<Scalar>(0.5));

  Quaternion q;
  q.w = cr * cp * cy + sr * sp * sy;
  q.x = sr * cp * cy - cr * sp * sy;
  q.y = cr * sp * cy + sr * cp * sy;
  q.z = cr * cp * sy - sr * sp * cy;

  return q;
}

__host__ __device__ Quaternion Quaternion::Conjugate() const {
  return Quaternion(-x, -y, -z, w);
}

// Refer to "https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles" for more details
__host__ __device__ Vector3 Quaternion::Rotate(const Vector3& v) const {
  // Extract the vector part of the quaternion
  Vector3 u(x, y, z);

  // Extract the scalar part of the quaternion
  Scalar s = w;

  // Do the math
  return 2 * glm::dot(u, v) * u + (s * s - glm::dot(u, u)) * v + 2 * s * glm::cross(u, v);
}

__host__ __device__ Mat3x3 Quaternion::ToMatrix3x3() const {
  Scalar x2 = x + x, y2 = y + y, z2 = z + z;
  Scalar xx = x2 * x, yy = y2 * y, zz = z2 * z;
  Scalar xy = x2 * y, xz = x2 * z, xw = x2 * w;
  Scalar yz = y2 * z, yw = y2 * w, zw = z2 * w;

  return Mat3x3(1 - yy - zz, xy + zw, xz - yw, xy - zw, 1 - xx - zz, yz + xw, xz + yw, yz - xw,
                1 - xx - yy);
}

__host__ __device__ Mat4x4 Quaternion::ToMatrix4x4() const {
  Scalar x2 = x + x, y2 = y + y, z2 = z + z;
  Scalar xx = x2 * x, yy = y2 * y, zz = z2 * z;
  Scalar xy = x2 * y, xz = x2 * z, xw = x2 * w;
  Scalar yz = y2 * z, yw = y2 * w, zw = z2 * w;
  Scalar entries[16];
  entries[0] = 1 - yy - zz;
  entries[1] = xy - zw;
  entries[2] = xz + yw, entries[3] = 0;
  entries[4] = xy + zw;
  entries[5] = 1 - xx - zz;
  entries[6] = yz - xw;
  entries[7] = 0;
  entries[8] = xz - yw;
  entries[9] = yz + xw;
  entries[10] = 1 - xx - yy;
  entries[11] = 0;
  entries[12] = 0;
  entries[13] = 0;
  entries[14] = 0;
  entries[15] = 1;
  // GLM matrix is column major
  // row major
  //return Mat4x4(
  //	entries[0], entries[1], entries[2], entries[3],
  //	entries[4], entries[5], entries[6], entries[7],
  //	entries[8], entries[9], entries[10], entries[11],
  //	entries[12], entries[13], entries[14], entries[15]);
  return Mat4x4(entries[0], entries[4], entries[8], entries[12], entries[1], entries[5], entries[9],
                entries[13], entries[2], entries[6], entries[10], entries[14], entries[3],
                entries[7], entries[11], entries[15]);
}

__host__ __device__ Quaternion::Quaternion(const Mat3x3& matrix) {
  Scalar tr = matrix[0][0] + matrix[1][1] + matrix[2][2];
  if (tr > 0.0) {
    Scalar s = glm::sqrt(tr + 1);
    w = s * 0.5f;
    if (s != 0.0)
      s = 0.5f / s;
    x = s * (matrix[2][1] - matrix[1][2]);
    y = s * (matrix[0][2] - matrix[2][0]);
    z = s * (matrix[1][0] - matrix[0][1]);
  } else {
    int i = 0, j, k;
    int next[3] = {1, 2, 0};
    Scalar q[4];
    if (matrix[1][1] > matrix[0][0])
      i = 1;
    if (matrix[2][2] > matrix[i][i])
      i = 2;
    j = next[i];
    k = next[j];
    Scalar s = glm::sqrt(matrix[i][i] - matrix[j][j] - matrix[k][k] + 1);
    q[i] = s * 0.5f;
    if (s != 0.0)
      s = 0.5f / s;
    q[3] = s * (matrix[k][j] - matrix[j][k]);
    q[j] = s * (matrix[j][i] + matrix[i][j]);
    q[k] = s * (matrix[k][i] + matrix[i][k]);
    x = q[0];
    y = q[1];
    z = q[2];
    w = q[3];
  }
}

__host__ __device__ Quaternion::Quaternion(const Mat4x4& matrix) {
  Scalar tr = matrix[0][0] + matrix[1][1] + matrix[2][2];
  if (tr > 0.0) {
    Scalar s = glm::sqrt(tr + 1);
    w = s * 0.5f;
    if (s != 0.0)
      s = 0.5f / s;
    x = s * (matrix[2][1] - matrix[1][2]);
    y = s * (matrix[0][2] - matrix[2][0]);
    z = s * (matrix[1][0] - matrix[0][1]);
  } else {
    int i = 0, j, k;
    int next[3] = {1, 2, 0};
    Scalar q[4];
    if (matrix[1][1] > matrix[0][0])
      i = 1;
    if (matrix[2][2] > matrix[i][i])
      i = 2;
    j = next[i];
    k = next[j];
    Scalar s = glm::sqrt(matrix[i][i] - matrix[j][j] - matrix[k][k] + 1);
    q[i] = s * 0.5f;
    if (s != 0.0)
      s = 0.5f / s;
    q[3] = s * (matrix[k][j] - matrix[j][k]);
    q[j] = s * (matrix[j][i] + matrix[i][j]);
    q[k] = s * (matrix[k][i] + matrix[i][k]);
    x = q[0];
    y = q[1];
    z = q[2];
    w = q[3];
  }
}

__host__ __device__ void Quaternion::ToRotationAxis(Scalar& rot, Vector3& axis) const {
  rot = 2 * glm::acos(w);
  if (glm::abs(rot) < EPSILON) {
    axis[0] = 0;
    axis[1] = 0;
    axis[2] = 1;
    return;
  }
  axis[0] = x;
  axis[1] = y;
  axis[2] = z;
  if (glm::length(axis) > EPSILON)
    axis = glm::normalize(axis);
  else
    axis = Vector3(0);
}

}  // namespace XRTailor