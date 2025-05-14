#include <xrtailor/math/MathFunctions.cuh>
#include <glm/ext/matrix_transform.hpp>

namespace XRTailor {
namespace MathFunctions {

__host__ __device__ Scalar CotTheta(const Vector3& v, const Vector3& w) {
  const Scalar cos_theta = glm::dot(v, w);
  const Scalar sin_theta = glm::length(glm::cross(v, w));

  return (cos_theta / sin_theta);
}

__host__ __device__ Vector3 FaceNormal(const Vector3& a, const Vector3& b, const Vector3& c) {
  // OpenGL considers polygons have counter clockwise winding to be front facing by default
  Vector3 e1 = b - a;
  Vector3 e2 = c - a;

  return glm::normalize(glm::cross(e1, e2));
}

__host__ __device__ Scalar TriangleArea(const Vector3& e0, const Vector3& e1) {
  return static_cast<Scalar>(0.5) * glm::length(glm::cross(e0, e1));
}

__host__ __device__ Scalar TriangleArea(const Vector3& v0, const Vector3& v1, const Vector3& v2) {
  return static_cast<Scalar>(0.5) * glm::length(glm::cross(v1 - v0, v2 - v0));
}

__host__ __device__ Scalar CopySign(Scalar number, Scalar sign) {
#ifdef XRTAILOR_USE_DOUBLE
  return copysign(number, sign);
#else
  return copysignf(number, sign);
#endif  // XRTAILOR_USE_DOUBLE
}

__host__ __device__ Scalar Length2(Vector3 vec) {
  return glm::dot(vec, vec);
}

__host__ __device__ Mat4 RotateX(Mat4 target, const Scalar& degree) {
  return glm::rotate(target, glm::radians(degree), Vector3(1, 0, 0));
}

__host__ __device__ Scalar Determinant3x3(const Mat3x3& A) noexcept {
  return A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
         A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
         A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
}

__host__ __device__ Scalar ScalarTripleProduct(const Vector3& u, const Vector3& v,
                                               const Vector3& w) {
  return glm::dot(glm::cross(u, v), w);
}

__host__ __device__ bool SameSign(const Scalar& a, const Scalar& b) {
  return ((a * b) > static_cast<Scalar>(0.0));
}

__host__ __device__ bool LineTriangleIntersects(const Vector3& p, const Vector3& q,
                                                const Vector3& a, const Vector3& b,
                                                const Vector3& c, Scalar& u, Scalar& v, Scalar& w) {
  Vector3 pq = q - p;
  Vector3 pa = a - p;
  Vector3 pb = b - p;
  Vector3 pc = c - p;

  Vector3 m = glm::cross(pq, pc);
  u = glm::dot(pb, m);
  v = -glm::dot(pa, m);
  if (!SameSign(u, v))
    return false;

  w = ScalarTripleProduct(pq, pb, pa);
  if (!SameSign(u, w))
    return false;

  Scalar denom = static_cast<Scalar>(1.0) / (u + v + w);
  u *= denom;
  v *= denom;
  w *= denom;

  return true;
}

__host__ __device__ Vector3 XVPos(const Vector3& x, const Vector3& v, Scalar t) {
  return x + v * t;
}

__host__ __device__ Scalar NewtonsMethod(Scalar a, Scalar b, Scalar c, Scalar d, Scalar x0,
                                         int dir) {
  if (dir != 0) {
    Scalar y0 = d + x0 * (c + x0 * (b + x0 * a));
    Scalar ddy0 = static_cast<Scalar>(2.0) * b + x0 * (static_cast<Scalar>(6.0) * a);
    x0 += dir * sqrt(MathFunctions::abs(static_cast<Scalar>(2.0) * y0 / ddy0));
  }
  for (int iter = 0; iter < 100; iter++) {
    Scalar y = d + x0 * (c + x0 * (b + x0 * a));
    Scalar dy = c + x0 * (static_cast<Scalar>(2.0) * b + static_cast<Scalar>(3.0) * x0 * a);
    if (dy == 0)
      return x0;
    Scalar x1 = x0 - y / dy;
    if (MathFunctions::abs(x0 - x1) < static_cast<Scalar>(1e-6))
      return x0;
    x0 = x1;
  }
  return x0;
}

__host__ __device__ int SolveQuadratic(Scalar a, Scalar b, Scalar c, Scalar x[2]) {
  Scalar d = b * b - static_cast<Scalar>(4.0) * a * c;
  if (d < 0) {
    x[0] = -b / (static_cast<Scalar>(2.0) * a);
    return 0;
  }
  Scalar q = -(b + MathFunctions::Sign(b) * sqrt(d)) * static_cast<Scalar>(0.5);
  int i = 0;
  if (MathFunctions::abs(a) > static_cast<Scalar>(1e-12) * MathFunctions::abs(q))
    x[i++] = q / a;
  if (MathFunctions::abs(q) > static_cast<Scalar>(1e-12) * MathFunctions::abs(c))
    x[i++] = c / q;
  if (i == 2 && x[0] > x[1])
    MathFunctions::MySwap(x[0], x[1]);
  return i;
}

__host__ __device__ int SolveCubic(Scalar a, Scalar b, Scalar c, Scalar d, Scalar x[]) {
  Scalar xc[2];
  int n = SolveQuadratic(static_cast<Scalar>(3.0) * a, static_cast<Scalar>(2.0) * b, c, xc);
  if (n == 0) {
    x[0] = NewtonsMethod(a, b, c, d, xc[0], 0);
    return 1;
  } else if (n == 1)
    return SolveQuadratic(b, c, d, x);
  else {
    Scalar yc[2] = {d + xc[0] * (c + xc[0] * (b + xc[0] * a)),
                    d + xc[1] * (c + xc[1] * (b + xc[1] * a))};
    int i = 0;
    if (yc[0] * a >= 0)
      x[i++] = NewtonsMethod(a, b, c, d, xc[0], -1);
    if (yc[0] * yc[1] <= 0) {
      int closer = MathFunctions::abs(yc[0]) < MathFunctions::abs(yc[1]) ? 0 : 1;
      x[i++] = NewtonsMethod(a, b, c, d, xc[closer], closer == 0 ? 1 : -1);
    }
    if (yc[1] * a <= 0)
      x[i++] = NewtonsMethod(a, b, c, d, xc[1], 1);
    return i;
  }
}

}  // namespace MathFunctions
}  // namespace XRTailor