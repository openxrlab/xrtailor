#pragma once

#include <vector_types.h>
#include <cuda_runtime.h>

#include <xrtailor/core/Scalar.cuh>

namespace XRTailor {
namespace MathFunctions {

__host__ __device__ Scalar CotTheta(const Vector3& v, const Vector3& w);

__host__ __device__ Vector3 FaceNormal(const Vector3& a, const Vector3& b, const Vector3& c);

__host__ __device__ Scalar TriangleArea(const Vector3& e0, const Vector3& e1);

__host__ __device__ Scalar TriangleArea(const Vector3& v0, const Vector3& v1, const Vector3& v2);

__host__ __device__ Scalar CopySign(Scalar number, Scalar sign);

__host__ __device__ Scalar Length2(Vector3 vec);

template <typename T>
__host__ __device__ static T min(T a, T b) {
  return a < b ? a : b;
}

template <typename T>
__host__ __device__ static T min(T a, T b, T c) {
  return min(a, min(b, c));
}

template <typename T>
__host__ __device__ static T min(T a, T b, T c, T d) {
  return min(min(a, b), min(c, d));
}

template <typename T>
__host__ __device__ static T max(T a, T b) {
  return a > b ? a : b;
}

template <typename T>
__host__ __device__ static T max(T a, T b, T c) {
  return max(a, max(b, c));
}

template <typename T>
__host__ __device__ static T max(T a, T b, T c, T d) {
  return max(max(a, b), max(c, d));
}

template <typename T>
__host__ __device__ static T sqr(T x) {
  return x * x;
}

template <typename T>
__host__ __device__ static T abs(T x) {
  return x < static_cast<T>(0) ? -x : x;
}

__host__ __device__ static Scalar Mixed(const Vector3& a, const Vector3& b, const Vector3& c) {
  return glm::dot(a, glm::cross(b, c));
}

template <typename T>
__host__ __device__ static void MySwap(T& a, T& b) {
  T t = a;
  a = b;
  b = t;
}

template <typename T>
__host__ __device__ static T Clamp(T x, T a, T b) {
  return x < a ? a : (x > b ? b : x);
}

template <typename T>
__host__ __device__ static int Sign(T x) {
  return x < static_cast<T>(0) ? -1 : 1;
}

__host__ __device__ Mat4 RotateX(Mat4 target, const Scalar& degree);

__host__ __device__ Scalar Determinant3x3(const Mat3x3& A) noexcept;

__host__ __device__ Scalar ScalarTripleProduct(const Vector3& u, const Vector3& v,
                                               const Vector3& w);

__host__ __device__ bool LineTriangleIntersects(const Vector3& p, const Vector3& q,
                                                const Vector3& a, const Vector3& b,
                                                const Vector3& c, Scalar& u, Scalar& v, Scalar& w);

__host__ __device__ Vector3 XVPos(const Vector3& x, const Vector3& v, Scalar t);

__host__ __device__ Scalar NewtonsMethod(Scalar a, Scalar b, Scalar c, Scalar d, Scalar x0,
                                         int dir);

__host__ __device__ int SolveQuadratic(Scalar a, Scalar b, Scalar c, Scalar x[2]);

__host__ __device__ int SolveCubic(Scalar a, Scalar b, Scalar c, Scalar d, Scalar x[]);

}  // namespace MathFunctions
}  // namespace XRTailor