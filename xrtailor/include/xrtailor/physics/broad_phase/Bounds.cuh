#pragma once

#include <cuda_runtime.h>

#include <xrtailor/math/MathFunctions.cuh>
#include <xrtailor/physics/broad_phase/Primitive.cuh>

namespace XRTailor {

class Bounds {
 public:
  Vector3 lower, upper;

  __host__ __device__ Bounds();

  __host__ __device__ Bounds(const Vector3& lower, const Vector3& upper);

  __host__ __device__ Bounds(const Primitive& p, const Scalar& scr = 0);

  ~Bounds() = default;

  __host__ __device__ Bounds operator+(const Bounds& b) const;

  __host__ __device__ void operator+=(const Vector3& v);

  __host__ __device__ void operator+=(const Bounds& b);

  __host__ __device__ Vector3 Center() const;

  __host__ __device__ int MajorAxis() const;

  __host__ __device__ Bounds Dilate(Scalar thickness) const;

  __host__ __device__ Scalar Distance(const Vector3& x) const;

  __host__ __device__ bool Overlap(const Bounds& b) const;

  __host__ __device__ bool Overlap(const Bounds& b, Scalar thickness) const;

  __host__ __device__ bool Overlap(const Vector3& p) const;

  /**
	 * @brief Test whether a ray intersects with bounding box, 
	 * @param p1 Start point of the line segment
	 * @param p2 End point of the line segment
	 * @param hit The intersection point
	 * @return True if the ray intersects
	*/
  __host__ __device__ bool Overlap(const Vector3& p1, const Vector3& p2, Vector3& hit) const;

  __host__ __device__ Bounds Merge(const Bounds& b) const;

  __host__ __device__ Bounds Merge(const Vector3& p) const;

  __host__ __device__ Vector3 MinVector(const Vector3& a, const Vector3& b) const;

  __host__ __device__ Vector3 MaxVector(const Vector3& a, const Vector3& b) const;
};

}  // namespace XRTailor