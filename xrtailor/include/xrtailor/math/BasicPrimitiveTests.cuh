#pragma once

#include <xrtailor/memory/Face.cuh>

namespace XRTailor {
namespace BasicPrimitiveTests {

__host__ __device__ bool InEdge(Scalar w, const Edge* edge0, const Edge* edge1);

__host__ __device__ Scalar NearestPoint(const Vector3& p, const Primitive& f, Vector3& qs,
                                        Scalar& _u, Scalar& _v, Scalar& _w) noexcept;

__host__ __device__ Scalar NearestPoint(const Vector3& x, const Vector3& y0, const Vector3& y1,
                                        const Vector3& y2, Vector3& qs, Scalar& _u, Scalar& _v,
                                        Scalar& _w) noexcept;

__host__ __device__ Scalar RayIntersect(const Vector3& o, const Vector3& d, Scalar ray_length,
                                        const Primitive& f) noexcept;

__host__ __device__ bool RayIntersect(const Vector3& ev0, const Vector3& ev1, const Vector3& fv0,
                                      const Vector3& fv1, const Vector3& fv2,
                                      Scalar& dist) noexcept;

/* \brief Computes closest points C1 and C2 of S1(s)=P1+s*(Q1-P1) and
*         S2(t)=P2+t*(Q2-P2), returning s and t. Function result is
*         distance between between S1(s) and S2(t).
*         Reference: Real Time Collision Detection
*/
__device__ Scalar ClosestPtSegmentSegment(const Vector3& p1, const Vector3& q1, const Vector3& p2,
                                          const Vector3& q2, Scalar& s, Scalar& t, Vector3& c1,
                                          Vector3& c2);

/* \brief Compute the minimum distance between two vectors 
*         projected onto the same axis.
*/
__device__ Scalar MinDist(const Vector3& p1, const Vector3& p2, const Vector3& q1,
                          const Vector3& q2);

/**
 * @brief Compute signed VF distance
 * @param x The vertex
 * @param y0 The face vertex 0
 * @param y1 The face vertex 1
 * @param y2 The face vertex 2
 * @param n The face normal
 * @param w The barycentric coordinate weights
 * @return distance
*/
__host__ __device__ Scalar SignedVertexFaceDistance(const Vector3& x, const Vector3& y0,
                                                    const Vector3& y1, const Vector3& y2,
                                                    Vector3& n, Scalar* w);

__host__ __device__ void GetVertexFaceBarycentricCoordinates(const Vector3& v1, const Vector3& v2,
                                                             const Vector3& v3, const Vector3& v4,
                                                             Scalar* bary);

__host__ __device__ void GetEdgeEdgeBarycentricCoordinates(const Vector3& v1, const Vector3& v2,
                                                           const Vector3& v3, const Vector3& v4,
                                                           Scalar* bary);

__host__ __device__ Scalar SignedEdgeEdgeDistance(const Vector3& x0, const Vector3& x1,
                                                  const Vector3& y0, const Vector3& y1, Vector3& n,
                                                  Scalar* w);

__host__ __device__ Scalar UnsignedVertexEdgeDistance(const Vector3& x, const Vector3& y0,
                                                      const Vector3& y1, Vector3& n, Scalar& wx,
                                                      Scalar& wy0, Scalar& wy1);

__host__ __device__ Scalar UnsignedVertexFaceDistance(const Vector3& x, const Vector3& y0,
                                                      const Vector3& y1, const Vector3& y2,
                                                      Vector3& n, Scalar* w);

}  // namespace BasicPrimitiveTests
}  // namespace XRTailor