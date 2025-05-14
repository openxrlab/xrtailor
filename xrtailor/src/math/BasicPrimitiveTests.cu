#include <xrtailor/math/BasicPrimitiveTests.cuh>

namespace XRTailor {
namespace BasicPrimitiveTests {
// Project the point onto triangle and find projetion point
__host__ __device__ Scalar NearestPoint(const Vector3& p, const Primitive& f, Vector3& qs,
                                        Scalar& _u, Scalar& _v, Scalar& _w) noexcept {
  const Vector3 a = Vector3(f.pred1.x, f.pred1.y, f.pred1.z);
  const Vector3 b = Vector3(f.pred2.x, f.pred2.y, f.pred2.z);
  const Vector3 c = Vector3(f.pred3.x, f.pred3.y, f.pred3.z);

  Vector3 e1 = b - a;
  Vector3 e2 = c - a;
  Vector3 e3 = c - b;

  // check if p is outside vertex region a
  Vector3 v1 = p - a;
  Scalar d1 = glm::dot(e1, v1), d2 = glm::dot(e2, v1);

  if (d1 <= 0 && d2 <= 0) {
    qs = a;

    _u = 1;
    _v = 0;
    _w = 0;
    return glm::length(p - qs);
  }

  // check if p is outside vertex region b
  Vector3 v2 = p - b;
  Scalar d3 = glm::dot(e1, v2), d4 = glm::dot(e2, v2);
  if (d3 >= 0 && d4 <= d3) {
    qs = b;

    _u = 0;
    _v = 1;
    _w = 0;
    return glm::length(p - qs);
  }

  // check if p is in edge region e1, if so return projection of p onto e1
  Scalar vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    Scalar v = d1 / (d1 - d3);
    qs = a + v * e1;

    _u = 1 - v;
    _v = v;
    _w = 0;
    return glm::length(p - qs);
  }

  // check if p in vertex region outside c
  Vector3 v3 = p - c;
  Scalar d5 = glm::dot(e1, v3), d6 = glm::dot(e2, v3);
  if (d6 >= 0 && d5 <= d6) {
    qs = c;

    _u = 0;
    _v = 0;
    _w = 1;
    return glm::length(p - qs);
  }

  // check if p is in edge region e2, if so return projection of p onto e2
  Scalar vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    Scalar w = d2 / (d2 - d6);
    qs = a + w * e2;

    _u = 1 - w;
    _v = 0;
    _w = w;
    return glm::length(p - qs);
  }

  // check if p is in edge region e3, if so return projection of p onto e3
  Scalar va = d3 * d6 - d5 * d4;
  if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
    Scalar w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    qs = b + w * e3;

    _u = 0;
    _v = 1 - w;
    _w = w;
    return glm::length(p - qs);
  }

  // p inside face region. Compute point through its barycentric coordinates (u,v,w)
  Scalar d = 1 / (va + vb + vc);
  Scalar v = vb * d;
  Scalar w = vc * d;
  qs = a + e1 * v + e2 * w;

  _u = 1 - v - w;
  _v = v;
  _w = w;

  return glm::length(p - qs);
}

__host__ __device__ Scalar NearestPoint(const Vector3& x, const Vector3& y0, const Vector3& y1,
                                        const Vector3& y2, Vector3& qs, Scalar& _u, Scalar& _v,
                                        Scalar& _w) noexcept {
  Vector3 e1 = y1 - y0;
  Vector3 e2 = y2 - y0;
  Vector3 e3 = y2 - y1;

  // check if x is outside vertex region y0
  Vector3 v1 = x - y0;
  Scalar d1 = glm::dot(e1, v1), d2 = glm::dot(e2, v1);

  if (d1 <= 0 && d2 <= 0) {
    qs = y0;

    _u = 1;
    _v = 0;
    _w = 0;
    return glm::length(x - qs);
  }

  // check if x is outside vertex region y1
  Vector3 v2 = x - y1;
  Scalar d3 = glm::dot(e1, v2), d4 = glm::dot(e2, v2);
  if (d3 >= 0 && d4 <= d3) {
    qs = y1;

    _u = 0;
    _v = 1;
    _w = 0;
    return glm::length(x - qs);
  }

  // check if x is in edge region e1, if so return projection of x onto e1
  Scalar vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    Scalar v = d1 / (d1 - d3);
    qs = y0 + v * e1;

    _u = 1 - v;
    _v = v;
    _w = 0;
    return glm::length(x - qs);
  }

  // check if p in vertex region outside c
  Vector3 v3 = x - y2;
  Scalar d5 = glm::dot(e1, v3), d6 = glm::dot(e2, v3);
  if (d6 >= 0 && d5 <= d6) {
    qs = y2;

    _u = 0;
    _v = 0;
    _w = 1;
    return glm::length(x - qs);
  }

  // check if x is in edge region e2, if so return projection of x onto e2
  Scalar vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    Scalar w = d2 / (d2 - d6);
    qs = y0 + w * e2;

    _u = 1 - w;
    _v = 0;
    _w = w;
    return glm::length(x - qs);
  }

  // check if x is in edge region e3, if so return projection of x onto e3
  Scalar va = d3 * d6 - d5 * d4;
  if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
    Scalar w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    qs = y1 + w * e3;

    _u = 0;
    _v = 1 - w;
    _w = w;
    return glm::length(x - qs);
  }

  // p inside face region. Compute point through its barycentric coordinates (u,v,w)
  Scalar d = 1 / (va + vb + vc);
  Scalar v = vb * d;
  Scalar w = vc * d;
  qs = y0 + e1 * v + e2 * w;

  _u = 1 - v - w;
  _v = v;
  _w = w;

  return glm::length(x - qs);
}

__device__ bool RayIntersect(const Vector3& ev0, const Vector3& ev1, const Vector3& fv0,
                             const Vector3& fv1, const Vector3& fv2, Scalar& dist) noexcept {
  Vector3 d = ev1 - ev0;
  // Möller–Trumbore intersection algorithm

  Vector3 e1 = fv1 - fv0;
  Vector3 e2 = fv2 - fv0;
  Vector3 n = glm::cross(d, e2);
  Scalar det = glm::dot(e1, n);

  // ray does not lie in the plane
  if (MathFunctions::abs(det) < EPSILON) {
    dist = INFINITY;
    return false;
  }

  Scalar invDet = 1 / det;
  Vector3 t = ev0 - fv0;
  Scalar u = glm::dot(t, n) * invDet;

  // ray lies outside triangle
  if (u < 0 || u > 1) {
    dist = INFINITY;
    return false;
  }

  Vector3 q = glm::cross(t, e1);
  Scalar v = glm::dot(d, q) * invDet;

  // ray lies outside the triangle
  if (v < 0 || v + u > 1) {
    dist = INFINITY;
    return false;
  }

  // check for intersection
  Scalar s = glm::dot(e2, q) * invDet;
  if (s > 0 && s < glm::length(d)) {
    dist = s;
    return true;
  }

  // no hit
  dist = INFINITY;
  return true;
}

__device__ Scalar RayIntersect(const Vector3& o, const Vector3& d, Scalar ray_length,
                               const Primitive& f) noexcept {
  // Möller–Trumbore intersection algorithm
  const Vector3& a(f.pred1);
  const Vector3& b(f.pred2);
  const Vector3& c(f.pred3);

  Vector3 e1 = b - a;
  Vector3 e2 = c - a;
  Vector3 n = glm::cross(d, e2);
  Scalar det = glm::dot(e1, n);

  // ray does not lie in the plane
  if (MathFunctions::abs(det) < EPSILON) {
    return SCALAR_MAX;
  }

  Scalar invDet = 1 / det;
  Vector3 t = o - a;
  Scalar u = glm::dot(t, n) * invDet;

  // ray lies outside triangle
  if (u < 0 || u > 1) {
    return SCALAR_MAX;
  }

  Vector3 q = glm::cross(t, e1);
  Scalar v = glm::dot(d, q) * invDet;

  // ray lies outside the triangle
  if (v < 0 || v + u > 1) {
    return SCALAR_MAX;
  }

  // check for intersection
  Scalar s = glm::dot(e2, q) * invDet;
  if (s > 0 && s < ray_length) {
    return s;
  }

  // no hit
  return INFINITY;
}

__host__ __device__ bool InEdge(Scalar w, const Edge* edge0, const Edge* edge1) {
  Vector3 x = (1 - w) * edge0->nodes[0]->x + w * edge0->nodes[1]->x;
  bool in = true;
  for (int i = 0; i < MAX_EF_ADJACENTS; i++) {
    Face* face = edge1->adjacents[i];
    if (face == nullptr)
      continue;
    Node* node0 = edge1->nodes[0];
    Node* node1 = edge1->nodes[1];
    Vector3 e = node1->x - node0->x;
    Vector3 n = face->n;
    Vector3 r = x - node0->x;
    in &= (MathFunctions::Mixed(e, n, r) >= 0);
  }
  return in;
}

__device__ Scalar ClosestPtSegmentSegment(const Vector3& p1, const Vector3& q1, const Vector3& p2,
                                          const Vector3& q2, Scalar& s, Scalar& t, Vector3& c1,
                                          Vector3& c2) {
  Vector3 d1 = q1 - p1;  // Direction vector of segment S1
  Vector3 d2 = q2 - p2;  // Direction vector of segment S2
  Vector3 r = p1 - p2;
  Scalar a = glm::dot(d1, d1);  // Squared length of segment S1, always nonnegative
  Scalar e = glm::dot(d2, d2);  // Squared length of segment S2, always nonnegative
  Scalar f = glm::dot(d2, r);
  // Check if either or both segments degenerate into points
  if (a <= EPSILON && e <= EPSILON) {
    // Both segments degenerate into points
    s = t = 0;
    c1 = p1;
    c2 = p2;
    return glm::length(c1 - c2);
  }
  if (a <= EPSILON) {
    // First segment degenerates into a point
    s = 0;
    t = f / e;  // s = 0 => t = (b*s + f) / e = f / e
    t = MathFunctions::Clamp(t, static_cast<Scalar>(0), static_cast<Scalar>(1));
  } else {
    Scalar c = glm::dot(d1, r);
    if (e <= EPSILON) {
      // Second segment degenerates into a point
      t = 0;
      s = MathFunctions::Clamp(-c / a, static_cast<Scalar>(0),
                               static_cast<Scalar>(1));  // t = 0 => s = (b*t - c) / a = -c / a
    } else {
      // The general nondegenerate case starts here
      Scalar b = glm::dot(d1, d2);
      Scalar denom = a * e - b * b;  // Always nonnegative
      // If segments not parallel, compute closest point on L1 to L2 and
      // clamp to segment S1. Else pick arbitrary s (here 0)
      if (denom > EPSILON) {
        s = MathFunctions::Clamp((b * f - c * e) / denom, static_cast<Scalar>(0),
                                 static_cast<Scalar>(1));
      } else
        s = 0;
      // Compute point on L2 closest to S1(s) using
      // t = Dot((P1 + D1*s) - P2,D2) / Dot(D2,D2) = (b*s + f) / e
      t = (b * s + f) / e;
      // If t in [0,1] done. Else clamp t, recompute s for the new value
      // of t using s = Dot((P2 + D2*t) - P1,D1) / Dot(D1,D1)= (t*b - c) / a
      // and clamp s to [0, 1]
      if (t < EPSILON) {
        t = 0;
        s = MathFunctions::Clamp(-c / a, static_cast<Scalar>(0), static_cast<Scalar>(1));
      } else if (t > 1) {
        t = 1;
        s = MathFunctions::Clamp((b - c) / a, static_cast<Scalar>(0), static_cast<Scalar>(1));
      }
    }
  }
  c1 = p1 + d1 * s;
  c2 = p2 + d2 * t;
  return glm::length(c1 - c2);
}

__device__ Scalar MinDist(const Vector3& p1, const Vector3& p2, const Vector3& q1,
                          const Vector3& q2) {
  return MathFunctions::min(MathFunctions::min(glm::length(p1 - q1), glm::length(p1 - q2)),
                            MathFunctions::min(glm::length(p2 - q1), glm::length(p2 - q2)));
}

Scalar SignedVertexFaceDistance(const Vector3& x, const Vector3& y0, const Vector3& y1,
                                const Vector3& y2, Vector3& n, Scalar* w) {
  n = glm::cross(glm::normalize(y1 - y0), glm::normalize(y2 - y0));
  if (glm::dot(n, n) < static_cast<Scalar>(1e-6))
    return INFINITY;
  n = glm::normalize(n);
  Scalar h = glm::dot(x - y0, n);
  // barycentric coordinates
  Scalar b0 = MathFunctions::Mixed(y1 - x, y2 - x, n);
  Scalar b1 = MathFunctions::Mixed(y2 - x, y0 - x, n);
  Scalar b2 = MathFunctions::Mixed(y0 - x, y1 - x, n);
  w[0] = static_cast<Scalar>(1.0);
  // negative normalized coordinates
  w[1] = -b0 / (b0 + b1 + b2);
  w[2] = -b1 / (b0 + b1 + b2);
  w[3] = -b2 / (b0 + b1 + b2);
  return h;
}

void GetVertexFaceBarycentricCoordinates(const Vector3& v1, const Vector3& v2, const Vector3& v3,
                                         const Vector3& v4, Scalar* bary) {
  Vector3 x13 = v1 - v3;
  Vector3 x23 = v2 - v3;
  Vector3 x43 = v4 - v3;
  Scalar A00 = glm::dot(x13, x13);
  Scalar A01 = glm::dot(x13, x23);
  Scalar A11 = glm::dot(x23, x23);
  Scalar b0 = glm::dot(x13, x43);
  Scalar b1 = glm::dot(x23, x43);
  Scalar detA = A00 * A11 - A01 * A01;
  bary[0] = (A11 * b0 - A01 * b1) / (detA + static_cast<Scalar>(1e-6));
  bary[1] = (-A01 * b0 + A00 * b1) / (detA + static_cast<Scalar>(1e-6));
  bary[2] = 1 - bary[0] - bary[1];
}

void GetEdgeEdgeBarycentricCoordinates(const Vector3& v1, const Vector3& v2, const Vector3& v3,
                                       const Vector3& v4, Scalar* bary) {
  Vector3 x21 = v2 - v1;
  Vector3 x43 = v4 - v3;
  Vector3 x31 = v3 - v1;
  Scalar A00 = glm::dot(x21, x21);
  Scalar A01 = -glm::dot(x21, x43);
  Scalar A11 = glm::dot(x43, x43);
  Scalar b0 = glm::dot(x21, x31);
  Scalar b1 = -glm::dot(x43, x31);
  Scalar detA = A00 * A11 - A01 * A01;

  bary[0] = (A11 * b0 - A01 * b1) / (detA + static_cast<Scalar>(1e-6));
  bary[1] = (-A01 * b0 + A00 * b1) / (detA + static_cast<Scalar>(1e-6));
}

Scalar SignedEdgeEdgeDistance(const Vector3& x0, const Vector3& x1, const Vector3& y0,
                              const Vector3& y1, Vector3& n, Scalar* w) {
  n = glm::cross(glm::normalize(x1 - x0), glm::normalize(y1 - y0));
  if (glm::dot(n, n) < static_cast<Scalar>(1e-6))
    return INFINITY;
  glm::normalize(n);
  Scalar h = glm::dot(x0 - y0, n);
  Scalar a0 = MathFunctions::Mixed(y1 - x1, y0 - x1, n);
  Scalar a1 = MathFunctions::Mixed(y0 - x0, y1 - x0, n);
  Scalar b0 = MathFunctions::Mixed(x0 - y1, x1 - y1, n);
  Scalar b1 = MathFunctions::Mixed(x1 - y0, x0 - y0, n);
  w[0] = a0 / (a0 + a1);
  w[1] = a1 / (a0 + a1);
  w[2] = -b0 / (b0 + b1);
  w[3] = -b1 / (b0 + b1);
  return h;
}

Scalar UnsignedVertexEdgeDistance(const Vector3& x, const Vector3& y0, const Vector3& y1,
                                  Vector3& n, Scalar& wx, Scalar& wy0, Scalar& wy1) {
  Scalar t = MathFunctions::Clamp(glm::dot(x - y0, y1 - y0) / glm::dot(y1 - y0, y1 - y0),
                                  static_cast<Scalar>(0.0), static_cast<Scalar>(1.0));
  Vector3 y = y0 + t * (y1 - y0);
  Scalar d = sqrt(glm::dot(x - y, x - y));
  n = glm::normalize(x - y);
  wx = static_cast<Scalar>(1.0);
  wy0 = static_cast<Scalar>(1.0) - t;
  wy1 = t;
  return d;
}

Scalar UnsignedVertexFaceDistance(const Vector3& x, const Vector3& y0, const Vector3& y1,
                                  const Vector3& y2, Vector3& n, Scalar* w) {
  Vector3 nt = glm::normalize(glm::cross(y1 - y0, y2 - y0));
  Scalar d = abs(glm::dot(x - y0, nt));
  Scalar b0 = MathFunctions::Mixed(y1 - x, y2 - x, nt);
  Scalar b1 = MathFunctions::Mixed(y2 - x, y0 - x, nt);
  Scalar b2 = MathFunctions::Mixed(y0 - x, y1 - x, nt);
  if (b0 >= static_cast<Scalar>(0.0) && b1 >= static_cast<Scalar>(0.0) &&
      b2 >= static_cast<Scalar>(0.0)) {
    n = nt;
    w[0] = static_cast<Scalar>(1.0);
    w[1] = -b0 / (b0 + b1 + b2);
    w[2] = -b1 / (b0 + b1 + b2);
    w[3] = -b2 / (b0 + b1 + b2);
    return d;
  }
  d = INFINITY;
  if (b0 < static_cast<Scalar>(0.0)) {
    Scalar dt = UnsignedVertexEdgeDistance(x, y1, y2, n, w[0], w[2], w[3]);
    if (dt < d) {
      d = dt;
      w[1] = static_cast<Scalar>(0.0);
    }
  }
  if (b1 < static_cast<Scalar>(0.0)) {
    Scalar dt = UnsignedVertexEdgeDistance(x, y2, y0, n, w[0], w[3], w[1]);
    if (dt < d) {
      d = dt;
      w[2] = static_cast<Scalar>(0.0);
    }
  }
  if (b2 < static_cast<Scalar>(0.0)) {
    Scalar dt = UnsignedVertexEdgeDistance(x, y0, y1, n, w[0], w[1], w[2]);
    if (dt < d) {
      d = dt;
      w[3] = static_cast<Scalar>(0.0);
    }
  }
  return d;
}

}  // namespace BasicPrimitiveTests
}  // namespace XRTailor