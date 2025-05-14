#include <xrtailor/physics/broad_phase/Bounds.cuh>

namespace XRTailor {

__host__ __device__ Bounds::Bounds() {
  const Scalar inf = std::numeric_limits<Scalar>::infinity();
  lower.x = inf;
  lower.y = inf;
  lower.z = inf;
  upper.x = -inf;
  upper.y = -inf;
  upper.z = -inf;
}

__host__ __device__ Bounds::Bounds(const Vector3& lower, const Vector3& upper)
    : lower(lower), upper(upper) {}

__host__ __device__ Bounds::Bounds(const Primitive& p, const Scalar& scr) {
  const Vector3 v1 = p.v1;
  const Vector3 v2 = p.v2;
  const Vector3 v3 = p.v3;

  const Vector3 pred1 = p.pred1;
  const Vector3 pred2 = p.pred2;
  const Vector3 pred3 = p.pred3;

  Scalar minX = MathFunctions::min(MathFunctions::min(v1.x, v2.x, v3.x),
                                   MathFunctions::min(pred1.x, pred2.x, pred3.x));
  Scalar minY = MathFunctions::min(MathFunctions::min(v1.y, v2.y, v3.y),
                                   MathFunctions::min(pred1.y, pred2.y, pred3.y));
  Scalar minZ = MathFunctions::min(MathFunctions::min(v1.z, v2.z, v3.z),
                                   MathFunctions::min(pred1.z, pred2.z, pred3.z));

  Scalar maxX = MathFunctions::max(MathFunctions::max(v1.x, v2.x, v3.x),
                                   MathFunctions::max(pred1.x, pred2.x, pred3.x));
  Scalar maxY = MathFunctions::max(MathFunctions::max(v1.y, v2.y, v3.y),
                                   MathFunctions::max(pred1.y, pred2.y, pred3.y));
  Scalar maxZ = MathFunctions::max(MathFunctions::max(v1.z, v2.z, v3.z),
                                   MathFunctions::max(pred1.z, pred2.z, pred3.z));

  lower = Vector3(minX, minY, minZ);
  upper = Vector3(maxX, maxY, maxZ);

  Vector3 center = (lower + upper) / static_cast<Scalar>(2.0);

  Vector3 n1 = glm::normalize(lower - center);
  Vector3 n2 = glm::normalize(upper - center);

  lower += scr * n1;
  upper += scr * n2;
}

__host__ __device__ Vector3 Bounds::MinVector(const Vector3& a, const Vector3& b) const {
  Vector3 ans;
  for (int i = 0; i < 3; i++)
    ans[i] = MathFunctions::min(a[i], b[i]);

  return ans;
}

__host__ __device__ Vector3 Bounds::MaxVector(const Vector3& a, const Vector3& b) const {
  Vector3 ans;
  for (int i = 0; i < 3; i++)
    ans[i] = MathFunctions::max(a[i], b[i]);

  return ans;
}

__host__ __device__ Bounds Bounds::operator+(const Bounds& b) const {
  return Bounds(MinVector(lower, b.lower), MaxVector(upper, b.upper));
}

__host__ __device__ void Bounds::operator+=(const Vector3& v) {
  lower = MinVector(lower, v);
  upper = MaxVector(upper, v);
}

__host__ __device__ void Bounds::operator+=(const Bounds& b) {
  lower = MinVector(lower, b.lower);
  upper = MaxVector(upper, b.upper);
}

__host__ __device__ Vector3 Bounds::Center() const {
  return static_cast<Scalar>(0.5) * (lower + upper);
}

__host__ __device__ int Bounds::MajorAxis() const {
  Vector3 d = upper - lower;
  if (d[0] >= d[1] && d[0] >= d[2])
    return 0;
  else if (d[1] >= d[0] && d[1] >= d[2])
    return 1;
  else
    return 2;
}

__host__ __device__ Bounds Bounds::Dilate(Scalar thickness) const {
  Bounds ans = *this;
  for (int i = 0; i < 3; i++) {
    ans.lower[i] -= thickness;
    ans.upper[i] += thickness;
  }

  return ans;
}

__host__ __device__ Scalar Bounds::Distance(const Vector3& x) const {
  Vector3 p;
  for (int i = 0; i < 3; i++)
    p[i] = MathFunctions::Clamp(x[i], lower[i], upper[i]);

  return sqrt(glm::dot(x - p, x - p));
}

__host__ __device__ bool Bounds::Overlap(const Bounds& b) const {
  for (int i = 0; i < 3; i++) {
    if (lower[i] > b.upper[i])
      return false;
    if (upper[i] < b.lower[i])
      return false;
  }

  return true;
}

__host__ __device__ bool Bounds::Overlap(const Bounds& b, Scalar thickness) const {
  return Overlap(b.Dilate(thickness));
}

__host__ __device__ bool Bounds::Overlap(const Vector3& p) const {
  if (p.x > lower.x && p.x < upper.x && p.y > lower.y && p.y < upper.y && p.z > lower.z &&
      p.z < upper.z)
    return true;

  return false;
}

__host__ __device__ inline int GetIntersection(Scalar fDst1, Scalar fDst2, Vector3 p1, Vector3 p2,
                                               Vector3& hit) {
  if ((fDst1 * fDst2) >= static_cast<Scalar>(0.0))
    return 0;
  if (fDst1 == fDst2)
    return 0;
  hit = p1 + (p2 - p1) * (-fDst1 / (fDst2 - fDst1));
  return 1;
}

__host__ __device__ inline int InBox(Vector3 hit, Vector3 lower, Vector3 upper, const int axis) {
  if (axis == 1 && hit.z > lower.z && hit.z < upper.z && hit.y > lower.y && hit.y < upper.y)
    return 1;
  if (axis == 2 && hit.z > lower.z && hit.z < upper.z && hit.x > lower.x && hit.x < upper.x)
    return 1;
  if (axis == 3 && hit.x > lower.x && hit.x < upper.x && hit.y > lower.y && hit.y < upper.y)
    return 1;
  return 0;
}

__host__ __device__ bool Bounds::Overlap(const Vector3& p1, const Vector3& p2, Vector3& hit) const {
  if (p2.x < lower.x && p1.x < lower.x)
    return false;
  if (p2.x > upper.x && p1.x > upper.x)
    return false;
  if (p2.y < lower.y && p1.y < lower.y)
    return false;
  if (p2.y > upper.y && p1.y > upper.y)
    return false;
  if (p2.z < lower.z && p1.z < lower.z)
    return false;
  if (p2.z > upper.z && p1.z > upper.z)
    return false;
  if (p1.x > lower.x && p1.x < upper.x && p1.y > lower.y && p1.y < upper.y && p1.z > lower.z &&
      p1.z < upper.z) {
    hit = p1;
    return true;
  }

  if ((GetIntersection(p1.x - lower.x, p2.x - lower.x, p1, p2, hit) &&
       InBox(hit, lower, upper, 1)) ||
      (GetIntersection(p1.y - lower.y, p2.y - lower.y, p1, p2, hit) &&
       InBox(hit, lower, upper, 2)) ||
      (GetIntersection(p1.z - lower.z, p2.z - lower.z, p1, p2, hit) &&
       InBox(hit, lower, upper, 3)) ||
      (GetIntersection(p1.x - upper.x, p2.x - upper.x, p1, p2, hit) &&
       InBox(hit, lower, upper, 1)) ||
      (GetIntersection(p1.y - upper.y, p2.y - upper.y, p1, p2, hit) &&
       InBox(hit, lower, upper, 2)) ||
      (GetIntersection(p1.z - upper.z, p2.z - upper.z, p1, p2, hit) && InBox(hit, lower, upper, 3)))
    return true;

  return false;
}

__host__ __device__ Bounds Bounds::Merge(const Bounds& b) const {
  Bounds ans;

  for (int i = 0; i < 3; i++) {
    ans.lower[i] = MathFunctions::min(lower[i], b.lower[i]);
    ans.upper[i] = MathFunctions::max(upper[i], b.upper[i]);
  }

  return ans;
}

__host__ __device__ Bounds Bounds::Merge(const Vector3& p) const {
  Bounds ans;

  for (int i = 0; i < 3; i++) {
    ans.lower[i] = MathFunctions::min(lower[i], p[i]);
    ans.upper[i] = MathFunctions::max(upper[i], p[i]);
  }

  return ans;
}

}  // namespace XRTailor