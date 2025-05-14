#include <xrtailor/memory/Node.cuh>

namespace XRTailor {
__host__ __device__ Node::Node() {}

__host__ __device__ Node::Node(const Vector3& x, bool is_free, bool is_cloth)
    : index(0), x0(x), x1(x), x(x), is_free(is_free), is_cloth(is_cloth), color(0) {}

__host__ __device__ Bounds Node::ComputeBounds(bool ccd) const {
  Bounds ans;
  ans += x;
  if (ccd)
    ans += x0;

  return ans;
}

__host__ __device__ Vector3 Node::Position(Scalar t) const {
  return x0 + t * (x - x0);
}

}  // namespace XRTailor