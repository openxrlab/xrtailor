#include <xrtailor/memory/Edge.cuh>
#include <xrtailor/memory/Face.cuh>
#include <xrtailor/math/MathFunctions.cuh>

namespace XRTailor {

__host__ __device__ Edge::Edge() : adjacents{nullptr, nullptr, nullptr, nullptr} {}

__host__ __device__ Edge::Edge(const Node* node0, const Node* node1)
    : nodes{const_cast<Node*>(node0), const_cast<Node*>(node1)},
      adjacents{nullptr, nullptr, nullptr, nullptr},
      index(0) {}

__host__ __device__ void Edge::ReplaceNode(const Node* src_node, const Node* tgt_node) {
  for (int i = 0; i < 2; i++) {
    if (nodes[i] == src_node) {
      nodes[i] = const_cast<Node*>(tgt_node);
    }
  }
}

__host__ __device__ bool Edge::IsFree() const {
  return nodes[0]->is_free && nodes[1]->is_free;
}

__host__ __device__ Scalar Edge::Length() const {
  return glm::length(nodes[1]->x - nodes[0]->x);
}

__host__ __device__ Bounds Edge::ComputeBounds(bool ccd) const {
  Bounds ans;
  for (int i = 0; i < 2; i++) {
    const Node* node = nodes[i];
    ans += node->x;
    if (ccd)
      ans += node->x0;
  }

  return ans;
}

__host__ __device__ Vector3 Edge::ComputeNormal() const {
  Vector3 ans(0);

  int cnt = 0;

  for (int i = 0; i < MAX_EF_ADJACENTS; i++) {
    if (adjacents[i] == nullptr)
      break;
    Face* face = adjacents[i];
    ans += MathFunctions::FaceNormal(face->nodes[0]->x, face->nodes[1]->x, face->nodes[2]->x);
    cnt++;
  }

  if (cnt == 0)
    return ans;

  return ans / Scalar(cnt);
}

}  // namespace XRTailor