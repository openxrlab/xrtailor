#include <xrtailor/memory/Face.cuh>

namespace XRTailor {
__host__ __device__ Face::Face() {}

__host__ __device__ Face::Face(const Node* v0, const Node* v1, const Node* v2)
    : nodes{const_cast<Node*>(v0), const_cast<Node*>(v1), const_cast<Node*>(v2)} {}

__host__ __device__ void Face::SetEdge(const Edge* edge) {
  Node* tgt_node0 = edge->nodes[0];
  Node* tgt_node1 = edge->nodes[1];
  for (int i = 0; i < 3; i++) {
    Node* src_node0 = nodes[i];
    Node* src_node1 = nodes[(i + 1) % 3];
    if (src_node0 == tgt_node0 && src_node1 == tgt_node1 ||
        src_node0 == tgt_node1 && src_node1 == tgt_node0) {
      edges[i] = const_cast<Edge*>(edge);
      return;
    }
  }
}

__host__ __device__ void Face::SetEdges(const Edge* edge0, const Edge* edge1, const Edge* edge2) {
  edges[0] = const_cast<Edge*>(edge0);
  edges[1] = const_cast<Edge*>(edge1);
  edges[2] = const_cast<Edge*>(edge2);
}

__host__ __device__ bool Face::IsFree() const {
  return nodes[0]->is_free && nodes[1]->is_free && nodes[2]->is_free;
}

__host__ __device__ bool Face::Contain(const Node* node) const {
  return nodes[0] == node || nodes[1] == node || nodes[2] == node;
}

__host__ __device__ bool Face::Contain(const Edge* edge) const {
  return edges[0] == edge || edges[1] == edge || edges[2] == edge;
}

__host__ __device__ bool Face::Adjacent(const Face* face) const {
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      if (nodes[i] == face->nodes[j])
        return true;
  return false;
}

__host__ __device__ void Face::Update() {
  Vector3 d1 = nodes[1]->x - nodes[0]->x;
  Vector3 d2 = nodes[2]->x - nodes[0]->x;
  n = glm::normalize(glm::cross(d1, d2));
}

__host__ __device__ Bounds Face::ComputeBounds(bool ccd) const {
  Bounds ans;
  for (int i = 0; i < 3; i++) {
    const Node* node = nodes[i];
    ans += node->x;
    if (ccd)
      ans += node->x0;
  }

  return ans;
}

__host__ __device__ Node* Face::OppositeNode(Node* node0, Node* node1) const {
  if (nodes[0] == node0 && nodes[1] == node1 || nodes[0] == node1 && nodes[1] == node0)
    return nodes[2];
  if (nodes[0] == node0 && nodes[2] == node1 || nodes[0] == node1 && nodes[2] == node0)
    return nodes[1];
  if (nodes[1] == node0 && nodes[2] == node1 || nodes[1] == node1 && nodes[2] == node0)
    return nodes[0];

  return nullptr;
}

}  // namespace XRTailor