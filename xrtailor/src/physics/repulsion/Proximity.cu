#include <xrtailor/physics/repulsion/Proximity.cuh>

#include <xrtailor/math/BasicPrimitiveTests.cuh>
namespace XRTailor {

__host__ __device__ Proximity::Proximity() {
  stiffness = static_cast<Scalar>(-1.0);
};

__host__ __device__ Proximity::Proximity(Node* node, Face* face, Scalar stiffness)
    : nodes{node, face->nodes[0], face->nodes[1], face->nodes[2]},
      is_fv(true),
      stiffness(stiffness) {
  Scalar w[4];
  Scalar d = BasicPrimitiveTests::SignedVertexFaceDistance(nodes[0]->x0, nodes[1]->x0, nodes[2]->x0,
                                                           nodes[3]->x0, n, w);
  if (d < static_cast<Scalar>(0.0) && face->IsFree())  // vertex enters from below the face
    n = -n;
}

__host__ __device__ Proximity::Proximity(Edge* edge0, Edge* edge1, Scalar stiffness)
    : nodes{edge0->nodes[0], edge0->nodes[1], edge1->nodes[0], edge1->nodes[1]},
      is_fv(false),
      stiffness(stiffness) {
  Scalar w[4];
  Scalar d = BasicPrimitiveTests::SignedEdgeEdgeDistance(nodes[0]->x0, nodes[1]->x0, nodes[2]->x0,
                                                         nodes[3]->x0, n, w);

  if (!edge1->IsFree())  // obstacle, using dihedral normal of the edge on the collider
  {
    n = edge1->ComputeNormal();
  } else {
    if (d < static_cast<Scalar>(0.0))  // edge0 enters from below the edge1
      n = -n;
  }
}

__host__ __device__ bool ProximityIsNull::operator()(Proximity prox) {
  return prox.stiffness < static_cast<Scalar>(0.0);
}

__host__ __device__ VFProximity::VFProximity() : node(nullptr), face(nullptr) {}

__host__ __device__ bool VFProximity::operator<(const VFProximity& p) const {
  if (node == p.node)
    return d < p.d;

  return node < p.node;
}

__host__ __device__ EEProximity::EEProximity() : edge(nullptr) {}

__host__ __device__ RTProximity::RTProximity() : node(nullptr), face(nullptr) {}

}  // namespace XRTailor