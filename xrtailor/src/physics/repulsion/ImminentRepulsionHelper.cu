#include <xrtailor/physics/repulsion/ImminentRepulsionHelper.cuh>
#include <xrtailor/math/BasicPrimitiveTests.cuh>

namespace XRTailor {

__host__ __device__ bool QuadIsNull::operator()(Quadrature quad) const {
  return quad.nodes[0] == nullptr;
}

__host__ __device__ void CheckImminentVFQuadrature(Node* node, Face* face, Quadrature& quad) {
  Node* node0 = face->nodes[0];
  Node* node1 = face->nodes[1];
  Node* node2 = face->nodes[2];

  if (node == node0 || node == node1 || node == node2)
    return;

  Node* nodes[4] = {node0, node1, node2, node};

  Vector3 ps[4], vs[4];
  for (int i = 0; i < 4; i++) {
    ps[i] = nodes[i]->x0;
    vs[i] = nodes[i]->v;
  }

  Vector3 fn;
  Scalar w[4], bary[4];
  BasicPrimitiveTests::SignedVertexFaceDistance(ps[3], ps[0], ps[1], ps[2], fn, w);

  bool inside = (MathFunctions::min(-w[1], -w[2], -w[3]) >= static_cast<Scalar>(1e-6));

  if (!inside)
    return;

  for (int i = 0; i < 4; i++) {
    bary[i] = w[(i + 1) % 4];
  }

  for (int i = 0; i < 4; i++) {
    quad.bary[i] = bary[i];
    quad.nodes[i] = nodes[i];
  }
}

__device__ void CheckImminentEEQuadrature(Edge* edge0, Edge* edge1, Quadrature& quad) {
  Node* node0 = edge0->nodes[0];
  Node* node1 = edge0->nodes[1];
  Node* node2 = edge1->nodes[0];
  Node* node3 = edge1->nodes[1];

  if (node0 == node2 || node0 == node3 || node1 == node2 || node1 == node3)
    return;

  Node* nodes[4] = {node0, node1, node2, node3};

  Vector3 ps[4], vs[4];
  for (int i = 0; i < 4; i++) {
    ps[i] = nodes[i]->x0;
    vs[i] = nodes[i]->v;
  }

  Scalar w[4];
  Vector3 fn;
  BasicPrimitiveTests::SignedEdgeEdgeDistance(ps[0], ps[1], ps[2], ps[3], fn, w);
  bool inside = (MathFunctions::min(w[0], w[1], -w[2], -w[3]) >= static_cast<Scalar>(1e-6) &&
                 BasicPrimitiveTests::InEdge(w[1], edge0, edge1) &&
                 BasicPrimitiveTests::InEdge(-w[3], edge1, edge0));
  if (!inside)
    return;

  for (int i = 0; i < 4; i++) {
    quad.bary[i] = w[i];
    quad.nodes[i] = nodes[i];
  }
}

__host__ __device__ bool EvaluateImminentImpulse(Quadrature& quad, Impulse& impulse,
                                                 const Scalar& thickness, const bool& add_repulsion,
                                                 const Scalar& dt) {
  Vector3 pr(0);
  Vector3 vr(0);

  for (int i = 0; i < 4; i++) {
    pr += quad.bary[i] * quad.nodes[i]->x0;
    vr += quad.bary[i] * quad.nodes[i]->v;
  }

  Scalar npr = glm::length(pr);

  if (npr > thickness)
    return false;

  if (npr < static_cast<Scalar>(1e-5))
    return false;

  pr = glm::normalize(pr);

  Scalar vr_nrm = glm::dot(vr, pr);

  Scalar target_repulsive_dist = 0;

  if (add_repulsion) {
    Scalar extra_project_dist_need = thickness - npr;
    target_repulsive_dist = extra_project_dist_need * static_cast<Scalar>(0.1);
  }

  if (vr_nrm > target_repulsive_dist)
    return false;

  Vector3 I = vr_nrm * dt * pr;
  //Vector3 I = (target_repulsive_dist - vr_nrm) * pr;

  Vector3 vr_tan = vr - vr_nrm * pr;
  if (glm::length(vr_tan) > thickness * static_cast<Scalar>(0.01)) {
    Scalar imp_norm = glm::length(I);

    Vector3 vr_tan_dir = glm::normalize(vr_tan);

    Scalar friction_coeff = static_cast<Scalar>(0);
    Scalar friction_scale = static_cast<Scalar>(1);
    Vector3 friction(0);
    if (imp_norm * friction_coeff > glm::length(vr_tan))
      friction = -vr_tan;
    else
      friction = -vr_tan_dir * imp_norm * friction_coeff;

    I += friction * friction_scale;
  }

  Scalar inv_cm = static_cast<Scalar>(0);
  for (int i = 0; i < 4; i++) {
    inv_cm += quad.bary[i] * quad.bary[i] * quad.nodes[i]->inv_mass;
  }

  if (inv_cm < static_cast<Scalar>(1e-5))
    return false;

  for (int i = 0; i < 4; i++) {
    Scalar beta = -quad.nodes[i]->inv_mass * quad.bary[i] / inv_cm;
    impulse.corrs[i] = I * beta;
    impulse.nodes[i] = quad.nodes[i];
    quad.nodes[i]->color = 1;
  }

  return true;
}

__device__ bool evaluateImminentRepulsiveImpulse(Quadrature& quad, Impulse& impulse,
                                                 const Scalar& thickness,
                                                 const Scalar& repulsive_strength,
                                                 const Scalar& max_repel_dist) {
  Vector3 pr(0);
  Vector3 vr(0);

  for (int i = 0; i < 4; i++) {
    pr += quad.bary[i] * quad.nodes[i]->x;
    vr += quad.bary[i] * quad.nodes[i]->v;
  }

  Scalar dist = glm::length(pr);

  pr = glm::normalize(pr);

  Scalar vr_nrm = glm::dot(pr, vr);  // project vr onto npr

  Vector3 impl(0);
  Scalar d = thickness - dist;

  if (vr_nrm < max_repel_dist * d) {
    Scalar I = repulsive_strength * d;
    Scalar I_max = (max_repel_dist * d - vr_nrm);

    I = I > I_max ? I_max : I;
    impl = I * pr;
  }

  Scalar inv_cm = 0;
  for (int i = 0; i < 4; i++) {
    inv_cm += quad.bary[i] * quad.bary[i] * quad.nodes[i]->inv_mass;
  }

  if (inv_cm < EPSILON * 10.0f)
    return false;

  for (int i = 0; i < 4; i++) {
    Scalar beta = quad.nodes[i]->inv_mass * quad.bary[i] / inv_cm;
    impulse.corrs[i] = -impl * beta;
    impulse.nodes[i] = quad.nodes[i];
    quad.nodes[i]->color = 1;
  }

  return true;
}

__device__ void AccumulateImpulses(Vector3* deltas, int* delta_counts, Impulse& impulse) {
  for (int i = 0; i < 4; i++) {
    if (glm::length(impulse.corrs[i]) < EPSILON)
      continue;

    if (!impulse.nodes[i]->is_free)
      continue;

    const int& node_idx = impulse.nodes[i]->index;

    AtomicAdd(deltas, node_idx, impulse.corrs[i]);
    atomicAdd(&delta_counts[node_idx], 1);
  }
}

}  // namespace XRTailor