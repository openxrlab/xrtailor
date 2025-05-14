#include <xrtailor/physics/pbd_collision/Constraint.cuh>
#include "cusparse.h"
#include <xrtailor/core/DeviceHelper.cuh>
#include <xrtailor/math/BasicPrimitiveTests.cuh>
#include <xrtailor/physics/broad_phase/lbvh/BVH.cuh>

namespace XRTailor {
namespace PBDCollision {
namespace DCD {
VFConstraint::VFConstraint(uint n_verts) {
  this->contacts.resize(n_verts, VFContact());
  this->n_verts = n_verts;
}

VFConstraint::~VFConstraint() {}

__global__ void GenerateVF_DCD_Kernel(BasicDeviceBVH bvh_dev, Node** nodes, VFContact* contacts,
                                      uint num_verts) {
  GET_CUDA_ID(idx, num_verts);

  contacts[idx] = {-1, Vector3(0)};

  auto pred = nodes[idx]->x;

  unsigned int buffer[LBVH_MAX_BUFFER_SIZE];
  // broad-phase culling
  const auto num_found = QueryDeviceStackless(bvh_dev, pred, buffer);

  Scalar hit = SCALAR_MAX;
  int hit_idx = -1;
  Vector3 hit_qs(0);

  for (unsigned int i = 0; i < num_found; i++) {
    // Narrow-phase test
    Vector3 qs;
    Scalar u, v, w;
    Scalar distance =
        BasicPrimitiveTests::NearestPoint(pred, bvh_dev.objects[buffer[i]], qs, u, v, w);
    if (distance < hit) {
      hit = distance;
      hit_qs = qs;
      hit_idx = buffer[i];
    }
  }
  contacts[idx] = {hit_idx, hit_qs};
}

void VFConstraint::Generate(Node** nodes, std::shared_ptr<BVH> bvh) {
  auto bvh_dev = bvh->GetDeviceRepr();
  if (!bvh->IsActive())
    return;

  CUDA_CALL(GenerateVF_DCD_Kernel, this->n_verts)
  (bvh_dev, nodes, pointer(this->contacts), this->n_verts);
}

__global__ void SolveVFDCD_Kernel(BasicDeviceBVH bvh_dev, Node** nodes, VFContact* contacts,
                                  uint num_verts) {
  GET_CUDA_ID(idx, num_verts);

  auto contact = contacts[idx];

  auto pred = nodes[idx]->x;

  int hit_idx = contact.index;

  if (hit_idx == -1)
    return;

  Vector3 hit_qs = contact.qs;
  auto primitive = bvh_dev.objects[hit_idx];
  Vector3 ns = MathFunctions::FaceNormal(primitive.pred1, primitive.pred2, primitive.pred3);

  Scalar lambda = glm::dot(ns, pred - hit_qs) - static_cast<Scalar>(1e-3);
  if (lambda < 0) {
    Vector3 corr = -lambda * ns;
    for (int j = 0; j < 3; j++)
      atomicAdd(&(nodes[idx]->x[j]), corr[j]);
  }
}

void VFConstraint::Solve(Node** nodes, std::shared_ptr<BVH> bvh) {
  XRTailor::ScopedTimerGPU timer("Solver_SolveVFDCD");
  auto bvh_dev = bvh->GetDeviceRepr();
  if (!bvh->IsActive())
    return;

  CUDA_CALL(SolveVFDCD_Kernel, this->n_verts)
  (bvh_dev, nodes, pointer(this->contacts), this->n_verts);
}

EEConstraint::EEConstraint(uint n_edges, std::shared_ptr<XRTailor::RTriangle> obstacle_r_tri) {
  this->contacts.resize(n_edges, EEContact());
  this->n_edges = n_edges;
  this->obstacle_r_tri = obstacle_r_tri;
}

EEConstraint::~EEConstraint() {}

__global__ void GenerateEE_DCD_Kernel(BasicDeviceBVH bvh_dev, Node** nodes, Edge** edges,
                                      CONST(uint*) object_rTris, EEContact* contacts,
                                      uint n_edges) {
  GET_CUDA_ID(idx, n_edges);

  contacts[idx] = {
      SCALAR_MAX, Vector3(0), Vector3(0), static_cast<Scalar>(0.0), static_cast<Scalar>(0.0),
      Vector3(0)};

  unsigned int buffer[LBVH_MAX_BUFFER_SIZE];

  uint idx1 = edges[idx]->nodes[0]->index;
  uint idx2 = edges[idx]->nodes[1]->index;
  Vector3 p_cloth = nodes[idx1]->x;
  Vector3 q_cloth = nodes[idx2]->x;

  // Broad-phase culling
  uint num_found = QueryDeviceStackless(bvh_dev, p_cloth, q_cloth, buffer);

  Scalar s, t;
  Vector3 p2, q2, fn;
  Scalar minDist = SCALAR_MAX;

  for (uint aabb_idx = 0u; aabb_idx < num_found; aabb_idx++) {
    auto f_idx = buffer[aabb_idx];
    auto primitive = bvh_dev.objects[f_idx];
    auto info = object_rTris[f_idx];
    Vector3 obstacle_verts[3] = {primitive.pred1, primitive.pred2, primitive.pred3};
    for (uint i = 0u; i < 3u; i++) {
      if (XRTailor::RTriEdge(info, i)) {
        Scalar _s, _t;
        Vector3 c1, c2;
        uint j = (i + 1u) % 3u;
        if (BasicPrimitiveTests::ClosestPtSegmentSegment(p_cloth, q_cloth, obstacle_verts[i],
                                                         obstacle_verts[j], _s, _t, c1, c2)) {
          Vector3 d = c1 - c2;
          Scalar dist = glm::length(d);

          if (dist < minDist) {
            minDist = dist;
            p2 = obstacle_verts[i];
            q2 = obstacle_verts[j];
            s = _s;
            t = _t;
            fn = MathFunctions::FaceNormal(obstacle_verts[0], obstacle_verts[1], obstacle_verts[2]);
          }
        }
      }
    }
  }
  contacts[idx] = {minDist, p2, q2, s, t, fn};
}

void EEConstraint::Generate(Node** nodes, Edge** edges, std::shared_ptr<BVH> bvh) {
  auto bvh_dev = bvh->GetDeviceRepr();
  if (!bvh->IsActive())
    return;

  CUDA_CALL(GenerateEE_DCD_Kernel, this->n_edges)
  (bvh_dev, nodes, edges, pointer(this->obstacle_r_tri->r_tris), pointer(this->contacts),
   this->n_edges);
}

__global__ void SolveEEDCD_Kernel(BasicDeviceBVH bvh_dev, Edge** edges, Vector3* deltas,
                                  int* delta_counts, EEContact* contacts, CONST(uint*) object_rTris,
                                  uint n_edges) {
  GET_CUDA_ID(idx, n_edges);

  auto contact = contacts[idx];

  if (contact.dist > 1e6f)
    return;

  Node* node1 = edges[idx]->nodes[0];
  Node* node2 = edges[idx]->nodes[1];
  uint idx1 = node1->index;
  uint idx2 = node2->index;

  Vector3 p1 = node1->x;
  Vector3 q1 = node2->x;
  Vector3 p2 = contact.p2;
  Vector3 q2 = contact.q2;
  Scalar s = contact.s;
  Scalar t = contact.t;
  Vector3 fn = contact.fn;

  Vector3 c1 = (1 - s) * p1 + s * q1;
  Vector3 c2 = (1 - t) * p2 + t * q2;

  Vector3 d = c1 - c2;
  Scalar length = glm::length(d);
  if (length < EPSILON)
    return;

  Vector3 dn = d / length;

  Scalar proj = glm::dot(d, contact.fn) - static_cast<Scalar>(1e-3);

  Scalar lambda = proj;

  if (lambda > 0) {
    // (p1, q1) is on the upper side of (p2, q2)
    return;
  }
  Vector3 corr1 = -(1 - s) * lambda * fn;
  Vector3 corr2 = -(s)*lambda * fn;
  int reorder = idx1 + idx2;
  AtomicAdd(deltas, idx1, corr1, reorder);
  AtomicAdd(deltas, idx2, corr2, reorder);

  atomicAdd(&delta_counts[idx1], 1);
  atomicAdd(&delta_counts[idx2], 1);
}

void EEConstraint::Solve(Edge** edges, Vector3* deltas, int* delta_counts,
                         std::shared_ptr<BVH> bvh) {
  auto bvh_dev = bvh->GetDeviceRepr();
  if (!bvh->IsActive())
    return;

  CUDA_CALL(SolveEEDCD_Kernel, this->n_edges)
  (bvh_dev, edges, deltas, delta_counts, pointer(this->contacts),
   pointer(this->obstacle_r_tri->r_tris), this->n_edges);
}

}  // namespace DCD

namespace CCD {

RTConstraint::RTConstraint(uint n_verts) {
  this->contacts.resize(n_verts, RTContact());
  this->n_verts = n_verts;
}

RTConstraint::~RTConstraint() {}

__global__ void GenerateVFCCD_Kernel(BasicDeviceBVH bvh_dev, Node** nodes, RTContact* contacts,
                                     uint num_verts) {
  GET_CUDA_ID(idx, num_verts);

  contacts[idx] = {-1, Vector3(0)};

  unsigned int buffer[LBVH_MAX_BUFFER_SIZE];

  Vector3 p1 = nodes[idx]->x0;
  Vector3 p2 = nodes[idx]->x;
  Vector3 d = p2 - p1;
  const Scalar length = glm::length(d);
  if (length < EPSILON)
    return;
  d /= length;

  // Broad-phase culling
  const auto num_found = QueryDeviceStackless(bvh_dev, p1, p2, buffer);
  Scalar hit = SCALAR_MAX;
  int hit_idx = -1;
  Vector3 qc(0);
  for (unsigned int i = 0; i < num_found; i++) {
    // Narrow-phase test
    Scalar dist = BasicPrimitiveTests::RayIntersect(p1, d, length, bvh_dev.objects[buffer[i]]);
    if (dist < hit) {
      qc = p1 + dist * d;
      hit = dist;
      hit_idx = buffer[i];
    }
  }
  contacts[idx] = {hit_idx, qc};
}

void RTConstraint::Generate(Node** nodes, std::shared_ptr<BVH> bvh) {
  auto bvh_dev = bvh->GetDeviceRepr();
  if (!bvh->IsActive())
    return;

  CUDA_CALL(GenerateVFCCD_Kernel, this->n_verts)
  (bvh_dev, nodes, pointer(this->contacts), this->n_verts);
  CUDA_CHECK_LAST();
}

__global__ void SolveVFCCD_Kernel(BasicDeviceBVH bvh_dev, Node** nodes, RTContact* contacts,
                                  uint num_verts) {
  GET_CUDA_ID(idx, num_verts);

  const RTContact& contact = contacts[idx];

  if (contact.index == -1)
    return;

  const Primitive& primitive = bvh_dev.objects[contact.index];
  Vector3 nc = MathFunctions::FaceNormal(primitive.pred1, primitive.pred2, primitive.pred3);

  Scalar lambda = glm::dot(nc, nodes[idx]->x - contact.qc);

  if (lambda < 0) {
    Vector3 corr = -lambda * nc;
    for (int j = 0; j < 3; j++)
      atomicAdd(&nodes[idx]->x[j], corr[j]);
  }
}

void RTConstraint::Solve(Node** nodes, std::shared_ptr<BVH> bvh) {
  XRTailor::ScopedTimerGPU timer("Solver_SolveVFCCD");
  auto bvh_dev = bvh->GetDeviceRepr();
  if (!bvh->IsActive())
    return;

  CUDA_CALL(SolveVFCCD_Kernel, this->n_verts)
  (bvh_dev, nodes, pointer(this->contacts), this->n_verts);
  CUDA_CHECK_LAST();
}

}  // namespace CCD

}  // namespace PBDCollision
}  // namespace XRTailor