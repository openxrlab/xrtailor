#include <xrtailor/physics/predictive_contact/Solver.cuh>

#include "thrust/for_each.h"
#include "thrust/device_ptr.h"
#include "thrust/remove.h"
#include <thrust/execution_policy.h>

#include <xrtailor/utils/Timer.hpp>
#include <xrtailor/math/BasicPrimitiveTests.cuh>
#include <xrtailor/physics/broad_phase/lbvh/BVH.cuh>

#define DELTA_MAX_LENGTH static_cast<Scalar>(0.01)

namespace XRTailor {
namespace PredictiveContact {

__device__ __constant__ SimParams d_params;

void SetSelfCollisionParams(SimParams* host_params) {
  XRTailor::ScopedTimerGPU timer("Solver_SetParams");
  checkCudaErrors(cudaMemcpyToSymbolAsync(d_params, host_params, sizeof(SimParams)));
}

VVConstraint::VVConstraint() {
  this->contacts.resize(10000u, VVContact());
  this->n_contacts.push_back(0u);
  this->insertion_idx.push_back(0u);
}

VVConstraint::~VVConstraint() {}

__global__ void GenerateVV_Kernel(BasicDeviceBVH bvh_dev, uint* r_tris, uint* overlaps,
                                  uint* overlap_nums, VVContact* contacts, uint* n_contacts,
                                  uint* insertion_idx, uint num_internal_nodes, uint num_objects) {
  GET_CUDA_ID(idx, num_objects);
  auto node_idx = idx + num_internal_nodes;
  auto object_idx = bvh_dev.nodes[node_idx].object_idx;
  auto query_aabb = bvh_dev.aabbs[node_idx];
  auto query_primitive = bvh_dev.objects[object_idx];
  auto query_info = r_tris[object_idx];
  Bounds aabb(query_primitive);

  VVContact local_contacts[512];
  uint n_local_contacts = 0u;

  Vector3 src_pos[3] = {query_primitive.v1, query_primitive.v2, query_primitive.v3};
  Vector3 src_pred[3] = {query_primitive.pred1, query_primitive.pred2, query_primitive.pred3};
  uint src_indices[3] = {query_primitive.idx1, query_primitive.idx2, query_primitive.idx3};
  auto num_found = overlap_nums[object_idx];

  for (uint src_idx = 0u; src_idx < 3u; src_idx++) {
    if (!RTriVertex(query_info, src_idx))
      continue;

    Vector3 p_start = src_pos[src_idx];
    Vector3 p_end = src_pred[src_idx];

    for (uint i = 0u; i < num_found; i++) {
      uint f_idx = overlaps[object_idx * LBVH_MAX_BUFFER_SIZE + i];

      if (f_idx == object_idx)
        continue;

      auto tgt_primitive = bvh_dev.objects[f_idx];
      auto tgt_info = r_tris[f_idx];

      Vector3 tgt_pos[3] = {tgt_primitive.v1, tgt_primitive.v2, tgt_primitive.v3};
      Vector3 tgt_pred[3] = {tgt_primitive.pred1, tgt_primitive.pred2, tgt_primitive.pred3};
      uint tgt_indices[3] = {tgt_primitive.idx1, tgt_primitive.idx2, tgt_primitive.idx3};

      for (uint tgt_idx = 0u; tgt_idx < 3u; tgt_idx++) {
        if (!RTriVertex(tgt_info, tgt_idx))
          continue;

        Vector3 q_start = tgt_pos[tgt_idx];
        Vector3 q_end = tgt_pred[tgt_idx];

        Scalar s, t = static_cast<Scalar>(0.0);
        Vector3 c1, c2;
        Scalar d = BasicPrimitiveTests::ClosestPtSegmentSegment(p_start, p_end, q_start, q_end, s,
                                                                t, c1, c2);

        Vector3 n = glm::normalize(c2 - c1);

        if (d >= d_params.scr + 2 * d_params.radius)
          continue;

        VVContact contact = {n, src_indices[src_idx], tgt_indices[tgt_idx]};
        local_contacts[n_local_contacts++] = contact;
      }
    }
  }

  atomicAdd(&*n_contacts, n_local_contacts);
  uint start_idx = atomicAdd(&*insertion_idx, n_local_contacts);
  for (uint i = 0u; i < n_local_contacts; i++) {
    contacts[start_idx + i] = local_contacts[i];
  }
}

void VVConstraint::GenerateStackless(std::shared_ptr<BVH> bvh, std::shared_ptr<RTriangle> r_tri,
                                     uint* overlaps, uint* overlap_nums) {
  auto bvh_dev = bvh->GetDeviceRepr();
  if (!bvh->IsActive())
    return;

  auto num_internal_nodes = bvh->NumInternalNodes();
  auto num_objects = bvh->NumObjects();

  uint h_n_contacts = 0u, h_insertion_idx = 0u;

  n_contacts.resize(1, 0);
  insertion_idx.resize(1, 0);

  CUDA_CALL(GenerateVV_Kernel, num_objects)
  (bvh_dev, pointer(r_tri->r_tris), overlaps, overlap_nums, pointer(contacts), pointer(n_contacts),
   pointer(insertion_idx), num_internal_nodes, num_objects);

  cudaError_t kernelError = cudaGetLastError();
  if (kernelError != cudaSuccess) {
    fprintf(stderr, "CUDA Error after VVonstraint::GenerateStackless kernel launch: %s\n",
            cudaGetErrorString(kernelError));
  }

  cudaError_t syncError = cudaDeviceSynchronize();
  if (syncError != cudaSuccess) {
    fprintf(stderr, "CUDA Error after VVConstraint::GenerateStackless cudaDeviceSynchronize: %s\n",
            cudaGetErrorString(syncError));
  }
}

__global__ void SolveVV_Kernel(const Vector3* positions, Vector3* predicted,
                               const Scalar* inv_masses, Vector3* deltas, int* delta_counts,
                               VVContact* contacts, uint num_contacts) {
  GET_CUDA_ID(idx, num_contacts);

  auto contact = contacts[idx];
  uint idx0 = contact.p0;
  uint idx1 = contact.p1;
  Vector3 p0 = predicted[idx0];
  Vector3 p1 = predicted[idx1];
  Scalar w0 = inv_masses[idx0];
  Scalar w1 = inv_masses[idx1];

  Vector3 n = contact.normal;

  Scalar proj = fabs(glm::dot(n, p0 - p1));
  Scalar constraint = proj - static_cast<Scalar>(2.0) * static_cast<Scalar>(d_params.radius);

  if (constraint >= static_cast<Scalar>(0.0))
    return;

  if (w0 <= EPSILON && w1 <= EPSILON)
    return;

  Scalar delta_lambda = -constraint / (w0 + w1 + EPSILON);

  Vector3 grad_p0 = n;
  Vector3 grad_p1 = -n;

  Vector3 delta_p0 = delta_lambda * grad_p0 * w0;
  Vector3 delta_p1 = delta_lambda * grad_p1 * w1;

  int reorder = idx0 + idx1;

  AtomicAdd(deltas, idx0, -delta_p0, reorder);
  AtomicAdd(deltas, idx1, -delta_p1, reorder);

  atomicAdd(&delta_counts[idx0], 1);
  atomicAdd(&delta_counts[idx1], 1);
}

void VVConstraint::Solve(const Vector3* positions, Vector3* predicted, const Scalar* inv_masses,
                         Vector3* deltas, int* delta_counts, Scalar radius) {
  uint h_n_contacts = 0u;
  checkCudaErrors(
      cudaMemcpy(&h_n_contacts, pointer(this->n_contacts), sizeof(uint), cudaMemcpyDeviceToHost));

  CUDA_CALL(SolveVV_Kernel, h_n_contacts)
  (positions, predicted, inv_masses, deltas, delta_counts, pointer(contacts), h_n_contacts);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    fprintf(stderr, "CUDA Error after SolveVV_Kernel: %s\n", cudaGetErrorString(error));
  }
}

VFConstraint::VFConstraint() {
  this->contacts.resize(10000u, VFContact());
  this->n_contacts.push_back(0u);
  this->insertion_idx.push_back(0u);
}

VFConstraint::~VFConstraint() {}

__global__ void GenerateVF_Kernel(BasicDeviceBVH bvh_dev, uint* r_tris, uint* overlaps,
                                  uint* overlap_nums, uint num_internal_nodes, uint num_objects,
                                  VFContact* contacts, uint* n_contacts, uint* insertion_idx) {
  GET_CUDA_ID(idx, num_objects);
  auto node_idx = idx + num_internal_nodes;

  auto object_idx = bvh_dev.nodes[node_idx].object_idx;
  auto query_aabb = bvh_dev.aabbs[node_idx];
  auto query_primitive = bvh_dev.objects[object_idx];
  auto query_info = r_tris[object_idx];
  Bounds aabb(query_primitive);

  VFContact local_contacts[LBVH_MAX_BUFFER_SIZE * 3];
  uint n_local_contacts = 0u;

  Vector3 src_pos[3] = {query_primitive.v1, query_primitive.v2, query_primitive.v3};
  Vector3 src_pred[3] = {query_primitive.pred1, query_primitive.pred2, query_primitive.pred3};
  uint src_indices[3] = {query_primitive.idx1, query_primitive.idx2, query_primitive.idx3};

  for (uint src_idx = 0u; src_idx < 3u; src_idx++) {
    if (!RTriVertex(query_info, src_idx))
      continue;

    Vector3 p_start = src_pos[src_idx];
    Vector3 p_end = src_pred[src_idx];
    Vector3 d = p_end - p_start;

    Scalar _min_dist = SCALAR_MAX;
    int _point_idx = -1;
    int _triangle_idx = 0;
    int _is_back_face = 0;

    for (uint i = 0u; i < overlap_nums[object_idx]; i++) {
      uint f_idx = overlaps[object_idx * LBVH_MAX_BUFFER_SIZE + i];

      auto tgt_primitive = bvh_dev.objects[f_idx];
      auto tgt_info = r_tris[f_idx];

      Vector3 tgt_pos[3] = {tgt_primitive.v1, tgt_primitive.v2, tgt_primitive.v3};
      Vector3 tgt_pred[3] = {tgt_primitive.pred1, tgt_primitive.pred2, tgt_primitive.pred3};
      uint tgt_indices[3] = {tgt_primitive.idx1, tgt_primitive.idx2, tgt_primitive.idx3};

      if (src_indices[src_idx] == tgt_indices[0] || src_indices[src_idx] == tgt_indices[1] ||
          src_indices[src_idx] == tgt_indices[2])
        continue;

      Vector3 pc;
      Scalar u, v, w;
      BasicPrimitiveTests::NearestPoint(p_start, tgt_pos[0], tgt_pos[1], tgt_pos[2], pc, u, v, w);
      Vector3 n = glm::normalize(pc - p_start);

      Vector3 d0 = tgt_pred[0] - tgt_pos[0];
      Vector3 d1 = tgt_pred[1] - tgt_pos[1];
      Vector3 d2 = tgt_pred[2] - tgt_pos[2];
      Vector3 dc = u * d0 + v * d1 + w * d2;

      Scalar proj1 = glm::dot(d, n);
      Scalar proj2 = glm::dot(dc, n);

      Scalar min_dist =
          BasicPrimitiveTests::MinDist(p_start, p_start + proj1 * n, pc, pc + proj2 * n);

      if (min_dist > d_params.scr + static_cast<Scalar>(2.0) * d_params.radius)
        continue;

      Vector3 fn = MathFunctions::FaceNormal(tgt_pos[0], tgt_pos[1], tgt_pos[2]);

      uint is_back_face = (glm::dot(n, fn) > 0) ? 1 : 0;

      if (min_dist < _min_dist) {
        _min_dist = min_dist;
        _point_idx = src_indices[src_idx];
        _triangle_idx = f_idx;
        _is_back_face = is_back_face;
      }
    }
    if (_point_idx != -1) {
      local_contacts[n_local_contacts++] = VFContact{_point_idx, _triangle_idx, _is_back_face};
    }
  }

  atomicAdd(&*n_contacts, n_local_contacts);
  uint start_idx = atomicAdd(&*insertion_idx, n_local_contacts);
  for (uint i = 0u; i < n_local_contacts; i++) {
    contacts[start_idx + i] = local_contacts[i];
  }
}

void VFConstraint::GenerateStackless(std::shared_ptr<BVH> bvh, std::shared_ptr<RTriangle> r_tri,
                                     uint* overlaps, uint* overlapNums) {
  auto bvh_dev = bvh->GetDeviceRepr();
  if (!bvh->IsActive())
    return;

  auto num_internal_nodes = bvh->NumInternalNodes();
  auto num_objects = bvh->NumObjects();

  uint h_n_contacts = 0u, h_insertion_idx = 0u;

  n_contacts.resize(1, 0);
  insertion_idx.resize(1, 0);

  CUDA_CALL(GenerateVF_Kernel, num_objects)
  (bvh_dev, pointer(r_tri->r_tris), overlaps, overlapNums, num_internal_nodes, num_objects,
   pointer(contacts), pointer(n_contacts), pointer(insertion_idx));

  cudaError_t kernelError = cudaGetLastError();
  if (kernelError != cudaSuccess) {
    fprintf(stderr, "CUDA Error after VFonstraint::GenerateStackless kernel launch: %s\n",
            cudaGetErrorString(kernelError));
  }

  cudaError_t syncError = cudaDeviceSynchronize();
  if (syncError != cudaSuccess) {
    fprintf(stderr, "CUDA Error after VFConstraint::GenerateStackless cudaDeviceSynchronize: %s\n",
            cudaGetErrorString(syncError));
  }
}

__global__ void SolveVF_Kernel(const Vector3* positions, Vector3* predicted, uint* indices,
                               const Scalar* invMasses, Vector3* deltas, int* delta_counts,
                               VFContact* contacts, uint num_contacts) {
  GET_CUDA_ID(idx, num_contacts);

  auto contact = contacts[idx];
  uint point_idx = contact.point_index;
  uint face_idx = contact.triangle_index;
  uint is_back_face = contact.is_back_face;

  uint idx0 = indices[face_idx * 3u];
  uint idx1 = indices[face_idx * 3u + 1u];
  uint idx2 = indices[face_idx * 3u + 2u];

  uint w = invMasses[point_idx];
  uint w0 = invMasses[idx0];
  uint w1 = invMasses[idx1];
  uint w2 = invMasses[idx2];

  Vector3 p = predicted[point_idx];
  Vector3 p0 = predicted[idx0];
  Vector3 p1 = predicted[idx1];
  Vector3 p2 = predicted[idx2];

  // https://github.com/vasumahesh1/azura/blob/master/Source/Samples/3_ClothSim/Shaders/SolvingPass_Cloth_ApplyConstraints.cs.slang
  Vector3 n = glm::cross(p1 - p0, p2 - p0);
  Vector3 n_hat =
      (is_back_face ? static_cast<Scalar>(-1.0) : static_cast<Scalar>(1.0)) * glm::normalize(n);
  Scalar n_mag = glm::length(n);

  const Scalar a = n_hat.x;
  const Scalar b = n_hat.y;
  const Scalar c = n_hat.z;

  const Scalar a2 = n_hat.x * n_hat.x;
  const Scalar b2 = n_hat.y * n_hat.y;
  const Scalar c2 = n_hat.z * n_hat.z;

  Vector3 n_vec = Vector3(a - (a2 * a) - (a * b2) - (a * c2), (-a2 * b) + b - (b2 * b) - (b * c2),
                          (-a2 * c) - (b2 * c) + c - (c2 * c));

  n_vec = n_vec / n_mag;

  //const glm::vec3 grad_p0 = glm::cross(p2 - p0, n_vec) - n_hat;
  //const glm::vec3 grad_p1 = glm::cross(p1 - p0, n_vec) - n_hat;
  //const glm::vec3 grad_p2 = glm::cross(p1 - p2, n_vec) - n_hat;
  const Vector3 grad_p1 = glm::cross(p2 - p0, n_vec);
  const Vector3 grad_p2 = glm::cross(p1 - p0, n_vec);
  const Vector3 grad_p0 = glm::cross(p1 - p2, n_vec) - n_hat;
  const Vector3 grad_p = n_hat;

  Scalar D = glm::dot(n_hat, p - p0);
  Scalar constraint = D - (static_cast<Scalar>(2.0) * d_params.radius);

  if (constraint > static_cast<Scalar>(0.0))
    return;

  Scalar denom = w * glm::dot(n_hat, n_hat) + w0 * glm::dot(grad_p0, grad_p0) +
                 w1 * glm::dot(grad_p1, grad_p1) + w2 * glm::dot(grad_p2, grad_p2);

  if (fabs(denom) > EPSILON) {
    const Scalar lambda = -constraint / denom;

    Vector3 dp = lambda * w * grad_p;
    Vector3 dp1 = lambda * w1 * grad_p1;
    Vector3 dp2 = lambda * w2 * grad_p2;
    Vector3 dp0 = lambda * w0 * grad_p0;

    int reorder = point_idx + idx0 + idx1 + idx2;
    AtomicAdd(deltas, point_idx, dp, reorder);
    AtomicAdd(deltas, idx0, dp0, reorder);
    AtomicAdd(deltas, idx1, dp1, reorder);
    AtomicAdd(deltas, idx2, dp2, reorder);

    atomicAdd(&delta_counts[point_idx], 1);
    atomicAdd(&delta_counts[idx0], 1);
    atomicAdd(&delta_counts[idx1], 1);
    atomicAdd(&delta_counts[idx2], 1);
  }
}

void VFConstraint::Solve(const Vector3* positions, Vector3* predicted, uint* indices,
                         const Scalar* inv_masses, Vector3* deltas, int* delta_counts,
                         Scalar radius) {
  uint h_n_contacts = 0u;
  checkCudaErrors(
      cudaMemcpy(&h_n_contacts, pointer(n_contacts), sizeof(uint), cudaMemcpyDeviceToHost));
  CUDA_CALL(SolveVF_Kernel, h_n_contacts)
  (positions, predicted, indices, inv_masses, deltas, delta_counts, pointer(contacts),
   h_n_contacts);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    fprintf(stderr, "CUDA Error after SolveVF_Kernel: %s\n", cudaGetErrorString(error));
  }
}

EEConstraint::EEConstraint() {
  this->contacts.resize(100000u, EEContact());
  this->n_contacts.push_back(0u);
  this->insertion_idx.push_back(0u);
}

EEConstraint::~EEConstraint() {}

__global__ void GenerateEE_Kernel(BasicDeviceBVH bvh_dev, uint* r_tris, uint* overlaps,
                                  uint* overlap_nums, EEContact* contacts, uint* n_contacts,
                                  uint* insertion_idx, uint num_internal_nodes, uint num_objects) {
  GET_CUDA_ID(idx, num_objects);
  auto node_idx = idx + num_internal_nodes;
  auto object_idx = bvh_dev.nodes[node_idx].object_idx;
  auto query_aabb = bvh_dev.aabbs[node_idx];
  auto query_primitive = bvh_dev.objects[object_idx];
  auto query_info = r_tris[object_idx];

  EEContact local_contacts[LBVH_MAX_BUFFER_SIZE * 9];
  int n_local_contacts = 0u;

  Vector3 src_pos[3] = {query_primitive.v1, query_primitive.v2, query_primitive.v3};
  Vector3 src_pred[3] = {query_primitive.pred1, query_primitive.pred2, query_primitive.pred3};
  int src_indices[3] = {query_primitive.idx1, query_primitive.idx2, query_primitive.idx3};

  for (int src_idx = 0u; src_idx < 3u; src_idx++) {
    if (!RTriEdge(query_info, src_idx))
      continue;
    int src_idx0 = src_idx;
    int src_idx1 = (src_idx + 1u) % 3u;
    int e0idx0 = src_indices[src_idx0];
    int e0idx1 = src_indices[src_idx1];

    Vector3 e0p0 = src_pos[src_idx0];
    Vector3 e0p1 = src_pos[src_idx1];

    Vector3 d_e0p0 = src_pred[src_idx0] - src_pos[src_idx0];
    Vector3 d_e0p1 = src_pred[src_idx1] - src_pos[src_idx1];

    for (uint i = 0u; i < overlap_nums[object_idx]; i++) {
      int f_idx = overlaps[object_idx * LBVH_MAX_BUFFER_SIZE + i];

      if (f_idx == object_idx)
        continue;

      auto tgt_primitive = bvh_dev.objects[f_idx];
      auto tgt_info = r_tris[f_idx];

      Vector3 tgt_pos[3] = {tgt_primitive.v1, tgt_primitive.v2, tgt_primitive.v3};
      Vector3 tgt_pred[3] = {tgt_primitive.pred1, tgt_primitive.pred2, tgt_primitive.pred3};
      int tgt_indices[3] = {tgt_primitive.idx1, tgt_primitive.idx2, tgt_primitive.idx3};

      for (int tgt_idx = 0u; tgt_idx < 3u; tgt_idx++) {
        if (!RTriEdge(tgt_info, tgt_idx))
          continue;
        int tgt_idx0 = tgt_idx;
        int tgt_idx1 = (tgt_idx + 1u) % 3u;
        Vector3 e1p0 = tgt_pos[tgt_idx0];
        Vector3 e1p1 = tgt_pos[tgt_idx1];

        int e1idx0 = tgt_indices[tgt_idx0];
        int e1idx1 = tgt_indices[tgt_idx1];

        if (e0idx0 == e1idx0 || e0idx0 == e1idx1 || e0idx1 == e1idx0 || e0idx1 == e1idx1)
          continue;

        Scalar alpha, beta;
        Vector3 c1, c2;
        Scalar dist = BasicPrimitiveTests::ClosestPtSegmentSegment(e0p0, e0p1, e1p0, e1p1, alpha,
                                                                   beta, c1, c2);

        Vector3 d_e1p0 = tgt_pred[tgt_idx0] - tgt_pos[tgt_idx0];
        Vector3 d_e1p1 = tgt_pred[tgt_idx1] - tgt_pos[tgt_idx1];

        Vector3 d_alpha = (1.0f - alpha) * d_e1p0 + alpha * d_e1p1;
        Vector3 d_beta = (1.0f - beta) * d_e1p0 + beta * d_e1p1;

        // project the lerped displacements onto the discrete separation vector
        Vector3 n = glm::normalize(c2 - c1);
        Vector3 c1_end = c1 + glm::dot(d_alpha, n) * n;
        Vector3 c2_end = c2 + glm::dot(d_beta, n) * n;

        Scalar s, t;
        Vector3 q1, q2;

        Scalar min_dist = BasicPrimitiveTests::MinDist(c1, c1_end, c2, c2_end);
        if (min_dist >= d_params.scr + static_cast<Scalar>(2.0) * d_params.radius)
          continue;

        local_contacts[n_local_contacts++] = {n, alpha, beta, e0idx0, e0idx1, e1idx0, e1idx1};
      }
    }
  }

  atomicAdd(&*n_contacts, n_local_contacts);
  uint start_idx = atomicAdd(&*insertion_idx, n_local_contacts);
  for (uint i = 0u; i < n_local_contacts; i++) {
    contacts[start_idx + i] = local_contacts[i];
  }
}

void EEConstraint::GenerateStackless(std::shared_ptr<BVH> bvh, std::shared_ptr<RTriangle> r_tri,
                                     uint* overlaps, uint* overlap_nums) {
  auto num_internal_nodes = bvh->NumInternalNodes();
  auto num_objects = bvh->NumObjects();

  auto bvh_dev = bvh->GetDeviceRepr();
  if (!bvh->IsActive())
    return;

  uint h_n_contacts = 0u, h_insertion_idx = 0u;

  checkCudaErrors(
      cudaMemcpy(pointer(n_contacts), &h_n_contacts, sizeof(uint), cudaMemcpyHostToDevice));
  checkCudaErrors(
      cudaMemcpy(pointer(insertion_idx), &h_insertion_idx, sizeof(uint), cudaMemcpyHostToDevice));

  CUDA_CALL(GenerateEE_Kernel, num_objects)
  (bvh_dev, pointer(r_tri->r_tris), overlaps, overlap_nums, pointer(contacts), pointer(n_contacts),
   pointer(insertion_idx), num_internal_nodes, num_objects);

  cudaError_t kernelError = cudaGetLastError();
  if (kernelError != cudaSuccess) {
    fprintf(stderr, "CUDA Error after EEConstraint::GenerateStackless kernel launch: %s\n",
            cudaGetErrorString(kernelError));
  }

  cudaError_t syncError = cudaDeviceSynchronize();
  if (syncError != cudaSuccess) {
    fprintf(stderr, "CUDA Error after EEConstraint::GenerateStackless cudaDeviceSynchronize: %s\n",
            cudaGetErrorString(syncError));
  }
}

__global__ void SolveEE_Kernel(const Vector3* positions, Vector3* predicted, uint* indices,
                               const Scalar* inv_masses, Vector3* deltas, int* delta_counts,
                               EEContact* contacts, uint num_contacts) {
  GET_CUDA_ID(idx, num_contacts);

  auto contact = contacts[idx];
  uint idxa = contact.e0p0;
  uint idxb = contact.e0p1;
  uint idxc = contact.e1p0;
  uint idxd = contact.e1p1;
  Vector3 n = contact.normal;
  Scalar s = contact.s;
  Scalar t = contact.t;
  Scalar wa = inv_masses[idxa];
  Scalar wb = inv_masses[idxb];
  Scalar wc = inv_masses[idxc];
  Scalar wd = inv_masses[idxd];

  if (wa <= EPSILON || wb <= EPSILON || wc <= EPSILON || wd <= EPSILON)
    return;

  Vector3 pa = predicted[idxa];
  Vector3 pb = predicted[idxb];
  Vector3 pc = predicted[idxc];
  Vector3 pd = predicted[idxd];

  Vector3 p0 = (static_cast<Scalar>(1) - s) * pa + s * pb;
  Vector3 p1 = (static_cast<Scalar>(1) - t) * pc + t * pd;
  Scalar w0 = (static_cast<Scalar>(1) - s) * wa + s * wb;
  Scalar w1 = (static_cast<Scalar>(1) - s) * wc + s * wd;

  Scalar constraint = glm::dot(n, p1 - p0) - static_cast<Scalar>(2) * d_params.radius;

  if (constraint >= static_cast<Scalar>(0))
    return;

  Vector3 grad_pa = -(static_cast<Scalar>(1) - s) * n;
  Vector3 grad_pb = -s * n;
  Vector3 grad_pc = (static_cast<Scalar>(1) - t) * n;
  Vector3 grad_pd = t * n;

  Scalar denom = wa * (static_cast<Scalar>(1) - s) * (static_cast<Scalar>(1) - s) + wb * s * s +
                 wc * (static_cast<Scalar>(1) - t) * (static_cast<Scalar>(1) - t) + wd * t * t;

  if (fabs(denom) > EPSILON) {
    Scalar delta_lambda = -constraint / denom;

    Vector3 dpa = delta_lambda * wa * grad_pa;
    Vector3 dpb = delta_lambda * wb * grad_pb;
    Vector3 dpc = delta_lambda * wc * grad_pc;
    Vector3 dpd = delta_lambda * wd * grad_pd;

    int reorder = idxa + idxb + idxc + idxd;

    AtomicAdd(deltas, idxa, dpa, reorder);
    AtomicAdd(deltas, idxb, dpb, reorder);
    AtomicAdd(deltas, idxc, dpc, reorder);
    AtomicAdd(deltas, idxd, dpd, reorder);

    atomicAdd(&delta_counts[idxa], 1u);
    atomicAdd(&delta_counts[idxb], 1u);
    atomicAdd(&delta_counts[idxc], 1u);
    atomicAdd(&delta_counts[idxd], 1u);
  }
}

void EEConstraint::Solve(const Vector3* positions, Vector3* predicted, uint* indices,
                         const Scalar* invMasses, Vector3* deltas, int* delta_counts,
                         Scalar radius) {
  thrust::host_vector<uint> h_n_contacts = n_contacts;

  CUDA_CALL(SolveEE_Kernel, h_n_contacts[0])
  (positions, predicted, indices, invMasses, deltas, delta_counts, pointer(contacts),
   h_n_contacts[0]);
}

}  // namespace PredictiveContact
}  // namespace XRTailor
