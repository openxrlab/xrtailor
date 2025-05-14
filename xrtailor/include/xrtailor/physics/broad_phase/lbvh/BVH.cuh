#pragma once

#include <iostream>
#include <vector>
#include <memory>

#include <vector_types.h>
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <xrtailor/core/DeviceHelper.cuh>
#include <xrtailor/physics/broad_phase/lbvh/BVHHelper.cuh>
#include <xrtailor/core/Common.cuh>
#include <xrtailor/memory/Node.cuh>
#include <xrtailor/memory/Face.cuh>
#include <xrtailor/memory/Edge.cuh>
#include <xrtailor/memory/RenderableVertex.cuh>
#include <xrtailor/memory/MemoryPool.cuh>
#include <xrtailor/physics/broad_phase/Primitive.cuh>
#include <xrtailor/physics/broad_phase/Bounds.cuh>
#include <xrtailor/memory/Pair.cuh>

namespace XRTailor {

struct BVNode {
  std::uint32_t parent_idx;  // parent node
  std::uint32_t left_idx;    // index of left  child node
  std::uint32_t right_idx;   // index of right child node
  std::uint32_t object_idx;  // == 0xFFFFFFFF if internal node.
};

// a set of pointers to use it on device.
struct BasicDeviceBVH {
  uint num_nodes;    // (# of internal node) + (# of leaves), 2N+1
  uint num_objects;  // (# of leaves), the same as the number of objects

  BVNode* nodes;
  Bounds* aabbs;
  Primitive* objects;

  int* i_escape_index_table;  // stores escape indices for internal nodes
  int* e_escape_index_table;  // stores escape indices for external nodes
};

// in 'cbvh', 'c' is for const
struct BasicDeviceCBVH {
  uint num_nodes;    // (# of internal node) + (# of leaves), 2N+1
  uint num_objects;  // (# of leaves), the same as the number of objects

  BVNode const* nodes;
  Bounds const* aabbs;
  Primitive const* objects;

  int const* i_escape_index_table;
  int const* e_escape_index_table;
};

struct DefaultMortonCodeCalculator {
  __host__ __device__ DefaultMortonCodeCalculator(Bounds w);
  DefaultMortonCodeCalculator() = default;
  ~DefaultMortonCodeCalculator() = default;
  DefaultMortonCodeCalculator(DefaultMortonCodeCalculator const&) = default;
  DefaultMortonCodeCalculator(DefaultMortonCodeCalculator&&) = default;
  DefaultMortonCodeCalculator& operator=(DefaultMortonCodeCalculator const&) = default;
  DefaultMortonCodeCalculator& operator=(DefaultMortonCodeCalculator&&) = default;

  __host__ __device__ uint operator()(const Primitive&, const Bounds& box);

  Bounds whole;  // root AABB
};

using BVHDevice = BasicDeviceBVH;
using CBVHDevice = BasicDeviceCBVH;

class BVH {
 public:
  BVH();
  ~BVH();
  BVH(const BVH&) = default;
  BVH(BVH&&) = default;
  BVH& operator=(const BVH&) = default;
  BVH& operator=(BVH&&) = default;

  BVHDevice GetDeviceRepr();

  CBVHDevice GetDeviceRepr() const;

  void UpdateData(const Face* const* faces, Scalar torlence, bool ccd, int n_faces);

  void Construct(Scalar scr = 0);

  void RefitMidstep(const Face* const* faces, Scalar torlence);

  void Update(Face** faces, bool ccd);

  Bounds GetRootAABB();

  thrust::host_vector<Primitive> ObjectsHost() const;
  thrust::host_vector<BVNode> NodesHost() const;
  thrust::host_vector<int> InternalEscapeIndexTableHost() const;
  thrust::host_vector<int> ExternalEscapeIndexTableHost() const;
  thrust::host_vector<Bounds> AABBsHost() const;
  thrust::host_vector<Bounds> DefaultAABBsHost() const;

  uint NumInternalNodes() const;
  uint NumNodes() const;
  uint NumObjects() const;

  bool IsActive() const;

 private:
  uint num_internal_nodes_;
  uint num_nodes_;

  thrust::device_vector<Primitive> objects_;  // primitives, |F|
  thrust::device_vector<Bounds> aabbs_;       // AABBs, 2*|F| - 1
  thrust::device_vector<int> i_escape_index_table_;
  thrust::device_vector<int> e_escape_index_table_;
  thrust::device_vector<Bounds> default_aabbs_;  // AABBs before sorting, |F|
  thrust::device_vector<BVNode> nodes_;          // tree nodes, 2*|F| - 1
  Bounds root_aabb_;
  bool is_active_;
};

__device__ thrust::pair<uint, Scalar> QueryDevice(const BasicDeviceBVH& bvh, const Vector3& target,
                                                  distance_calculator calc_dist, Scalar& nearestU,
                                                  Scalar& nearestV, Scalar& nearestW);

__device__ uint QueryDeviceStackless(const BasicDeviceBVH& bvh, const Vector3& p, uint* outiter);

__device__ uint QueryDeviceStackless(const BasicDeviceBVH& bvh, const Bounds& query_aabb,
                                     uint* overlaps, const uint& primitive_idx);

__device__ uint QueryDeviceStackless(const BasicDeviceBVH& bvh, const Vector3& p0,
                                     const Vector3& p1, uint* outiter);

thrust::device_vector<PairFF> Traverse(std::shared_ptr<BVH> bvh_lhs, std::shared_ptr<BVH> bvh_rhs,
                                       Face** faces_lhs, Face** faces_rhs, const Scalar& thickness);

thrust::device_vector<PairFF> Traverse(BVH* bvh_lhs, BVH* bvh_rhs, Face** faces_lhs,
                                       Face** faces_rhs, const Scalar& thickness);

thrust::device_vector<PairFF> TraverseStack(std::shared_ptr<BVH> bvh_lhs,
                                            std::shared_ptr<BVH> bvh_rhs, Face** faces_lhs,
                                            Face** faces_rhs, const Scalar& thickness);

thrust::device_vector<PairFF> TraverseStack(BVH* bvh_lhs, BVH* bvh_rhs, Face** faces_lhs,
                                            Face** faces_rhs, const Scalar& thickness);

thrust::device_vector<Pairii> Traverse(std::shared_ptr<BVH> bvh, const Scalar& thickness);

}  // namespace XRTailor
