#include <xrtailor/physics/broad_phase/lbvh/BVH.cuh>

#include <xrtailor/utils/Timer.hpp>

#include <thrust/execution_policy.h>
#include <thrust/pair.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/unique.h>
#include <thrust/system_error.h>
#include <thrust/for_each.h>

#include <xrtailor/math/BasicPrimitiveTests.cuh>

//#define BVH_DEBUG

namespace XRTailor {
// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
__host__ __device__ inline std::uint32_t expand_bits(std::uint32_t v) noexcept {
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000011u) & 0xC30C30C3u;
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
//__host__ __device__ inline std::uint32_t morton_code(glm::vec3 xyz, float resolution = 1024.0f) noexcept
__host__ __device__ inline std::uint32_t morton_code(Vector3 xyz,
                                                     Scalar resolution = 1024) noexcept {
  xyz.x = MathFunctions::min(MathFunctions::max(xyz.x * resolution, static_cast<Scalar>(0.0)),
                             resolution - static_cast<Scalar>(1.0));
  xyz.y = MathFunctions::min(MathFunctions::max(xyz.y * resolution, static_cast<Scalar>(0.0)),
                             resolution - static_cast<Scalar>(1.0));
  xyz.z = MathFunctions::min(MathFunctions::max(xyz.z * resolution, static_cast<Scalar>(0.0)),
                             resolution - static_cast<Scalar>(1.0));

  const std::uint32_t xx = expand_bits(static_cast<std::uint32_t>(xyz.x));
  const std::uint32_t yy = expand_bits(static_cast<std::uint32_t>(xyz.y));
  const std::uint32_t zz = expand_bits(static_cast<std::uint32_t>(xyz.z));
  return xx * 4 + yy * 2 + zz;
}

__device__ inline int common_upper_bits(const unsigned int lhs, const unsigned int rhs) noexcept {
  return ::__clz(lhs ^ rhs);
}

__device__ inline int common_upper_bits(const unsigned long long int lhs,
                                        const unsigned long long int rhs) noexcept {
  return ::__clzll(lhs ^ rhs);
}

__device__ inline uint2 determine_range(const unsigned int* node_code,
                                        const unsigned int num_leaves, unsigned int idx) {
  if (idx == 0) {
    return make_uint2(0, num_leaves - 1);
  }

  // determine direction of the range
  const unsigned int self_code = node_code[idx];
  const int L_delta = common_upper_bits(self_code, node_code[idx - 1]);
  const int R_delta = common_upper_bits(self_code, node_code[idx + 1]);
  const int d = (R_delta > L_delta) ? 1 : -1;

  // Compute upper bound for the length of the range
  const int delta_min = thrust::min(L_delta, R_delta);
  int l_max = 2;
  int delta = -1;
  int i_tmp = idx + d * l_max;
  if (0 <= i_tmp && i_tmp < num_leaves) {
    delta = common_upper_bits(self_code, node_code[i_tmp]);
  }
  while (delta > delta_min) {
    l_max <<= 1;
    i_tmp = idx + d * l_max;
    delta = -1;
    if (0 <= i_tmp && i_tmp < num_leaves) {
      delta = common_upper_bits(self_code, node_code[i_tmp]);
    }
  }

  // Find the other end by binary search
  int l = 0;
  int t = l_max >> 1;
  while (t > 0) {
    i_tmp = idx + (l + t) * d;
    delta = -1;
    if (0 <= i_tmp && i_tmp < num_leaves) {
      delta = common_upper_bits(self_code, node_code[i_tmp]);
    }
    if (delta > delta_min) {
      l += t;
    }
    t >>= 1;
  }
  unsigned int jdx = idx + l * d;
  if (d < 0) {
    thrust::swap(idx, jdx);  // make it sure that idx < jdx
  }
  return make_uint2(idx, jdx);
}

__device__ inline uint2 determine_range(const unsigned long long* node_code,
                                        const unsigned int num_leaves, unsigned int idx) {
  if (idx == 0) {
    return make_uint2(0, num_leaves - 1);
  }

  // determine direction of the range
  const unsigned long long int self_code = node_code[idx];
  const int L_delta = common_upper_bits(self_code, node_code[idx - 1]);
  const int R_delta = common_upper_bits(self_code, node_code[idx + 1]);
  const int d = (R_delta > L_delta) ? 1 : -1;

  // Compute upper bound for the length of the range

  const int delta_min = thrust::min(L_delta, R_delta);
  int l_max = 2;
  int delta = -1;
  int i_tmp = idx + d * l_max;
  if (0 <= i_tmp && i_tmp < num_leaves) {
    delta = common_upper_bits(self_code, node_code[i_tmp]);
  }
  while (delta > delta_min) {
    l_max <<= 1;
    i_tmp = idx + d * l_max;
    delta = -1;
    if (0 <= i_tmp && i_tmp < num_leaves) {
      delta = common_upper_bits(self_code, node_code[i_tmp]);
    }
  }

  // Find the other end by binary search
  int l = 0;
  int t = l_max >> 1;
  while (t > 0) {
    i_tmp = idx + (l + t) * d;
    delta = -1;
    if (0 <= i_tmp && i_tmp < num_leaves) {
      delta = common_upper_bits(self_code, node_code[i_tmp]);
    }
    if (delta > delta_min) {
      l += t;
    }
    t >>= 1;
  }
  unsigned int jdx = idx + l * d;
  if (d < 0) {
    thrust::swap(idx, jdx);  // make it sure that idx < jdx
  }
  return make_uint2(idx, jdx);
}

__device__ inline unsigned int find_split(const unsigned int* node_code,
                                          const unsigned int num_leaves, const unsigned int first,
                                          const unsigned int last) {
  const unsigned int first_code = node_code[first];
  const unsigned int last_code = node_code[last];
  if (first_code == last_code) {
    return (first + last) >> 1;
  }
  const int delta_node = common_upper_bits(first_code, last_code);

  // binary search...
  int split = first;
  int stride = last - first;
  do {
    stride = (stride + 1) >> 1;
    const int middle = split + stride;
    if (middle < last) {
      const int delta = common_upper_bits(first_code, node_code[middle]);
      if (delta > delta_node) {
        split = middle;
      }
    }
  } while (stride > 1);

  return split;
}

__device__ inline unsigned int find_split(const unsigned long long* node_code,
                                          const unsigned int num_leaves, const unsigned int first,
                                          const unsigned int last) {
  const unsigned long long first_code = node_code[first];
  const unsigned long long last_code = node_code[last];
  if (first_code == last_code) {
    return (first + last) >> 1;
  }
  const int delta_node = common_upper_bits(first_code, last_code);

  // binary search...
  int split = first;
  int stride = last - first;
  do {
    stride = (stride + 1) >> 1;
    const int middle = split + stride;
    if (middle < last) {
      const int delta = common_upper_bits(first_code, node_code[middle]);
      if (delta > delta_node) {
        split = middle;
      }
    }
  } while (stride > 1);

  return split;
}

__global__ void BuildInternalEscapeIndexTable_Kernel(BasicDeviceBVH bvh_dev,
                                                     uint num_internal_nodes) {
  GET_CUDA_ID(idx, num_internal_nodes);

  auto node = bvh_dev.nodes[idx];
  unsigned int parent = node.parent_idx;

  // node is root
  if (parent == 0xFFFFFFFF) {
    bvh_dev.i_escape_index_table[idx] = -1;
    return;
  }

  if (idx == bvh_dev.nodes[parent].left_idx) {
    // left child
    bvh_dev.i_escape_index_table[idx] = bvh_dev.nodes[parent].right_idx;
  } else {
    // right child
    auto curr_idx = parent;
    auto curr_node = bvh_dev.nodes[curr_idx];
    while (curr_idx != 0xFFFFFFFF) {
      if (curr_node.parent_idx == 0xFFFFFFFF) {
        bvh_dev.i_escape_index_table[idx] = -1;
        break;
      }

      if (curr_idx == bvh_dev.nodes[curr_node.parent_idx].right_idx) {
        // current node is a right child, continue traceback
        curr_idx = curr_node.parent_idx;
        curr_node = bvh_dev.nodes[curr_idx];
      } else {
        // left child
        bvh_dev.i_escape_index_table[idx] = bvh_dev.nodes[curr_node.parent_idx].right_idx;
        break;
      }
    }
  }
}

__global__ void BuildExternalEscapeIndexTable_Kernel(BasicDeviceBVH bvh_dev,
                                                     uint num_internal_nodes, uint num_objects) {
  GET_CUDA_ID(idx, num_objects);
  auto node = bvh_dev.nodes[idx + num_internal_nodes];
  unsigned int parent = node.parent_idx;

  if ((idx + num_internal_nodes) == bvh_dev.nodes[parent].left_idx) {
    // left child
    bvh_dev.e_escape_index_table[idx] = bvh_dev.nodes[parent].right_idx;
  } else {
    // right child
    bvh_dev.e_escape_index_table[idx] = bvh_dev.i_escape_index_table[parent];
  }
}

void BuildEscapeIndexTable(const BasicDeviceBVH& bvh_dev, const unsigned int num_internal_nodes,
                           const unsigned int num_objects) {
  CUDA_CALL(BuildInternalEscapeIndexTable_Kernel, num_internal_nodes)
  (bvh_dev, num_internal_nodes);

  CUDA_CALL(BuildExternalEscapeIndexTable_Kernel, num_objects)
  (bvh_dev, num_internal_nodes, num_objects);
}

__global__ void ConstructInternalNodes32Bits_Kernel(BasicDeviceBVH bvh_dev,
                                                    const unsigned int* node_code,
                                                    const uint num_objects) {
  GET_CUDA_ID(idx, num_objects - 1);

  bvh_dev.nodes[idx].object_idx = 0xFFFFFFFF;  //  internal nodes

  const uint2 ij = determine_range(node_code, num_objects, idx);
  const int gamma = find_split(node_code, num_objects, ij.x, ij.y);

  bvh_dev.nodes[idx].left_idx = gamma;
  bvh_dev.nodes[idx].right_idx = gamma + 1;
  if (thrust::min(ij.x, ij.y) == gamma) {
    bvh_dev.nodes[idx].left_idx += num_objects - 1;
  }
  if (thrust::max(ij.x, ij.y) == gamma + 1) {
    bvh_dev.nodes[idx].right_idx += num_objects - 1;
  }
  bvh_dev.nodes[bvh_dev.nodes[idx].left_idx].parent_idx = idx;
  bvh_dev.nodes[bvh_dev.nodes[idx].right_idx].parent_idx = idx;
}

__global__ void ConstructInternalNodes64Bits_Kernel(BasicDeviceBVH bvh_dev,
                                                    const unsigned long long* node_code,
                                                    const uint num_objects) {
  GET_CUDA_ID(idx, num_objects - 1);

  bvh_dev.nodes[idx].object_idx = 0xFFFFFFFF;  //  internal nodes

  const uint2 ij = determine_range(node_code, num_objects, idx);
  const int gamma = find_split(node_code, num_objects, ij.x, ij.y);

  bvh_dev.nodes[idx].left_idx = gamma;
  bvh_dev.nodes[idx].right_idx = gamma + 1;
  if (thrust::min(ij.x, ij.y) == gamma) {
    bvh_dev.nodes[idx].left_idx += num_objects - 1;
  }
  if (thrust::max(ij.x, ij.y) == gamma + 1) {
    bvh_dev.nodes[idx].right_idx += num_objects - 1;
  }
  bvh_dev.nodes[bvh_dev.nodes[idx].left_idx].parent_idx = idx;
  bvh_dev.nodes[bvh_dev.nodes[idx].right_idx].parent_idx = idx;
}

void ConstructInternalNodesCUDA(const BasicDeviceBVH& bvh_dev, const unsigned int* node_code,
                                const unsigned int num_objects) {
  CUDA_CALL(ConstructInternalNodes32Bits_Kernel, num_objects - 1)
  (bvh_dev, node_code, num_objects);
}

void ConstructInternalNodesCUDA(const BasicDeviceBVH& bvh_dev, const unsigned long long* node_code,
                                const unsigned int num_objects) {
  CUDA_CALL(ConstructInternalNodes64Bits_Kernel, num_objects - 1)
  (bvh_dev, node_code, num_objects);
}

__global__ void MapHierarchy_Kernel(BasicDeviceBVH bvh_dev, int* left_indices, int* right_indices,
                                    int* parent_indices, int* object_indices, int n_nodes) {
  GET_CUDA_ID(i, n_nodes);

  BVNode* node = &bvh_dev.nodes[i];
  left_indices[i] = node->left_idx;
  right_indices[i] = node->right_idx;
  parent_indices[i] = node->parent_idx;
  object_indices[i] = node->object_idx;
}

__host__ __device__ DefaultMortonCodeCalculator::DefaultMortonCodeCalculator(Bounds w) : whole(w) {}

__host__ __device__ unsigned int DefaultMortonCodeCalculator::operator()(const Primitive&,
                                                                         const Bounds& box) {
  Vector3 p = box.Center();
  // transform to origin
  p.x -= whole.lower.x;
  p.y -= whole.lower.y;
  p.z -= whole.lower.z;
  // scale to [0, 1]
  p.x /= (whole.upper.x - whole.lower.x);
  p.y /= (whole.upper.y - whole.lower.y);
  p.z /= (whole.upper.z - whole.lower.z);

  return morton_code(p);
}

BVH::BVH() : is_active_(false) {};

BVH::~BVH() {};

BVHDevice BVH::GetDeviceRepr() {
  return BVHDevice{static_cast<unsigned int>(nodes_.size()),
                   static_cast<unsigned int>(objects_.size()),
                   nodes_.data().get(),
                   aabbs_.data().get(),
                   objects_.data().get(),
                   i_escape_index_table_.data().get(),
                   e_escape_index_table_.data().get()};
}
CBVHDevice BVH::GetDeviceRepr() const {
  return CBVHDevice{static_cast<unsigned int>(nodes_.size()),
                    static_cast<unsigned int>(objects_.size()),
                    nodes_.data().get(),
                    aabbs_.data().get(),
                    objects_.data().get(),
                    i_escape_index_table_.data().get(),
                    e_escape_index_table_.data().get()};
}

__global__ void UpdateObjects_Kernel(const Face* const* faces, Primitive* objects, uint num_objects,
                                     Scalar torlence, bool ccd) {
  GET_CUDA_ID(i, num_objects);

  const Face* face = faces[i];

  Node* node1 = face->nodes[0];
  Node* node2 = face->nodes[1];
  Node* node3 = face->nodes[2];

  uint idx1 = node1->index;
  uint idx2 = node2->index;
  uint idx3 = node3->index;

  Vector3 v1 = node1->x0 + torlence * node1->n;
  Vector3 v2 = node2->x0 + torlence * node2->n;
  Vector3 v3 = node3->x0 + torlence * node3->n;

  Vector3 pred1, pred2, pred3;
  if (ccd) {
    pred1 = node1->x + torlence * node1->n;
    pred2 = node2->x + torlence * node2->n;
    pred3 = node3->x + torlence * node3->n;
  } else {
    pred1 = v1;
    pred2 = v2;
    pred3 = v3;
  }

  Primitive& p = objects[i];
  p.idx1 = idx1;
  p.idx2 = idx2;
  p.idx3 = idx3;
  p.v1 = v1;
  p.v2 = v2;
  p.v3 = v3;
  p.pred1 = pred1;
  p.pred2 = pred2;
  p.pred3 = pred3;
}

__global__ void InitAABBs_Kernel(Primitive* objects, Bounds* aabbs, uint num_internal_nodes,
                                 uint num_objects, Scalar scr) {
  GET_CUDA_ID(idx, num_objects);
  // calculate aabb of object
  auto f = objects[idx];
  const Vector3& v1 = f.v1;
  const Vector3& v2 = f.v2;
  const Vector3& v3 = f.v3;

  const Vector3& pred1 = f.pred1;
  const Vector3& pred2 = f.pred2;
  const Vector3& pred3 = f.pred3;

  Bounds& aabb = aabbs[idx + num_internal_nodes];

  Scalar min_x = MathFunctions::min(MathFunctions::min(v1.x, v2.x, v3.x),
                                    MathFunctions::min(pred1.x, pred2.x, pred3.x));
  Scalar min_y = MathFunctions::min(MathFunctions::min(v1.y, v2.y, v3.y),
                                    MathFunctions::min(pred1.y, pred2.y, pred3.y));
  Scalar min_z = MathFunctions::min(MathFunctions::min(v1.z, v2.z, v3.z),
                                    MathFunctions::min(pred1.z, pred2.z, pred3.z));

  Scalar max_x = MathFunctions::max(MathFunctions::max(v1.x, v2.x, v3.x),
                                    MathFunctions::max(pred1.x, pred2.x, pred3.x));
  Scalar max_y = MathFunctions::max(MathFunctions::max(v1.y, v2.y, v3.y),
                                    MathFunctions::max(pred1.y, pred2.y, pred3.y));
  Scalar max_z = MathFunctions::max(MathFunctions::max(v1.z, v2.z, v3.z),
                                    MathFunctions::max(pred1.z, pred2.z, pred3.z));

  aabb.lower = Vector3(min_x, min_y, min_z);
  aabb.upper = Vector3(max_x, max_y, max_z);

  Vector3 center = (aabb.lower + aabb.upper) * static_cast<Scalar>(0.5);

  Vector3 n1 = glm::normalize(aabb.lower - center);
  Vector3 n2 = glm::normalize(aabb.upper - center);

  aabb.lower += scr * n1;
  aabb.upper += scr * n2;
}

__global__ void ComputeMortons_Kernel(Primitive* objects, Bounds* aabbs, Bounds aabb_whole,
                                      uint* mortons, uint num_internal_nodes, uint num_objects) {
  GET_CUDA_ID(idx, num_objects);

  const Bounds& box = aabbs[idx + num_internal_nodes];

  Vector3 p = box.Center();
  // transform to origin
  p.x -= aabb_whole.lower.x;
  p.y -= aabb_whole.lower.y;
  p.z -= aabb_whole.lower.z;
  // scale to [0, 1]
  p.x /= (aabb_whole.upper.x - aabb_whole.lower.x);
  p.y /= (aabb_whole.upper.y - aabb_whole.lower.y);
  p.z /= (aabb_whole.upper.z - aabb_whole.lower.z);

  mortons[idx] = morton_code(p);
}

__global__ void InitObjectIndices_Kernel(uint* indices, uint num_objects) {
  GET_CUDA_ID(idx, num_objects);

  indices[idx] = idx;
}

__global__ void ConstructLeafNodes_Kernel(uint* indices, BVNode* nodes, uint num_internal_nodes,
                                          uint num_objects) {
  GET_CUDA_ID(idx, num_objects);

  uint object_idx = indices[idx];

  BVNode n;
  n.parent_idx = 0xFFFFFFFF;
  n.left_idx = 0xFFFFFFFF;
  n.right_idx = 0xFFFFFFFF;
  n.object_idx = object_idx;

  nodes[idx + num_internal_nodes] = n;
}

__host__ __device__ inline Bounds Merge(const Bounds& lhs, const Bounds& rhs) noexcept {
  Bounds merged;
  merged.upper.x = ::fmaxf(lhs.upper.x, rhs.upper.x);
  merged.upper.y = ::fmaxf(lhs.upper.y, rhs.upper.y);
  merged.upper.z = ::fmaxf(lhs.upper.z, rhs.upper.z);
  merged.lower.x = ::fminf(lhs.lower.x, rhs.lower.x);
  merged.lower.y = ::fminf(lhs.lower.y, rhs.lower.y);
  merged.lower.z = ::fminf(lhs.lower.z, rhs.lower.z);
  return merged;
}

__global__ void AssignAABBCAS_Kernel(BasicDeviceBVH self, int* flags, uint num_objects) {
  GET_CUDA_ID(idx, num_objects);

  unsigned int parent = self.nodes[idx + num_objects - 1].parent_idx;
  while (parent != 0xFFFFFFFF)  // means idx == 0
  {
    const int old = atomicCAS(flags + parent, 0, 1);
    if (old == 0) {
      // this is the first thread entered here.
      // wait the other thread from the other child node.
      return;
    }
    assert(old == 1);
    // here, the flag has already been 1. it means that this
    // thread is the 2nd thread. merge AABB of both childlen.
    const auto lidx = self.nodes[parent].left_idx;
    const auto ridx = self.nodes[parent].right_idx;
    const auto lbox = self.aabbs[lidx];
    const auto rbox = self.aabbs[ridx];
    self.aabbs[parent] = Merge(lbox, rbox);
    // look the next parent...
    parent = self.nodes[parent].parent_idx;
  }
}

__global__ void AssignAABBAtomic_Kernel(BasicDeviceBVH self, uint num_nodes) {
  GET_CUDA_ID(idx, num_nodes);

  Bounds& bounds = self.aabbs[idx];
  unsigned int parent = self.nodes[idx].parent_idx;

  while (parent != 0xFFFFFFFF)  // means idx == 0
  {
    for (int j = 0; j < 3; j++) {
      AtomicMin(&self.aabbs[parent].lower[j], bounds.lower[j]);
      AtomicMax(&self.aabbs[parent].upper[j], bounds.upper[j]);
    }

    // look the next parent...
    parent = self.nodes[parent].parent_idx;
  }
  return;
}

__global__ void CopyAABB_Kernel(Bounds* src_aabbs, Bounds* tgt_aabbs, uint num_internal_nodes,
                                uint num_objects) {
  GET_CUDA_ID(idx, num_objects);
  tgt_aabbs[idx] = src_aabbs[idx + num_internal_nodes];
}

void BVH::UpdateData(const Face* const* faces, Scalar torlence, bool ccd, int n_faces) {
  objects_.resize(n_faces, Primitive());
  aabbs_.resize(0);
  nodes_.resize(0);

  CUDA_CALL(UpdateObjects_Kernel, n_faces)
  (faces, pointer(objects_), n_faces, torlence, ccd);
  CUDA_CHECK_LAST();
}

Bounds BVH::GetRootAABB() {
  return root_aabb_;
}

thrust::host_vector<Primitive> BVH::ObjectsHost() const {
  thrust::host_vector<Primitive> h_objects = objects_;
  return h_objects;
}
thrust::host_vector<BVNode> BVH::NodesHost() const {
  thrust::host_vector<BVNode> h_nodes = nodes_;
  return h_nodes;
}
thrust::host_vector<Bounds> BVH::AABBsHost() const {
  thrust::host_vector<Bounds> h_aabbs = aabbs_;
  return h_aabbs;
}
thrust::host_vector<Bounds> BVH::DefaultAABBsHost() const {
  thrust::host_vector<Bounds> h_default_aabbs = default_aabbs_;
  return h_default_aabbs;
}
thrust::host_vector<int> BVH::InternalEscapeIndexTableHost() const {
  thrust::host_vector<int> h_i_escape_index_table = i_escape_index_table_;
  return h_i_escape_index_table;
}
thrust::host_vector<int> BVH::ExternalEscapeIndexTableHost() const {
  thrust::host_vector<int> h_e_escape_index_table = e_escape_index_table_;
  return h_e_escape_index_table;
}

unsigned int BVH::NumInternalNodes() const {
  return num_internal_nodes_;
}
unsigned int BVH::NumNodes() const {
  return num_nodes_;
}
unsigned int BVH::NumObjects() const {
  return num_internal_nodes_ + 1u;
}

bool BVH::IsActive() const {
  return is_active_;
}

void BVH::Construct(Scalar scr) {
  const uint num_objects = objects_.size();
  num_internal_nodes_ = num_objects - 1;
  num_nodes_ = num_objects * 2 - 1;
  /*
        * Construction of LBVH:
        *   1. assign Morton code for each primitive
        *   2. sort the Morton codes
        *   3. construct BRT
        *   4. assign AABB for each internal node
        *   5. build escape index table
        */
  if (num_objects == 0)
    return;  // no object data available
  // --------------------------------------------------------------------
  // Step1: calculate morton code of each primitive
  Bounds default_aabb;
  this->aabbs_.resize(num_nodes_, default_aabb);

  // compute AABB using Primitives
  CUDA_CALL(InitAABBs_Kernel, num_objects)
  (pointer(objects_), pointer(aabbs_), num_internal_nodes_, num_objects, scr);
  CUDA_CHECK_LAST();

  const auto aabb_whole = thrust::reduce(
      aabbs_.begin() + num_internal_nodes_, aabbs_.end(), default_aabb,
      [] __device__(const Bounds& lhs, const Bounds& rhs) { return lhs.Merge(rhs); });
  CUDA_CHECK_LAST();

  root_aabb_ = aabb_whole;

  // compute morton code utilizing unit box
  thrust::device_vector<uint> morton(num_objects);
  CUDA_CALL(ComputeMortons_Kernel, num_objects)
  (pointer(objects_), pointer(aabbs_), aabb_whole, pointer(morton), num_internal_nodes_,
   num_objects);
  CUDA_CHECK_LAST();

  i_escape_index_table_.resize(num_objects, -1);
  e_escape_index_table_.resize(num_objects, -1);

  // --------------------------------------------------------------------
  // Step2: sort object-indices by morton code
  thrust::device_vector<uint> indices(num_objects);
  CUDA_CALL(InitObjectIndices_Kernel, num_objects)
  (pointer(indices), num_objects);

  // Sort indices by morton code. Keep indices ascending order
  thrust::stable_sort_by_key(morton.begin(), morton.end(),
                             thrust::make_zip_iterator(thrust::make_tuple(
                                 aabbs_.begin() + num_internal_nodes_, indices.begin())));

  // check morton codes are unique
  thrust::device_vector<unsigned long long int> morton64(num_objects);
  const auto uniqued = thrust::unique_copy(morton.begin(), morton.end(), morton64.begin());
  const bool morton_code_is_unique = (morton64.end() == uniqued);
  if (!morton_code_is_unique) {
    thrust::transform(
        morton.begin(), morton.end(), indices.begin(), morton64.begin(),
        [] __device__(const unsigned int m, const unsigned int idx) {
          unsigned long long int m64 = m;
          m64 <<= 32;  // expand 32-bit morton code to 64 bit
          m64 |= idx;  // assgin index to morton code so that the merged morton code is unique
          return m64;
        });
  }

  // construct leaf nodes and aabbs
  BVNode default_node;
  default_node.parent_idx = 0xFFFFFFFF;
  default_node.left_idx = 0xFFFFFFFF;
  default_node.right_idx = 0xFFFFFFFF;
  default_node.object_idx = 0xFFFFFFFF;
  this->nodes_.resize(num_nodes_, default_node);

  CUDA_CALL(ConstructLeafNodes_Kernel, num_objects)
  (pointer(indices), pointer(nodes_), num_internal_nodes_, num_objects);

  // --------------------------------------------------------------------
  // Step3: Construct internal nodes
  const auto self = this->GetDeviceRepr();
  if (morton_code_is_unique)  // 32-bit version
  {
    // unsigned int has 4-bytes, which corresponds to 32-bit
    const unsigned int* node_code = morton.data().get();
    ConstructInternalNodesCUDA(self, node_code, num_objects);
  } else  // 64-bit version
  {
    // unsigned long long has 8-bytes, which corresponds to 64-bit
    const unsigned long long int* node_code = morton64.data().get();
    ConstructInternalNodesCUDA(self, node_code, num_objects);
  }
  // --------------------------------------------------------------------
  // Step4: assign AABB for each node by bottom-up strategy
  CUDA_CALL(AssignAABBAtomic_Kernel, num_nodes_)
  (self, num_nodes_);
  CUDA_CHECK_LAST();

  // --------------------------------------------------------------------
  // Step5: build escape index table
  BuildEscapeIndexTable(self, num_internal_nodes_, num_objects);
  CUDA_CHECK_LAST();

  is_active_ = true;
}

__global__ void RefitExternalNode_Kernel(BasicDeviceBVH self, const Vector3* predicted,
                                         const Vector3* normals, uint num_internal_nodes,
                                         uint num_objects, Scalar torlence) {
  GET_CUDA_ID(idx, num_objects);

  uint node_idx = idx + num_internal_nodes;
  uint object_idx = self.nodes[node_idx].object_idx;

  uint idx1 = self.objects[object_idx].idx1;
  uint idx2 = self.objects[object_idx].idx2;
  uint idx3 = self.objects[object_idx].idx3;

  self.objects[object_idx].pred1 = predicted[idx1] + torlence * normals[idx1];
  self.objects[object_idx].pred2 = predicted[idx2] + torlence * normals[idx2];
  self.objects[object_idx].pred3 = predicted[idx3] + torlence * normals[idx3];

  Bounds b = self.aabbs[node_idx];
  b = b.Merge(self.objects[object_idx].pred1);
  b = b.Merge(self.objects[object_idx].pred2);
  b = b.Merge(self.objects[object_idx].pred3);
  self.aabbs[node_idx] = b;
}

__global__ void RefitExternalNode_Kernel(BasicDeviceBVH self, const Vector3* predicted,
                                         uint num_internal_nodes, uint num_objects) {
  GET_CUDA_ID(idx, num_objects);

  uint node_idx = idx + num_internal_nodes;
  uint object_idx = self.nodes[node_idx].object_idx;

  uint idx1 = self.objects[object_idx].idx1;
  uint idx2 = self.objects[object_idx].idx2;
  uint idx3 = self.objects[object_idx].idx3;

  self.objects[object_idx].pred1 = predicted[idx1];
  self.objects[object_idx].pred2 = predicted[idx2];
  self.objects[object_idx].pred3 = predicted[idx3];

  Bounds b = self.aabbs[node_idx];
  b = b.Merge(self.objects[object_idx].pred1);
  b = b.Merge(self.objects[object_idx].pred2);
  b = b.Merge(self.objects[object_idx].pred3);

  self.aabbs[node_idx] = b;
}

__global__ void RefitExternalNodeMidstep_Kernel(BasicDeviceBVH self, const Face* const* faces,
                                                uint num_internal_nodes, uint num_objects,
                                                Scalar torlence) {
  GET_CUDA_ID(idx, num_objects);

  uint node_idx = idx + num_internal_nodes;
  uint object_idx = self.nodes[node_idx].object_idx;

  uint idx1 = self.objects[object_idx].idx1;
  uint idx2 = self.objects[object_idx].idx2;
  uint idx3 = self.objects[object_idx].idx3;

  self.objects[object_idx].pred1 =
      faces[object_idx]->nodes[0]->x1 + torlence * faces[object_idx]->nodes[0]->n1;
  self.objects[object_idx].pred2 =
      faces[object_idx]->nodes[1]->x1 + torlence * faces[object_idx]->nodes[1]->n1;
  self.objects[object_idx].pred3 =
      faces[object_idx]->nodes[2]->x1 + torlence * faces[object_idx]->nodes[2]->n1;

  Bounds& b = self.aabbs[node_idx];
  b += self.objects[object_idx].pred1;
  b += self.objects[object_idx].pred2;
  b += self.objects[object_idx].pred3;
}

void BVH::RefitMidstep(const Face* const* faces, Scalar torlence) {
  /*
        * To refit the bvh:
        *   1. update predicted positions & AABBs of the external nodes
        *   2. update internal nodes
        */
  ScopedTimerGPU("Solver_BVHRefit");
  // ----------- Step1: update external nodes -----------
  const auto self = this->GetDeviceRepr();
  if (!this->IsActive())
    return;

  const auto num_primitives = this->objects_.size();
  auto num_internal_nodes = this->num_internal_nodes_;

  CUDA_CALL(RefitExternalNodeMidstep_Kernel, num_primitives)
  (self, faces, num_internal_nodes, num_primitives, torlence);

  // ----------- Step2: update internal nodes -----------
  thrust::device_vector<int> flag_container(num_internal_nodes, 0);
  const auto flags = flag_container.data().get();
  CUDA_CALL(AssignAABBCAS_Kernel, num_primitives)
  (self, flags, num_primitives);
}

__global__ void Update_Kernel(BasicDeviceBVH self, Face** faces, int num_internal_nodes,
                              int num_objects, bool ccd) {
  GET_CUDA_ID(idx, num_objects);

  uint node_idx = idx + num_internal_nodes;
  uint object_idx = self.nodes[node_idx].object_idx;

  Face* face = faces[object_idx];

  self.objects[object_idx].pred1 = face->nodes[0]->x;
  self.objects[object_idx].pred2 = face->nodes[1]->x;
  self.objects[object_idx].pred3 = face->nodes[2]->x;

  self.aabbs[node_idx] = self.aabbs[node_idx].Merge(face->ComputeBounds(ccd));
}

void BVH::Update(Face** faces, bool ccd) {
  const auto self = this->GetDeviceRepr();
  if (!this->IsActive())
    return;

  const auto num_primitives = this->objects_.size();
  auto num_internal_nodes = this->num_internal_nodes_;
  int num_nodes = this->num_nodes_;

  CUDA_CALL(Update_Kernel, num_primitives)
  (self, faces, num_internal_nodes, num_primitives, ccd);
  CUDA_CHECK_LAST();

  CUDA_CALL(AssignAABBAtomic_Kernel, num_nodes)
  (self, num_nodes);
  CUDA_CHECK_LAST();
}

__global__ void PredictiveRefitExternalNode_Kernel(BasicDeviceBVH self, const Vector3* positions,
                                                   const Vector3* velocities,
                                                   uint num_internal_nodes, uint num_objects,
                                                   Scalar time_step) {
  GET_CUDA_ID(idx, num_objects);

  uint node_idx = idx + num_internal_nodes;
  uint object_idx = self.nodes[node_idx].object_idx;

  uint idx1 = self.objects[object_idx].idx1;
  uint idx2 = self.objects[object_idx].idx2;
  uint idx3 = self.objects[object_idx].idx3;

  self.objects[object_idx].v1 = positions[idx1];
  self.objects[object_idx].v2 = positions[idx2];
  self.objects[object_idx].v3 = positions[idx3];

  self.objects[object_idx].pred1 = positions[idx1] + velocities[idx1] * time_step;
  self.objects[object_idx].pred2 = positions[idx2] + velocities[idx2] * time_step;
  self.objects[object_idx].pred3 = positions[idx3] + velocities[idx3] * time_step;

  Bounds& b = self.aabbs[node_idx];
  b.Merge(self.objects[object_idx].pred1);
  b.Merge(self.objects[object_idx].pred2);
  b.Merge(self.objects[object_idx].pred3);
}

__device__ thrust::pair<unsigned int, Scalar> QueryDevice(const BasicDeviceBVH& bvh,
                                                          const Vector3& target,
                                                          distance_calculator calc_dist,
                                                          Scalar& nearestU, Scalar& nearestV,
                                                          Scalar& nearestW) {
  // pair of {node_idx, mindist}
  thrust::pair<unsigned int, Scalar> stack[64];
  thrust::pair<unsigned int, Scalar>* stack_ptr = stack;
  *stack_ptr++ = thrust::make_pair(0, MinDist(bvh.aabbs[0], target));

  unsigned int nearest = 0xFFFFFFFF;
  Scalar dist_to_nearest_object = Infinity();

  Vector3 qs(0);
  Scalar u = 0, v = 0, w = 0;

  do {
    const auto node = *--stack_ptr;
    if (node.second > dist_to_nearest_object) {
      // if aabb mindist > already_found_mindist, it cannot have a nearest
      continue;
    }

    const unsigned int L_idx = bvh.nodes[node.first].left_idx;
    const unsigned int R_idx = bvh.nodes[node.first].right_idx;

    const Bounds& L_box = bvh.aabbs[L_idx];
    const Bounds& R_box = bvh.aabbs[R_idx];

    const Scalar L_mindist = MinDist(L_box, target);
    const Scalar R_mindist = MinDist(R_box, target);

    const Scalar L_minmaxdist = MinMaxDist(L_box, target);
    const Scalar R_minmaxdist = MinMaxDist(R_box, target);

    // there should be an object that locates within minmaxdist.

    if (L_mindist <= R_minmaxdist)  // L is worth considering
    {
      const auto obj_idx = bvh.nodes[L_idx].object_idx;
      if (obj_idx != 0xFFFFFFFF)  // leaf node
      {
        const Scalar dist = calc_dist(target, bvh.objects[obj_idx], qs, u, v, w);
        if (dist <= dist_to_nearest_object) {
          dist_to_nearest_object = dist;
          nearest = obj_idx;
          nearestU = u;
          nearestV = v;
          nearestW = w;
        }
      } else {
        *stack_ptr++ = thrust::make_pair(L_idx, L_mindist);
      }
    }
    if (R_mindist <= L_minmaxdist)  // R is worth considering
    {
      const auto obj_idx = bvh.nodes[R_idx].object_idx;
      if (obj_idx != 0xFFFFFFFF)  // leaf node
      {
        const Scalar dist = calc_dist(target, bvh.objects[obj_idx], qs, u, v, w);
        if (dist <= dist_to_nearest_object) {
          dist_to_nearest_object = dist;
          nearest = obj_idx;
          nearestU = u;
          nearestV = v;
          nearestW = w;
        }
      } else {
        *stack_ptr++ = thrust::make_pair(R_idx, R_mindist);
      }
    }
    assert(stack_ptr < stack + 64);
  } while (stack < stack_ptr);
  return thrust::make_pair(nearest, dist_to_nearest_object);
}

__device__ unsigned int QueryDeviceStackless(const BasicDeviceBVH& bvh, const Vector3& p,
                                             uint* outiter) {
  int node_idx = 0;
  uint num_found = 0u;
  BVNode node;
  Bounds aabb;
  uint num_internal_nodes = bvh.num_objects - 1u;
  while (node_idx != -1) {
    node = bvh.nodes[node_idx];
    aabb = bvh.aabbs[node_idx];

    // maximum potential collisions cached, stop escaping
    if (num_found >= LBVH_MAX_BUFFER_SIZE)
      break;

    if (aabb.Overlap(p)) {
      if (node.object_idx != 0xFFFFFFFF)  // external node
      {
        outiter[num_found++] = node.object_idx;
        node_idx = bvh.e_escape_index_table[node_idx - num_internal_nodes];
      } else  // internal node
        node_idx = node.left_idx;
    } else {
      if (node.object_idx != 0xFFFFFFFF)  // external node
        node_idx = bvh.e_escape_index_table[node_idx - num_internal_nodes];
      else  // internal node
        node_idx = bvh.i_escape_index_table[node_idx];
    }
  }
  return num_found;
}

__device__ unsigned int QueryDeviceStackless(const BasicDeviceBVH& bvh, const Vector3& p0,
                                             const Vector3& p1, uint* outiter) {
  int node_idx = 0;
  uint num_found = 0u;
  BVNode node;
  Bounds aabb;
  uint num_internal_nodes = bvh.num_objects - 1u;
  Vector3 hit(SCALAR_MAX);
  //glm::vec3 hit(INFINITY);
  while (node_idx != -1) {
    node = bvh.nodes[node_idx];
    aabb = bvh.aabbs[node_idx];

    // maximum potential collisions cached, stop escaping
    if (num_found >= LBVH_MAX_BUFFER_SIZE)
      break;

    if (aabb.Overlap(p0, p1, hit)) {
      if (node.object_idx != 0xFFFFFFFF)  // external node
      {
        outiter[num_found++] = node.object_idx;
        node_idx = bvh.e_escape_index_table[node_idx - num_internal_nodes];
      } else  // internal node
        node_idx = node.left_idx;
    } else {
      if (node.object_idx != 0xFFFFFFFF)  // external node
        node_idx = bvh.e_escape_index_table[node_idx - num_internal_nodes];
      else  // internal node
        node_idx = bvh.i_escape_index_table[node_idx];
    }
  }
  return num_found;
}

__device__ unsigned int QueryDeviceStackless(const BasicDeviceBVH& bvh, const Bounds& query_aabb,
                                             uint* overlaps, const uint& primitive_idx) {
  int node_idx = 0;
  uint num_found = 0u;
  BVNode node;
  Bounds aabb;
  uint idx_offset = primitive_idx * LBVH_MAX_BUFFER_SIZE;
  uint num_internal_nodes = bvh.num_objects - 1u;
  while (node_idx != -1) {
    node = bvh.nodes[node_idx];
    aabb = bvh.aabbs[node_idx];

    // maximum potential collisions cached, stop escaping
    if (num_found >= LBVH_MAX_BUFFER_SIZE)
      break;

    if (aabb.Overlap(query_aabb)) {
      if (node.object_idx != 0xFFFFFFFF)  // external node
      {
        overlaps[idx_offset + num_found] = node.object_idx;
        num_found++;
        node_idx = bvh.e_escape_index_table[node_idx - num_internal_nodes];
      } else  // internal node
        node_idx = node.left_idx;
    } else {
      if (node.object_idx != 0xFFFFFFFF)  // external node
        node_idx = bvh.e_escape_index_table[node_idx - num_internal_nodes];
      else  // internal node
        node_idx = bvh.i_escape_index_table[node_idx];
    }
  }
  return num_found;
}

__global__ void CountPairsStack_Kernel(BasicDeviceBVH bvh_lhs, BasicDeviceBVH bvh_rhs,
                                       Scalar thickness, int* num, uint num_objects_lhs) {
  GET_CUDA_ID(idx, num_objects_lhs);

  auto node_idx_lhs = idx + (bvh_lhs.num_objects - 1);
  auto aabb_lhs = bvh_lhs.aabbs[node_idx_lhs];
  int node_idx_rhs = 0;

  int& n = num[idx];
  n = 0;

  int stack[64];
  int* stack_ptr = stack;
  *stack_ptr++ = 0;

  do {
    const int node_idx = *--stack_ptr;
    const int L_idx = bvh_rhs.nodes[node_idx].left_idx;
    const int R_idx = bvh_rhs.nodes[node_idx].right_idx;

    if (aabb_lhs.Overlap(bvh_rhs.aabbs[L_idx], thickness)) {
      const int object_idx = bvh_rhs.nodes[L_idx].object_idx;
      if (object_idx != 0xFFFFFFFF) {
        n++;
      } else {
        *stack_ptr++ = L_idx;
      }
    }
    if (aabb_lhs.Overlap(bvh_rhs.aabbs[R_idx], thickness)) {
      const int object_idx = bvh_rhs.nodes[R_idx].object_idx;
      if (object_idx != 0xFFFFFFFF) {
        n++;
      } else {
        *stack_ptr++ = R_idx;
      }
    }
  } while (stack < stack_ptr);
}

__global__ void FindPairsStack_Kernel(BasicDeviceBVH bvh_lhs, BasicDeviceBVH bvh_rhs,
                                      Face** faces_lhs, Face** faces_rhs, Scalar thickness,
                                      const int* num, PairFF* pairs, uint num_objects_lhs) {
  GET_CUDA_ID(id, num_objects_lhs);

  auto node_idx_lhs = id + (bvh_lhs.num_objects - 1);
  auto aabb_lhs = bvh_lhs.aabbs[node_idx_lhs];
  auto object_idx_lhs = bvh_lhs.nodes[node_idx_lhs].object_idx;

  int node_idx_rhs = 0;
  int index = num[id] - 1;  // index offset of the proximity

  int stack[64];
  int* stack_ptr = stack;
  *stack_ptr++ = 0;

  do {
    const int node_idx = *--stack_ptr;
    const int L_idx = bvh_rhs.nodes[node_idx].left_idx;
    const int R_idx = bvh_rhs.nodes[node_idx].right_idx;

    if (aabb_lhs.Overlap(bvh_rhs.aabbs[L_idx], thickness)) {
      const int object_idx_rhs = bvh_rhs.nodes[L_idx].object_idx;
      if (object_idx_rhs != 0xFFFFFFFF) {
        PairFF& pair = pairs[index--];
        pair.first = faces_lhs[object_idx_lhs];
        pair.second = faces_rhs[object_idx_rhs];
      } else {
        *stack_ptr++ = L_idx;
      }
    }
    if (aabb_lhs.Overlap(bvh_rhs.aabbs[R_idx], thickness)) {
      const int object_idx_rhs = bvh_rhs.nodes[R_idx].object_idx;
      if (object_idx_rhs != 0xFFFFFFFF) {
        PairFF& pair = pairs[index--];
        pair.first = faces_lhs[object_idx_lhs];
        pair.second = faces_rhs[object_idx_rhs];
      } else {
        *stack_ptr++ = R_idx;
      }
    }
  } while (stack < stack_ptr);
}

/**
     * @brief Evaluate number of proximities for inter-object collision.
     * @param bvh_dev_lhs The source device LBVH
     * @param bvh_dev_rhs The target device LBVH
     * @param thickness Cloth thickness
     * @param num Number of estimated proximities
     * @param num_objects Number of objects(external nodes)
     * @return void
    */
__global__ void CountPairs_Kernel(BasicDeviceBVH bvh_lhs, BasicDeviceBVH bvh_rhs, Scalar thickness,
                                  int* num, uint num_objects_lhs) {
  GET_CUDA_ID(idx, num_objects_lhs);

  int node_idx_lhs = idx + (bvh_lhs.num_objects - 1);
  Bounds& aabb_lhs = bvh_lhs.aabbs[node_idx_lhs];
  int node_idx_rhs = 0;

  int& n = num[idx];
  n = 0;

  uint num_internal_nodes = bvh_rhs.num_objects - 1u;
  while (node_idx_rhs != -1) {
    BVNode& node_rhs = bvh_rhs.nodes[node_idx_rhs];
    Bounds& aabb_rhs = bvh_rhs.aabbs[node_idx_rhs];

    if (aabb_rhs.Overlap(aabb_lhs, thickness)) {
      if (node_rhs.object_idx != 0xFFFFFFFF)  // external node
      {
        n++;
        node_idx_rhs = bvh_rhs.e_escape_index_table[node_idx_rhs - num_internal_nodes];
      } else  // internal node
        node_idx_rhs = node_rhs.left_idx;
    } else {
      if (node_rhs.object_idx != 0xFFFFFFFF)  // external node
        node_idx_rhs = bvh_rhs.e_escape_index_table[node_idx_rhs - num_internal_nodes];
      else  // internal node
        node_idx_rhs = bvh_rhs.i_escape_index_table[node_idx_rhs];
    }
  }
}

/**
     * @brief Evaluate proximities for inter-object collision.
     * @param bvh_lhs The source device LBVH
     * @param bvh_rhs The target device LBVH
     * @param thickness Cloth thickness
     * @param num Number of estimated proximities
     * @param pairs The resulting FF proximity
     * @param num_objects Number of objects(external nodes) for bvh_lhs
     * @return void
    */
__global__ void FindPairs_Kernel(BasicDeviceBVH bvh_lhs, BasicDeviceBVH bvh_rhs, Face** faces_lhs,
                                 Face** faces_rhs, Scalar thickness, const int* num, PairFF* pairs,
                                 uint num_objects_lhs) {
  GET_CUDA_ID(id, num_objects_lhs);

  auto node_idx_lhs = id + (bvh_lhs.num_objects - 1);
  auto aabb_lhs = bvh_lhs.aabbs[node_idx_lhs];
  auto object_idx_lhs = bvh_lhs.nodes[node_idx_lhs].object_idx;

  int node_idx_rhs = 0;
  int index = num[id] - 1;  // index offset of the proximity

  BVNode node_rhs;
  Bounds aabb_rhs;
  uint num_internal_nodes = bvh_rhs.num_objects - 1u;
  while (node_idx_rhs != -1) {
    node_rhs = bvh_rhs.nodes[node_idx_rhs];
    aabb_rhs = bvh_rhs.aabbs[node_idx_rhs];

    if (aabb_rhs.Overlap(aabb_lhs, thickness)) {
      if (node_rhs.object_idx != 0xFFFFFFFF)  // external node
      {
        PairFF& pair = pairs[index--];
        pair.first = faces_lhs[object_idx_lhs];
        pair.second = faces_rhs[node_rhs.object_idx];

        node_idx_rhs = bvh_rhs.e_escape_index_table[node_idx_rhs - num_internal_nodes];
      } else  // internal node
        node_idx_rhs = node_rhs.left_idx;
    } else {
      if (node_rhs.object_idx != 0xFFFFFFFF)  // external node
        node_idx_rhs = bvh_rhs.e_escape_index_table[node_idx_rhs - num_internal_nodes];
      else  // internal node
        node_idx_rhs = bvh_rhs.i_escape_index_table[node_idx_rhs];
    }
  }
}

/**
     * @brief Evaluate number of proximities for self collision.
     * @param bvh_dev The device LBVH
     * @param thickness Cloth thickness
     * @param num Number of estimated proximities
     * @param num_objects Number of objects(external nodes)
     * @return void
    */
__global__ void CountPairsSelf_Kernel(BasicDeviceBVH bvh_dev, Scalar thickness, int* num,
                                      uint num_objects) {
  GET_CUDA_ID(idx, num_objects);

  auto query_node_idx = idx + (bvh_dev.num_objects - 1);
  auto query_aabb = bvh_dev.aabbs[query_node_idx];
  int node_idx = 0;

  int& n = num[idx];
  n = 0;

  BVNode node;
  Bounds aabb;
  uint num_internal_nodes = bvh_dev.num_objects - 1u;
  while (node_idx != -1) {
    node = bvh_dev.nodes[node_idx];
    aabb = bvh_dev.aabbs[node_idx];

    if (aabb.Overlap(query_aabb, thickness)) {
      if (node.object_idx != 0xFFFFFFFF)  // external node
      {
        n++;
        node_idx = bvh_dev.e_escape_index_table[node_idx - num_internal_nodes];
      } else  // internal node
        node_idx = node.left_idx;
    } else {
      if (node.object_idx != 0xFFFFFFFF)  // external node
        node_idx = bvh_dev.e_escape_index_table[node_idx - num_internal_nodes];
      else  // internal node
        node_idx = bvh_dev.i_escape_index_table[node_idx];
    }
  }
}

__global__ void FindPairsSelf_Kernel(BasicDeviceBVH bvh_dev, Scalar thickness, const int* num,
                                     Pairii* pairs, uint num_objects) {
  GET_CUDA_ID(id, num_objects);

  auto query_node_idx = id + (bvh_dev.num_objects - 1);
  auto query_aabb = bvh_dev.aabbs[query_node_idx];
  auto query_object_idx = bvh_dev.nodes[query_node_idx].object_idx;
  int node_idx = 0;

  int index = num[id] - 1;  // index offset of the proximity

  BVNode node;
  Bounds aabb;
  uint num_internal_nodes = bvh_dev.num_objects - 1u;
  while (node_idx != -1) {
    node = bvh_dev.nodes[node_idx];
    aabb = bvh_dev.aabbs[node_idx];

    if (aabb.Overlap(query_aabb, thickness)) {
      if (node.object_idx != 0xFFFFFFFF)  // external node
      {
        Pairii& pair = pairs[index--];
        pair.first = query_object_idx;
        pair.second = node.object_idx;

        node_idx = bvh_dev.e_escape_index_table[node_idx - num_internal_nodes];
      } else  // internal node
        node_idx = node.left_idx;
    } else {
      if (node.object_idx != 0xFFFFFFFF)  // external node
        node_idx = bvh_dev.e_escape_index_table[node_idx - num_internal_nodes];
      else  // internal node
        node_idx = bvh_dev.i_escape_index_table[node_idx];
    }
  }
}

/**
     * @brief Traverse the BVH to get inter-object proximities
     * @param bvh_lhs The source LBVH
     * @param bvh_rhs The target LBVH
     * @param thickness The cloth thickness
     * @return
    */
thrust::device_vector<PairFF> Traverse(std::shared_ptr<BVH> bvh_lhs, std::shared_ptr<BVH> bvh_rhs,
                                       Face** faces_lhs, Face** faces_rhs,
                                       const Scalar& thickness) {
  auto bvh_dev_lhs = bvh_lhs->GetDeviceRepr();
  auto bvh_dev_rhs = bvh_rhs->GetDeviceRepr();
  if (!bvh_lhs->IsActive() || !bvh_rhs->IsActive())
    return thrust::device_vector<PairFF>();

  auto n_objects = bvh_lhs->NumObjects();
  thrust::device_vector<int> num(n_objects);
  int* num_ptr = pointer(num);

  // evaluate the size of proximities
  CUDA_CALL(CountPairs_Kernel, n_objects)
  (bvh_dev_lhs, bvh_dev_rhs, thickness, num_ptr, n_objects);

  // do prefix sum for number of proximities
  thrust::inclusive_scan(num.begin(), num.end(), num.begin());
  thrust::device_vector<PairFF> ans(num.back());

  CUDA_CALL(FindPairs_Kernel, n_objects)
  (bvh_dev_lhs, bvh_dev_rhs, faces_lhs, faces_rhs, thickness, num_ptr, pointer(ans), n_objects);
  CUDA_CHECK_LAST();

  return ans;
}

thrust::device_vector<PairFF> Traverse(BVH* bvh_lhs, BVH* bvh_rhs, Face** faces_lhs,
                                       Face** faces_rhs, const Scalar& thickness) {
  auto bvh_dev_lhs = bvh_lhs->GetDeviceRepr();
  auto bvh_dev_rhs = bvh_rhs->GetDeviceRepr();
  if (!bvh_lhs->IsActive() || !bvh_rhs->IsActive())
    return thrust::device_vector<PairFF>();

  auto n_objects = bvh_lhs->NumObjects();
  thrust::device_vector<int> num(n_objects);
  int* num_ptr = pointer(num);
  // evaluate the size of proximities
  CUDA_CALL(CountPairs_Kernel, n_objects)
  (bvh_dev_lhs, bvh_dev_rhs, thickness, num_ptr, n_objects);

  // do prefix sum for number of proximities
  thrust::inclusive_scan(num.begin(), num.end(), num.begin());
  thrust::device_vector<PairFF> ans(num.back());

  CUDA_CALL(FindPairs_Kernel, n_objects)
  (bvh_dev_lhs, bvh_dev_rhs, faces_lhs, faces_rhs, thickness, num_ptr, pointer(ans), n_objects);
  CUDA_CHECK_LAST();

  return ans;
}

thrust::device_vector<PairFF> TraverseStack(std::shared_ptr<BVH> bvh_lhs,
                                            std::shared_ptr<BVH> bvh_rhs, Face** faces_lhs,
                                            Face** faces_rhs, const Scalar& thickness) {
  auto bvh_dev_lhs = bvh_lhs->GetDeviceRepr();
  auto bvh_dev_rhs = bvh_rhs->GetDeviceRepr();
  if (!bvh_lhs->IsActive() || !bvh_rhs->IsActive())
    return thrust::device_vector<PairFF>();

  auto n_objects = bvh_lhs->NumObjects();
  thrust::device_vector<int> num(n_objects);
  int* num_ptr = pointer(num);

  // evaluate the size of proximities
  CUDA_CALL(CountPairsStack_Kernel, n_objects)
  (bvh_dev_lhs, bvh_dev_rhs, thickness, num_ptr, n_objects);

  // do prefix sum for number of proximities
  thrust::inclusive_scan(num.begin(), num.end(), num.begin());
  thrust::device_vector<PairFF> ans(num.back());

  CUDA_CALL(FindPairsStack_Kernel, n_objects)
  (bvh_dev_lhs, bvh_dev_rhs, faces_lhs, faces_rhs, thickness, num_ptr, pointer(ans), n_objects);
  CUDA_CHECK_LAST();

  return ans;
}

thrust::device_vector<PairFF> TraverseStack(BVH* bvh_lhs, BVH* bvh_rhs, Face** faces_lhs,
                                            Face** faces_rhs, const Scalar& thickness) {
  auto bvh_dev_lhs = bvh_lhs->GetDeviceRepr();
  auto bvh_dev_rhs = bvh_rhs->GetDeviceRepr();
  if (!bvh_lhs->IsActive() || !bvh_rhs->IsActive())
    return thrust::device_vector<PairFF>();

  auto n_objects = bvh_lhs->NumObjects();
  thrust::device_vector<int> num(n_objects);
  int* num_ptr = pointer(num);

  // evaluate the size of proximities
  CUDA_CALL(CountPairsStack_Kernel, n_objects)
  (bvh_dev_lhs, bvh_dev_rhs, thickness, num_ptr, n_objects);

  // do prefix sum for number of proximities
  thrust::inclusive_scan(num.begin(), num.end(), num.begin());
  thrust::device_vector<PairFF> ans(num.back());

  CUDA_CALL(FindPairsStack_Kernel, n_objects)
  (bvh_dev_lhs, bvh_dev_rhs, faces_lhs, faces_rhs, thickness, num_ptr, pointer(ans), n_objects);
  CUDA_CHECK_LAST();

  return ans;
}

/**
     * @brief Traverse the BVH to get self proximities
     * @param BVH The LBVH
     * @param thickness The cloth thickness
     * @return
    */
thrust::device_vector<Pairii> Traverse(std::shared_ptr<BVH> bvh, const Scalar& thickness) {
  auto bvh_dev = bvh->GetDeviceRepr();
  if (!bvh->IsActive())
    return thrust::device_vector<Pairii>();

  auto n_objects = bvh->NumObjects();
  thrust::device_vector<int> num(n_objects);
  int* num_ptr = pointer(num);

  // evaluate the size of proximities
  CUDA_CALL(CountPairsSelf_Kernel, n_objects)
  (bvh_dev, thickness, num_ptr, n_objects);
  CUDA_CHECK_LAST();

  // do prefix sum for number of proximities
  thrust::inclusive_scan(num.begin(), num.end(), num.begin());
  thrust::device_vector<Pairii> ans(num.back());
  CUDA_CALL(FindPairsSelf_Kernel, n_objects)
  (bvh_dev, thickness, num_ptr, pointer(ans), n_objects);
  CUDA_CHECK_LAST();

  return ans;
}

}  // namespace XRTailor