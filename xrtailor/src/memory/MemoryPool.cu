#include <xrtailor/memory/MemoryPool.cuh>

#include <iostream>

#include <xrtailor/core/DeviceHelper.cuh>
#include <xrtailor/core/ErrorDefs.hpp>

namespace XRTailor {

MemoryPool::MemoryPool()
    : node_ptr_(0),
      edge_ptr_(0),
      face_ptr_(0) {
  node_pool_.resize(NODE_POOL_SIZE);
  edge_pool_.resize(EDGE_POOL_SIZE);
  face_pool_.resize(FACE_POOL_SIZE);
}

MemoryPool::~MemoryPool() {}

Node* MemoryPool::CreateNodes(int n_nodes) {
  if (node_ptr_ + n_nodes > NODE_POOL_SIZE) {
    std::cerr << "Node pool run out" << std::endl;
    exit(TAILOR_EXIT::NODE_POOL_RUN_OUT);
  }

  node_ptr_ += n_nodes;

  return pointer(node_pool_, node_ptr_ - n_nodes);
}

Edge* MemoryPool::CreateEdges(int n_edges) {
  if (edge_ptr_ + n_edges > EDGE_POOL_SIZE) {
    std::cerr << "Edge pool run out" << std::endl;
    exit(TAILOR_EXIT::EDGE_POOL_RUN_OUT);
  }

  edge_ptr_ += n_edges;

  return pointer(edge_pool_, edge_ptr_ - n_edges);
}
Face* MemoryPool::CreateFaces(int n_faces) {
  if (face_ptr_ + n_faces > FACE_POOL_SIZE) {
    std::cerr << "Face pool run out" << std::endl;
    exit(TAILOR_EXIT::FACE_POOL_RUN_OUT);
  }

  face_ptr_ += n_faces;

  return pointer(face_pool_, face_ptr_ - n_faces);
}

int MemoryPool::NodeOffset() {
  return node_ptr_;
}

int MemoryPool::EdgeOffset() {
  return edge_ptr_;
}

int MemoryPool::FaceOffset() {
  return face_ptr_;
}

void MemoryPool::SetupInitialPositions(const std::vector<Vector3>& positions) {
  thrust::device_vector<Vector3> newPositions = positions;
  initial_positions.insert(initial_positions.end(), newPositions.begin(), newPositions.end());
}

void MemoryPool::SetupTmpPositions(int newSize) {
  int size = tmp_positions.size();
  tmp_positions.resize(size + newSize, Vector3(0));
}

void MemoryPool::SetupDeltas(int newSize) {
  int size = deltas.size();
  deltas.resize(size + newSize, Vector3(0));
  delta_counts.resize(size + newSize, 0);
}

void MemoryPool::AddEuclidean(int particle_index, int slot_index, Scalar distance) {
  attach_particle_ids.push_back(particle_index);
  attach_slot_ids.push_back(slot_index);
  attach_distances.push_back(distance);
}

void MemoryPool::AddGeodesic(unsigned int src_index, unsigned int tgt_index, Scalar rest_length) {
  geodesic_src_indices.push_back(src_index);
  geodesic_tgt_indices.push_back(tgt_index);
  geodesic_rest_length.push_back(rest_length);
}

void MemoryPool::AddSDFCollider(const SDFCollider& sc) {
  sdf_colliders.push_back(sc);
}

void MemoryPool::UpdateColliders(std::vector<Collider*>& colliders, Scalar dt) {
  sdf_colliders.resize(colliders.size());
  for (int i = 0; i < colliders.size(); i++) {
    const Collider* c = colliders[i];
    if (!c->enabled)
      continue;
    SDFCollider sc;
    sc.type = c->type;
    sc.position = c->actor->transform->position;
    sc.scale = c->actor->transform->scale;
    sc.cur_transform = c->cur_transform;
    sc.inv_cur_transform = glm::inverse(c->cur_transform);
    sc.last_transform = c->last_transform;
    sc.delta_time = dt;
    sdf_colliders[i] = sc;
  }
}

int MemoryPool::NumGeodesics() {
  return geodesic_rest_length.size();
}

void MemoryPool::SetupObstacleMaskedIndex(const std::vector<uint>& indices) {
  thrust::device_vector<uint> new_indices = indices;
  obstacle_masked_indices.insert(obstacle_masked_indices.end(), new_indices.begin(), new_indices.end());
}

void MemoryPool::SetupObstacleMaskedIndex(const thrust::host_vector<uint>& indices) {
  thrust::device_vector<uint> new_indices = indices;
  obstacle_masked_indices.insert(obstacle_masked_indices.end(), new_indices.begin(), new_indices.end());
}

void MemoryPool::SetupSkinnedObstacle(int size) {
  skin_params.resize(size);
}

}  // namespace XRTailor