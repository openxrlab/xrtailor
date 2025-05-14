#include <xrtailor/physics/ClothSolverHelper.cuh>
#include <xrtailor/utils/Timer.hpp>

#include <glm/ext/matrix_transform.hpp>

#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/count.h>
#include <thrust/sort.h>

#include <iomanip>

#include <xrtailor/physics/broad_phase/lbvh/BVH.cuh>

namespace XRTailor {
__device__ __constant__ SimParams d_params;
SimParams h_params;

void SetSimulationParams(SimParams* host_params) {
  ScopedTimerGPU timer("Solver_SetParams");
  checkCudaErrors(cudaMemcpyToSymbolAsync(d_params, host_params, sizeof(SimParams)));
  h_params = *host_params;
}

__global__ void QueryNearestBarycentricClothToObstacle_Kernel(Node** nodes,
                                                              BasicDeviceBVH bvh_dev,
                                                              SkinParam* skin_params,
                                                              int n_cloth_nodes) {
  GET_CUDA_ID(i, n_cloth_nodes);

  auto queryGarmentVertex = nodes[i]->x0;
  const int MAX_BUFFER_SIZE = 64;
  unsigned int buffer[MAX_BUFFER_SIZE];
  for (unsigned int j = 0; j < MAX_BUFFER_SIZE; ++j) {
    buffer[j] = 0xFFFFFFFF;
  }
  Scalar hit = SCALAR_MAX;
  int hit_idx = -1;
  Scalar hit_u, hit_v, hit_w;

  const auto nearest =
      QueryDevice(bvh_dev, queryGarmentVertex, distance_calculator(), hit_u, hit_v, hit_w);

  hit_idx = nearest.first;
  hit = nearest.second;

  skin_params[i].idx0 = bvh_dev.objects[hit_idx].idx1;
  skin_params[i].idx1 = bvh_dev.objects[hit_idx].idx2;
  skin_params[i].idx2 = bvh_dev.objects[hit_idx].idx3;
  skin_params[i].u = static_cast<smplx::Scalar>(hit_u);
  skin_params[i].v = static_cast<smplx::Scalar>(hit_v);
  skin_params[i].w = static_cast<smplx::Scalar>(hit_w);
}

void QueryNearestBarycentricClothToObstacle(std::shared_ptr<PhysicsMesh> cloth,
                                            SkinParam* skin_params, std::shared_ptr<BVH> bvh) {
  int n_cloth_nodes = cloth->nodes.size();
  auto bvh_dev = bvh->GetDeviceRepr();
  if (!bvh->IsActive())
    return;

  CUDA_CALL(QueryNearestBarycentricClothToObstacle_Kernel, n_cloth_nodes)
  (pointer(cloth->nodes), bvh_dev, skin_params, n_cloth_nodes);
  CUDA_CHECK_LAST();
}

__host__ Mat4 RotateX(Mat4& target, const Scalar& degree) {
  return glm::rotate(target, glm::radians(degree), Vector3(1, 0, 0));
}

__global__ void UpdateObstaclePredicted_Kernel(Node** nodes, smplx::Scalar* src_positions,
                                               Mat4 model_matrix, Mat4 align_matrix, int n_nodes) {
  GET_CUDA_ID(i, n_nodes);

  Vector3 src_position(static_cast<Scalar>(src_positions[i * 3]),
                       static_cast<Scalar>(src_positions[i * 3 + 1]),
                       static_cast<Scalar>(src_positions[i * 3 + 2]));
  nodes[i]->x = model_matrix * align_matrix * Vector4(src_position, 1);
}

void UpdateObstaclePredicted(Node** nodes, smplx::Points src_points, const int tgt_count,
                             Mat4& model_matrix, Scalar X_degree) {
  smplx::Scalar* src_positions = nullptr;
  cudaMalloc((void**)&src_positions, src_points.size() * sizeof(smplx::Scalar));
  cudaMemcpy(src_positions, src_points.data(), src_points.size() * sizeof(smplx::Scalar),
             cudaMemcpyHostToDevice);
  // Due to different coordinate system, all AMASS dada are rotated 90 degs
  // CCW on x-axis; we undo this rotation using the model matrix
  Mat4 target = Mat4(1);
  Mat4 align_matrix = RotateX(target, X_degree);

  CUDA_CALL(UpdateObstaclePredicted_Kernel, tgt_count)
  (nodes, src_positions, model_matrix, align_matrix, tgt_count);
  CUDA_CHECK_LAST();

  cudaFree(src_positions);
}

__global__ void updateX_Kernel(Node** nodes, Mat4 model_matrix, int n_nodes) {
  GET_CUDA_ID(id, n_nodes);

  nodes[id]->x = model_matrix * Vector4(nodes[id]->x, 1);
}

void UpdateX(Node** nodes, Mat4 model_matrix, int n_nodes) {
  CUDA_CALL(updateX_Kernel, n_nodes)
  (nodes, model_matrix, n_nodes);
  CUDA_CHECK_LAST();
}

__global__ void InitializePositions_Kernel(Node** nodes, const int start, const int count,
                                           const Mat4 model_matrix) {
  GET_CUDA_ID(id, count);
  nodes[start + id]->x0 = model_matrix * Vector4(nodes[start + id]->x0, 1);
}

void InitializePositions(std::shared_ptr<PhysicsMesh> physics_mesh, const int start, const int count,
                         Mat4& model_matrix) {
  ScopedTimerGPU timer("Solver_Initialize");
  CUDA_CALL(InitializePositions_Kernel, count)
  (pointer(physics_mesh->nodes), start, count, model_matrix);
  CUDA_CHECK_LAST();
}

__global__ void PredictPositions_Kernel(Node** nodes, const Scalar delta_time, int n_nodes) {
  GET_CUDA_ID(i, n_nodes);

  if (nodes[i]->inv_mass > EPSILON)
    nodes[i]->v += static_cast<Vector3>(d_params.gravity) * delta_time;
  nodes[i]->x = nodes[i]->x0 + nodes[i]->v * delta_time;
}

void PredictPositions(Node** nodes, const Scalar delta_time, int n_nodes) {
  ScopedTimerGPU timer("Solver_Predict");
  CUDA_CALL(PredictPositions_Kernel, n_nodes)
  (nodes, delta_time, n_nodes);
}

__global__ void ApplyDeltas_Kernel(Node** nodes, Vector3* deltas, int* delta_counts) {
  GET_CUDA_ID(id, d_params.num_particles);

  Scalar count = static_cast<Scalar>(delta_counts[id]);
  if (count > static_cast<Scalar>(0.0)) {
    nodes[id]->x += deltas[id] / count * static_cast<Scalar>(d_params.relaxation_factor);
    deltas[id] = Vector3(0);
    delta_counts[id] = 0;
  }
}

void ApplyDeltas(Node** nodes, Vector3* deltas, int* delta_counts) {
  ScopedTimerGPU timer("Solver_ApplyDeltas");
  CUDA_CALL(ApplyDeltas_Kernel, h_params.num_particles)
  (nodes, deltas, delta_counts);
}

__device__ Vector3 ComputeFriction(Vector3 correction, Vector3 rel_vel) {
  Vector3 friction = Vector3(0);
  Scalar correction_length = glm::length(correction);
  if (d_params.sdf_friction > static_cast<Scalar>(0) && correction_length > static_cast<Scalar>(0)) {
    Vector3 norm = correction / correction_length;

    Vector3 tan_vel = rel_vel - norm * glm::dot(rel_vel, norm);
    Scalar tan_length = glm::length(tan_vel);
    Scalar max_tan_length = correction_length * d_params.sdf_friction;

    friction = -tan_vel * MathFunctions::min(max_tan_length / tan_length, static_cast<Scalar>(1));
  }
  return friction;
}

__global__ void CollideParticles_Kernel(Vector3* deltas, int* delta_counts, Node** nodes,
                                        CONST(uint*) neighbors) {
  GET_CUDA_ID(id, d_params.num_particles);

  Vector3 position_delta = Vector3(0);
  int delta_count = 0;
  Vector3 pred_i = nodes[id]->x;
  Vector3 vel_i = (pred_i - nodes[id]->x0);
  Scalar w_i = nodes[id]->inv_mass;

  for (int neighbor = id; neighbor < d_params.num_particles * d_params.max_num_neighbors;
       neighbor += d_params.num_particles) {
    uint j = neighbors[neighbor];
    if (j > d_params.num_particles)
      break;

    Scalar w_j = nodes[j]->inv_mass;
    Scalar denom = w_i + w_j;
    if (denom <= 0)
      continue;

    Vector3 pred_j = nodes[j]->x;
    Vector3 diff = pred_i - pred_j;
    Scalar distance = glm::length(diff);
    if (distance >= d_params.particle_diameter)
      continue;

    Vector3 gradient = diff / (distance + EPSILON);
    Scalar lambda = (distance - d_params.particle_diameter) / denom;
    Vector3 common = lambda * gradient;

    delta_count++;
    position_delta -= w_i * common;

    Vector3 relative_velocity = vel_i - (pred_j - nodes[j]->x0);
    Vector3 friction = ComputeFriction(common, relative_velocity);
    position_delta += w_i * friction;
  }
  deltas[id] = position_delta;
  delta_counts[id] = delta_count;
}

void CollideParticles(Vector3* deltas, int* delta_counts, Node** nodes, CONST(uint*) neighbors) {

  ScopedTimerGPU timer("Solver_CollideParticles");
  CUDA_CALL(CollideParticles_Kernel, h_params.num_particles)
  (deltas, delta_counts, nodes, neighbors);
  CUDA_CALL(ApplyDeltas_Kernel, h_params.num_particles)
  (nodes, deltas, delta_counts);
}

__global__ void Finalize_Kernel(Node** nodes, const Scalar delta_time) {
  GET_CUDA_ID(id, d_params.num_particles);

  Vector3 new_pos = nodes[id]->x;
  Vector3 raw_vel = (new_pos - nodes[id]->x0) / delta_time;
  nodes[id]->v = raw_vel * (1 - d_params.damping * delta_time);
  nodes[id]->x0 = new_pos;
}

void Finalize(Node** nodes, const Scalar delta_time) {
  ScopedTimerGPU timer("Solver_Finalize");
  CUDA_CALL(Finalize_Kernel, h_params.num_particles)
  (nodes, delta_time);
  CUDA_CHECK_LAST();
}

__global__ void UpdateSkinnedVertex_Kernel(Node** nodes, CONST(smplx::Scalar*) skinned_verts,
                                           const unsigned int n_nodes, Mat4 align_matrix) {
  GET_CUDA_ID(id, n_nodes);
  Vector3 skinned_vertex = align_matrix * Vector4(static_cast<Scalar>(skinned_verts[id * 3]),
                                                 static_cast<Scalar>(skinned_verts[id * 3 + 1]),
                                                 static_cast<Scalar>(skinned_verts[id * 3 + 2]), 1);
  nodes[id]->x0 = skinned_vertex;
}

void UpdateSkinnedVertices(std::shared_ptr<PhysicsMesh> physics_mesh,
                           CONST(smplx::Scalar*) skinned_verts, Scalar X_degree) {
  int n_cloth_nodes = physics_mesh->nodes.size();
  Mat4 align_martrix = MathFunctions::RotateX(Mat4(1), X_degree);
  CUDA_CALL(UpdateSkinnedVertex_Kernel, n_cloth_nodes)
  (pointer(physics_mesh->nodes), skinned_verts, n_cloth_nodes, align_martrix);
}

__global__ void UpdatePredicted_Kernel(Vector3* tgt_positions, Node** src_nodes) {
  GET_CUDA_ID(id, d_params.num_particles);
  tgt_positions[id] = src_nodes[id]->x;
}

void UpdatePredicted(Vector3* tgt_positions, Node** src_nodes) {
  CUDA_CALL(UpdatePredicted_Kernel, h_params.num_particles)
  (tgt_positions, src_nodes);
}

__global__ void UpdateVertexKernel(Vector3* tgt_positions, Node** nodes, uint n_nodes) {
  GET_CUDA_ID(i, n_nodes);

  tgt_positions[i] = nodes[i]->x0;
}

void UpdateVertices(Vector3* tgt_positions, std::shared_ptr<PhysicsMesh> physics_mesh) {
  int n_nodes = physics_mesh->nodes.size();
  CUDA_CALL(UpdateVertexKernel, n_nodes)
  (tgt_positions, pointer(physics_mesh->nodes), n_nodes);
  CUDA_CHECK_LAST();
}

__global__ void SetupNode_Kernel(Node* nodes, Node** nodes2, Vector3* positions, Vector3* predicted,
                                 Scalar* inv_masses, int num_verts) {
  GET_CUDA_ID(id, num_verts);

  Node& node = nodes[id];

  node.x0 = positions[id];
  node.x = predicted[id];
  node.x1 = predicted[id];
  node.v = Vector3(0);

  node.inv_mass = inv_masses[id];
  node.index = id;
  node.area = static_cast<Scalar>(0.0);
  node.is_free = true;
  node.min_index = -1;
  node.color = 0;
  node.removed = false;

  nodes2[id] = &nodes[id];
}

void SetupNodes(Node* nodes, Node** nodes2, Vector3* positions, Vector3* predicted,
                Scalar* inv_masses, int num_verts) {
  CUDA_CALL(SetupNode_Kernel, num_verts)
  (nodes, nodes2, positions, predicted, inv_masses, num_verts);
  CUDA_CHECK_LAST();
}

__global__ void FinalizeNodes_Kernel(Node* nodes, Vector3* predicted, int num_verts) {
  GET_CUDA_ID(id, num_verts);

  Node& node = nodes[id];

  predicted[id] = node.x;
}

void FinalizeNodes(Node* nodes, Vector3* predicted, int num_verts) {
  CUDA_CALL(FinalizeNodes_Kernel, num_verts)
  (nodes, predicted, num_verts);
  CUDA_CHECK_LAST();
}

void BuildDeviceEFAdjacency(std::vector<std::set<uint>>& raw_nb_ef,
                            thrust::device_vector<uint>& nb_ef,
                            thrust::device_vector<uint>& nb_ef_prefix) {
  printf("Build device EF adjacency\n");
  uint num_edges = raw_nb_ef.size();

  thrust::host_vector<uint> h_nbEF;
  thrust::host_vector<uint> h_nb_ef_prefix;

  h_nb_ef_prefix.push_back(0);
  uint idx_offset = 0u;
  for (uint e_idx = 0u; e_idx < num_edges; e_idx++) {
    idx_offset += raw_nb_ef[e_idx].size();
    h_nb_ef_prefix.push_back(idx_offset);
    for (auto f_idx : raw_nb_ef[e_idx]) {
      h_nbEF.push_back(f_idx);
    }
  }

  nb_ef.insert(nb_ef.end(), h_nbEF.begin(), h_nbEF.end());
  nb_ef_prefix.insert(nb_ef_prefix.end(), h_nb_ef_prefix.begin(), h_nb_ef_prefix.end());
}

void BuildDeviceVFAdjacency(std::vector<std::set<uint>>& raw_nb_vf,
                            thrust::device_vector<uint>& nb_vf,
                            thrust::device_vector<uint>& nb_vf_prefix) {
  printf("Build device VF adjacency\n");
  uint num_verts = raw_nb_vf.size();

  thrust::host_vector<uint> h_nb_vf;
  thrust::host_vector<uint> h_nb_vf_prefix;

  h_nb_vf_prefix.push_back(0);
  uint idx_offset = 0u;
  for (uint v_idx = 0u; v_idx < num_verts; v_idx++) {
    uint numNeighborFaces = raw_nb_vf[v_idx].size();
    idx_offset += numNeighborFaces;
    h_nb_vf_prefix.push_back(idx_offset);
  }

  idx_offset = 0u;
  for (uint v_idx = 0u; v_idx < num_verts; v_idx++) {
    for (auto f_idx : raw_nb_vf[v_idx]) {
      h_nb_vf.push_back(f_idx);
    }
  }
  nb_vf.insert(nb_vf.end(), h_nb_vf.begin(), h_nb_vf.end());
  nb_vf_prefix.insert(nb_vf_prefix.end(), h_nb_vf_prefix.begin(), h_nb_vf_prefix.end());
}

void BuildVFAdjacency(std::vector<std::set<uint>>& h_nb_vf, std::shared_ptr<PhysicsMesh> physics_mesh,
                      thrust::device_vector<uint>& nb_vf, thrust::device_vector<uint>& nb_vf_prefix) {
  printf("Build adjacency\n");
  thrust::host_vector<uint> h_indices = physics_mesh->HostIndices();

  uint numVerts = physics_mesh->NumNodes();
  uint num_faces = physics_mesh->NumFaces();

  h_nb_vf.resize(numVerts);

  for (uint f_idx = 0u; f_idx < num_faces; f_idx++) {
    uint f_idx_offset = f_idx * 3u;
    uint v_idx1 = h_indices[f_idx_offset + 0u];
    uint v_idx2 = h_indices[f_idx_offset + 1u];
    uint v_idx3 = h_indices[f_idx_offset + 2u];

    h_nb_vf[v_idx1].insert(f_idx);
    h_nb_vf[v_idx2].insert(f_idx);
    h_nb_vf[v_idx3].insert(f_idx);
  }

  BuildDeviceVFAdjacency(h_nb_vf, nb_vf, nb_vf_prefix);
}

void BuildVFAdjacency(std::shared_ptr<PhysicsMesh> physics_mesh, thrust::device_vector<uint>& nb_vf,
                      thrust::device_vector<uint>& nb_vf_prefix) {
  printf("Build VF adjacency\n");
  thrust::host_vector<uint> h_indices = physics_mesh->HostIndices();

  printf("node size: %d, face size: %d\n", physics_mesh->nodes.size(), h_indices.size() / 3);

  uint num_verts = physics_mesh->NumNodes();
  uint num_faces = physics_mesh->NumFaces();

  printf("node size: %d, face size: %d\n", num_verts, num_faces);

  std::vector<std::set<uint>> h_nb_vf(num_verts);

  for (uint f_idx = 0u; f_idx < num_faces; f_idx++) {
    uint f_idx_offset = f_idx * 3u;
    uint v_idx1 = h_indices[f_idx_offset + 0u];
    uint v_idx2 = h_indices[f_idx_offset + 1u];
    uint v_idx3 = h_indices[f_idx_offset + 2u];

    h_nb_vf[v_idx1].insert(f_idx);
    h_nb_vf[v_idx2].insert(f_idx);
    h_nb_vf[v_idx3].insert(f_idx);
  }

  BuildDeviceVFAdjacency(h_nb_vf, nb_vf, nb_vf_prefix);
}
}  // namespace XRTailor