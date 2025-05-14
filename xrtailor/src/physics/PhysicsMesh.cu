#include <xrtailor/physics/PhysicsMesh.cuh>

#include "cuda_gl_interop.h"
#include <thrust/copy.h>

#include <xrtailor/core/DeviceHelper.cuh>
#include <xrtailor/physics/PhysicsMeshHelper.cuh>

//#define PHYSICS_MESH_DEBUG_LOG

namespace XRTailor {

template <typename T>
RegisteredBuffer<T>::RegisteredBuffer() {}

template <typename T>
RegisteredBuffer<T>::~RegisteredBuffer() {
  Destroy();
}

template <typename T>
T* RegisteredBuffer<T>::Data() const {
  return buffer_;
}

template <typename T>
RegisteredBuffer<T>::operator T*() const {
  return buffer_;
}

template <typename T>
T& RegisteredBuffer<T>::operator[](size_t index) {
  assert(buffer_cpu_);
  assert(index < count_);
  return buffer_cpu_[index];
}

template <typename T>
size_t RegisteredBuffer<T>::Size() const {
  return count_;
}

template <typename T>
void RegisteredBuffer<T>::Destroy() {
  if (cuda_vbo_resource_ != nullptr) {
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource_));
  }
  if (buffer_cpu_) {
    cudaFree(buffer_cpu_);
  }
  count_ = 0;
  buffer_ = nullptr;
  buffer_cpu_ = nullptr;
  cuda_vbo_resource_ = nullptr;
}

template <typename T>
void RegisteredBuffer<T>::RegisterBuffer(GLuint vbo) {
  checkCudaErrors(
      cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource_, vbo, cudaGraphicsRegisterFlagsNone));
  // map (example 'gl_cuda_interop_pingpong_st' says map and unmap only needs to be done once)
  checkCudaErrors(cudaGraphicsMapResources(1, &cuda_vbo_resource_, 0));
  checkCudaErrors(
      cudaGraphicsResourceGetMappedPointer((void**)&buffer_, &num_bytes_, cuda_vbo_resource_));
  count_ = num_bytes_ / sizeof(T);

  // unmap
  checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource_, 0));
}

//__global__ void updateRenderingData_Kernel(const Face* const* faces, RenderableVertex* vertices, int n_faces)
//{
//	GET_CUDA_ID(i, n_faces);
//
//	const Face* face = faces[i];
//	for (int j = 0; j < 3; j++)
//	{
//		Vertex* vertex = face->vertices[j];
//		Node* node = vertex->node;
//		int idx = 3 * i + j;
//		vertices[idx].x = node->x0;
//		vertices[idx].n = node->n;
//		vertices[idx].uv = vertex->uv;
//	}
//}
__global__ void updateRenderingData_Kernel(const Node* const* nodes, RenderableVertex* vertices,
                                           int n_nodes) {
  GET_CUDA_ID(i, n_nodes);
  const Node* node = nodes[i];
  // XXX: we employ single-precision float for rendering. Hence glm::vec3 is used here
  vertices[i].x = static_cast<glm::vec3>(node->x0);
  vertices[i].n = static_cast<glm::vec3>(node->n);
  // TODO: map uv correctly
  vertices[i].uv = glm::vec2(0);
}

PhysicsMesh::PhysicsMesh(std::shared_ptr<MemoryPool> memory_pool)
    : memory_pool_(memory_pool),
      node_offset_(0),
      vertex_offset_(0),
      edge_offset_(0),
      face_offset_(0) {}

void PhysicsMesh::Destroy() {
  v_buffer_.clear();
  r_buffers_.clear();
}

PhysicsMesh::~PhysicsMesh() {
  Destroy();
}

__global__ void FinalizeBuffer_Kernel(RenderableVertex* tgt_verts, RenderableVertex* src_verts,
                                      int offset, int n_new_verts) {
  GET_CUDA_ID(i, n_new_verts);

  tgt_verts[i + offset] = src_verts[i];
}

void PhysicsMesh::RegisterNewBuffer(GLuint vbo) {
  auto rbuf = std::make_shared<RegisteredBuffer<RenderableVertex>>();
  rbuf->RegisterBuffer(vbo);
  r_buffers_.push_back(rbuf);

  size_t last = offsets_.size() - 1;
  size_t offset = offsets_.empty() ? 0 : offsets_[last] + r_buffers_[last]->Size();
  offsets_.push_back(offset);
#ifdef PHYSICS_MESH_DEBUG_LOG
  std::cout << "offset: " << offset << std::endl;
  std::cout << "face size: " << faces.size() << std::endl;
  std::cout << "rbuf size: " << rbuf->size() << std::endl;
#endif  // PHYSICS_MESH_DEBUG_LOG

  // copy from rbuffers to vbuffer
  v_buffer_.resize(v_buffer_.size() + rbuf->Size());
  int nNewVerts = rbuf->Size();
  CUDA_CALL(FinalizeBuffer_Kernel, nNewVerts)
  (pointer(v_buffer_), rbuf->Data(), offset, nNewVerts);
  CUDA_CHECK_LAST();
}

void PhysicsMesh::RegisterNewMesh(const std::vector<Vector3>& positions,
                                  //const std::vector<glm::vec2>& uvs,
                                  const std::vector<uint>& indices,
                                  //const std::vector<uint>& uv_indices,
                                  const thrust::host_vector<uint>& h_edges,
                                  const thrust::host_vector<uint>& h_fe_indices,
                                  int submesh_vertex_offset, const bool& is_cloth) {
  this->submesh_vertex_offsets_.push_back(submesh_vertex_offset);
  this->submesh_indices_.push_back(indices);
#ifdef PHYSICS_MESH_DEBUG_LOG
  std::cout << "Register new mesh" << std::endl;
#endif  // PHYSICS_MESH_DEBUG_LOG
  thrust::device_vector<Vector3> d_x = positions;
  //thrust::device_vector<glm::vec2> d_uv = uvs;
  thrust::device_vector<uint> d_x_indices = indices;
  //thrust::device_vector<uint> d_uvIndices = uv_indices;
  thrust::device_vector<uint> d_edges = h_edges;
  thrust::device_vector<uint> d_feIndices = h_fe_indices;

  int n_nodes = d_x.size();
  //int nVertices = d_uv.size();
  int n_faces = d_x_indices.size() / 3;
  int n_edges = h_edges.size() / 2;

  int prev_num_particles = memory_pool_->NodeOffset();
  //int prev_num_vertices = m_memoryPool->vertexOffset();
  int prev_num_edges = memory_pool_->EdgeOffset();
  int prev_num_faces = memory_pool_->FaceOffset();

#ifdef PHYSICS_MESH_DEBUG_LOG
  std::cout << "Prev #node " << prev_num_particles
            << ", "
               //"#vert " << prev_num_vertices << ", "
               "#edge "
            << prev_num_edges
            << ", "
               "#face "
            << prev_num_faces << std::endl;
#endif  // PHYSICS_MESH_DEBUG_LOG

  int new_num_particles = n_nodes;
  //int new_num_vertices = nVertices;
  int new_num_faces = n_faces;
  int new_num_edges = n_edges;

  this->num_submesh_vertices_.push_back(n_nodes);

#ifdef PHYSICS_MESH_DEBUG_LOG
  std::cout << "New #node " << new_num_particles
            << ", "
               //"#vert " << new_num_vertices << ", "
               "#edge "
            << new_num_edges
            << ", "
               "#face "
            << new_num_faces << std::endl;
#endif  // PHYSICS_MESH_DEBUG_LOG
  checkCudaErrors(cudaDeviceSynchronize());

  nodes.resize(nodes.size() + new_num_particles);
  Node** nodes_pointer = pointer(nodes);
  InitializeNodes(pointer(d_x), is_cloth, nodes_pointer, memory_pool_->CreateNodes(new_num_particles),
                  node_offset_, prev_num_particles, new_num_particles);
#ifdef PHYSICS_MESH_DEBUG_LOG
  std::cout << "Create node done" << std::endl;
#endif  // PHYSICS_MESH_DEBUG_LOG
  checkCudaErrors(cudaDeviceSynchronize());

  //vertices.resize(vertices.size() + new_num_vertices);
  //Vertex** verticesPointer = pointer(vertices);
  //initializeVertices(pointer(d_uv), verticesPointer, m_memoryPool->createVertices(new_num_vertices),
  //	m_vertex_offset, prev_num_vertices, new_num_vertices);
  ////std::cout << "Create vertex done" << std::endl;
  //checkCudaErrors(cudaDeviceSynchronize());

  edges.resize(edges.size() + new_num_edges);
  Edge** edges_pointer = pointer(edges);
  InitializeEdges(pointer(d_edges), nodes_pointer, edges_pointer,
                  memory_pool_->CreateEdges(new_num_edges), edge_offset_, prev_num_edges, new_num_edges);
#ifdef PHYSICS_MESH_DEBUG_LOG
  checkCudaErrors(cudaDeviceSynchronize());
  std::cout << "Create edge done" << std::endl;
#endif  // PHYSICS_MESH_DEBUG_LOG

  faces.resize(faces.size() + new_num_faces);
  Face** faces_pointer = pointer(faces);
  InitializeFaces(pointer(d_x_indices), pointer(d_feIndices), nodes_pointer, edges_pointer,
                  faces_pointer, memory_pool_->CreateFaces(new_num_faces), node_offset_,
                  edge_offset_, face_offset_, prev_num_particles, prev_num_edges, prev_num_faces,
                  new_num_faces);
#ifdef PHYSICS_MESH_DEBUG_LOG
  checkCudaErrors(cudaDeviceSynchronize());
  std::cout << "Create face done" << std::endl;
#endif  // PHYSICS_MESH_DEBUG_LOG
  UpdateFaceGeometries(faces_pointer, face_offset_, prev_num_faces, new_num_faces);

#ifdef PHYSICS_MESH_DEBUG_LOG
  checkCudaErrors(cudaDeviceSynchronize());
  std::cout << "Create face geometry done" << std::endl;
#endif  // PHYSICS_MESH_DEBUG_LOG
  UpdateNodeGeometriesLocal(nodes_pointer, faces_pointer, node_offset_, face_offset_,
                            prev_num_particles, new_num_particles, prev_num_faces, new_num_faces);
#ifdef PHYSICS_MESH_DEBUG_LOG
  checkCudaErrors(cudaDeviceSynchronize());
  std::cout << "Create node geometry done" << std::endl;
#endif  // PHYSICS_MESH_DEBUG_LOG

#ifdef PHYSICS_MESH_DEBUG_LOG
  checkCudaErrors(cudaDeviceSynchronize());
  std::cout << "Memory pool offset, "
            << "#node " << memory_pool_->NodeOffset()
            << ", "
            //<< "#vert " << m_memoryPool->vertexOffset() << ", "
            << "#edge " << memory_pool_->EdgeOffset() << ", "
            << "#face " << memory_pool_->FaceOffset() << std::endl;
#endif  // PHYSICS_MESH_DEBUG_LOG
}

void PhysicsMesh::UpdateIndices(int prev_num_nodes, int new_num_nodes, int prev_num_vertices,
                                int new_num_vertices) {
  UpdateNodeIndices(pointer(nodes), node_offset_, prev_num_nodes, new_num_nodes);
}

void PhysicsMesh::Sync() {
  CUDA_CALL(updateRenderingData_Kernel, nodes.size())
  (pointer(nodes), pointer(v_buffer_), nodes.size());
  CUDA_CHECK_LAST();

  // copy from vbuffer to rbuffers
  for (int i = 0; i < r_buffers_.size(); i++) {
#ifdef PHYSICS_MESH_DEBUG_LOG
    std::cout << "vbuffer to rbuffer " << i << " offset " << offsets_[i] << std::endl;
#endif  // PHYSICS_MESH_DEBUG_LOG
    cudaMemcpy(r_buffers_[i]->Data(), pointer(v_buffer_) + offsets_[i],
               r_buffers_[i]->Size() * sizeof(RenderableVertex), cudaMemcpyDefault);
  }
}

std::vector<size_t> PhysicsMesh::GetOffsets() {
  return offsets_;
}

thrust::host_vector<int> PhysicsMesh::NumSubmeshVertices() const {
  return num_submesh_vertices_;
}

thrust::host_vector<int> PhysicsMesh::SubmeshVertexOffsets() const {
  return submesh_vertex_offsets_;
}

thrust::host_vector<thrust::host_vector<int>> PhysicsMesh::SubmeshIndices() const {
  return submesh_indices_;
}

void PhysicsMesh::UpdateNodeGeometries() {
  UpdateNodeGeometriesGlobal(pointer(nodes), pointer(faces), nodes.size(), faces.size());
}

void PhysicsMesh::UpdateNormals() {
  UpdateNodeNormals(pointer(nodes), pointer(faces), nodes.size(), faces.size());
}

void PhysicsMesh::UpdateMidstepNormals() {
  UpdateNodeMidstepNormals(pointer(nodes), pointer(faces), nodes.size(), faces.size());
}

__global__ void MapPositions_Kernel(Node** nodes, Vector3* positions, int n_nodes) {
  GET_CUDA_ID(i, n_nodes);

  positions[i] = nodes[i]->x0;
}

__global__ void MapPredicted_Kernel(Node** nodes, Vector3* predicted, int n_nodes) {
  GET_CUDA_ID(i, n_nodes);

  predicted[i] = nodes[i]->x;
}

__global__ void MapNormals_Kernel(Node** nodes, Vector3* normals, int n_nodes) {
  GET_CUDA_ID(i, n_nodes);

  normals[i] = nodes[i]->n;
}

thrust::host_vector<Vector3> PhysicsMesh::HostPositions() {
  int n_nodes = nodes.size();
  thrust::host_vector<Vector3> ans(n_nodes);
  thrust::device_vector<Vector3> d_ans(n_nodes);

  CUDA_CALL(MapPositions_Kernel, n_nodes)
  (pointer(nodes), pointer(d_ans), n_nodes);
  CUDA_CHECK_LAST();

  thrust::copy(d_ans.begin(), d_ans.end(), ans.begin());

  return ans;
}

thrust::host_vector<Vector3> PhysicsMesh::HostPredicted() {
  int n_nodes = nodes.size();
  thrust::host_vector<Vector3> ans(n_nodes);
  thrust::device_vector<Vector3> d_ans(n_nodes);

  CUDA_CALL(MapPredicted_Kernel, n_nodes)
  (pointer(nodes), pointer(d_ans), n_nodes);
  CUDA_CHECK_LAST();

  thrust::copy(d_ans.begin(), d_ans.end(), ans.begin());

  return ans;
}

thrust::device_vector<Vector3> PhysicsMesh::DevicePositions() {
  int n_nodes = nodes.size();
  thrust::device_vector<Vector3> d_ans(n_nodes);

  CUDA_CALL(MapPositions_Kernel, n_nodes)
  (pointer(nodes), pointer(d_ans), n_nodes);
  CUDA_CHECK_LAST();

  return d_ans;
}

thrust::host_vector<Vector3> PhysicsMesh::HostNormals() {
  int n_nodes = nodes.size();
  thrust::host_vector<Vector3> ans(n_nodes);
  thrust::device_vector<Vector3> d_ans(n_nodes);

  CUDA_CALL(MapNormals_Kernel, n_nodes)
  (pointer(nodes), pointer(d_ans), n_nodes);
  CUDA_CHECK_LAST();

  thrust::copy(d_ans.begin(), d_ans.end(), ans.begin());

  return ans;
}

__global__ void MapIndices_Kernel(Face** faces, uint* indices, int n_faces) {
  GET_CUDA_ID(i, n_faces);

  Face* face = faces[i];
  indices[i * 3] = face->nodes[0]->index;
  indices[i * 3 + 1] = face->nodes[1]->index;
  indices[i * 3 + 2] = face->nodes[2]->index;
}

//__global__ void mapUVIndices_Kernel
//(
//	Face** faces,
//	uint* indices,
//	int n_faces
//)
//{
//	GET_CUDA_ID(i, n_faces);
//
//	const Face* face = faces[i];
//	indices[i * 3 + 0] = face->vertices[0]->index;
//	indices[i * 3 + 1] = face->vertices[1]->index;
//	indices[i * 3 + 2] = face->vertices[2]->index;
//}

thrust::host_vector<uint> PhysicsMesh::HostIndices() {
  checkCudaErrors(cudaDeviceSynchronize());

  int n_faces = faces.size();
  thrust::device_vector<uint> d_ans(n_faces * 3);
  CUDA_CALL(MapIndices_Kernel, n_faces)
  (pointer(faces), pointer(d_ans), n_faces);

  thrust::host_vector<uint> ans = std::move(d_ans);

  return ans;
}

//thrust::host_vector<uint> PhysicsMesh::h_uv_indices()
//{
//	checkCudaErrors(cudaDeviceSynchronize());
//
//	int n_faces = faces.size();
//	thrust::device_vector<uint> d_ans(n_faces * 3);
//	CUDA_CALL(mapUVIndices_Kernel, n_faces)
//		(pointer(faces), pointer(d_ans), n_faces);
//
//	thrust::host_vector<uint> ans = std::move(d_ans);
//
//	return ans;
//
//}

__global__ void mapColors_Kernel(Node** nodes, uint* colors, int n_nodes) {
  GET_CUDA_ID(i, n_nodes);

  colors[i] = nodes[i]->color;
}

thrust::host_vector<uint> PhysicsMesh::HostColors() {
  checkCudaErrors(cudaDeviceSynchronize());

  int n_nodes = nodes.size();
  thrust::host_vector<uint> ans(n_nodes);
  thrust::device_vector<uint> d_ans(n_nodes);

  CUDA_CALL(mapColors_Kernel, n_nodes)
  (pointer(nodes), pointer(d_ans), n_nodes);

  thrust::copy(d_ans.begin(), d_ans.end(), ans.begin());

  return ans;
}

thrust::device_vector<uint> PhysicsMesh::DeviceIndices() {
  int n_faces = faces.size();
  thrust::device_vector<uint> d_ans(n_faces * 3);

  CUDA_CALL(MapIndices_Kernel, n_faces)
  (pointer(faces), pointer(d_ans), n_faces);

  return d_ans;
}

__global__ void GetInvMass_Kernel(Node** nodes, Scalar* inv_mass, int node_index) {
  GET_CUDA_ID(i, 1);

  inv_mass[0] = nodes[node_index]->inv_mass;
}

__global__ void SetInvMass_Kernel(Node** nodes, Scalar inv_mass, int node_index) {
  GET_CUDA_ID(i, 1);

  nodes[node_index]->inv_mass = inv_mass;
}

__global__ void GetX0_Kernel(Node** nodes, Vector3* x0, int node_index) {
  GET_CUDA_ID(i, 1);

  x0[0] = nodes[node_index]->x0;
}

__global__ void SetX0_Kernel(Node** nodes, Vector3 pos, int node_index) {
  GET_CUDA_ID(i, 1);

  nodes[node_index]->x0 = pos;
}

void PhysicsMesh::SetX0(Vector3 pos, int node_index) {
  CUDA_CALL(SetX0_Kernel, 1)
  (pointer(nodes), pos, node_index);
  CUDA_CHECK_LAST();
}

Vector3 PhysicsMesh::GetX0(int node_index) {
  Vector3* d_pos;
  checkCudaErrors(cudaMalloc(&d_pos, sizeof(Vector3)));

  CUDA_CALL(GetX0_Kernel, 1)
  (pointer(nodes), d_pos, node_index);
  CUDA_CHECK_LAST();

  Vector3 h_pos;
  checkCudaErrors(cudaMemcpy(&h_pos, d_pos, sizeof(Vector3), cudaMemcpyDeviceToHost));
  cudaFree(d_pos);

  return h_pos;
}

Scalar PhysicsMesh::GetInvMass(int node_index) {
  Scalar* d_invMass;
  checkCudaErrors(cudaMalloc(&d_invMass, sizeof(Scalar)));

  CUDA_CALL(GetInvMass_Kernel, 1)
  (pointer(nodes), d_invMass, node_index);
  CUDA_CHECK_LAST();

  Scalar h_invMass;

  checkCudaErrors(cudaMemcpy(&h_invMass, d_invMass, sizeof(Scalar), cudaMemcpyDeviceToHost));
  cudaFree(d_invMass);

  return h_invMass;
}

void PhysicsMesh::SetInvMass(Scalar inv_mass, int node_index) {
  CUDA_CALL(SetInvMass_Kernel, 1)
  (pointer(nodes), inv_mass, node_index);
  CUDA_CHECK_LAST();
}

__global__ void setFree_Kernel(Node** nodes, bool is_free, int node_index) {
  GET_CUDA_ID(i, 1);

  nodes[node_index]->is_free = is_free;
}

void PhysicsMesh::SetFree(bool is_free, int node_index) {
  CUDA_CALL(setFree_Kernel, 1)
  (pointer(nodes), is_free, node_index);
  CUDA_CHECK_LAST();
}

__global__ void SetVelocity_Kernel(Node** nodes, Vector3 velocity, int node_index) {
  GET_CUDA_ID(i, 1);

  nodes[node_index]->v = velocity;
}

__global__ void GetVelocity_Kernel(Node** nodes, Vector3* v, int node_index) {
  GET_CUDA_ID(i, 1);

  v[0] = nodes[node_index]->v;
}

void PhysicsMesh::SetVelocity(Vector3 velocity, int node_index) {
  CUDA_CALL(SetVelocity_Kernel, 1)
  (pointer(nodes), velocity, node_index);
  CUDA_CHECK_LAST();
}

Vector3 PhysicsMesh::GetVelocity(int node_index) {
  Vector3* d_v;
  checkCudaErrors(cudaMalloc(&d_v, sizeof(Vector3)));

  CUDA_CALL(GetVelocity_Kernel, 1)
  (pointer(nodes), d_v, node_index);
  CUDA_CHECK_LAST();

  Vector3 h_v;
  checkCudaErrors(cudaMemcpy(&h_v, d_v, sizeof(Vector3), cudaMemcpyDeviceToHost));
  cudaFree(d_v);

  return h_v;
}

__global__ void ResetVelocity_Kernel(Node** nodes, int n_nodes) {
  GET_CUDA_ID(i, n_nodes);
  nodes[i]->v = Vector3(static_cast<Scalar>(0.0));
}

void PhysicsMesh::ResetVelocities() {
  int n_nodes = nodes.size();
  CUDA_CALL(ResetVelocity_Kernel, n_nodes)
  (pointer(nodes), n_nodes);
  CUDA_CHECK_LAST();
}

__global__ void resetX_Kernel(Node** nodes, int n_nodes) {
  GET_CUDA_ID(i, n_nodes);
  nodes[i]->x = Vector3(static_cast<Scalar>(0.0));
}

void PhysicsMesh::ResetX() {
  int n_nodes = nodes.size();
  CUDA_CALL(resetX_Kernel, n_nodes)
  (pointer(nodes), n_nodes);
  CUDA_CHECK_LAST();
}

__global__ void predictedToPosition_Kernel(Node** nodes, int n_nodes) {
  GET_CUDA_ID(i, n_nodes);

  nodes[i]->x0 = nodes[i]->x;
}

void PhysicsMesh::PredictedToPositions() {
  CUDA_CALL(predictedToPosition_Kernel, nodes.size())
  (pointer(nodes), nodes.size());
  CUDA_CHECK_LAST();
}

__global__ void Interpolate_Kernel(Node** nodes, int n_nodes, int step, int num_steps) {
  GET_CUDA_ID(i, n_nodes);
  Scalar alpha = (static_cast<Scalar>(step) + static_cast<Scalar>(1.0)) / num_steps;
  nodes[i]->x1 = (static_cast<Scalar>(1.0) - alpha) * nodes[i]->x0 + alpha * nodes[i]->x;
}

void PhysicsMesh::Interpolate(int step, int num_steps) {
  CUDA_CALL(Interpolate_Kernel, nodes.size())
  (pointer(nodes), nodes.size(), step, num_steps);
  CUDA_CHECK_LAST();
}

void PhysicsMesh::SetPtrOffset(int node_offset, int edge_offset, int face_offset) {
  node_offset_ = node_offset;
  edge_offset_ = edge_offset;
  face_offset_ = face_offset;
}

int PhysicsMesh::NumNodes() {
  return this->nodes.size();
}

//int PhysicsMesh::n_verts()
//{
//	return this->vertices.size();
//}

int PhysicsMesh::NumEdges() {
  return this->edges.size();
}

int PhysicsMesh::NumFaces() {
  return this->faces.size();
}

__global__ void RegisterRTri_Kernel(const uint* rTris, Face** faces, int new_num_faces) {
  GET_CUDA_ID(i, new_num_faces);

  faces[i]->r_tri = rTris[i];
}

void PhysicsMesh::RegisterRTris(const thrust::device_vector<uint>& r_tris) {
  int n_faces = NumFaces();
  CUDA_CALL(RegisterRTri_Kernel, n_faces)
  (pointer(r_tris), pointer(faces), n_faces);

  CUDA_CHECK_LAST();
}

__global__ void RegisterEFAdjacency_Kernel(Edge** edges, Face** faces, const uint* nb_ef,
                                           const uint* nb_ef_prefix, int n_edges) {
  GET_CUDA_ID(eid, n_edges);

  Edge* edge = edges[eid];

  uint n_nb_fs = nb_ef_prefix[eid + 1] - nb_ef_prefix[eid];
  uint nb_fs_idx_offset = nb_ef_prefix[eid];

  for (int i = 0; i < n_nb_fs; i++) {
    edge->adjacents[i] = faces[nb_ef[nb_fs_idx_offset + i]];
  }
}

void PhysicsMesh::RegisterEFAdjacency(const thrust::device_vector<uint>& nb_ef,
                                      const thrust::device_vector<uint>& nb_ef_prefix) {
  int n_edges = nb_ef_prefix.size() - 1;
  CUDA_CALL(RegisterEFAdjacency_Kernel, n_edges)
  (pointer(edges), pointer(faces), pointer(nb_ef), pointer(nb_ef_prefix), n_edges);
  CUDA_CHECK_LAST();
  checkCudaErrors(cudaDeviceSynchronize());
}

}  // namespace XRTailor