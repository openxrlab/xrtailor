#pragma once

#include <iostream>
#include <vector>
#include <memory>

#include <vector_types.h>
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <xrtailor/core/Common.cuh>
#include <xrtailor/core/Common.hpp>
#include <xrtailor/memory/Node.cuh>
#include <xrtailor/memory/Face.cuh>
#include <xrtailor/memory/Edge.cuh>
#include <xrtailor/memory/RenderableVertex.cuh>
#include <xrtailor/memory/MemoryPool.cuh>

namespace XRTailor {

template <class T>
class RegisteredBuffer {
 public:
  RegisteredBuffer();

  RegisteredBuffer(const RegisteredBuffer&) = delete;

  RegisteredBuffer& operator=(const RegisteredBuffer&) = delete;

  ~RegisteredBuffer();

  T* Data() const;

  operator T*() const;

  T& operator[](size_t index);

  size_t Size() const;

  void Destroy();

  // CUDA interop with OpenGL
  void RegisterBuffer(GLuint vbo);

 private:
  size_t count_ = 0;
  size_t num_bytes_ = 0;
  T* buffer_ = nullptr;
  T* buffer_cpu_ = nullptr;
  struct cudaGraphicsResource* cuda_vbo_resource_ = nullptr;
};

class PhysicsMesh {
 public:
  PhysicsMesh(std::shared_ptr<MemoryPool> memory_pool);

  ~PhysicsMesh();

  void Destroy();

  void RegisterNewBuffer(GLuint vbo);

  void RegisterNewMesh(const std::vector<Vector3>& positions, const std::vector<uint>& indices,
                       const thrust::host_vector<uint>& h_edges,
                       const thrust::host_vector<uint>& h_fe_indices, int submesh_vertex_offset,
                       const bool& is_cloth = true);

  void RegisterRTris(const thrust::device_vector<uint>& rTris);

  void RegisterEFAdjacency(const thrust::device_vector<uint>& nb_ef,
                           const thrust::device_vector<uint>& nb_ef_prefix);

  void Sync();

  void UpdateIndices(int prev_num_nodes, int new_num_nodes, int prev_num_vertices,
                     int new_num_vertices);

  void UpdateNodeGeometries();

  void UpdateNormals();

  void UpdateMidstepNormals();

  std::vector<size_t> GetOffsets();

  thrust::host_vector<int> SubmeshVertexOffsets() const;

  thrust::host_vector<int> NumSubmeshVertices() const;

  thrust::host_vector<thrust::host_vector<int>> SubmeshIndices() const;

  thrust::host_vector<Vector3> HostPositions();

  thrust::host_vector<Vector3> HostPredicted();

  thrust::host_vector<Vector3> HostNormals();

  thrust::host_vector<uint> HostIndices();

  thrust::host_vector<uint> HostColors();

  thrust::device_vector<Vector3> DevicePositions();

  thrust::device_vector<uint> DeviceIndices();

  Scalar GetInvMass(int node_index);

  void SetInvMass(Scalar inv_mass, int node_index);

  void SetFree(bool is_free, int node_index);

  Vector3 GetVelocity(int node_index);

  void SetVelocity(Vector3 velocity, int node_index);

  Vector3 GetX0(int node_index);

  void SetX0(Vector3 pos, int node_index);

  void ResetVelocities();

  void ResetX();

  void PredictedToPositions();

  void Interpolate(int step, int num_steps);

  int NumNodes();

  int NumEdges();

  int NumFaces();

  void SetPtrOffset(int node_offset, int edge_offset, int face_offset);

  thrust::device_vector<Node*> nodes;
  thrust::device_vector<Edge*> edges;
  thrust::device_vector<Face*> faces;

 private:
  // render buffers
  std::vector<std::shared_ptr<RegisteredBuffer<RenderableVertex>>> r_buffers_;
  // merged render buffers
  thrust::device_vector<RenderableVertex> v_buffer_;
  // index offset of render buffers
  std::vector<size_t> offsets_;

  std::shared_ptr<MemoryPool> memory_pool_;

  // indicate the pointer offset relative to the memory pool
  int node_offset_, vertex_offset_, edge_offset_, face_offset_;

  // indicate the sub mesh index offset relative the merged mesh
  thrust::host_vector<int> submesh_vertex_offsets_;
  thrust::host_vector<thrust::host_vector<int>> submesh_indices_;
  thrust::host_vector<int> num_submesh_vertices_;
};

}  // namespace XRTailor