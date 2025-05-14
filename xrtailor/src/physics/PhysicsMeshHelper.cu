#include <xrtailor/physics/PhysicsMeshHelper.cuh>

#include <xrtailor/core/DeviceHelper.cuh>
#include <xrtailor/memory/Pair.cuh>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>

namespace XRTailor {

__global__ void InitializeNodes_Kernel(const Vector3* x, bool is_cloth, Node** nodes, Node* pool,
                                       int ptr_offset, int prev_num_nodes, int new_num_nodes) {
  GET_CUDA_ID(i, new_num_nodes);

  Node* node = &pool[prev_num_nodes + i];
  *node = Node(x[i], is_cloth, is_cloth);
  node->v = Vector3(0.0);
  node->index = i + prev_num_nodes - ptr_offset;
  nodes[prev_num_nodes - ptr_offset + i] = node;
}

void InitializeNodes(const Vector3* x, bool is_free, Node** nodes, Node* pool, int ptr_offset,
                     int prev_num_nodes, int new_num_nodes) {
  CUDA_CALL(InitializeNodes_Kernel, new_num_nodes)
  (x, is_free, nodes, pool, ptr_offset, prev_num_nodes, new_num_nodes);
  CUDA_CHECK_LAST();
}

__global__ void InitializeFaces_Kernel(const unsigned int* x_indices, const unsigned int* fe_indices,
                                       Node** nodes, Edge** edges, Face** faces, Face* pool,
                                       int node_ptr_offset, int edge_ptr_offset,
                                       int face_ptr_offset, int prev_num_particles, int prev_num_edges,
                                       int prev_num_faces, int new_num_faces) {
  GET_CUDA_ID(i, new_num_faces);

  int x_idx0 = x_indices[i * 3u + 0u] + prev_num_particles - node_ptr_offset;
  int x_idx1 = x_indices[i * 3u + 1u] + prev_num_particles - node_ptr_offset;
  int x_idx2 = x_indices[i * 3u + 2u] + prev_num_particles - node_ptr_offset;
  int e_idx0 = fe_indices[i * 3u + 0u] + prev_num_edges - edge_ptr_offset;
  int e_idx1 = fe_indices[i * 3u + 1u] + prev_num_edges - edge_ptr_offset;
  int e_idx2 = fe_indices[i * 3u + 2u] + prev_num_edges - edge_ptr_offset;

  Node* node0 = nodes[x_idx0];
  Node* node1 = nodes[x_idx1];
  Node* node2 = nodes[x_idx2];

  Face* face = &pool[i + prev_num_faces];
  *face = Face(node0, node1, node2);
  const Edge* edge0 = edges[e_idx0];
  const Edge* edge1 = edges[e_idx1];
  const Edge* edge2 = edges[e_idx2];
  face->SetEdges(edge0, edge1, edge2);
  face->index = i + prev_num_faces - face_ptr_offset;
  face->type = (node0->is_free && node1->is_free && node2->is_free) ? 0 : 1;


  faces[i + prev_num_faces - face_ptr_offset] = face;
}

void InitializeFaces(const unsigned int* x_indices, const unsigned int* feIndices, Node** nodes,
                     Edge** edges, Face** faces, Face* pool, int node_ptr_offset,
                     int edge_ptr_offset, int face_ptr_offset, int prev_num_particles,
                     int prev_num_edges, int prev_num_faces, int new_num_faces) {
  CUDA_CALL(InitializeFaces_Kernel, new_num_faces)
  (x_indices, feIndices, nodes, edges, faces, pool, node_ptr_offset, edge_ptr_offset,
   face_ptr_offset, prev_num_particles, prev_num_edges, prev_num_faces, new_num_faces);
  CUDA_CHECK_LAST();
}

__global__ void initializeEdges_Kernel(const unsigned int* edgeIndices, const Node* const* nodes,
                                       Edge** edges, Edge* pool, int ptr_offset, int prev_num_edges,
                                       int new_num_edges) {
  GET_CUDA_ID(i, new_num_edges);

  int idx0 = edgeIndices[i * 2];
  int idx1 = edgeIndices[i * 2 + 1];

  Edge* edge = &pool[i + prev_num_edges];
  *edge = Edge(nodes[idx0], nodes[idx1]);
  edge->index = i + prev_num_edges - ptr_offset;

  edges[i + prev_num_edges - ptr_offset] = edge;
}

void InitializeEdges(const unsigned int* edgeIndices, const Node* const* nodes, Edge** edges,
                     Edge* pool, int ptr_offset, int prev_num_edges, int new_num_edges) {
  CUDA_CALL(initializeEdges_Kernel, new_num_edges)
  (edgeIndices, nodes, edges, pool, ptr_offset, prev_num_edges, new_num_edges);
  CUDA_CHECK_LAST();
}

__global__ void updateNodeIndices_Kernel(Node** nodes, int ptr_offset, int prev_num_nodes,
                                         int new_num_nodes) {
  GET_CUDA_ID(i, new_num_nodes);

  nodes[i + prev_num_nodes - ptr_offset]->index = i + prev_num_nodes - ptr_offset;
}

void UpdateNodeIndices(Node** nodes, int ptr_offset, int prev_num_nodes, int new_num_nodes) {
  CUDA_CALL(updateNodeIndices_Kernel, new_num_nodes)
  (nodes, ptr_offset, prev_num_nodes, new_num_nodes);
  CUDA_CHECK_LAST();
}

//__global__ void updateVertexIndices_Kernel(Vertex** vertices, int ptr_offset, int prevNumVertices, int newNumVertices)
//{
//	GET_CUDA_ID(i, newNumVertices);
//
//	vertices[i + prevNumVertices - ptr_offset]->index = i + prevNumVertices;
//}

//void updateVertexIndices(Vertex** vertices, int ptr_offset, int prevNumVertices, int newNumVertices)
//{
//	CUDA_CALL(updateVertexIndices_Kernel, newNumVertices)
//		(vertices, ptr_offset, prevNumVertices, newNumVertices);
//	CUDA_CHECK_LAST();
//}

__global__ void updateFaceGeometries_Kernel(Face** faces, int ptr_offset, int prev_num_faces,
                                            int new_num_faces) {
  GET_CUDA_ID(i, new_num_faces);

  faces[i + prev_num_faces - ptr_offset]->Update();
}

void UpdateFaceGeometries(Face** faces, int ptr_offset, int prev_num_faces, int new_num_faces) {
  CUDA_CALL(updateFaceGeometries_Kernel, new_num_faces)
  (faces, ptr_offset, prev_num_faces, new_num_faces);
  CUDA_CHECK_LAST();
}

__global__ void InitializeNodeGeometriesLocal_Kernel(Node** nodes, int ptr_offset, int prev_num_nodes,
                                                     int new_num_nodes) {
  GET_CUDA_ID(i, new_num_nodes);

  Node* node = nodes[i + prev_num_nodes - ptr_offset];
  node->x1 = node->x;
  node->n = Vector3(0.0);
  node->area = 0.0f;
  node->inv_mass = node->is_free ? 1.0f : 1.0f / 1e3f;
}

__global__ void InitializeNodeGeometriesGlobal_Kernel(Node** nodes, int n_nodes) {
  GET_CUDA_ID(i, n_nodes);

  Node* node = nodes[i];
  node->n = Vector3(0.0);
  node->area = 0.0f;
  node->inv_mass = node->is_free ? 1.0f : 1.0f / 1e3f;
}

__global__ void InitializeNodeNormals_Kernel(Node** nodes, int n_nodes) {
  GET_CUDA_ID(i, n_nodes);

  Node* node = nodes[i];
  node->n = Vector3(0.0f);
}

__global__ void InitializeNodeMidstepNormals_Kernel(Node** nodes, int n_nodes) {
  GET_CUDA_ID(i, n_nodes);

  Node* node = nodes[i];
  node->n1 = Vector3(0.0f);
}

__global__ void UpdateNodeGeometriesLocal_Kernel(const Face* const* faces, int ptr_offset,
                                                 int prev_num_faces, int new_num_faces) {
  GET_CUDA_ID(i, new_num_faces);

  const Face* face = faces[i + prev_num_faces - ptr_offset];

  for (int j = 0; j < 3; j++) {
    Node* node = face->nodes[j];
    Vector3 e0 = face->nodes[(j + 1) % 3]->x - node->x;
    Vector3 e1 = face->nodes[(j + 2) % 3]->x - node->x;
    Vector3 n = glm::cross(e0, e1) / (glm::dot(e0, e0) * glm::dot(e1, e1));
    for (int k = 0; k < 3; k++)
      atomicAdd(&node->n[k], n[k]);
  }
}

__global__ void UpdateNodeGeometriesGlobal_Kernel(const Face* const* faces, int n_faces) {
  GET_CUDA_ID(i, n_faces);

  const Face* face = faces[i];

  for (int j = 0; j < 3; j++) {
    Node* node = face->nodes[j];
    Vector3 e0 = face->nodes[(j + 1) % 3]->x - node->x;
    Vector3 e1 = face->nodes[(j + 2) % 3]->x - node->x;
    Vector3 n = glm::cross(e0, e1) / (glm::dot(e0, e0) * glm::dot(e1, e1));
    for (int k = 0; k < 3; k++)
      atomicAdd(&node->n[k], n[k]);
  }
}

__global__ void updateNodeNormals_Kernel(const Face* const* faces, int n_faces) {
  GET_CUDA_ID(i, n_faces);

  const Face* face = faces[i];

  for (int j = 0; j < 3; j++) {
    Node* node = face->nodes[j];
    Vector3 e0 = face->nodes[(j + 1) % 3]->x0 - node->x0;
    Vector3 e1 = face->nodes[(j + 2) % 3]->x0 - node->x0;
    Vector3 n = glm::cross(e0, e1) / (glm::dot(e0, e0) * glm::dot(e1, e1));

    for (int k = 0; k < 3; k++)
      atomicAdd(&node->n[k], n[k]);
  }
}

__global__ void UpdateNodeMidstepNormals_Kernel(const Face* const* faces, int n_faces) {
  GET_CUDA_ID(i, n_faces);

  const Face* face = faces[i];

  for (int j = 0; j < 3; j++) {
    Node* node = face->nodes[j];
    Vector3 e0 = face->nodes[(j + 1) % 3]->x1 - node->x1;
    Vector3 e1 = face->nodes[(j + 2) % 3]->x1 - node->x1;
    Vector3 n = glm::cross(e0, e1) / (glm::dot(e0, e0) * glm::dot(e1, e1));
    for (int k = 0; k < 3; k++)
      atomicAdd(&node->n1[k], n[k]);
  }
}

__global__ void FinalizeNodeGeometriesLocal_Kernel(Node** nodes, int ptr_offset, int prev_num_nodes,
                                                   int new_num_nodes) {
  GET_CUDA_ID(i, new_num_nodes);

  nodes[i + prev_num_nodes - ptr_offset]->n = glm::normalize(nodes[i + prev_num_nodes - ptr_offset]->n);
}

__global__ void FinalizeNodeGeometriesGlobal_Kernel(Node** nodes, int n_nodes) {
  GET_CUDA_ID(i, n_nodes);

  nodes[i]->n = glm::normalize(nodes[i]->n);
}

__global__ void FinalizeNodeNormals_Kernel(Node** nodes, int n_nodes) {
  GET_CUDA_ID(i, n_nodes);

  nodes[i]->n = glm::normalize(nodes[i]->n);
}

__global__ void FinalizeNodeMidstepNormals_Kernel(Node** nodes, int n_nodes) {
  GET_CUDA_ID(i, n_nodes);

  nodes[i]->n1 = glm::normalize(nodes[i]->n1);
}

void UpdateNodeGeometriesLocal(Node** nodes, Face** faces, int node_ptr_offset, int face_ptr_offset,
                               int prev_num_nodes, int new_num_nodes, int prev_num_faces,
                               int new_num_faces) {
  CUDA_CALL(InitializeNodeGeometriesLocal_Kernel, new_num_nodes)
  (nodes, node_ptr_offset, prev_num_nodes, new_num_nodes);
  CUDA_CHECK_LAST();

  CUDA_CALL(UpdateNodeGeometriesLocal_Kernel, new_num_faces)
  (faces, face_ptr_offset, prev_num_faces, new_num_faces);
  CUDA_CHECK_LAST();

  CUDA_CALL(FinalizeNodeGeometriesLocal_Kernel, new_num_nodes)
  (nodes, node_ptr_offset, prev_num_nodes, new_num_nodes);
  CUDA_CHECK_LAST();
}

void UpdateNodeGeometriesGlobal(Node** nodes, Face** faces, int n_nodes, int n_faces) {
  CUDA_CALL(InitializeNodeGeometriesGlobal_Kernel, n_nodes)
  (nodes, n_nodes);
  CUDA_CHECK_LAST();

  CUDA_CALL(UpdateNodeGeometriesGlobal_Kernel, n_faces)
  (faces, n_faces);
  CUDA_CHECK_LAST();

  CUDA_CALL(FinalizeNodeGeometriesGlobal_Kernel, n_nodes)
  (nodes, n_nodes);
  CUDA_CHECK_LAST();
}

void UpdateNodeNormals(Node** nodes, Face** faces, int n_nodes, int n_faces) {
  CUDA_CHECK_LAST();
  CUDA_CALL(InitializeNodeNormals_Kernel, n_nodes)
  (nodes, n_nodes);
  CUDA_CHECK_LAST();

  CUDA_CALL(updateNodeNormals_Kernel, n_faces)
  (faces, n_faces);
  CUDA_CHECK_LAST();

  CUDA_CALL(FinalizeNodeNormals_Kernel, n_nodes)
  (nodes, n_nodes);
  CUDA_CHECK_LAST();
}

void UpdateNodeMidstepNormals(Node** nodes, Face** faces, int n_nodes, int n_faces) {
  CUDA_CHECK_LAST();
  CUDA_CALL(InitializeNodeMidstepNormals_Kernel, n_nodes)
  (nodes, n_nodes);
  CUDA_CHECK_LAST();

  CUDA_CALL(UpdateNodeMidstepNormals_Kernel, n_faces)
  (faces, n_faces);
  CUDA_CHECK_LAST();

  CUDA_CALL(FinalizeNodeMidstepNormals_Kernel, n_nodes)
  (nodes, n_nodes);
  CUDA_CHECK_LAST();
}

__global__ void UpdateFaceTypeAndIndex_Kernel(const PairFF* pairs, int* types, int* indices,
                                              int n_pairs) {
  GET_CUDA_ID(i, n_pairs);

  PairFF p = pairs[i];
  Face* f1 = p.first;
  Face* f2 = p.second;

  types[i * 2] = f1->type;
  indices[i * 2] = f1->index;

  types[i * 2 + 1] = f2->type;
  indices[i * 2 + 1] = f2->index;
}

void UpdateFaceTypeAndIndices(const thrust::device_vector<PairFF>& pairs,
                              thrust::host_vector<int>& types, thrust::host_vector<int>& indices) {
  int n_pairs = pairs.size();

  thrust::device_vector<int> d_types(n_pairs * 2);
  thrust::device_vector<int> d_indices(n_pairs * 2);

  CUDA_CALL(UpdateFaceTypeAndIndex_Kernel, n_pairs)
  (pointer(pairs), pointer(d_types), pointer(d_indices), n_pairs);
  CUDA_CHECK_LAST();

  types = std::move(d_types);
  indices = std::move(d_indices);
}

}  // namespace XRTailor