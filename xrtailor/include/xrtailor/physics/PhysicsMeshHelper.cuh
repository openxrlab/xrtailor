#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <xrtailor/memory/Node.cuh>
#include <xrtailor/memory/Face.cuh>
#include <xrtailor/memory/Edge.cuh>
#include <xrtailor/memory/Pair.cuh>

namespace XRTailor {

void InitializeNodes(const Vector3* x, bool is_free, Node** nodes, Node* pool, int ptr_offset,
                     int prev_num_nodes, int new_num_nodes);

void InitializeFaces(const unsigned int* x_indices, const unsigned int* fe_indices, Node** nodes,
                     Edge** edges, Face** faces, Face* pool, int node_ptr_offset,
                     int edge_ptr_offset, int face_ptr_offset, int prev_num_particles,
                     int prev_num_edges, int prev_num_faces, int new_num_faces);

void InitializeEdges(const unsigned int* edgeIndices, const Node* const* nodes, Edge** edges,
                     Edge* pool, int ptr_offset, int prev_num_edges, int newNumEdges);

void UpdateNodeIndices(Node** nodes, int ptr_offset, int prev_num_nodes, int new_num_nodes);

void UpdateFaceGeometries(Face** faces, int ptr_offset, int prev_num_faces, int new_num_faces);

void UpdateNodeGeometriesLocal(Node** nodes, Face** faces, int node_ptr_offset, int face_ptr_offset,
                               int prev_num_nodes, int new_num_nodes, int prev_num_faces,
                               int new_num_faces);

void UpdateNodeGeometriesGlobal(Node** nodes, Face** faces, int n_nodes, int n_faces);

void UpdateNodeNormals(Node** nodes, Face** faces, int n_nodes, int n_faces);

void UpdateNodeMidstepNormals(Node** nodes, Face** faces, int n_nodes, int n_faces);

void UpdateFaceTypeAndIndices(const thrust::device_vector<PairFF>& pairs,
                              thrust::host_vector<int>& types, thrust::host_vector<int>& indices);

}  // namespace XRTailor