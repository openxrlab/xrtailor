#pragma once

#include <glm/glm.hpp>

#include <xrtailor/runtime/rag_doll/smpl/smplx.hpp>
#include <xrtailor/runtime/rag_doll/smpl/sequence.hpp>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <xrtailor/runtime/mesh/MeshIO.hpp>
#include <xrtailor/physics/PhysicsMesh.cuh>

class Mesh;

namespace XRTailor {
template <typename ModelConfig>
void SaveBodyAsObj(const std::string& path, std::shared_ptr<smplx::Body<ModelConfig>> body,
                   Vector3* h_positions);

template <class ModelConfig>
void SaveBodyAsObj(const std::string& path, std::shared_ptr<smplx::Body<ModelConfig>> body,
                   std::shared_ptr<PhysicsMesh> physics_mesh);

void SaveGarmentAsObj(const std::string& path, std::shared_ptr<PhysicsMesh> physics_mesh);

void SaveGarmentAsObj(const std::string& path, Vector3* positions, uint* indices,
                      const uint num_vertices, const uint num_triangles);

void SaveGarmentAsObj(const std::string& path, Vector3* positions, Vector3* colors, uint* indices,
                      const uint num_vertices, const uint num_triangles);

void SaveMeshAsObj(const std::string& path, const std::vector<Vector3>& positions,
                   const std::vector<uint>& indices);

void SaveMeshDataAsObj(const std::string& path, const MeshData& mesh_data);

void SaveMeshDataAsObjWithUVNormal(const std::string& path, const MeshData& mesh_data);
}  // namespace XRTailor
