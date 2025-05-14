#pragma once

#include <iostream>
#include <vector>
#include <set>

#include <xrtailor/core/Common.cuh>
#include <xrtailor/core/Common.hpp>
#include <xrtailor/runtime/rag_doll/smpl/smplx.hpp>
#include <xrtailor/core/DeviceHelper.cuh>
#include <xrtailor/memory/Node.cuh>
#include <xrtailor/physics/sdf/SDFCollider.cuh>
#include <xrtailor/physics/PhysicsMesh.cuh>
#include <xrtailor/physics/PhysicsMeshHelper.cuh>
#include <xrtailor/math/MathFunctions.cuh>

namespace XRTailor {
class BVH;

void SetSimulationParams(SimParams* host_params);

__device__ Vector3 ComputeFriction(Vector3 correction, Vector3 rel_vel);

void QueryNearestBarycentricClothToObstacle(std::shared_ptr<PhysicsMesh> cloth,
                                            SkinParam* skin_params, std::shared_ptr<BVH> bvh);

void UpdateObstaclePredicted(Node** nodes, smplx::Points src_points, const int tgt_count,
                             Mat4& model_matrix, Scalar X_degree);

void UpdateX(Node** nodes, Mat4 model_matrix, int n_nodes);

void InitializePositions(std::shared_ptr<PhysicsMesh> physics_mesh, const int start, const int count,
                         Mat4& model_matrix);

void PredictPositions(Node** nodes, const Scalar delta_time, int n_nodes);

void ApplyDeltas(Node** nodes, Vector3* deltas, int* delta_counts);

void CollideParticles(Vector3* deltas, int* delta_counts, Node** nodes, CONST(uint*) neighbors);

void Finalize(Node** nodes, const Scalar delta_time);

void UpdateSkinnedVertices(std::shared_ptr<PhysicsMesh> physics_mesh,
                           CONST(smplx::Scalar*) skinned_verts, Scalar X_degree);

void UpdateVertices(Vector3* tgt_positions, std::shared_ptr<PhysicsMesh> physics_mesh);

void UpdatePredicted(Vector3* tgt_positions, Node** src_nodes);

void SetupNodes(Node* nodes, Node** nodes2, Vector3* positions, Vector3* predicted,
                Scalar* inv_masses, int num_verts);

void FinalizeNodes(Node* nodes, Vector3* predicted, int num_verts);

void BuildDeviceEFAdjacency(std::vector<std::set<uint>>& h_nb_ef, thrust::device_vector<uint>& nb_ef,
                            thrust::device_vector<uint>& nb_ef_prefix);

void BuildDeviceVFAdjacency(std::vector<std::set<uint>>& h_nb_vf, thrust::device_vector<uint>& nb_vf,
                            thrust::device_vector<uint>& nb_vf_prefix);

void BuildVFAdjacency(std::shared_ptr<PhysicsMesh> physics_mesh, thrust::device_vector<uint>& nb_vf,
                      thrust::device_vector<uint>& nb_vf_prefix);

}  // namespace XRTailor
