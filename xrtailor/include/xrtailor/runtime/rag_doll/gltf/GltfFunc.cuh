#pragma once

#include <tiny_gltf.h>

#include <string>
#include <algorithm>

#include <xrtailor/core/Scalar.hpp>
#include <xrtailor/math/Quaternion.cuh>
#include <xrtailor/math/Transform3x3.cuh>
#include <xrtailor/physics/broad_phase/Bounds.cuh>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define NULL_TIME (-9599.99)

namespace XRTailor {

typedef unsigned char byte;
typedef int joint;
typedef int scene;

void GetBoundingBoxByName(tinygltf::Model& model, const tinygltf::Primitive& primitive,
                          const std::string& attribute_name, Bounds& bound,
                          Transform3x3& transform);

void GetVec3ByAttributeName(tinygltf::Model& model, const tinygltf::Primitive& primitive,
                            const std::string& attribute_name,
                            thrust::host_vector<Vector3>& vertices);

void GetVec4ByAttributeName(tinygltf::Model& model, const tinygltf::Primitive& primitive,
                            const std::string& attribute_name,
                            thrust::host_vector<Vector4>& vec4_data);

void GetScalarByIndex(tinygltf::Model& model, int index, thrust::host_vector<Scalar>& result);

void GetVec3ByIndex(tinygltf::Model& model, int index, thrust::host_vector<Vector3>& result);

void GetQuatByIndex(tinygltf::Model& model, int index, thrust::host_vector<Quaternion>& result);

void GetTriangles(tinygltf::Model& model, const tinygltf::Primitive& primitive,
                  thrust::host_vector<uint>& indices, int point_offest);

void GetVertexBindJoint(tinygltf::Model& model, const tinygltf::Primitive& primitive,
                        const std::string& attribute_name, thrust::host_vector<Vector4>& vec4_data,
                        const thrust::host_vector<int>& skin_joints);

void GetNodesAndHierarchy(tinygltf::Model& model,
                          std::map<scene, thrust::host_vector<int>> scene_joints_nodes_id,
                          thrust::host_vector<joint>& all_nodes,
                          std::map<joint, thrust::host_vector<int>>& id_dir);

void TraverseNode(tinygltf::Model& model, joint id, thrust::host_vector<joint>& joint_nodes,
                  std::map<joint, thrust::host_vector<int>>& dir,
                  thrust::host_vector<joint> current_dir);

void GetJointsTransformData(const thrust::host_vector<int>& all_joints,
                            std::map<int, std::string>& joint_name,
                            thrust::host_vector<thrust::host_vector<int>>& joint_child,
                            std::map<int, Quaternion>& joint_rotation,
                            std::map<int, Vector3>& joint_scale,
                            std::map<int, Vector3>& joint_translation,
                            std::map<int, Mat4x4>& joint_matrix, tinygltf::Model model);

void ImportAnimation(tinygltf::Model model, std::map<joint, Vector3i>& joint_output,
                     std::map<joint, Vector3>& joint_input,
                     std::map<joint, thrust::host_vector<Vector3>>& joint_T_f_anim,
                     std::map<joint, thrust::host_vector<Scalar>>& joint_T_Time,
                     std::map<joint, thrust::host_vector<Vector3>>& joint_S_f_anim,
                     std::map<joint, thrust::host_vector<Scalar>>& joint_S_Time,
                     std::map<joint, thrust::host_vector<Quaternion>>& joint_R_f_anim,
                     std::map<joint, thrust::host_vector<Scalar>>& joint_R_Time);

void BuildInverseBindMatrices(const std::vector<joint>& all_joints,
                              std::map<joint, Mat4x4>& joint_matrix, int& max_joint_id,
                              tinygltf::Model& model, std::map<joint, Quaternion>& joint_rotation,
                              std::map<joint, Vector3>& joint_translation,
                              std::map<joint, Vector3>& joint_scale,
                              std::map<joint, Mat4x4>& joint_inverse_bind_matrix,
                              std::map<joint, thrust::host_vector<int>> joint_id_joint_dir);

void UpdateJointMeshCameraDir(tinygltf::Model& model, int& joint_num, int& mesh_num,
                              std::map<joint, thrust::host_vector<int>>& joint_id_joint_dir,
                              thrust::host_vector<joint>& all_joints,
                              thrust::host_vector<int>& all_nodes,
                              std::map<joint, thrust::host_vector<int>> node_id_dir,
                              std::map<int, thrust::host_vector<int>>& mesh_id_dir,
                              thrust::host_vector<int>& all_meshes,
                              thrust::device_vector<int>& d_joints, int& max_joint_id);

thrust::host_vector<int> GetJointDirByJointIndex(
    int index, std::map<joint, thrust::host_vector<int>> joint_id_joint_dir);

void GetMeshMatrix(tinygltf::Model& model, const thrust::host_vector<int>& all_mesh_node_ids,
                   int& max_mesh_id, thrust::host_vector<Mat4x4>& mesh_matrix);
}  // namespace XRTailor
