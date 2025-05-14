#pragma once

#include <memory>
#include <string>
#include <vector>
#include <algorithm>

#include <thrust/host_vector.h>

#include <xrtailor/runtime/mesh/MeshIO.hpp>
#include <xrtailor/utils/FileSystemUtils.hpp>
#include <xrtailor/math/Quaternion.cuh>
#include <xrtailor/runtime/rag_doll/gltf/JointInfo.cuh>
#include <xrtailor/runtime/rag_doll/gltf/SkinInfo.cuh>

namespace XRTailor {

class Node;

class GLTFLoader {
 public:
  typedef unsigned char byte;
  typedef int joint;
  typedef int shape;
  typedef int mesh;
  typedef int primitive;
  typedef int scene;

  GLTFLoader();

  ~GLTFLoader();

  void LoadGltf(std::string file_path, MeshData& mesh_data);

  thrust::host_vector<Vector3> TestLBS(int frame_number);

  void UpdateAnimation(int frame_number, Node** nodes, int n_nodes);

 public:
  bool import_animation = false;
  Mat4 transform;

  std::map<joint, std::string> joint_name;

  thrust::host_vector<int> all_nodes;

  thrust::host_vector<int> all_meshes;

  std::map<joint, thrust::host_vector<int>> node_id_dir;
  std::map<int, thrust::host_vector<int>> mesh_id_dir;

  std::map<int, Mat4> node_matrix;

  thrust::host_vector<Mat4> mesh_matrix;

  int max_mesh_id = -1;
  int max_joint_id = -1;
  int joint_num = -1;
  int mesh_num = -1;

  std::map<int, thrust::host_vector<Vector2u>> skin_verts_range;

  std::shared_ptr<SkinInfo> skin;
  std::shared_ptr<JointInfo> joints_data;
  std::shared_ptr<JointAnimationInfo> animation;

  std::map<int, std::string> node_name;

  thrust::host_vector<joint> all_joints;
  // [translation, scale, rotation]
  std::map<joint, Vector3i> joint_output;
  std::map<joint, Vector3> joint_input;
  std::map<joint, Mat4> joint_inverse_bind_matrix_map;
  std::map<joint, Mat4> joint_animation_matrix;
  std::map<joint, Quaternion> joint_rotation;
  std::map<joint, Vector3> joint_scale;
  std::map<joint, Vector3> joint_translation;
  std::map<joint, Mat4> joint_matrix;
  std::map<joint, thrust::host_vector<int>> joint_id_joint_dir;

  std::map<joint, thrust::host_vector<Vector3>> joint_T_f_anim;
  std::map<joint, thrust::host_vector<Quaternion>> joint_R_f_anim;
  std::map<joint, thrust::host_vector<Vector3>> joint_S_f_anim;
  std::map<joint, thrust::host_vector<Scalar>> joint_T_Time;
  std::map<joint, thrust::host_vector<Scalar>> joint_S_Time;
  std::map<joint, thrust::host_vector<Scalar>> joint_R_Time;

  thrust::device_vector<Vector3> initial_position;
  thrust::device_vector<Vector3> initial_normal;
  thrust::device_vector<int> d_joints;

  thrust::device_vector<Vector2> tex_coord_0;
  thrust::device_vector<Vector2> tex_coord_1;
  thrust::device_vector<Mat4> initial_matrix;
  thrust::device_vector<Vector3> joint_world_position;
  thrust::device_vector<Mat4> joint_inverse_bind_matrix;
  thrust::device_vector<Mat4> joint_local_matrix;
  thrust::device_vector<Mat4> joint_world_matrix;

  thrust::device_vector<Vector3> d_shape_center;

  thrust::device_vector<Mat4> d_mesh_matrix;
  thrust::device_vector<int> d_shape_mesh_id;

 private:
  void UpdateAnimation(int frameNumber, thrust::device_vector<Vector3>& world_positons);

  void InitializeData();

  void UpdateAnimationMatrix(const thrust::host_vector<joint>& all_joints, int current_frame);

  Vector3 GetVertexLocationWithJointTransform(joint joint_id, Vector3 in_point,
                                              std::map<joint, Mat4> j_matrix);

  void UpdateJointWorldMatrix(const thrust::host_vector<joint>& all_joints,
                              std::map<joint, Mat4> j_matrix);

  void BuildInverseBindMatrices(const thrust::host_vector<joint>& all_joints);

  void UpdateTransformState();

  void InsertMidstepAnimation(std::map<joint, Vector3> joint_translation,
                              std::map<joint, Vector3> joint_scale,
                              std::map<joint, Quaternion> joint_rotation,
                              std::map<joint, thrust::host_vector<Vector3>>& joint_T_f_anim,
                              std::map<joint, thrust::host_vector<Scalar>>& joint_T_Time,
                              std::map<joint, thrust::host_vector<Vector3>>& joint_S_f_anim,
                              std::map<joint, thrust::host_vector<Scalar>>& joint_S_Time,
                              std::map<joint, thrust::host_vector<Quaternion>>& joint_R_f_anim,
                              std::map<joint, thrust::host_vector<Scalar>>& joint_R_Time,
                              Scalar n_midstep_frames, Scalar fps);
};
}  // namespace XRTailor