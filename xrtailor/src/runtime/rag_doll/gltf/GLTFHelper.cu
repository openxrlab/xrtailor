#include <xrtailor/runtime/rag_doll/gltf/GLTFHelper.cuh>

namespace XRTailor {

__global__ void JointAnimation_Kernel(Vector3* world_position, Mat4* world_matrix, int* joints,
                                      Mat4 transform, int n_joint) {
  GET_CUDA_ID(p_Id, n_joint);

  Vector4 result = Vector4(0, 0, 0, 1);
  int joint_id = joints[p_Id];

  result = world_matrix[joint_id] * result;


  world_position[p_Id][0] = result[0];
  world_position[p_Id][1] = result[1];
  world_position[p_Id][2] = result[2];
}

void JointAnimation(thrust::device_vector<Vector3>& world_position,
                    thrust::device_vector<Mat4>& world_matrix, thrust::device_vector<int>& joints,
                    Mat4 transform) {
  int n_joint = joints.size();
  CUDA_CALL(JointAnimation_Kernel, n_joint)
  (pointer(world_position), pointer(world_matrix), pointer(joints), transform, n_joint);
  CUDA_CHECK_LAST();
}

__global__ void PointsAnimation(Vector3* intial_position, Vector3* world_position,
                                Mat4* joint_inverse_bind_matrix, Mat4* world_matrix,
                                Vector4* bind_joints_0, Vector4* bind_joints_1, Vector4* weights_0,
                                Vector4* weights_1,
                                Mat4 transform, bool is_normal,
                                Vector2u range, int n_bind_joints0, int n_bind_joints1, int n_node) {
  GET_CUDA_ID(p_id, n_node);

  if (p_id < range[0] || p_id > range[1])
    return;

  Vector4 result(0, 0, 0, Scalar(!is_normal));

  int skin_info_vertex_id = p_id - range[0];

  Vector3 offest;

  bool j0 = n_bind_joints0;
  bool j1 = n_bind_joints1;

  if (j0) {
    for (unsigned int i = 0; i < 4; i++) {
      int joint_id = int(bind_joints_0[skin_info_vertex_id][i]);
      Scalar weight = weights_0[skin_info_vertex_id][i];

      offest = intial_position[p_id];
      Vector4 v_bone_space = joint_inverse_bind_matrix[joint_id] *
                             Vector4(offest[0], offest[1], offest[2], Scalar(!is_normal));  //

      result += (world_matrix[joint_id] * v_bone_space) * weight;
    }
  }
  if (j1) {
    for (unsigned int i = 0; i < 4; i++) {
      int joint_id = int(bind_joints_1[skin_info_vertex_id][i]);
      Scalar weight = weights_1[skin_info_vertex_id][i];

      offest = intial_position[p_id];
      Vector4 v_bone_space = joint_inverse_bind_matrix[joint_id] *
                             Vector4(offest[0], offest[1], offest[2], Scalar(!is_normal));  //

      result += (world_matrix[joint_id] * v_bone_space) * weight;
    }
  }

  if (j0 | j1) {
    world_position[p_id][0] = result[0];
    world_position[p_id][1] = result[1];
    world_position[p_id][2] = result[2];
  }

  if (is_normal)
    world_position[p_id] = glm::normalize(world_position[p_id]);
}

__global__ void Skin_Kernel(Vector3* intial_position, Node** nodes, Mat4* joint_inverse_bind_matrix,
                            Mat4* world_matrix,
                            Vector4* bind_joints_0, Vector4* bind_joints_1, Vector4* weights_0,
                            Vector4* weights_1,
                            Mat4 transform,
                            Vector2u range, int n_bind_joints0, int n_bind_joints1, int n_node) {
  GET_CUDA_ID(p_id, n_node);

  if (p_id < range[0] || p_id > range[1])
    return;

  Vector4 result(0, 0, 0, 1);

  int skin_info_vertex_id = p_id - range[0];

  Vector3 offest;

  bool j0 = n_bind_joints0;
  bool j1 = n_bind_joints1;

  if (j0) {
    for (unsigned int i = 0; i < 4; i++) {
      int joint_id = int(bind_joints_0[skin_info_vertex_id][i]);
      Scalar weight = weights_0[skin_info_vertex_id][i];

      offest = intial_position[p_id];
      Vector4 v_bone_space =
          joint_inverse_bind_matrix[joint_id] * Vector4(offest[0], offest[1], offest[2], 1);  //

      result += (world_matrix[joint_id] * v_bone_space) * weight;
    }
  }
  if (j1) {
    for (unsigned int i = 0; i < 4; i++) {
      int joint_id = int(bind_joints_1[skin_info_vertex_id][i]);
      Scalar weight = weights_1[skin_info_vertex_id][i];

      offest = intial_position[p_id];
      Vector4 v_bone_space =
          joint_inverse_bind_matrix[joint_id] * Vector4(offest[0], offest[1], offest[2], 1);  //

      result += (world_matrix[joint_id] * v_bone_space) * weight;
    }
  }

  if (j0 | j1) {
    nodes[p_id]->x[0] = result[0];
    nodes[p_id]->x[1] = result[1];
    nodes[p_id]->x[2] = result[2];
  }
}

void SkinAnimation(Vector3* intial_position, Node** nodes, Mat4x4* joint_inverse_bind_matrix,
                   Mat4x4* world_matrix,
                   Vector4* bind_joints_0, Vector4* bind_joints_1, Vector4* weights_0,
                   Vector4* weights_1,
                   int n_node, int n_bind_joints0, int n_bind_joints1,
                   Mat4x4 transform,
                   Vector2u range) {
  CUDA_CALL(Skin_Kernel, n_node)
  (intial_position, nodes, joint_inverse_bind_matrix, world_matrix, bind_joints_0, bind_joints_1,
   weights_0, weights_1, transform, range, n_bind_joints0, n_bind_joints1, n_node);
  CUDA_CHECK_LAST();
}

void SkinAnimation(thrust::device_vector<Vector3>& intial_position,
                   thrust::device_vector<Vector3>& world_position,
                   thrust::device_vector<Mat4x4>& joint_inverse_bind_matrix,
                   thrust::device_vector<Mat4x4>& world_matrix,
                   thrust::device_vector<Vector4>& bind_joints_0,
                   thrust::device_vector<Vector4>& bind_joints_1,
                   thrust::device_vector<Vector4>& weights_0,
                   thrust::device_vector<Vector4>& weights_1,
                   Mat4x4 transform, bool is_normal,
                   Vector2u range) {
  int n_node = intial_position.size();
  int n_bind_joints0 = bind_joints_0.size();
  int n_bind_joints1 = bind_joints_1.size();
  CUDA_CALL(PointsAnimation, n_node)
  (pointer(intial_position), pointer(world_position), pointer(joint_inverse_bind_matrix),
   pointer(world_matrix), pointer(bind_joints_0), pointer(bind_joints_1), pointer(weights_0),
   pointer(weights_1), transform, is_normal, range, n_bind_joints0, n_bind_joints1, n_node);
  CUDA_CHECK_LAST();
}

}  // namespace XRTailor
