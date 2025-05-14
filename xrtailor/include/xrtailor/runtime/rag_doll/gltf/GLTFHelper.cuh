#pragma once

#include <xrtailor/core/DeviceHelper.cuh>
#include <xrtailor/memory/Node.cuh>

namespace XRTailor {

void JointAnimation(thrust::device_vector<Vector3>& world_position,
                    thrust::device_vector<Mat4>& world_matrix, thrust::device_vector<int>& joints,
                    Mat4 transform);

void SkinAnimation(
    thrust::device_vector<Vector3>& intial_position, thrust::device_vector<Vector3>& world_position,
    thrust::device_vector<Mat4x4>& joint_inverse_bind_matrix,
    thrust::device_vector<Mat4x4>& world_matrix, thrust::device_vector<Vector4>& bind_joints_0,
    thrust::device_vector<Vector4>& bind_joints_1, thrust::device_vector<Vector4>& weights_0,
    thrust::device_vector<Vector4>& weights_1, Mat4x4 transform, bool isNormal, Vector2u range);

void SkinAnimation(Vector3* intial_position, Node** nodes, Mat4x4* joint_inverse_bind_matrix,
                   Mat4x4* world_matrix, Vector4* bind_joints_0, Vector4* bind_joints_1,
                   Vector4* weights_0, Vector4* weights_1, int n_node, int n_bind_joints0,
                   int n_bind_joints1, Mat4x4 transform, Vector2u range);

}  // namespace XRTailor
