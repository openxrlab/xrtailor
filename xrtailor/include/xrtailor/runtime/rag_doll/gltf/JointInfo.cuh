#pragma once

#include <set>
#include <map>
#include <string>
#include <algorithm>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <xrtailor/core/Scalar.cuh>

#include <xrtailor/math/Quaternion.cuh>
#include <xrtailor/math/Transform3x3.cuh>

namespace XRTailor {

struct Pose {
  Pose();

  Pose(thrust::host_vector<Vector3> t, thrust::host_vector<Vector3> s,
       thrust::host_vector<Quaternion> r);

  ~Pose();

  int Size();

  void Resize(int newSize);

  thrust::host_vector<Vector3> t;
  thrust::host_vector<Quaternion> r;
  thrust::host_vector<Vector3> s;
};

class JointInfo {
  typedef int joint;

 public:
  JointInfo();

  JointInfo(thrust::device_vector<Mat4x4>& inverse_bind_matrix,
            thrust::device_vector<Mat4x4>& local_matrix, thrust::device_vector<Mat4x4>& world_matrix,
            thrust::host_vector<int>& all_joints,
            std::map<joint, thrust::host_vector<joint>>& joint_dir,
            std::map<joint, Vector3>& bind_pose_translation, std::map<joint, Vector3>& bind_pose_scale,
            std::map<joint, Quaternion>& bind_pose_rotation);

  ~JointInfo();

  void UpdateJointInfo(thrust::device_vector<Mat4x4>& inverse_bind_matrix,
                       thrust::device_vector<Mat4x4>& local_matrix,
                       thrust::device_vector<Mat4x4>& world_matrix,
                       thrust::host_vector<int>& all_joints,
                       std::map<joint, thrust::host_vector<joint>>& joint_dir,
                       std::map<joint, Vector3>& bind_pose_translation,
                       std::map<joint, Vector3>& bind_pose_scale,
                       std::map<joint, Quaternion>& bind_pose_rotation);

  void SetJoint(const JointInfo& j);

  bool IsEmpty();

  void UpdateWorldMatrixByTransform();

  void SetJointName(const std::map<int, std::string> name);

 public:
  std::map<int, std::string> joint_name;

  thrust::device_vector<Mat4x4> joint_inverse_bind_matrix;
  thrust::device_vector<Mat4x4> joint_local_matrix;
  thrust::device_vector<Mat4x4> joint_world_matrix;

  thrust::host_vector<Vector3> bind_pose_translation;
  thrust::host_vector<Vector3> bind_pose_scale;
  thrust::host_vector<Quaternion> bind_pose_rotation;

  // Animation
  thrust::device_vector<Vector3> current_translation;
  thrust::device_vector<Quaternion> current_rotation;
  thrust::device_vector<Vector3> current_scale;

  thrust::host_vector<joint> all_joints;
  std::map<joint, thrust::host_vector<joint>> joint_dir;

  int max_joint_id = -1;
};

class JointAnimationInfo {
  typedef int joint;

 public:
  JointAnimationInfo();

  ~JointAnimationInfo();

  void SetAnimationData(std::map<joint, thrust::host_vector<Vector3>>& joint_translation,
                        std::map<joint, thrust::host_vector<Scalar>>& joint_time_code_translation,
                        std::map<joint, thrust::host_vector<Vector3>>& joint_scale,
                        std::map<joint, thrust::host_vector<Scalar>>& joint_index_time_code_scale,
                        std::map<joint, thrust::host_vector<Quaternion>>& joint_rotation,
                        std::map<joint, thrust::host_vector<Scalar>>& joint_index_rotation,
                        std::shared_ptr<JointInfo> skeleton);

  void UpdateJointsTransform(Scalar time);

  Transform3x3 UpdateTransform(joint joint_id, Scalar time);

  thrust::host_vector<Vector3> GetJointsTranslation(Scalar time);

  thrust::host_vector<Quaternion> GetJointsRotation(Scalar time);

  thrust::host_vector<Vector3> GetJointsScale(Scalar time);

  Scalar GetTotalTime();

  int FindMaxSmallerIndex(const thrust::host_vector<Scalar>& arr, Scalar v);

  Vector3 Lerp(Vector3 v0, Vector3 v1, Scalar weight);

  Quaternion Normalize(const Quaternion&);

  Quaternion SLerp(const Quaternion& q1, const Quaternion& q2, Scalar weight);

  Quaternion NLerp(const Quaternion& q1, const Quaternion& q2, Scalar weight);

  thrust::host_vector<int> GetJointDir(int index,
                                       std::map<int, thrust::host_vector<int>> joint_dir);

  void SetLoop(bool loop);

  Pose GetPose(Scalar in_time);

  Scalar GetCurrentAnimationTime();

  Scalar& GetBlendInTime();

  Scalar& GetBlendOutTime();

  Scalar& GetPlayRate();

 public:
  std::shared_ptr<JointInfo> skeleton = NULL;

 private:
  // Animation and time stamp
  std::map<joint, thrust::host_vector<Vector3>> joint_index_translation_;
  std::map<joint, thrust::host_vector<Scalar>> joint_index_time_code_translation_;

  std::map<joint, thrust::host_vector<Vector3>> joint_index_scale_;
  std::map<joint, thrust::host_vector<Scalar>> joint_index_time_code_scale_;

  std::map<joint, thrust::host_vector<Quaternion>> joint_index_rotation_;
  std::map<joint, thrust::host_vector<Scalar>> joint_index_time_code_rotation_;

  // Animation data at right now
  // Notice that only skeletons that have moved will be recorded
  thrust::host_vector<Vector3> t_;
  thrust::host_vector<Vector3> s_;
  thrust::host_vector<Quaternion> r_;

  thrust::device_vector<Mat4x4> joint_world_matrix;

  Scalar total_time_ = 0;
  Scalar current_time_ = -1;

  bool loop_ = true;
  Scalar blend_in_time_ = 0;
  Scalar blend_out_time_ = 0;
  Scalar play_rate_ = 1;

  Scalar animation_time_ = 0;
};

}  // namespace XRTailor
