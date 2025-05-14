#include <xrtailor/runtime/rag_doll/gltf/JointInfo.cuh>

#include <xrtailor/runtime/rag_doll/gltf/GltfFunc.cuh>

#include <xrtailor/core/DeviceHelper.cuh>

namespace XRTailor {

Pose::Pose() {}

Pose::Pose(thrust::host_vector<Vector3> t, thrust::host_vector<Vector3> s,
           thrust::host_vector<Quaternion> r) {
  t = t;
  r = r;
  s = s;
}

Pose::~Pose() {
  t.clear();
  r.clear();
  s.clear();
}

int Pose::Size() {
  return static_cast<int>(t.size());
}

void Pose::Resize(int newSize) {
  t.resize(newSize);
  r.resize(newSize);
  s.resize(newSize);
}

JointInfo::JointInfo(){};

__global__ void UpdateLocalMatrix(  // need to consider original binding
    Vector3* translation, Vector3* scale, Quaternion* rotation, Mat4* local_matrix, int* joint_ids,
    int n_joint) {
  GET_CUDA_ID(p_id, n_joint);

  int joint = joint_ids[p_id];

  Mat4 r = rotation[joint].ToMatrix4x4();
  Mat4 s = Mat4(scale[joint][0], 0, 0, 0, 0, scale[joint][1], 0, 0, 0, 0, scale[joint][2], 0, 0, 0,
                0, 1);
  Mat4 t = Mat4(1, 0, 0, translation[joint][0], 0, 1, 0, translation[joint][1], 0, 0, 1,
                translation[joint][2], 0, 0, 0, 1);
  local_matrix[joint] = t * s * r;

  Mat4 c = local_matrix[joint];
}

void JointInfo::UpdateWorldMatrixByTransform() {
  thrust::device_vector<int> joint_ids = all_joints;

  int n_joint = joint_ids.size();

  CUDA_CALL(UpdateLocalMatrix, n_joint)
  (pointer(current_translation), pointer(current_scale), pointer(current_rotation),
   pointer(joint_local_matrix), pointer(joint_ids), n_joint);
  CUDA_CHECK_LAST();

  thrust::host_vector<Mat4> c_joint_mat4;
  c_joint_mat4.resize(max_joint_id + 1);

  thrust::host_vector<Mat4> c_joint_local_matrix = joint_local_matrix;

  for (size_t i = 0; i < n_joint; i++) {
    joint joint_id = all_joints[i];
    const thrust::host_vector<int>& j_d = GetJointDirByJointIndex(joint_id, joint_dir);

    Mat4 temp_matrix(1);

    for (int k = j_d.size() - 1; k >= 0; k--) {
      joint select = j_d[k];
      temp_matrix *= c_joint_local_matrix[select];
    }
    c_joint_mat4[joint_id] = temp_matrix;
  }
}

void JointInfo::SetJointName(const std::map<int, std::string> name) {
  this->joint_name = name;
}

void JointInfo::SetJoint(const JointInfo& j) {
  joint_inverse_bind_matrix = j.joint_inverse_bind_matrix;
  joint_local_matrix = j.joint_local_matrix;
  joint_world_matrix = j.joint_world_matrix;
  all_joints = j.all_joints;
  joint_dir = j.joint_dir;
  max_joint_id = j.max_joint_id;
  bind_pose_translation = j.bind_pose_translation;
  bind_pose_scale = j.bind_pose_scale;
  bind_pose_rotation = j.bind_pose_rotation;

  current_translation = j.current_translation;
  current_rotation = j.current_rotation;
  current_scale = j.current_scale;

  joint_name = j.joint_name;
}

bool JointInfo::IsEmpty() {
  if (joint_inverse_bind_matrix.size() == 0 || joint_local_matrix.size() == 0 ||
      joint_world_matrix.size() == 0)
    return true;

  return false;
}

void JointInfo::UpdateJointInfo(thrust::device_vector<Mat4x4>& _inverse_bind_matrix,
                                thrust::device_vector<Mat4x4>& _local_matrix,
                                thrust::device_vector<Mat4x4>& _world_matrix,
                                thrust::host_vector<int>& _all_joints,
                                std::map<joint, thrust::host_vector<joint>>& _joint_dir,
                                std::map<joint, Vector3>& _bind_pose_translation,
                                std::map<joint, Vector3>& _bind_pose_scale,
                                std::map<joint, Quaternion>& _bind_pose_rotation) {
  joint_inverse_bind_matrix = _inverse_bind_matrix;
  joint_local_matrix = _local_matrix;
  joint_world_matrix = _world_matrix;
  all_joints = _all_joints;
  joint_dir = _joint_dir;
  if (all_joints.size())
    max_joint_id = *(std::max_element(all_joints.begin(), all_joints.end()));

  std::vector<Vector3> tempT;
  std::vector<Vector3> tempS;
  std::vector<Quaternion> tempR;

  bind_pose_translation.resize(max_joint_id + 1);
  bind_pose_scale.resize(max_joint_id + 1);
  bind_pose_rotation.resize(max_joint_id + 1);

  for (auto it : all_joints) {
    auto iterT = _bind_pose_translation.find(it);
    if (iterT != _bind_pose_translation.end())
      bind_pose_translation[it] = _bind_pose_translation[it];
    else
      bind_pose_translation[it] = Vector3(0);

    auto iterS = _bind_pose_scale.find(it);
    if (iterS != _bind_pose_scale.end())
      bind_pose_scale[it] = _bind_pose_scale[it];
    else
      bind_pose_scale[it] = Vector3(1);

    auto iterR = _bind_pose_rotation.find(it);
    if (iterR != _bind_pose_rotation.end())
      bind_pose_rotation[it] = _bind_pose_rotation[it];
    else
      bind_pose_rotation[it] = Quaternion();
  }
}

JointInfo::~JointInfo() {
  joint_inverse_bind_matrix.clear();
  joint_inverse_bind_matrix.clear();
  joint_local_matrix.clear();
  joint_world_matrix.clear();

  bind_pose_translation.clear();
  bind_pose_scale.clear();
  bind_pose_rotation.clear();

  current_translation.clear();
  current_rotation.clear();
  current_scale.clear();

  all_joints.clear();
  joint_dir.clear();
}

JointInfo::JointInfo(thrust::device_vector<Mat4x4>& inverse_bind_matrix,
                     thrust::device_vector<Mat4x4>& local_matrix,
                     thrust::device_vector<Mat4x4>& world_matrix,
                     thrust::host_vector<int>& all_joints,
                     std::map<joint, thrust::host_vector<joint>>& joint_dir,
                     std::map<joint, Vector3>& bind_pose_translation,
                     std::map<joint, Vector3>& bind_pose_scale,
                     std::map<joint, Quaternion>& bind_pose_rotation) {
  UpdateJointInfo(inverse_bind_matrix, local_matrix, world_matrix, all_joints, joint_dir,
                  bind_pose_translation, bind_pose_scale, bind_pose_rotation);
}

JointAnimationInfo::JointAnimationInfo(){};

void JointAnimationInfo::SetAnimationData(
    std::map<joint, thrust::host_vector<Vector3>>& _joint_translation,
    std::map<joint, thrust::host_vector<Scalar>>& _joint_time_code_translation,
    std::map<joint, thrust::host_vector<Vector3>>& _joint_scale,
    std::map<joint, thrust::host_vector<Scalar>>& _joint_index_time_code_scale,
    std::map<joint, thrust::host_vector<Quaternion>>& _joint_rotation,
    std::map<joint, thrust::host_vector<Scalar>>& _joint_index_rotation,
    std::shared_ptr<JointInfo> skeleton) {
  joint_index_translation_ = _joint_translation;
  joint_index_time_code_translation_ = _joint_time_code_translation;

  joint_index_scale_ = _joint_scale;
  joint_index_time_code_scale_ = _joint_index_time_code_scale;

  joint_index_rotation_ = _joint_rotation;
  joint_index_time_code_rotation_ = _joint_index_rotation;
  skeleton = skeleton;

  t_.resize(skeleton->max_joint_id + 1);
  s_.resize(skeleton->max_joint_id + 1);
  r_.resize(skeleton->max_joint_id + 1);

  float start_R = NULL;
  float end_R = NULL;
  for (auto it : joint_index_time_code_rotation_) {
    {
      float temp_min = *std::min_element(it.second.begin(), it.second.end());

      if (start_R == NULL)
        start_R = temp_min;
      else
        start_R = start_R < temp_min ? start_R : temp_min;
    }

    {
      float temp_max = *std::max_element(it.second.begin(), it.second.end());

      if (end_R == NULL)
        end_R = temp_max;
      else
        end_R = end_R > temp_max ? end_R : temp_max;
    }
  }

  float start_T = NULL;
  float end_T = NULL;
  for (auto it : joint_index_time_code_translation_) {
    {
      float temp_min = *std::min_element(it.second.begin(), it.second.end());

      if (start_T == NULL)
        start_T = temp_min;
      else
        start_T = start_T < temp_min ? start_T : temp_min;
    }

    {
      float temp_max = *std::max_element(it.second.begin(), it.second.end());

      if (end_T == NULL)
        end_T = temp_max;
      else
        end_T = end_T > temp_max ? end_T : temp_max;
    }
  }

  float start_S = NULL;
  float end_S = NULL;
  for (auto it : joint_index_time_code_scale_) {
    {
      float temp_min = *std::min_element(it.second.begin(), it.second.end());

      if (start_S == NULL)
        start_S = temp_min;
      else
        start_S = start_S < temp_min ? start_S : temp_min;
    }

    {
      float temp_max = *std::max_element(it.second.begin(), it.second.end());

      if (end_S == NULL)
        end_S = temp_max;
      else
        end_S = end_S > temp_max ? end_S : temp_max;
    }
  }

  float time_min =
      (start_T < start_R ? start_T : start_R) < start_S ? (start_T < start_R ? start_T : start_R) : start_S;
  float time_max = (end_T > end_R ? end_T : end_R) > end_S ? (end_T > end_R ? end_T : end_R) : end_S;

  this->total_time_ = time_max - time_min;
  if (time_min != 0) {
    for (auto it : joint_index_time_code_rotation_)
      for (size_t i = 0; i < it.second.size(); i++) {
        it.second[i] = it.second[i] - time_min;
      }
    for (auto it : joint_index_time_code_translation_)
      for (size_t i = 0; i < it.second.size(); i++) {
        it.second[i] = it.second[i] - time_min;
      }
    for (auto it : joint_index_time_code_scale_)
      for (size_t i = 0; i < it.second.size(); i++) {
        it.second[i] = it.second[i] - time_min;
      }
  }
}

void JointAnimationInfo::UpdateJointsTransform(Scalar time) {
  if (current_time_ == time * play_rate_)
    return;

  current_time_ = time * play_rate_;

  if (skeleton != NULL) {
    for (size_t i = 0; i < skeleton->all_joints.size(); i++) {
      joint select = skeleton->all_joints[i];
      UpdateTransform(select, time);
    }
  }
}

Transform3x3 JointAnimationInfo::UpdateTransform(joint select, Scalar time) {
  auto iter_R = joint_index_rotation_.find(select);
  if (iter_R != joint_index_rotation_.end())
    r_[select] = iter_R->second[(int)time];
  else
    r_[select] = skeleton->bind_pose_rotation[select];

  {
    //Rotation
    if (iter_R != joint_index_rotation_.end()) {
      const thrust::host_vector<Quaternion>& all_R = joint_index_rotation_[select];
      const thrust::host_vector<Scalar>& tTimeCode = joint_index_time_code_rotation_[select];

      int t_id = FindMaxSmallerIndex(tTimeCode, time);

      if (t_id >= all_R.size() - 1)  //   [size-1]<=[t_id]
      {
        r_[select] = all_R[all_R.size() - 1];
      } else {
        if (all_R[t_id] != all_R[t_id + 1]) {
          float weight = (time - tTimeCode[t_id]) / (tTimeCode[t_id + 1] - tTimeCode[t_id]);
          r_[select] = SLerp(all_R[t_id], all_R[t_id + 1], weight);
        }
      }
    } else {
      r_[select] = skeleton->bind_pose_rotation[select];
    }
  }

  //Translation
  auto iter_T = joint_index_translation_.find(select);
  if (iter_T != joint_index_translation_.end())
    t_[select] = iter_T->second[(int)time];
  else
    t_[select] = skeleton->bind_pose_translation[select];

  {
    //Translation
    if (iter_T != joint_index_translation_.end()) {
      const thrust::host_vector<Vector3>& all_T = joint_index_translation_[select];
      const thrust::host_vector<Scalar>& t_time_code = joint_index_time_code_translation_[select];

      int tId = FindMaxSmallerIndex(t_time_code, time);

      if (tId >= all_T.size() - 1) {
        t_[select] = all_T[all_T.size() - 1];
      } else {
        if (all_T[tId] != all_T[tId + 1]) {
          float weight = (time - t_time_code[tId]) / (t_time_code[tId + 1] - t_time_code[tId]);
          t_[select] = Lerp(all_T[tId], all_T[tId + 1], weight);
        }
      }
    } else {
      t_[select] = skeleton->bind_pose_translation[select];
    }
  }

  //Scale
  auto iter_S = joint_index_scale_.find(select);
  if (iter_S != joint_index_scale_.end())
    s_[select] = iter_S->second[(int)time];
  else
    s_[select] = skeleton->bind_pose_scale[select];

  {
    //Scale
    if (iter_S != joint_index_scale_.end()) {
      const thrust::host_vector<Vector3>& all_S = joint_index_scale_[select];
      const thrust::host_vector<Scalar>& t_time_code = joint_index_time_code_scale_[select];

      int t_id = FindMaxSmallerIndex(t_time_code, time);

      if (t_id >= all_S.size() - 1) {
        s_[select] = all_S[all_S.size() - 1];
      } else {
        if (all_S[t_id] != all_S[t_id + 1]) {
          float weight = (time - t_time_code[t_id]) / (t_time_code[t_id + 1] - t_time_code[t_id]);
          s_[select] = Lerp(all_S[t_id], all_S[t_id + 1], weight);
        }
      }
    } else {
      s_[select] = skeleton->bind_pose_scale[select];
    }
  }

  return Transform3x3(t_[select], r_[select].ToMatrix3x3(), s_[select]);
}

thrust::host_vector<Vector3> JointAnimationInfo::GetJointsTranslation(Scalar time) {
  UpdateJointsTransform(time);
  return t_;
}

thrust::host_vector<Quaternion> JointAnimationInfo::GetJointsRotation(Scalar time) {
  UpdateJointsTransform(time);
  return r_;
}

thrust::host_vector<Vector3> JointAnimationInfo::GetJointsScale(Scalar time) {
  UpdateJointsTransform(time);
  return s_;
}

Scalar JointAnimationInfo::GetTotalTime() {
  return total_time_;
}

int JointAnimationInfo::FindMaxSmallerIndex(const thrust::host_vector<Scalar>& arr, Scalar v) {
  int left = 0;
  int right = arr.size() - 1;
  int max_index = -1;

  if (arr.size() >= 1) {
    if (arr[0] > v)
      return 0;

    if (arr[arr.size() - 1] < v)
      return arr.size() - 1;
  }

  while (left <= right) {
    int mid = left + (right - left) / 2;

    if (arr[mid] <= v) {
      max_index = mid;
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }

  return max_index;
}

Quaternion JointAnimationInfo::NLerp(const Quaternion& q1, const Quaternion& q2, Scalar weight) {
  Quaternion temp_Q;

  if (q1.x * q2.x < 0 && q1.y * q2.y < 0 && q1.z * q2.z < 0 && q1.w * q2.w < 0) {
    temp_Q.x = -q2.x;
    temp_Q.y = -q2.y;
    temp_Q.z = -q2.z;
    temp_Q.w = -q2.w;
  } else {
    temp_Q = q2;
  }

  Quaternion result = (1 - weight) * q1 + weight * temp_Q;

  if (result.Norm() < 0.001)
    result = Quaternion();
  else
    result.Normalize();

  return result;
}

Quaternion JointAnimationInfo::SLerp(const Quaternion& q1, const Quaternion& q2, Scalar t) {
  double cos_theta = q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z;
  Quaternion result;

  if (abs(cos_theta) >= 1.0) {
    result.w = q1.w;
    result.x = q1.x;
    result.y = q1.y;
    result.z = q1.z;

    return result;
  }

  // set q2 = -q2 if inner product is negative to ensure the shortest path
  Quaternion q2_adjusted = q2;
  if (cos_theta < 0) {
    q2_adjusted.w = -q2.w;
    q2_adjusted.x = -q2.x;
    q2_adjusted.y = -q2.y;
    q2_adjusted.z = -q2.z;
    cos_theta = -cos_theta;
  }

  double theta = std::acos(cos_theta);
  double sin_theta = std::sin(theta);
  double weight1 = std::sin((1 - t) * theta) / sin_theta;
  double weight2 = std::sin(t * theta) / sin_theta;

  result.w = q1.w * weight1 + q2_adjusted.w * weight2;
  result.x = q1.x * weight1 + q2_adjusted.x * weight2;
  result.y = q1.y * weight1 + q2_adjusted.y * weight2;
  result.z = q1.z * weight1 + q2_adjusted.z * weight2;

  if (result.Norm() < 0.001)
    result = Quaternion();
  else
    result.Normalize();

  return result;
}

thrust::host_vector<int> JointAnimationInfo::GetJointDir(
    int index, std::map<int, thrust::host_vector<int>> joint_dir) {
  std::map<int, thrust::host_vector<int>>::const_iterator iter = joint_dir.find(index);
  if (iter == joint_dir.end()) {
    std::cout << "Error: not found JointIndex \n";

    std::vector<int> empty;
    return empty;
  }
  return iter->second;
}

void JointAnimationInfo::SetLoop(bool loop) {
  loop_ = loop;
}

Pose JointAnimationInfo::GetPose(Scalar in_time) {
  Scalar time = in_time;

  if (this->loop_) {
    time = fmod(time, total_time_);
  }

  auto t = this->GetJointsTranslation(time);
  auto s = this->GetJointsScale(time);
  auto r = this->GetJointsRotation(time);

  return Pose(t, s, r);
}

Scalar JointAnimationInfo::GetCurrentAnimationTime() {
  return current_time_;
}

Scalar& JointAnimationInfo::GetBlendInTime() {
  return blend_in_time_;
}

Scalar& JointAnimationInfo::GetBlendOutTime() {
  return blend_out_time_;
}

Scalar& JointAnimationInfo::GetPlayRate() {
  return play_rate_;
}

Quaternion JointAnimationInfo::Normalize(const Quaternion& q) {
  Scalar norm = sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
  return {q.w / norm, q.x / norm, q.y / norm, q.z / norm};
}

Vector3 JointAnimationInfo::Lerp(Vector3 v0, Vector3 v1, Scalar weight) {
  return v0 + (v1 - v0) * weight;
}

JointAnimationInfo::~JointAnimationInfo() {
  joint_index_translation_.clear();
  joint_index_time_code_translation_.clear();
  joint_index_scale_.clear();
  joint_index_time_code_scale_.clear();
  joint_index_rotation_.clear();
  joint_index_time_code_rotation_.clear();
  t_.clear();
  s_.clear();
  r_.clear();
  joint_world_matrix.clear();
  skeleton = NULL;
};

}  // namespace XRTailor
