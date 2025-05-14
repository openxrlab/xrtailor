#include <xrtailor/runtime/rag_doll/gltf/SkinInfo.cuh>

#include <xrtailor/math/Transform3x3.cuh>

namespace XRTailor {

SkinInfo::SkinInfo() {}

SkinInfo::~SkinInfo() {
  for (size_t i = 0; i < skin_num_; i++) {
    v_joint_weight_0[i].clear();
    v_joint_weight_1[i].clear();
    v_joint_id_0[i].clear();
    v_joint_id_1[i].clear();
  }
  v_joint_weight_0.clear();
  v_joint_weight_1.clear();
  v_joint_id_0.clear();
  v_joint_id_1.clear();

  for (auto it : skin_verts_range) {
    it.second.clear();
  }
  skin_verts_range.clear();
}

void SkinInfo::PushBackData(const thrust::host_vector<Vector4>& weight_0,
                             const thrust::host_vector<Vector4>& weight_1,
                             const thrust::host_vector<Vector4>& id_0,
                             const thrust::host_vector<Vector4>& id_1) {
  skin_num_++;

  v_joint_weight_0.resize(skin_num_);
  v_joint_weight_1.resize(skin_num_);
  v_joint_id_0.resize(skin_num_);
  v_joint_id_1.resize(skin_num_);

  this->v_joint_weight_0[skin_num_ - 1] = weight_0;
  this->v_joint_weight_1[skin_num_ - 1] = weight_1;
  this->v_joint_id_0[skin_num_ - 1] = id_0;
  this->v_joint_id_1[skin_num_ - 1] = id_1;
}

void SkinInfo::ClearSkinInfo() {
  for (size_t i = 0; i < skin_num_; i++) {
    v_joint_weight_0[i].clear();
    v_joint_weight_1[i].clear();
    v_joint_id_0[i].clear();
    v_joint_id_1[i].clear();
  }

  skin_num_ = 0;

  v_joint_weight_0.clear();
  v_joint_weight_1.clear();
  v_joint_id_0.clear();
  v_joint_id_1.clear();

  for (auto it : skin_verts_range) {
    it.second.clear();
  }
  skin_verts_range.clear();
}

int SkinInfo::Size() {
  return skin_num_;
}

void SkinInfo::SetInitialPosition(thrust::host_vector<Vector3> h_pos) {
  this->initial_position = h_pos;
}

void SkinInfo::SetInitialNormal(thrust::host_vector<Vector3> h_normal) {
  this->initial_normal = h_normal;
}

}  // namespace XRTailor