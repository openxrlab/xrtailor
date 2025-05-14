#pragma once

#include <set>
#include <map>
#include <string>
#include <algorithm>

#include <xrtailor/core/Scalar.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace XRTailor {

class SkinInfo {
 public:
  SkinInfo();

  ~SkinInfo();

  void PushBackData(const thrust::host_vector<Vector4>& weight_0,
                    const thrust::host_vector<Vector4>& weight_1,
                    const thrust::host_vector<Vector4>& id_0,
                    const thrust::host_vector<Vector4>& id_1);

  void ClearSkinInfo();

  int Size();

  void SetInitialPosition(thrust::host_vector<Vector3> h_pos);

  void SetInitialNormal(thrust::host_vector<Vector3> h_normal);

  thrust::host_vector<thrust::device_vector<Vector4>> v_joint_weight_0;
  thrust::host_vector<thrust::device_vector<Vector4>> v_joint_weight_1;

  thrust::host_vector<thrust::device_vector<Vector4>> v_joint_id_0;
  thrust::host_vector<thrust::device_vector<Vector4>> v_joint_id_1;

  std::map<int, thrust::host_vector<Vector2u>> skin_verts_range;

  thrust::device_vector<Vector3> initial_position;
  thrust::device_vector<Vector3> initial_normal;

 private:
  int skin_num_ = 0;
};

}  // namespace XRTailor
