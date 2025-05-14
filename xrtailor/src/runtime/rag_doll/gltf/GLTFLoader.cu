#include <xrtailor/runtime/rag_doll/gltf/GLTFLoader.cuh>

#include <iomanip>

#include <xrtailor/runtime/rag_doll/gltf/Defs.hpp>
#include <xrtailor/runtime/rag_doll/gltf/GltfFunc.cuh>
#include <xrtailor/runtime/rag_doll/gltf/GLTFHelper.cuh>
#include <xrtailor/core/DeviceHelper.cuh>

#include <thrust/transform.h>

namespace XRTailor {

void LogDeviceMat4(const thrust::device_vector<Mat4>& d_mat) {
  thrust::host_vector<Mat4> h_mat = d_mat;
  for (auto iter = h_mat.begin(); iter != h_mat.end(); ++iter) {
    int cnt = 0;
    std::cout << std::fixed << std::setprecision(4) << "[" << (*iter)[0][0] << ", " << (*iter)[0][1]
              << ", " << (*iter)[0][2] << ", " << (*iter)[0][3] << ", " << (*iter)[1][0] << ", "
              << (*iter)[1][1] << ", " << (*iter)[1][2] << ", " << (*iter)[1][3] << ", "
              << (*iter)[2][0] << ", " << (*iter)[2][1] << ", " << (*iter)[2][2] << ", "
              << (*iter)[2][3] << ", " << (*iter)[3][0] << ", " << (*iter)[3][1] << ", "
              << (*iter)[3][2] << ", " << (*iter)[3][3] << "] ";
    cnt++;
    if (cnt % 1 == 0)
      std::cout << std::endl;
  }
  std::cout << std::endl;
}

void LogDeviceInt(const thrust::device_vector<int>& d_vec) {
  thrust::host_vector<int> h_vec = d_vec;
  for (auto iter = h_vec.begin(); iter != h_vec.end(); ++iter) {
    std::cout << (*iter) << " ";
  }
  std::cout << std::endl;
}

void LogDeviceVector3(const thrust::device_vector<Vector3>& d_vec) {
  thrust::host_vector<Vector3> h_vec = d_vec;
  int cnt = 0;
  for (auto iter = h_vec.begin(); iter != h_vec.end(); ++iter) {
    std::cout << std::fixed << std::setprecision(4) << "(" << (*iter)[0] << ", " << (*iter)[1]
              << ", " << (*iter)[2] << ") ";
    cnt++;
    if (cnt % 3 == 0)
      std::cout << std::endl;
  }
  std::cout << std::endl;
}

void LogDeviceVector4(const thrust::device_vector<Vector4>& d_vec) {
  thrust::host_vector<Vector4> h_vec = d_vec;
  int cnt = 0;
  for (auto iter = h_vec.begin(); iter != h_vec.end(); ++iter) {
    std::cout << std::fixed << std::setprecision(4) << "(" << (*iter)[0] << ", " << (*iter)[1]
              << ", " << (*iter)[2] << ", " << (*iter)[3] << ") ";
    cnt++;
    if (cnt % 3 == 0)
      std::cout << std::endl;
  }
  std::cout << std::endl;
}

GLTFLoader::GLTFLoader() {
  this->skin = std::make_shared<SkinInfo>();
  this->joints_data = std::make_shared<JointInfo>();
  this->animation = std::make_shared<JointAnimationInfo>();
}

GLTFLoader::~GLTFLoader() {}

bool LoadImageData(tinygltf::Image* /* image */, const int /* image_idx */, std::string* /* err */,
                   std::string* /* warn */, int /* req_width */, int /* req_height */,
                   const unsigned char* /* bytes */, int /* size */, void* /*user_data */) {
  return true;
}

void GLTFLoader::InsertMidstepAnimation(
    std::map<joint, Vector3> init_joint_translation, std::map<joint, Vector3> init_joint_scale,
    std::map<joint, Quaternion> init_joint_rotation,
    std::map<joint, thrust::host_vector<Vector3>>& joint_T_f_anim,
    std::map<joint, thrust::host_vector<Scalar>>& joint_T_Time,
    std::map<joint, thrust::host_vector<Vector3>>& joint_S_f_anim,
    std::map<joint, thrust::host_vector<Scalar>>& joint_S_Time,
    std::map<joint, thrust::host_vector<Quaternion>>& joint_R_f_anim,
    std::map<joint, thrust::host_vector<Scalar>>& joint_R_Time, Scalar n_midstep_frames, Scalar fps) {

  Scalar delta_time = 1 / fps;
  Scalar total_time = delta_time * n_midstep_frames;

  thrust::host_vector<Vector3> midstep_T(n_midstep_frames);
  thrust::host_vector<Vector3> midstep_S(n_midstep_frames);
  thrust::host_vector<Quaternion> midstep_R(n_midstep_frames);

  thrust::host_vector<Scalar> midstep_T_Time(n_midstep_frames);
  thrust::host_vector<Scalar> midstep_S_Time(n_midstep_frames);
  thrust::host_vector<Scalar> midstep_R_Time(n_midstep_frames);

  for (int step = 0; step < n_midstep_frames; step++) {
    Scalar current_time_ = delta_time * step;
    midstep_T_Time[step] = current_time_;
    midstep_S_Time[step] = current_time_;
    midstep_R_Time[step] = current_time_;
  }

  Scalar new_overall_time = delta_time * n_midstep_frames;

  for (auto iter = joint_T_Time.begin(); iter != joint_T_Time.end(); ++iter) {
    thrust::transform(iter->second.begin(), iter->second.end(), iter->second.begin(),
                      [new_overall_time] __host__(Scalar x) { return x + new_overall_time; });
    iter->second.insert(iter->second.begin(), midstep_T_Time.begin(), midstep_T_Time.end());
  }

  for (auto iter = joint_S_Time.begin(); iter != joint_S_Time.end(); ++iter) {
    thrust::transform(iter->second.begin(), iter->second.end(), iter->second.begin(),
                      [new_overall_time] __host__(Scalar x) { return x + new_overall_time; });
    iter->second.insert(iter->second.begin(), midstep_S_Time.begin(), midstep_S_Time.end());
  }

  for (auto iter = joint_R_Time.begin(); iter != joint_R_Time.end(); ++iter) {
    thrust::transform(iter->second.begin(), iter->second.end(), iter->second.begin(),
                      [new_overall_time] __host__(Scalar x) { return x + new_overall_time; });
    iter->second.insert(iter->second.begin(), midstep_R_Time.begin(), midstep_R_Time.end());
  }

  // trans
  for (auto iter = joint_T_f_anim.begin(); iter != joint_T_f_anim.end(); ++iter) {

    auto res = init_joint_translation.find(iter->first);
    if (res == init_joint_translation.end()) {
      std::cout << "Cannot find init translation at joint " << iter->first << std::endl;
      exit(0);
    }
    Vector3 t0 = res->second;

    Vector3 t1 = iter->second[0];
    for (int step = 0; step < n_midstep_frames; step++) {
      Scalar weight = step / n_midstep_frames;
      Vector3 t = this->animation->Lerp(t0, t1, weight);
      midstep_T[step] = t;
    }
    iter->second.insert(iter->second.begin(), midstep_T.begin(), midstep_T.end());
  }

  // scale
  for (auto iter = joint_S_f_anim.begin(); iter != joint_S_f_anim.end(); ++iter) {
    auto res = init_joint_scale.find(iter->first);
    if (res == init_joint_scale.end()) {
      std::cout << "Cannot find init scale at joint " << iter->first << std::endl;
      exit(0);
    }
    Vector3 s0 = res->second;

    Vector3 s1 = iter->second[0];
    for (int step = 0; step < n_midstep_frames; step++) {
      Scalar weight = step / n_midstep_frames;
      Vector3 s = this->animation->Lerp(s0, s1, weight);
      midstep_S[step] = s;
    }
    iter->second.insert(iter->second.begin(), midstep_S.begin(), midstep_S.end());
  }

  // rot
  for (auto iter = joint_R_f_anim.begin(); iter != joint_R_f_anim.end(); ++iter) {
    auto res = init_joint_rotation.find(iter->first);
    if (res == init_joint_rotation.end()) {
      std::cout << "Cannot find init rotation at joint " << iter->first << std::endl;
      exit(0);
    }
    Quaternion r0 = res->second;
    Quaternion r1 = iter->second[0];

    for (int step = 0; step < n_midstep_frames; step++) {
      Scalar weight = step / n_midstep_frames;
      Quaternion r = this->animation->SLerp(r0, r1, weight);
      midstep_R[step] = r;
    }
    iter->second.insert(iter->second.begin(), midstep_R.begin(), midstep_R.end());
  }
}

void GLTFLoader::LoadGltf(std::string file_path,
                          MeshData& mesh_data) {

  this->UpdateTransformState();
  this->InitializeData();

  tinygltf::Model* new_model = new tinygltf::Model;
  tinygltf::Model model = *new_model;
  delete new_model;

  tinygltf::TinyGLTF loader;
  std::string err;
  std::string warn;

  loader.SetImageLoader(LoadImageData, nullptr);
  bool ret = loader.LoadBinaryFromFile(&model, &err, &warn, file_path);  // for binary glTF(.glb)

  if (!warn.empty()) {
    printf("Warn: %s\n", warn.c_str());
  }

  if (!err.empty()) {
    printf("Err: %s\n", err.c_str());
  }

  if (!ret) {
    printf("Failed to parse glTF\n");
    return;
  }

  ImportAnimation(model, joint_output, joint_input, joint_T_f_anim, joint_T_Time, joint_S_f_anim,
                  joint_S_Time, joint_R_f_anim, joint_R_Time);

  for (int i = 0; i < model.nodes.size(); i++) {
    node_name[i] = model.nodes[i].name;
  }

  // import scenes
  std::map<scene, thrust::host_vector<int>> scene_nodes;
  for (size_t i = 0; i < model.scenes.size(); i++) {
    std::vector<int> vec_scene_roots;
    vec_scene_roots = model.scenes[i].nodes;
    scene_nodes[i] = vec_scene_roots;
  }

  GetNodesAndHierarchy(model, scene_nodes, all_nodes, node_id_dir);

  UpdateJointMeshCameraDir(model, joint_num, mesh_num, joint_id_joint_dir, all_joints, all_nodes,
                              node_id_dir, mesh_id_dir, all_meshes, d_joints, max_joint_id);

  thrust::host_vector<thrust::host_vector<int>> joint_child;  //build edgeset;

  //get Local Transform T S R M
  GetJointsTransformData(all_nodes, joint_name, joint_child, joint_rotation, joint_scale,
                         joint_translation, joint_matrix, model);

  //get InverseBindMatrix (Global)
  BuildInverseBindMatrices(all_joints);

  thrust::host_vector<Mat4> local_matrix;
  local_matrix.resize(max_joint_id + 1);

  for (auto jId : all_joints) {
    local_matrix[jId] = joint_matrix[jId];
  }

  this->joint_local_matrix = local_matrix;

  // get joint world Location
  {
    thrust::host_vector<Vector3> joint_vertices;
    std::map<int, int> joint_id_v_id;

    for (size_t j = 0; j < joint_num; j++) {
      joint j_id = all_joints[j];
      joint_id_v_id[j_id] = joint_vertices.size();

      joint_vertices.push_back(
          GetVertexLocationWithJointTransform(j_id, Vector3(0, 0, 0), joint_matrix));
    }

    this->joint_world_position = joint_vertices;
  }

  thrust::host_vector<Vector3> raw_positions;
  thrust::host_vector<Vector3> raw_normals;
  thrust::host_vector<Vector3> raw_uvs;
  std::vector<std::vector<Index>> raw_indices;

  thrust::host_vector<unsigned int> face_indices;
  int shape_num = 0;

  for (auto mesh_id : model.meshes) {
    shape_num += mesh_id.primitives.size();
  }

  int primitive_point_offest;
  int current_shape = 0;
  std::map<int, int> shape_mesh_id;

  //skin verts range;
  {
    int temp_shape_id = 0;
    int temp_size = 0;
    for (int m_id = 0; m_id < model.meshes.size(); m_id++) {
      // import mesh
      thrust::host_vector<uint> vertex_index;
      thrust::host_vector<uint> normal_index;
      thrust::host_vector<uint> tex_coord_index;

      int prim_num = model.meshes[m_id].primitives.size();

      for (size_t p_id = 0; p_id < prim_num; p_id++)  //shape
      {
        primitive_point_offest = (raw_positions.size());

        //current primitive
        const tinygltf::Primitive& primitive = model.meshes[m_id].primitives[p_id];

        std::map<std::string, int> attributes_name = primitive.attributes;

        //Set Vertices
        GetVec3ByAttributeName(model, primitive, std::string("POSITION"), raw_positions);
        skin_verts_range[temp_shape_id].push_back(Vector2u(temp_size, raw_positions.size() - 1));
        temp_shape_id++;
        temp_size = raw_positions.size();

        //Set Normal
        GetVec3ByAttributeName(model, primitive, std::string("NORMAL"), raw_normals);

        //Set TexCoord
        GetVec3ByAttributeName(model, primitive, std::string("TEXCOORD_0"), raw_uvs);

        //Set Triangles
        if (primitive.mode == TINYGLTF_MODE_TRIANGLES) {

          thrust::host_vector<uint> temp_face_indices;

          GetTriangles(model, primitive, temp_face_indices, primitive_point_offest);
          vertex_index = (temp_face_indices);
          normal_index = (temp_face_indices);
          tex_coord_index = (temp_face_indices);

          shape_mesh_id[current_shape] = m_id;  // set shapeId - mesh_id;
          for (int i = 0; i < temp_face_indices.size() / 3; i++) {
            std::vector<Index> face_indices;
            face_indices.push_back(Index(static_cast<int>(vertex_index[i * 3 + 0]),
                                        static_cast<int>(normal_index[i * 3 + 0]),
                                        static_cast<int>(tex_coord_index[i * 3 + 0])));
            face_indices.push_back(Index(static_cast<int>(vertex_index[i * 3 + 1]),
                                        static_cast<int>(normal_index[i * 3 + 1]),
                                        static_cast<int>(tex_coord_index[i * 3 + 1])));
            face_indices.push_back(Index(static_cast<int>(vertex_index[i * 3 + 2]),
                                        static_cast<int>(normal_index[i * 3 + 2]),
                                        static_cast<int>(tex_coord_index[i * 3 + 2])));
            raw_indices.push_back(face_indices);
          }
        }
        current_shape++;
      }
    }
  }

  // glue code need to be removed later
  std::vector<Vector3> dirty_positions =
      std::vector<Vector3>{raw_positions.begin(), raw_positions.end()};
  std::vector<Vector3> dirty_normals = std::vector<Vector3>{raw_normals.begin(), raw_normals.end()};
  std::vector<Vector3> dirty_uvs = std::vector<Vector3>{raw_uvs.begin(), raw_uvs.end()};
  std::vector<std::vector<Index>> dirty_indices =
      std::vector<std::vector<Index>>{raw_indices.begin(), raw_indices.end()};

  mesh_data.positions = dirty_positions;
  mesh_data.normals = dirty_normals;
  mesh_data.uvs = dirty_uvs;
  mesh_data.indices = dirty_indices;

  // import skin;
  {
    std::map<int, std::vector<joint>> shape_skin_joint;

    for (size_t i = 0; i < model.skins.size(); i++) {
      auto joints = model.skins[i].joints;
      shape_skin_joint[i] = joints;
    }

    std::map<int, std::vector<joint>> skin_node_mesh_id;
    std::map<mesh, std::vector<shape>> mesh_node_primitive;

    for (size_t i = 0; i < model.nodes.size(); i++) {
      if (model.nodes[i].skin != -1) {
        skin_node_mesh_id[i].push_back(model.nodes[i].mesh);
      }
    }

    {
      int temp_shape_id = 0;

      for (int m_id = 0; m_id < model.meshes.size(); m_id++) {
        int prim_num = model.meshes[m_id].primitives.size();

        for (size_t p_id = 0; p_id < prim_num; p_id++) {
          mesh_node_primitive[m_id].push_back(temp_shape_id);
          temp_shape_id++;
        }
      }
    }
    this->skin->ClearSkinInfo();
    {
      int temp_shape_id = 0;
      for (int m_id = 0; m_id < model.meshes.size(); m_id++) {
        int prim_num = model.meshes[m_id].primitives.size();

        for (size_t p_id = 0; p_id < prim_num; p_id++) {
          std::vector<joint> skin_joints;

          if (shape_skin_joint.find(0) != shape_skin_joint.end())
            skin_joints = shape_skin_joint[temp_shape_id];

          if (skin_joints.size()) {

            thrust::host_vector<Vector4> weight0;
            thrust::host_vector<Vector4> weight1;

            thrust::host_vector<Vector4> joint0;
            thrust::host_vector<Vector4> joint1;

            GetVec4ByAttributeName(model, model.meshes[m_id].primitives[p_id],
                                   std::string("WEIGHTS_0"), weight0);  //

            GetVec4ByAttributeName(model, model.meshes[m_id].primitives[p_id],
                                   std::string("WEIGHTS_1"), weight1);  //

            GetVertexBindJoint(model, model.meshes[m_id].primitives[p_id], std::string("JOINTS_0"),
                               joint0, skin_joints);

            GetVertexBindJoint(model, model.meshes[m_id].primitives[p_id], std::string("JOINTS_1"),
                               joint1, skin_joints);


            this->skin->PushBackData(weight0, weight1, joint0, joint1);
          }
        }
      }
    }

    for (auto it : skin_verts_range) {
      this->skin->skin_verts_range[it.first] = it.second;
    }
  }

  thrust::host_vector<int> c_shape_mesh_id;

  c_shape_mesh_id.resize(shape_mesh_id.size());

  std::vector<int> mesh_node_ids;

  for (size_t n_id = 0; n_id < model.nodes.size(); n_id++) {
    int j = 0;

    if (model.nodes[n_id].mesh >= 0) {
      j++;
      mesh_node_ids.push_back(n_id);
    }
  }

  GetMeshMatrix(model, mesh_node_ids, max_mesh_id, mesh_matrix);

  for (auto it : shape_mesh_id) {
    c_shape_mesh_id[it.first] = mesh_node_ids[it.second];
  }
  d_shape_mesh_id = c_shape_mesh_id;

  initial_position = raw_positions;
  initial_normal = raw_normals;

  d_mesh_matrix = mesh_matrix;

  this->UpdateTransformState();
  this->skin->SetInitialPosition(initial_position);
  this->skin->SetInitialNormal(initial_normal);

  this->joints_data->UpdateJointInfo(joint_inverse_bind_matrix, joint_local_matrix, joint_world_matrix,
                                    all_joints, joint_id_joint_dir, joint_translation, joint_scale,
                                    joint_rotation);

  this->joints_data->SetJointName(joint_name);
  
  this->InsertMidstepAnimation(joint_translation, joint_scale, joint_rotation, joint_T_f_anim,
                               joint_T_Time, joint_S_f_anim, joint_S_Time, joint_R_f_anim,
                               joint_R_Time, 100, 24);

  this->animation->SetAnimationData(joint_T_f_anim, joint_T_Time, joint_S_f_anim, joint_S_Time,
                                    joint_R_f_anim, joint_R_Time, this->joints_data);
}

void GLTFLoader::UpdateTransformState() {
  Vector3 location = Vector3(0);
  Vector3 scale = Vector3(1);
  Mat4 m_T = Mat4(1, 0, 0, location[0], 0, 1, 0, location[1], 0, 0, 1, location[2], 0, 0, 0, 1);
  Mat4 m_S = Mat4(scale[0], 0, 0, 0, 0, scale[1], 0, 0, 0, 0, scale[2], 0, 0, 0, 0, 1);
  Quaternion rotation;
  Mat4 m_R = rotation.ToMatrix4x4();

  transform = m_T * m_S * m_R;
}

void GLTFLoader::BuildInverseBindMatrices(const thrust::host_vector<joint>& all_joints) {
  std::map<joint, Mat4> temp_joint_matrix = joint_matrix;
  std::vector<Mat4> temp;

  temp.resize(max_joint_id + 1);

  for (size_t i = 0; i < max_joint_id + 1; i++) {
    temp[i] = Mat4(1);
  }

  for (size_t i = 0; i < all_joints.size(); i++) {
    joint joint_id = all_joints[i];

    const thrust::host_vector<int>& j_d = GetJointDirByJointIndex(joint_id, joint_id_joint_dir);

    Mat4 temp_matrix(1);

    for (int k = 0; k < j_d.size(); k++) {
      joint select = j_d[k];

      Vector3 temp_VT(0, 0, 0);
      Vector3 temp_VS(1, 1, 1);
      Quaternion temp_QR = Quaternion(Mat3(1));

      if (joint_input.find(select) != joint_input.end()) {
        temp_QR = joint_rotation[select];

        temp_VT = joint_translation[select];

        temp_VS = joint_scale[select];

        Mat4 m_T = Mat4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, temp_VT[0], temp_VT[1], temp_VT[2], 1);
        Mat4 m_S = Mat4(temp_VS[0], 0, 0, 0, 0, temp_VS[1], 0, 0, 0, 0, temp_VS[2], 0, 0, 0, 0, 1);

        Mat4 m_R = temp_QR.ToMatrix4x4();

        temp_joint_matrix[select] = m_T * m_S * m_R;
      }

      temp_matrix *= glm::inverse(temp_joint_matrix[select]);
    }

    joint_inverse_bind_matrix_map[joint_id] = (temp_matrix);

    temp[joint_id] = temp_matrix;
  }

  this->joint_inverse_bind_matrix = temp;
}

Vector3 GLTFLoader::GetVertexLocationWithJointTransform(joint joint_id, Vector3 in_point,
                                                        std::map<joint, Mat4> j_matrix) {

  Vector3 result = Vector3(0);

  const thrust::host_vector<int>& j_d = GetJointDirByJointIndex(joint_id, joint_id_joint_dir);

  Mat4 temp_matrix = Mat4(1);

  for (int k = j_d.size() - 1; k >= 0; k--) {
    joint select = j_d[k];
    temp_matrix *= j_matrix[select];
  }

  Vector4 joint_location = temp_matrix * Vector4(in_point[0], in_point[1], in_point[2], 1);
  result = Vector3(joint_location[0], joint_location[1], joint_location[2]);

  return result;
};

void GLTFLoader::UpdateAnimationMatrix(const thrust::host_vector<joint>& all_joints,
                                       int current_frame) {
  joint_animation_matrix = joint_matrix;

  for (auto j_id : all_joints) {
    const thrust::host_vector<int>& j_d = GetJointDirByJointIndex(j_id, joint_id_joint_dir);

    Mat4 temp_matrix = Mat4(1);

    for (int k = j_d.size() - 1; k >= 0; k--) {

      joint select = j_d[k];

      Vector3 temp_VT = Vector3(0, 0, 0);
      Vector3 temp_VS = Vector3(1, 1, 1);
      Quaternion temp_QR = Quaternion(Mat3(1));

      if (joint_input.find(select) != joint_input.end())  //ֻ
      {
        auto iter_R = joint_R_f_anim.find(select);
        int temp_frame = 0;

        if (iter_R != joint_R_f_anim.end()) {

          if (current_frame > iter_R->second.size() - 1)
            temp_frame = iter_R->second.size() - 1;
          else if (current_frame < 0)
            temp_frame = 0;
          else
            temp_frame = current_frame;

          temp_QR = iter_R->second[temp_frame];
        } else {
          temp_QR = joint_rotation[select];
        }

        std::map<joint, thrust::host_vector<Vector3>>::const_iterator iter_T;
        iter_T = joint_T_f_anim.find(select);
        if (iter_T != joint_T_f_anim.end()) {
          if (current_frame > iter_T->second.size() - 1)
            temp_frame = iter_T->second.size() - 1;
          else if (current_frame < 0)
            temp_frame = 0;
          else
            temp_frame = current_frame;

          temp_VT = iter_T->second[temp_frame];
        } else {
          temp_VT = joint_translation[select];
        }

        std::map<joint, thrust::host_vector<Vector3>>::const_iterator iter_S;
        iter_S = joint_S_f_anim.find(select);
        if (iter_S != joint_S_f_anim.end()) {
          if (current_frame > iter_S->second.size() - 1)
            temp_frame = iter_S->second.size() - 1;
          else if (current_frame < 0)
            temp_frame = 0;
          else
            temp_frame = current_frame;

          temp_VS = iter_S->second[temp_frame];
        } else {
          temp_VS = joint_scale[select];
        }

        Mat4 m_T = Mat4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, temp_VT[0], temp_VT[1], temp_VT[2], 1);
        Mat4 m_S = Mat4(temp_VS[0], 0, 0, 0, 0, temp_VS[1], 0, 0, 0, 0, temp_VS[2], 0, 0, 0, 0, 1);
        Mat4 m_R = temp_QR.ToMatrix4x4();

        joint_animation_matrix[select] = m_T * m_S * m_R;
      }
      //
    }
  }
}

void GLTFLoader::UpdateJointWorldMatrix(const thrust::host_vector<joint>& all_joints,
                                        std::map<joint, Mat4> jMatrix) {
  thrust::host_vector<Mat4> c_joint_mat4f;

  c_joint_mat4f.resize(max_joint_id + 1, Mat4(1));

  for (size_t i = 0; i < all_joints.size(); i++) {
    joint jointId = all_joints[i];
    const thrust::host_vector<int>& j_d = GetJointDirByJointIndex(jointId, joint_id_joint_dir);

    Mat4 temp_matrix = Mat4(1);

    for (int k = j_d.size() - 1; k >= 0; k--) {
      joint select = j_d[k];
      temp_matrix *= jMatrix[select];  //
    }
    c_joint_mat4f[jointId] = temp_matrix;
  }

  this->joint_world_matrix = std::move(c_joint_mat4f);
}

thrust::host_vector<Vector3> GLTFLoader::TestLBS(int frame_number) {
  thrust::device_vector<Vector3> d_pos;
  d_pos.resize(this->initial_position.size());
  std::cout << "init " << this->initial_position.size() << std::endl;
  UpdateAnimation(frame_number, d_pos);

  checkCudaErrors(cudaDeviceSynchronize());

  thrust::host_vector<Vector3> h_pos = std::move(d_pos);
  return h_pos;
}

void GLTFLoader::UpdateAnimation(int frame_number, Node** nodes, int n_nodes) {
  if (joint_output.empty() || all_joints.empty() || joint_matrix.empty())
    return;

  this->UpdateAnimationMatrix(all_joints, frame_number);
  this->UpdateJointWorldMatrix(all_joints, joint_animation_matrix);

  JointAnimation(joint_world_position, joint_world_matrix, d_joints, transform);

  for (size_t i = 0; i < this->skin->Size(); i++)  //
  {
    auto& bind_joint0 = this->skin->v_joint_id_0[i];
    auto& bind_joint1 = this->skin->v_joint_id_1[i];

    auto& bind_weight0 = this->skin->v_joint_weight_0[i];
    auto& bind_weight1 = this->skin->v_joint_weight_1[i];

    for (size_t j = 0; j < this->skin->skin_verts_range[i].size(); j++) {
      Vector2u& range = this->skin->skin_verts_range[i][j];

      SkinAnimation(pointer(this->initial_position),
                    nodes, pointer(this->joint_inverse_bind_matrix), pointer(this->joint_world_matrix),

                    pointer(bind_joint0), pointer(bind_joint1), pointer(bind_weight0),
                    pointer(bind_weight1),

                    n_nodes, bind_joint0.size(), bind_joint1.size(), this->transform, range);
    }
  }
}

void GLTFLoader::UpdateAnimation(int frame_number, thrust::device_vector<Vector3>& world_positons) {
  if (joint_output.empty() || all_joints.empty() || joint_matrix.empty())
    return;

  this->UpdateAnimationMatrix(all_joints, frame_number);
  this->UpdateJointWorldMatrix(all_joints, joint_animation_matrix);

  JointAnimation(joint_world_position, joint_world_matrix, d_joints, transform);

  for (size_t i = 0; i < this->skin->Size(); i++)  //
  {
    auto& bind_joint0 = this->skin->v_joint_id_0[i];
    auto& bind_joint1 = this->skin->v_joint_id_1[i];

    auto& bind_weight0 = this->skin->v_joint_weight_0[i];
    auto& bind_weight1 = this->skin->v_joint_weight_1[i];

    for (size_t j = 0; j < this->skin->skin_verts_range[i].size(); j++) {
      Vector2u& range = this->skin->skin_verts_range[i][j];
      SkinAnimation(this->initial_position, world_positons, this->joint_inverse_bind_matrix,
                    this->joint_world_matrix,

                    bind_joint0, bind_joint1, bind_weight0, bind_weight1, this->transform, false,
                    range);
    }
  }
}

void GLTFLoader::InitializeData() {
  joint_rotation.clear();
  joint_scale.clear();
  joint_translation.clear();
  joint_matrix.clear();
  joint_id_joint_dir.clear();
  joint_T_f_anim.clear();
  joint_R_f_anim.clear();
  joint_S_f_anim.clear();
  joint_T_Time.clear();
  joint_S_Time.clear();
  joint_R_Time.clear();
  joint_output.clear();
  joint_input.clear();
  joint_inverse_bind_matrix_map.clear();
  joint_animation_matrix.clear();
  all_joints.clear();

  initial_position.clear();
  d_joints.clear();
  initial_normal.clear();

  all_nodes.clear();
  all_meshes.clear();
  node_id_dir.clear();

  mesh_id_dir.clear();
  node_matrix.clear();

  d_mesh_matrix.clear();
  d_shape_mesh_id.clear();
  skin_verts_range.clear();
  node_name.clear();

  max_joint_id = -1;
  joint_num = -1;
  mesh_num = -1;

  initial_matrix.clear();
  joint_inverse_bind_matrix.clear();
  joint_local_matrix.clear();
  joint_world_matrix.clear();

  tex_coord_0.clear();
  tex_coord_1.clear();
}

}  // namespace XRTailor