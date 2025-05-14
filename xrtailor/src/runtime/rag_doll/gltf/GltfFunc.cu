#include <xrtailor/runtime/rag_doll/gltf/GltfFunc.cuh>

namespace XRTailor {

void GetBoundingBoxByName(tinygltf::Model& model, const tinygltf::Primitive& primitive,
                          const std::string& attribute_name, Bounds& bound,
                          Transform3x3& transform) {
  std::map<std::string, int>::const_iterator iter;
  iter = primitive.attributes.find(attribute_name);

  if (iter == primitive.attributes.end()) {
    std::cout << attribute_name << " : not found !!! \n";
    return;
  }

  auto min = model.accessors[iter->second].minValues;
  auto max = model.accessors[iter->second].maxValues;
  if (min.size() != 3) {
    std::cout << attribute_name << " : not Vec3f !!! \n";
    return;
  }

  Vector3 v0(min[0], min[1], min[2]);
  Vector3 v1(max[0], max[1], max[2]);

  bound = Bounds(v0, v1);

  transform = Transform3x3(Vector3(bound.Center()), Mat3x3(1), Vector3(1));
}

void GetVec3ByAttributeName(tinygltf::Model& model, const tinygltf::Primitive& primitive,
                            const std::string& attribute_name,
                            thrust::host_vector<Vector3>& vertices) {
  std::map<std::string, int>::const_iterator iter;
  iter = primitive.attributes.find(attribute_name);

  if (iter == primitive.attributes.end()) {
    std::cout << attribute_name << " : not found !!! \n";
    return;
  }

  const tinygltf::Accessor& accessor_attribute = model.accessors[iter->second];
  const tinygltf::BufferView& buffer_view = model.bufferViews[accessor_attribute.bufferView];
  const tinygltf::Buffer& buffer = model.buffers[buffer_view.buffer];

  if (accessor_attribute.type == TINYGLTF_TYPE_VEC3) {
    const float* positions = reinterpret_cast<const float*>(
        &buffer.data[buffer_view.byteOffset + accessor_attribute.byteOffset]);
    for (size_t i = 0; i < accessor_attribute.count; ++i) {
      vertices.push_back(Vector3(positions[i * 3 + 0], positions[i * 3 + 1], positions[i * 3 + 2]));
    }
  } else if (accessor_attribute.type == TINYGLTF_TYPE_VEC2) {
    const float* positions = reinterpret_cast<const float*>(
        &buffer.data[buffer_view.byteOffset + accessor_attribute.byteOffset]);
    for (size_t i = 0; i < accessor_attribute.count; ++i) {
      vertices.push_back(Vector3(positions[i * 2 + 0], positions[i * 2 + 1], 0));
    }
  }
}

void GetVec4ByAttributeName(tinygltf::Model& model, const tinygltf::Primitive& primitive,
                            const std::string& attribute_name,
                            thrust::host_vector<Vector4>& vec4_data) {
  std::map<std::string, int>::const_iterator iter;
  iter = primitive.attributes.find(attribute_name);

  if (iter == primitive.attributes.end()) {
    return;
  }

  const tinygltf::Accessor& accessor_attribute = model.accessors[iter->second];
  const tinygltf::BufferView& buffer_view = model.bufferViews[accessor_attribute.bufferView];
  const tinygltf::Buffer& buffer = model.buffers[buffer_view.buffer];

  if (accessor_attribute.type == TINYGLTF_TYPE_VEC4) {
    if (accessor_attribute.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
      const unsigned short* data = reinterpret_cast<const unsigned short*>(
          &buffer.data[buffer_view.byteOffset + accessor_attribute.byteOffset]);
      for (size_t i = 0; i < accessor_attribute.count; ++i) {

        vec4_data.push_back(Vector4(float(data[i * 4 + 0]), float(data[i * 4 + 1]),
                                   float(data[i * 4 + 2]), float(data[i * 4 + 3])));
      }
    } else if (accessor_attribute.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
      const unsigned int* data = reinterpret_cast<const unsigned int*>(
          &buffer.data[buffer_view.byteOffset + accessor_attribute.byteOffset]);
      for (size_t i = 0; i < accessor_attribute.count; ++i) {
        vec4_data.push_back(Vector4(float(data[i * 4 + 0]), float(data[i * 4 + 1]),
                                   float(data[i * 4 + 2]), float(data[i * 4 + 3])));
      }
    } else if (accessor_attribute.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT) {
      const float* data = reinterpret_cast<const float*>(
          &buffer.data[buffer_view.byteOffset + accessor_attribute.byteOffset]);
      for (size_t i = 0; i < accessor_attribute.count; ++i) {
        vec4_data.push_back(
            Vector4(data[i * 4 + 0], data[i * 4 + 1], data[i * 4 + 2], data[i * 4 + 3]));
      }
    }
  }
}

void GetTriangles(tinygltf::Model& model, const tinygltf::Primitive& primitive,
                  thrust::host_vector<uint>& indices, int point_offest) {
  const tinygltf::Accessor& accessor_triangles = model.accessors[primitive.indices];
  const tinygltf::BufferView& buffer_view = model.bufferViews[accessor_triangles.bufferView];
  const tinygltf::Buffer& buffer = model.buffers[buffer_view.buffer];

  //get Triangle Vertex id
  if (accessor_triangles.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
    const byte* elements = reinterpret_cast<const byte*>(
        &buffer.data[accessor_triangles.byteOffset + buffer_view.byteOffset]);

    for (size_t k = 0; k < accessor_triangles.count / 3; k++) {
      indices.push_back(int(elements[k * 3]) + point_offest);
      indices.push_back(int(elements[k * 3 + 1]) + point_offest);
      indices.push_back(int(elements[k * 3 + 2]) + point_offest);
    }

  } else if (accessor_triangles.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
    const unsigned short* elements = reinterpret_cast<const unsigned short*>(
        &buffer.data[accessor_triangles.byteOffset + buffer_view.byteOffset]);

    for (size_t k = 0; k < accessor_triangles.count / 3; k++) {
      indices.push_back(int(elements[k * 3]) + point_offest);
      indices.push_back(int(elements[k * 3 + 1]) + point_offest);
      indices.push_back(int(elements[k * 3 + 2]) + point_offest);
    }

  } else if (accessor_triangles.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
    const unsigned int* elements = reinterpret_cast<const unsigned int*>(
        &buffer.data[accessor_triangles.byteOffset + buffer_view.byteOffset]);

    for (size_t k = 0; k < accessor_triangles.count / 3; k++) {
      indices.push_back(int(elements[k * 3]) + point_offest);
      indices.push_back(int(elements[k * 3 + 1]) + point_offest);
      indices.push_back(int(elements[k * 3 + 2]) + point_offest);
    }
  }
}

void GetScalarByIndex(tinygltf::Model& model, int index, thrust::host_vector<Scalar>& result) {
  const tinygltf::Accessor& accessor = model.accessors[index];
  const tinygltf::BufferView& buffer_view = model.bufferViews[accessor.bufferView];
  const tinygltf::Buffer& buffer = model.buffers[buffer_view.buffer];

  if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT) {
    const float* dataPtr =
        reinterpret_cast<const float*>(&buffer.data[buffer_view.byteOffset + accessor.byteOffset]);

    for (size_t i = 0; i < accessor.count; ++i) {
      result.push_back(static_cast<Scalar>(dataPtr[i]));
    }
  } else if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_DOUBLE) {
    const double* dataPtr =
        reinterpret_cast<const double*>(&buffer.data[buffer_view.byteOffset + accessor.byteOffset]);

    for (size_t i = 0; i < accessor.count; ++i) {
      result.push_back(static_cast<Scalar>(dataPtr[i]));
    }
  } else {
    printf("\n !!!!!!!!  Error ComponentType  !!!!!!!!\n");
  }

  return;
}

void GetVec3ByIndex(tinygltf::Model& model, int index, thrust::host_vector<Vector3>& result) {
  const tinygltf::Accessor& accessor = model.accessors[index];
  const tinygltf::BufferView& buffer_view = model.bufferViews[accessor.bufferView];
  const tinygltf::Buffer& buffer = model.buffers[buffer_view.buffer];

  if (accessor.type == TINYGLTF_TYPE_VEC3) {
    const float* data_ptr =
        reinterpret_cast<const float*>(&buffer.data[buffer_view.byteOffset + accessor.byteOffset]);

    for (size_t i = 0; i < accessor.count; ++i) {
      result.push_back(Vector3(data_ptr[i * 3 + 0], data_ptr[i * 3 + 1], data_ptr[i * 3 + 2]));
    }
  }
}

void GetQuatByIndex(tinygltf::Model& model, int index, thrust::host_vector<Quaternion>& result) {
  const tinygltf::Accessor& accessor = model.accessors[index];
  const tinygltf::BufferView& buffer_view = model.bufferViews[accessor.bufferView];
  const tinygltf::Buffer& buffer = model.buffers[buffer_view.buffer];

  if (accessor.type == TINYGLTF_TYPE_VEC4) {
    const float* data_ptr =
        reinterpret_cast<const float*>(&buffer.data[buffer_view.byteOffset + accessor.byteOffset]);

    for (size_t i = 0; i < accessor.count; ++i) {
      result.push_back(
          Quaternion(data_ptr[i * 4 + 0], data_ptr[i * 4 + 1], data_ptr[i * 4 + 2], data_ptr[i * 4 + 3])
              .Normalize());
    }
  }
}

void GetVertexBindJoint(tinygltf::Model& model, const tinygltf::Primitive& primitive,
                        const std::string& attribute_name, thrust::host_vector<Vector4>& vec4_data,
                        const thrust::host_vector<int>& skin_joints) {
  std::map<std::string, int>::const_iterator iter;
  iter = primitive.attributes.find(attribute_name);

  if (iter == primitive.attributes.end()) {
    return;
  }

  const tinygltf::Accessor& accessor_attribute = model.accessors[iter->second];
  const tinygltf::BufferView& buffer_view = model.bufferViews[accessor_attribute.bufferView];
  const tinygltf::Buffer& buffer = model.buffers[buffer_view.buffer];

  //std::cout << attribute_name << ", type: " << accessor_attribute.type << ", componentType: " << accessor_attribute.componentType << std::endl;

  if (accessor_attribute.type == TINYGLTF_TYPE_VEC4) {
    if (accessor_attribute.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
      const unsigned char* data = reinterpret_cast<const unsigned char*>(
          &buffer.data[buffer_view.byteOffset + accessor_attribute.byteOffset]);
      for (size_t i = 0; i < accessor_attribute.count; ++i) {

        vec4_data.push_back(
            Vector4(skin_joints[int(data[i * 4 + 0])], skin_joints[int(data[i * 4 + 1])],
                    skin_joints[int(data[i * 4 + 2])], skin_joints[int(data[i * 4 + 3])]));
      }
    }

    if (accessor_attribute.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
      const unsigned short* data = reinterpret_cast<const unsigned short*>(
          &buffer.data[buffer_view.byteOffset + accessor_attribute.byteOffset]);
      for (size_t i = 0; i < accessor_attribute.count; ++i) {

        vec4_data.push_back(
            Vector4(skin_joints[int(data[i * 4 + 0])], skin_joints[int(data[i * 4 + 1])],
                    skin_joints[int(data[i * 4 + 2])], skin_joints[int(data[i * 4 + 3])]));
      }
    } else if (accessor_attribute.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
      const unsigned int* data = reinterpret_cast<const unsigned int*>(
          &buffer.data[buffer_view.byteOffset + accessor_attribute.byteOffset]);
      for (size_t i = 0; i < accessor_attribute.count; ++i) {

        vec4_data.push_back(
            Vector4(skin_joints[int(data[i * 4 + 0])], skin_joints[int(data[i * 4 + 1])],
                    skin_joints[int(data[i * 4 + 2])], skin_joints[int(data[i * 4 + 3])]));
      }
    } else if (accessor_attribute.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT) {
      const float* data = reinterpret_cast<const float*>(
          &buffer.data[buffer_view.byteOffset + accessor_attribute.byteOffset]);
      for (size_t i = 0; i < accessor_attribute.count; ++i) {

        vec4_data.push_back(
            Vector4(skin_joints[int(data[i * 4 + 0])], skin_joints[int(data[i * 4 + 1])],
                    skin_joints[int(data[i * 4 + 2])], skin_joints[int(data[i * 4 + 3])]));
      }
    }
  }
}

void GetNodesAndHierarchy(tinygltf::Model& model,
                          std::map<scene, thrust::host_vector<int>> scene_joints_nodes_id,
                          thrust::host_vector<joint>& all_nodes,
                          std::map<joint, thrust::host_vector<int>>& id_dir) {
  for (auto it : scene_joints_nodes_id) {
    scene scene_id = it.first;
    thrust::host_vector<joint> scene_joint_roots = it.second;
    std::map<joint, int> root_joint_num;

    for (size_t n = 0; n < scene_joint_roots.size(); n++) {
      int root_node_id = scene_joint_roots[n];

      thrust::host_vector<int> null_vvec;
      TraverseNode(model, root_node_id, all_nodes, id_dir, null_vvec);
    }
  }
}

void TraverseNode(tinygltf::Model& model, joint id, thrust::host_vector<joint>& joint_nodes,
                  std::map<joint, thrust::host_vector<int>>& dir,
                  thrust::host_vector<joint> current_dir) {
  const tinygltf::Node& node = model.nodes[id];
  current_dir.push_back(id);
  joint_nodes.push_back(id);

  for (int child_index : node.children) {
    const tinygltf::Node& child_node = model.nodes[child_index];
    TraverseNode(model, child_index, joint_nodes, dir, current_dir);
  }

  std::reverse(current_dir.begin(), current_dir.end());
  dir[id] = current_dir;
}

void GetJointsTransformData(const thrust::host_vector<int>& all_joints,
                            std::map<int, std::string>& joint_name,
                            thrust::host_vector<thrust::host_vector<int>>& joint_child,
                            std::map<int, Quaternion>& joint_rotation,
                            std::map<int, Vector3>& joint_scale,
                            std::map<int, Vector3>& joint_translation,
                            std::map<int, Mat4x4>& joint_matrix, tinygltf::Model model) {
  for (size_t k = 0; k < all_joints.size(); k++) {
    joint j_id = all_joints[k];
    std::vector<int>& children = model.nodes[j_id].children;  //std::vector<int> children ;

    std::vector<double>& rotation = model.nodes[j_id].rotation;        //quat length must be 0 or 4
    std::vector<double>& scale = model.nodes[j_id].scale;              //length must be 0 or 3
    std::vector<double>& translation = model.nodes[j_id].translation;  //length must be 0 or 3
    std::vector<double>& matrix = model.nodes[j_id].matrix;            //length must be 0 or 16

    joint_name[j_id] = model.nodes[j_id].name;
    joint_child.push_back(children);

    Mat4x4 temp_T(1);
    Mat4x4 temp_R(1);
    Mat4x4 temp_S(1);

    if (!rotation.empty())
      joint_rotation[j_id] = (Quaternion(rotation[0], rotation[1], rotation[2], rotation[3]));
    else
      joint_rotation[j_id] = (Quaternion(0, 0, 0, 0));

    if (!scale.empty())
      joint_scale[j_id] = (Vector3(scale[0], scale[1], scale[2]));
    else
      joint_scale[j_id] = (Vector3(1.0f, 1.0f, 1.0f));

    if (!translation.empty())
      joint_translation[j_id] = (Vector3(translation[0], translation[1], translation[2]));
    else
      joint_translation[j_id] = (Vector3(0.0f, 0.0f, 0.0f));

    if (!matrix.empty()) {
      joint_matrix[j_id] = (Mat4x4(matrix[0], matrix[4], matrix[8], matrix[12], matrix[1], matrix[5],
                                  matrix[9], matrix[13], matrix[2], matrix[6], matrix[10],
                                  matrix[14], matrix[3], matrix[7], matrix[11], matrix[15]));
    } else {
      //Translation Matrix

      if (!translation.empty()) {
        temp_T = Mat4x4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, translation[0], translation[1],
                       translation[2], 1);
      } else
        temp_T = Mat4x4(1);

      if (!rotation.empty())
        temp_R = Quaternion(rotation[0], rotation[1], rotation[2], rotation[3]).ToMatrix4x4();
      else
        temp_R = Mat4x4(1);

      if (!scale.empty()) {
        temp_S = Mat4x4(scale[0], 0, 0, 0, 0, scale[1], 0, 0, 0, 0, scale[2], 0, 0, 0, 0, 1);
        temp_S = glm::transpose(temp_S);
      } else
        temp_S = Mat4x4(1);

      joint_matrix[j_id] = (temp_T * temp_R * temp_S);  // if jointmatrix not found, build it
    }
  }
}

thrust::host_vector<int> GetJointDirByJointIndex(
    int index, std::map<joint, thrust::host_vector<int>> joint_id_joint_dir) {
  thrust::host_vector<int> joint_dir;
  std::map<int, thrust::host_vector<int>>::const_iterator iter;

  //get skeletal chain
  iter = joint_id_joint_dir.find(index);
  if (iter == joint_id_joint_dir.end()) {
    std::cout << "Error: not found JointIndex \n";
    return joint_dir;
  }

  joint_dir = iter->second;
  return joint_dir;
}

void BuildInverseBindMatrices(const std::vector<joint>& all_joints,
                              std::map<joint, Mat4x4>& joint_matrix, int& max_joint_id,
                              tinygltf::Model& model, std::map<joint, Quaternion>& joint_rotation,
                              std::map<joint, Vector3>& joint_translation,
                              std::map<joint, Vector3>& joint_scale,
                              std::map<joint, Mat4x4>& joint_inverse_bind_matrix,
                              std::map<joint, thrust::host_vector<int>> joint_id_joint_dir) {
  std::map<joint, Mat4> temp_joint_matrix = joint_matrix;
  thrust::host_vector<Mat4> temp;

  temp.resize(max_joint_id + 1);

  for (size_t i = 0; i < max_joint_id + 1; i++) {
    temp.push_back(Mat4(1));
  }

  for (size_t i = 0; i < all_joints.size(); i++) {
    joint joint_id = all_joints[i];

    const thrust::host_vector<int>& jD = GetJointDirByJointIndex(joint_id, joint_id_joint_dir);

    Mat4 temp_matrix = Mat4(1);

    for (int k = 0; k < jD.size(); k++) {
      joint select = jD[k];

      Vector3 temp_VT(0, 0, 0);
      Vector3 temp_VS(1, 1, 1);
      Quaternion temp_QR = Quaternion(Mat3(1));

      if (model.nodes[select].matrix.empty()) {
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

    joint_inverse_bind_matrix[joint_id] = (temp_matrix);

    temp[joint_id] = temp_matrix;
  }
}

void UpdateJointMeshCameraDir(tinygltf::Model& model, int& joint_num, int& mesh_num,
                                 std::map<joint, thrust::host_vector<int>>& joint_id_joint_dir,
                                 thrust::host_vector<joint>& all_joints,
                                 thrust::host_vector<int>& all_nodes,
                                 std::map<joint, thrust::host_vector<int>> node_id_dir,
                                 std::map<int, thrust::host_vector<int>>& mesh_id_dir,
                                 thrust::host_vector<int>& all_meshs,
                                 thrust::device_vector<int>& d_joints, int& max_joint_id) {
  for (auto nId : all_nodes) {
    if (model.nodes[nId].mesh == -1 && model.nodes[nId].camera == -1) {
      all_joints.push_back(nId);
      joint_id_joint_dir[nId] = node_id_dir[nId];
    }
    if (model.nodes[nId].mesh == 1) {
      mesh_id_dir[nId] = node_id_dir[nId];
    }
  }

  joint_num = all_joints.size();
  mesh_num = all_meshs.size();

  d_joints = all_joints;

  if (all_joints.size())
    max_joint_id = *std::max_element(all_joints.begin(), all_joints.end());
  else
    max_joint_id = -1;
}

void GetMeshMatrix(tinygltf::Model& model, const thrust::host_vector<int>& all_mesh_node_ids,
                   int& max_mesh_id, thrust::host_vector<Mat4x4>& mesh_matrix) {
  max_mesh_id = *std::max_element(all_mesh_node_ids.begin(), all_mesh_node_ids.end());

  mesh_matrix.resize(max_mesh_id + 1);

  Mat4 temp_T;
  Mat4 temp_R;
  Mat4 temp_S;

  for (size_t k = 0; k < all_mesh_node_ids.size(); k++) {
    std::vector<double>& rotation =
        model.nodes[all_mesh_node_ids[k]].rotation;  //quat length must be 0 or 4
    std::vector<double>& scale = model.nodes[all_mesh_node_ids[k]].scale;  //length must be 0 or 3
    std::vector<double>& translation =
        model.nodes[all_mesh_node_ids[k]].translation;                       //length must be 0 or 3
    std::vector<double>& matrix = model.nodes[all_mesh_node_ids[k]].matrix;  //length must be 0 or 16

    if (!matrix.empty()) {
      mesh_matrix[all_mesh_node_ids[k]] =
          (Mat4(matrix[0], matrix[4], matrix[8], matrix[12], matrix[1], matrix[5], matrix[9],
                matrix[13], matrix[2], matrix[6], matrix[10], matrix[14], matrix[3], matrix[7],
                matrix[11], matrix[15]));
    }

    else {
      //Translation Matrix

      if (!translation.empty()) {
        temp_T = Mat4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, translation[0], translation[1],
                     translation[2], 1);
      } else
        temp_T = Mat4(1);

      if (!rotation.empty())
        temp_R = Quaternion(rotation[0], rotation[1], rotation[2], rotation[3]).ToMatrix4x4();
      else
        temp_R = Mat4(1);

      if (!scale.empty())
        temp_S = Mat4(scale[0], 0, 0, 0, 0, scale[1], 0, 0, 0, 0, scale[2], 0, 0, 0, 0, 1);
      else
        temp_S = Mat4(1);

      mesh_matrix[all_mesh_node_ids[k]] =
          (temp_T * temp_R * temp_S);  // if jointmatrix not found, build it
    }
  }
}

void ImportAnimation(tinygltf::Model model, std::map<joint, Vector3i>& joint_output,
                     std::map<joint, Vector3>& joint_input,
                     std::map<joint, thrust::host_vector<Vector3>>& joint_T_f_anim,
                     std::map<joint, thrust::host_vector<Scalar>>& joint_T_Time,
                     std::map<joint, thrust::host_vector<Vector3>>& joint_S_f_anim,
                     std::map<joint, thrust::host_vector<Scalar>>& joint_S_Time,
                     std::map<joint, thrust::host_vector<Quaternion>>& joint_R_f_anim,
                     std::map<joint, thrust::host_vector<Scalar>>& joint_R_Time) {
  using namespace tinygltf;
  //input output
  for (size_t i = 0; i < model.nodes.size(); i++) {
    joint_output[i] = Vector3i(-1, -1, -1);  //
    joint_input[i] = Vector3(NULL_TIME, NULL_TIME, NULL_TIME);
  }

  //Reset loading animation  ;
  for (size_t i = 0; i < model.animations.size(); i++) {
    std::string& name = model.animations[i].name;
    std::vector<AnimationChannel>& channels = model.animations[i].channels;
    std::vector<AnimationSampler>& samplers = model.animations[i].samplers;

    std::cout << "Importing animation " << name << std::endl;

    if (name == "mixamo.com" || name == "mixamo.com.001")
      continue;

    for (size_t j = 0; j < channels.size(); j++)  //channels
    {
      //get sampler info
      int& samplerId = channels[j].sampler;           // required
      joint& joint_nodeId = channels[j].target_node;  // required (index of the node to target)
      std::string& target_path =
          channels[j].target_path;  // required in ["translation", "rotation", "scale","weights"]

      //get
      int& input = samplers[samplerId].input;    //real time
      int& output = samplers[samplerId].output;  //transform bufferid
      std::string& interpolation = samplers[samplerId].interpolation;

      {

        if (target_path == "translation") {
          joint_output[joint_nodeId][0] = output;
          joint_input[joint_nodeId][0] = input;
        } else if (target_path == "scale") {
          joint_output[joint_nodeId][1] = output;
          joint_input[joint_nodeId][0] = input;
        } else if (target_path == "rotation") {
          joint_output[joint_nodeId][2] = output;
          joint_input[joint_nodeId][0] = input;
        }
      }

      //Reset
      {
        //out animation data
        thrust::host_vector<Vector3> frame_T_anim;
        thrust::host_vector<Quaternion> frame_R_anim;
        thrust::host_vector<Vector3> frame_S_anim;
        //
        thrust::host_vector<Scalar> frame_T_Time;
        thrust::host_vector<Scalar> frame_R_Time;
        thrust::host_vector<Scalar> frame_S_Time;

        // get joint transform data
        if (target_path == "translation") {
          GetVec3ByIndex(model, output, frame_T_anim);
          joint_T_f_anim[joint_nodeId] = frame_T_anim;

          GetScalarByIndex(model, input, frame_T_Time);
          joint_T_Time[joint_nodeId] = frame_T_Time;  //t
        } else if (target_path == "scale") {
          GetVec3ByIndex(model, output, frame_S_anim);
          joint_S_f_anim[joint_nodeId] = frame_S_anim;
          GetScalarByIndex(model, input, frame_S_Time);
          joint_S_Time[joint_nodeId] = frame_S_Time;  //s
        } else if (target_path == "rotation") {
          GetQuatByIndex(model, output, frame_R_anim);
          joint_R_f_anim[joint_nodeId] = frame_R_anim;
          GetScalarByIndex(model, input, frame_R_Time);
          joint_R_Time[joint_nodeId] = frame_R_Time;  //r
        }
      }
    }
  }
}

}  // namespace XRTailor
