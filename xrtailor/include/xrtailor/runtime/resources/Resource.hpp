#pragma once

#include <iostream>
#include <unordered_map>
#include <string>
#include "stb_image.h"
#include <xrtailor/runtime/mesh/Mesh.hpp>
#include <xrtailor/runtime/resources/Material.hpp>
#include <xrtailor/runtime/mesh/MeshIO.hpp>
#include <xrtailor/utils/Logger.hpp>
#include <xrtailor/utils/FileSystemUtils.hpp>
#include <xrtailor/utils/ObjUtils.hpp>

#include <xrtailor/runtime/rag_doll/gltf/GLTFLoader.cuh>

#include <thrust/host_vector.h>

namespace XRTailor {
class Resource {
 public:
  static uint LoadTexture(const std::string& filename) {
    if (texture_cache_.count(filename) > 0) {
      return texture_cache_[filename];
    }

    uint texture_id;
    glGenTextures(1, &texture_id);
    texture_cache_[filename] = texture_id;
    int width, height, nr_components;
    unsigned char* data = stbi_load(filename.c_str(), &width, &height, &nr_components, 0);
    if (data == nullptr) {
      filesystem::path texture_base_path = GetTextureDirectory();
      filesystem::path full_path = texture_base_path;
      full_path.append(filename);

      data = stbi_load(full_path.string().c_str(), &width, &height, &nr_components, 0);
      LOG_DEBUG("Texture full path: {}", full_path.string());
    }
    if (data) {
      GLenum internal_format = GL_RED;
      GLenum data_format = GL_RED;

      if (nr_components == 3) {
        internal_format = GL_SRGB;
        data_format = GL_RGB;
      } else if (nr_components == 4) {
        internal_format = GL_SRGB_ALPHA;
        data_format = GL_RGBA;
      }

      glBindTexture(GL_TEXTURE_2D, texture_id);
      glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0, data_format,
                   GL_UNSIGNED_BYTE, data);
      glGenerateMipmap(GL_TEXTURE_2D);

      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

      stbi_image_free(data);
    } else {
      LOG_ERROR("Failed to load texture: {}", filename);
      stbi_image_free(data);
    }

    return texture_id;
  }

  static std::vector<uint> LoadMask(const std::string& body_model) {
    filesystem::path full_path(GetBodyTemplateDirectory());

    std::string mask_name = "";
    if (body_model == "SMPL" || body_model == "SMPLH") {
      mask_name += "SMPLH";
    } else {
      mask_name += "SMPLX";
    }

    mask_name += "_mask.txt";
    full_path.append(mask_name);

    LOG_TRACE("Load mask from: {}", full_path.string());
    std::string mask_code;
    std::ifstream file;
    file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    try {
      file.open(full_path.string());
      std::stringstream mask_stream;
      mask_stream << file.rdbuf();
      file.close();
      mask_code = mask_stream.str();
    } catch (std::ifstream::failure& e) {
      LOG_ERROR("Mask::FILE_NOT_SUCCESFULLY_READ: {}", e.what());
      exit(TAILOR_EXIT::MASK_READ_ERROR);
    }
    std::vector<uint> res;
    for (auto& val : mask_code) {
      res.push_back(int(val - '0'));
    }
    return res;
  }

  static std::shared_ptr<Mesh> LoadMeshDataAndBuildStructure(const std::string& filepath,
                                                             uint ID = -1) {
    LOG_DEBUG("Load mesh from: {}", filepath);
    std::ifstream in(filepath.c_str());

    if (!in.is_open()) {
      LOG_ERROR("Error: Could not open file {} for reading", filepath);
      return nullptr;
    }

    MeshData data = MeshIO::Read(in);

    auto result = std::shared_ptr<Mesh>(new Mesh(data, ID));
    mesh_cache_[filepath] = result;
    return result;
  }

  static std::shared_ptr<Mesh> LoadGLBAndBuildStructure(std::shared_ptr<GLTFLoader> loader,
                                                        const std::string& filepath, uint id = -1) {
    LOG_DEBUG("Load glb from: {}", filepath);

    MeshData mesh_data;
    loader->LoadGltf(filepath, mesh_data);
    LOG_DEBUG("Load gltf done");

    auto result = std::shared_ptr<Mesh>(new Mesh(mesh_data, id));

    mesh_cache_[filepath] = result;
    return result;
  }

  static std::shared_ptr<Material> LoadMaterial(const std::string& path,
                                                bool include_geometry_shader = false) {
    if (mat_cache_.count(path)) {
      return mat_cache_[path];
    }

    filesystem::path shader_base_path = GetShaderDirectory();
    filesystem::path vertex_path(shader_base_path);
    vertex_path.append(path + ".vert");
    std::string vertex_code = LoadText(vertex_path.string());

    if (vertex_code.length() == 0)
      vertex_code = LoadText(path + ".vert");
    if (vertex_code.length() == 0) {
      LOG_ERROR("material.vertex not found ({})", path);
      exit(TAILOR_EXIT::SHADER_NOT_FOUND);
    }

    filesystem::path fragment_path(shader_base_path);
    fragment_path.append(path + ".frag");
    std::string fragment_code = LoadText(fragment_path.string());

    if (fragment_code.length() == 0)
      fragment_code = LoadText(path + ".frag");
    if (fragment_code.length() == 0) {
      LOG_ERROR("material.fragment not found ({})", path);
      exit(TAILOR_EXIT::SHADER_NOT_FOUND);
    }

    std::string geometry_code;
    if (include_geometry_shader) {
      filesystem::path geometry_path(shader_base_path);
      geometry_path.append(path + ".geom");
      geometry_code = LoadText(geometry_path.string());

      if (geometry_code.length() == 0)
        geometry_code = LoadText(path + ".geom");
      if (geometry_code.length() == 0) {
        LOG_ERROR("Error(Resource): material.geometry not found ({})", path);
        exit(TAILOR_EXIT::SHADER_NOT_FOUND);
      }
    }
    auto result = std::make_shared<Material>(vertex_code, fragment_code, geometry_code);
    mat_cache_[path] = result;
    result->name = path;
    return result;
  }

  static std::string LoadText(const std::string& path) {
    std::string code;
    std::ifstream file;
    file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    try {
      file.open(path);
      std::stringstream v_vhader_stream;
      v_vhader_stream << file.rdbuf();
      file.close();
      code = v_vhader_stream.str();
    } catch (std::ifstream::failure& e) {
      LOG_ERROR("SHADER::FILE_NOT_SUCCESFULLY_READ: {}", e.what());
      exit(TAILOR_EXIT::SHADER_READ_ERROR);
    }
    return code;
  }

  static void ClearCache() {
    mat_cache_.clear();
    mesh_cache_.clear();
  }

 private:
  static inline std::unordered_map<std::string, uint> texture_cache_;
  static inline std::unordered_map<std::string, std::shared_ptr<Mesh>> mesh_cache_;
  static inline std::unordered_map<std::string, std::shared_ptr<Material>> mat_cache_;
};
}  // namespace XRTailor