#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <unordered_map>

namespace XRTailor {
class Material {
 public:
  std::unordered_map<std::string, uint> textures;

  Material() {}

  Material(std::string& vertex_code, std::string& fragment_code, std::string& geometry_code) {
    const char* v_shader_code = vertex_code.c_str();
    const char* f_shader_code = fragment_code.c_str();
    const char* g_shader_code = geometry_code.c_str();
    shader_id_ = CompileShader(v_shader_code, f_shader_code, g_shader_code);
  }

  Material(const Material&) = delete;

  ~Material() { glDeleteProgram(shader_id_); }

  uint CompileShader(const char* v_shader_code, const char* f_shader_code,
                     const char* g_shader_code) const {
    uint vertex, fragment, geometry = 0;
    uint shader = glCreateProgram();
    // vertex shader
    vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, &v_shader_code, NULL);
    glCompileShader(vertex);
    CheckCompileErrors(vertex, "VERTEX");
    glAttachShader(shader, vertex);
    // fragment shader
    fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &f_shader_code, NULL);
    glCompileShader(fragment);
    CheckCompileErrors(fragment, "FRAGMENT");
    glAttachShader(shader, fragment);
    // geometry shader
    if (strlen(g_shader_code) > 0) {
      geometry = glCreateShader(GL_GEOMETRY_SHADER);
      glShaderSource(geometry, 1, &g_shader_code, NULL);
      glCompileShader(geometry);
      CheckCompileErrors(fragment, "GEOMETRY");
      glAttachShader(shader, geometry);
    }
    // shader program
    glLinkProgram(shader);
    CheckCompileErrors(shader, "PROGRAM");
    // delete the shaders as they're linked into our program now and no longer necessary
    glDeleteShader(vertex);
    glDeleteShader(fragment);
    if (strlen(g_shader_code) > 0)
      glDeleteShader(geometry);
    return shader;
  }

  uint shaderID() const { return shader_id_; }

  void Use() const { glUseProgram(shader_id_); }

  GLint GetLocation(const std::string& name) const {
    return glGetUniformLocation(shader_id_, name.c_str());
  }

  // utility uniform functions
  // ------------------------------------------------------------------------
  void SetTexture(const std::string& name, uint texture) { textures[name] = texture; }
  // ------------------------------------------------------------------------
  void SetBool(const std::string& name, bool value) const {
    Use();
    glUniform1i(GetLocation(name), static_cast<int>(value));
  }
  // ------------------------------------------------------------------------
  void SetInt(const std::string& name, int value) const {
    Use();
    glUniform1i(GetLocation(name), value);
  }
  // ------------------------------------------------------------------------
  void SetUInt(const std::string& name, uint value) const {
    Use();
    glUniform1ui(GetLocation(name), value);
  }
  // ------------------------------------------------------------------------
  void SetFloat(const std::string& name, float value) const {
    Use();
    auto err = glGetError();
    if (err != GL_NO_ERROR) {
      // possible reason: opengl buffer overflow (e.g. DrawArrays with too much item count)
      LOG_ERROR("Material::SetFloat: Code #{} in material({})", err, this->name);
    }
    glUniform1f(GetLocation(name), value);
  }
  // ------------------------------------------------------------------------
  void SetVec2(const std::string& name, const glm::vec2& value) const {
    Use();
    glUniform2fv(GetLocation(name), 1, &value[0]);
  }
  void SetVec2(const std::string& name, float x, float y) const {
    Use();
    glUniform2f(GetLocation(name), x, y);
  }
  // ------------------------------------------------------------------------
  void SetVec3(const std::string& name, const glm::vec3& value) const {
    Use();
    glUniform3fv(GetLocation(name), 1, &value[0]);
  }
  void SetVec3(const std::string& name, float x, float y, float z) const {
    Use();
    glUniform3f(GetLocation(name), x, y, z);
  }
  // ------------------------------------------------------------------------
  void SetVec4(const std::string& name, const glm::vec4& value) const {
    Use();
    glUniform4fv(GetLocation(name), 1, &value[0]);
  }
  void SetVec4(const std::string& name, float x, float y, float z, float w) {
    Use();
    glUniform4f(GetLocation(name), x, y, z, w);
  }
  // ------------------------------------------------------------------------
  void SetMat2(const std::string& name, const glm::mat2& mat) const {
    Use();
    glUniformMatrix2fv(GetLocation(name), 1, GL_FALSE, &mat[0][0]);
  }
  // ------------------------------------------------------------------------
  void SetMat3(const std::string& name, const glm::mat3& mat) const {
    Use();
    glUniformMatrix3fv(GetLocation(name), 1, GL_FALSE, &mat[0][0]);
  }
  // ------------------------------------------------------------------------
  void SetMat4(const std::string& name, const glm::mat4& mat) const {
    Use();
    glUniformMatrix4fv(GetLocation(name), 1, GL_FALSE, &mat[0][0]);
  }

  std::string name = "";
  float specular = 0.2f;
  float smoothness = 100.0f;
  bool double_sided = false;
  bool no_wireframe = false;

 private:
  uint shader_id_ = -1;

  void CheckCompileErrors(uint shader, std::string type) const {
    int success;
    char info_log[1024];
    if (type != "PROGRAM") {
      glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
      if (!success) {
        glGetShaderInfoLog(shader, 1024, NULL, info_log);
        LOG_ERROR("SHADER_COMPILATION_ERROR of type: {0}, {1}", type, info_log);
        exit(TAILOR_EXIT::SHADER_COMPILATION_ERROR);
      }
    } else {
      glGetProgramiv(shader, GL_LINK_STATUS, &success);
      if (!success) {
        glGetProgramInfoLog(shader, 1024, NULL, info_log);
        LOG_ERROR("PROGRAM_LINKING_ERROR of type: {0}, {1}", type, info_log);
        exit(TAILOR_EXIT::SHADER_LINKING_ERROR);
      }
    }
  }
};
}  // namespace XRTailor