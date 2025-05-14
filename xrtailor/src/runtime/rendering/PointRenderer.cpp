#include <xrtailor/runtime/rendering/PointRenderer.hpp>

#include <xrtailor/runtime/engine/GameInstance.hpp>
#include <xrtailor/core/Global.hpp>
#include <xrtailor/runtime/scene/Camera.hpp>
#include <xrtailor/runtime/scene/Actor.hpp>
#include <xrtailor/runtime/rendering/Light.hpp>
#include <xrtailor/runtime/resources/Resource.hpp>

namespace XRTailor {
PointRenderer::PointRenderer() {

  SET_COMPONENT_NAME;

  point_color = Vector3(0.34f, 0.45f, 0.93f);
  point_size = 5.0f;

  const char* vertex_shader_source =
      "#version 330 core\n"
      "layout (location = 0) in vec3 aPos;\n"
      "uniform mat4 MVP;\n"
      "void main()\n"
      "{\n"
      "   gl_Position = MVP * vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
      "}\0";
  const char* fragment_shader_source =
      "#version 330 core\n"
      "out vec4 FragColor;\n"
      "uniform vec3 color;\n"
      "void main()\n"
      "{\n"
      "   FragColor = vec4(color, 1.0f);\n"
      "}\n\0";

  // vertex shader
  unsigned int vertex_shader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertex_shader, 1, &vertex_shader_source, nullptr);
  glCompileShader(vertex_shader);
  // check for shader compile errors
  int success;
  char info_log[512];
  glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &success);
  if (success == 0) {
    glGetShaderInfoLog(vertex_shader, 512, nullptr, info_log);
    std::cout << "ERROR::SHADER::VERTEX::COMPILIATION_FAILED\n" << info_log << std::endl;
  }

  // fragment shader
  unsigned int fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragment_shader, 1, &fragment_shader_source, nullptr);
  glCompileShader(fragment_shader);
  // check for shader compile errors
  glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &success);
  if (success == 0) {
    glGetShaderInfoLog(fragment_shader, 512, nullptr, info_log);
    std::cout << "ERROR::SHADER::FRAGMENT::COMPILIATION_FAILED\n" << info_log << std::endl;
  }

  // link shaders
  shader_program = glCreateProgram();
  glAttachShader(shader_program, vertex_shader);
  glAttachShader(shader_program, fragment_shader);
  glLinkProgram(shader_program);
  // check for linking errors
  glGetShaderiv(shader_program, GL_LINK_STATUS, &success);
  if (success == 0) {
    glGetShaderInfoLog(shader_program, 512, nullptr, info_log);
    std::cout << "ERROR::SHADER_PROGRAM::LINK_FAILED\n" << info_log << std::endl;
  }

  glDeleteShader(vertex_shader);
  glDeleteShader(fragment_shader);

  glGenVertexArrays(1, &vao);
  glGenBuffers(1, &vbo);
  glGenBuffers(1, &ebo);
  // ---------- end initialization ----------
}

void PointRenderer::RebindBuffer() {
  glBindVertexArray(vao);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  //glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);

  // copy user defined data to the device memory
  // GL_STATIC_DRAW: data is rarely changed
  // GL_DYNAMIC_DRAW: data is often changed
  // GL_STEAM_DRAW: data is changed when draw calls
  glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

  //glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)nullptr);
  glEnableVertexAttribArray(0);
}

void PointRenderer::AddVertex(const glm::vec3& vert) {
  vertices.push_back(vert.x);
  vertices.push_back(vert.y);
  vertices.push_back(vert.z);
}

void PointRenderer::AddVertices(std::vector<glm::vec3>& verts) {
  for (auto vert_iter = verts.begin(); vert_iter != verts.end(); vert_iter++) {
    AddVertex(*vert_iter);
  }
}

void PointRenderer::SetVertices(std::vector<glm::vec3>& verts) {
  vertices.clear();
  for (auto vert_iter = verts.begin(); vert_iter != verts.end(); vert_iter++) {
    AddVertex(*vert_iter);
  }
  RebindBuffer();
}

void PointRenderer::SetDefaultConfig() {
  printf("[PointRenderer] Use default config\n");
  vertices.clear();
  AddVertex(Vector3(1.0, 0.1, 0.0));
  RebindBuffer();
}

int PointRenderer::SetColor(glm::vec3 color) {
  point_color = color;
  return 1;
}

int PointRenderer::SetPointSize(const float& size) {
  point_size = size;
  return 1;
}

void PointRenderer::Render() {
  auto model = actor->transform->matrix();
  auto view = Global::camera->View();
  auto projection = Global::camera->Projection();
  mvp = static_cast<glm::mat4>(projection * view * model);
  DrawCall();
}

int PointRenderer::DrawCall() {
  glUseProgram(shader_program);
  glUniformMatrix4fv(glGetUniformLocation(shader_program, "MVP"), 1, GL_FALSE, &mvp[0][0]);
  glUniform3fv(glGetUniformLocation(shader_program, "color"), 1, &point_color[0]);

  glBindVertexArray(vao);

  glPointSize(point_size);
  glDrawArrays(GL_POINTS, 0, vertices.size() / 3);
  glBindVertexArray(0);
  return 1;
}

void PointRenderer::Reset() {
  vertices.clear();
  RebindBuffer();
}

PointRenderer::~PointRenderer() {
  glDeleteVertexArrays(1, &vao);
  glDeleteBuffers(1, &vbo);
  //glDeleteBuffers(1, &ebo);
  glDeleteProgram(shader_program);
}
}  // namespace XRTailor