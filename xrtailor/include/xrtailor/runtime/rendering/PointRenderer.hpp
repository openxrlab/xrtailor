#pragma once

#include <vector>

#include <xrtailor/core/Scalar.hpp>
#include <xrtailor/runtime/scene/Component.hpp>

namespace XRTailor {
class PointRenderer : public Component {
 protected:
  uint shader_program;
  uint vbo, vao, ebo;
  std::vector<float> vertices;
  std::vector<uint> indices;
  glm::mat4 mvp;
  glm::vec3 point_color;
  float point_size;

 public:
  PointRenderer();

  void RebindBuffer();

  void AddVertex(const glm::vec3& vert);

  void AddVertices(std::vector<glm::vec3>& verts);

  void SetVertices(std::vector<glm::vec3>& verts);

  void SetDefaultConfig();

  int SetColor(glm::vec3 color);

  int SetPointSize(const float& size);

  void Render();

  int DrawCall();

  void Reset();

  ~PointRenderer();
};
}  // namespace XRTailor