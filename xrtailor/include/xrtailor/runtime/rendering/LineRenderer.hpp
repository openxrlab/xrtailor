#pragma once

#include <vector>


#include <xrtailor/core/Scalar.hpp>
#include <xrtailor/runtime/scene/Component.hpp>

namespace XRTailor {
class LineRenderer : public Component {
 public:
  uint shader_program;
  uint vbo, vao, ebo;
  std::vector<float> vertices;
  std::vector<uint> indices;
  glm::mat4 mvp;
  glm::vec3 line_color;
  float line_width;

 public:
  LineRenderer();

  void RebindBuffer();

  void AddVertex(const glm::vec3& vert);

  void AddVertices(std::vector<glm::vec3>& verts);

  void SetVertices(std::vector<glm::vec3>& verts);

  int SetColor(glm::vec3 color);

  int SetLineWidth(const float& width);

  void Render();

  int DrawCall();

  void Reset();

  ~LineRenderer();
};
}  // namespace XRTailor