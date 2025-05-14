#include <xrtailor/runtime/rendering/ArrowRenderer.hpp>

namespace XRTailor {
ArrowRenderer::ArrowRenderer() {
  SET_COMPONENT_NAME;
}

void ArrowRenderer::SetDefaultArrowConfig() {
  vertices.clear();
  AddArrow(Arrow(glm::vec3(1.0, 0.0, 0.0), glm::vec3(2.0, 1.0, 1.0)));
  RebindBuffer();

  //ShowDebugInfo();
}

void ArrowRenderer::SetCustomedArrowConfig() {
  vertices.clear();
  AddArrow(Arrow(glm::vec3(1.0, 0.0, 0.0), glm::vec3(2.0, 1.0, 1.0)));
  AddArrow(Arrow(glm::vec3(-1.0, 0.0, 0.0), glm::vec3(0.0, 1.0, 1.0)));
  RebindBuffer();

  //ShowDebugInfo();
}

void ArrowRenderer::ShowDebugInfo() {
  printf("buffer has %d float attributes, %d vertices, %d edges\n", vertices.size(),
         vertices.size() / 3, vertices.size() / 6);
  for (size_t i = 0; i < vertices.size(); i++) {
    printf("%f ", vertices[i]);
  }
}

void ArrowRenderer::AddArrow(glm::vec3 _start, glm::vec3 _end) {
  if (glm::length(_end - _start) < 1e-6f) {
    return;
  }

  Arrow arrow = Arrow(_start, _end);

  glm::vec3 v0 = arrow.start;
  glm::vec3 v1 = arrow.end;
  glm::vec3 v2 = arrow.lhs;
  glm::vec3 v3 = arrow.rhs;

  std::vector<glm::vec3> arrow_verts = {v0, v1, v1, v2, v1, v3};

  AddVertices(arrow_verts);
}

void ArrowRenderer::AddArrow(const Arrow& _arrow) {
  glm::vec3 v0 = _arrow.start;
  glm::vec3 v1 = _arrow.end;
  glm::vec3 v2 = _arrow.lhs;
  glm::vec3 v3 = _arrow.rhs;

  std::vector<glm::vec3> arrow_verts = {v0, v1, v1, v2, v1, v3};

  AddVertices(arrow_verts);
}

void ArrowRenderer::SetArrows(std::vector<Arrow> _arrows) {
  vertices.clear();
  for (auto arrow_iter = _arrows.begin(); arrow_iter != _arrows.end(); arrow_iter++) {
    AddArrow(*arrow_iter);
  }
  RebindBuffer();
}

ArrowRenderer::~ArrowRenderer() = default;
}  // namespace XRTailor