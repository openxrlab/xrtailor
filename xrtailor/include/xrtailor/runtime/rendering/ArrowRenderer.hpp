#pragma once

#include <vector>

#include <xrtailor/runtime/rendering/Arrow.hpp>
#include <xrtailor/runtime/rendering/LineRenderer.hpp>

namespace XRTailor {
class ArrowRenderer : public LineRenderer {
 public:
  ArrowRenderer();

  void SetDefaultArrowConfig();

  void SetCustomedArrowConfig();

  void AddArrow(glm::vec3 _start, glm::vec3 _end);

  void AddArrow(const Arrow& _arrow);

  void SetArrows(std::vector<Arrow> _arrows);

  void ShowDebugInfo();

  ~ArrowRenderer();
};
}  // namespace XRTailor