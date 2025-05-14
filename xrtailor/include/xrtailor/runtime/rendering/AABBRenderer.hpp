#pragma once

#include <vector>

#include <xrtailor/runtime/rendering/AABB.hpp>
#include <xrtailor/runtime/rendering/LineRenderer.hpp>

namespace XRTailor {
class AABBRenderer : public LineRenderer {
 public:
  AABBRenderer();

  void SetDefaultAABBConfig();

  void SetCustomedAABBConfig();

  void AddAABB(const AABB& aabb);

  void SetAABBs(std::vector<AABB> aabbs);

  void ShowDebugInfo();

  ~AABBRenderer();
};
}  // namespace XRTailor