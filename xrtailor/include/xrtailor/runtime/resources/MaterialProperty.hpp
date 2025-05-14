#pragma once

#include <functional>

#include <xrtailor/runtime/resources/Material.hpp>

namespace XRTailor {
// This class allows objects with same material to have different properties
struct MaterialProperty {
  std::function<void(Material*)> pre_rendering;
};
}  // namespace XRTailor