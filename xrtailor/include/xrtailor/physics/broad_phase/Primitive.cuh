#pragma once

#include <xrtailor/core/Common.cuh>
#include <xrtailor/core/Scalar.cuh>

namespace XRTailor {
struct Primitive {
  Vector3 v1, v2, v3;
  Vector3 pred1, pred2, pred3;
  uint idx1, idx2, idx3;
};
}  // namespace XRTailor