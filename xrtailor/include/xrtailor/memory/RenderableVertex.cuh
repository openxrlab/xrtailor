#pragma once

#include <cuda_runtime.h>

#include <xrtailor/core/Scalar.cuh>

namespace XRTailor {

class RenderableVertex {
 public:
  glm::vec3 x, n;
  glm::vec2 uv;
  __host__ __device__ RenderableVertex();
  ~RenderableVertex() = default;
};

}  // namespace XRTailor
