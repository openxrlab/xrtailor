#pragma once
#include <limits>
#include <glm/glm.hpp>
#include <xrtailor/core/Version.hpp>

namespace XRTailor {

#define M_PI 3.14159265358979323846

#ifdef XRTAILOR_USE_DOUBLE
using Scalar = double;
using Vector2 = glm::dvec2;
using Vector3 = glm::dvec3;
using Vector4 = glm::dvec4;
using Mat2 = glm::dmat2x2;
using Mat3 = glm::dmat3x3;
using Mat4 = glm::dmat4x4;
using Mat2x2 = glm::dmat2x2;
using Mat3x2 = glm::dmat3x2;
using Mat3x3 = glm::dmat3x3;
using Mat4x3 = glm::dmat4x3;
using Mat4x4 = glm::dmat4x4;
constexpr Scalar SCALAR_MAX_HOST = DBL_MAX;
constexpr Scalar EPSILON = 1e-6;
#else
using Scalar = float;
using Vector2 = glm::vec2;
using Vector3 = glm::vec3;
using Vector4 = glm::vec4;
using Mat2 = glm::mat2x2;
using Mat3 = glm::mat3x3;
using Mat4 = glm::mat4x4;
using Mat2x2 = glm::mat2x2;
using Mat3x2 = glm::mat3x2;
using Mat3x3 = glm::mat3x3;
using Mat4x3 = glm::mat4x3;
using Mat4x4 = glm::mat4x4;
constexpr Scalar SCALAR_MAX_HOST = FLT_MAX;
constexpr Scalar EPSILON = 1e-6f;
#endif

using uint = unsigned int;
using Vector2i = glm::ivec2;
using Vector2u = glm::uvec2;
using Vector3i = glm::ivec3;
using Vector3u = glm::uvec3;
using Vector4i = glm::ivec4;
using Vector4u = glm::uvec4;
}  // namespace XRTailor