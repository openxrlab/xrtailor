#pragma once

#include <xrtailor/core/Scalar.cuh>

namespace XRTailor {
namespace FEM {

struct StrainConstraint {
  int idx0;
  int idx1;
  int idx2;
  Scalar area;
  Mat2 inv_rest_mat;
  Scalar xx_stiffness;
  Scalar xy_stiffness;
  Scalar yy_stiffness;
  Scalar xy_poisson_ratio;
  Scalar yx_poisson_ratio;
};

struct IsometricBendingConstraint {
  int idx0;
  int idx1;
  int idx2;
  int idx3;
  Scalar stiffness;
  Mat4x4 Q;
  Scalar lambda;
};

}  // namespace FEM
}  // namespace XRTailor