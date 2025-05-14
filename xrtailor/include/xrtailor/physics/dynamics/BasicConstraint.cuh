#pragma once

#include <xrtailor/core/Scalar.cuh>

namespace XRTailor {
namespace BasicConstraint {

struct StretchConstraint {
  Scalar rest_length;
  int idx0;
  int idx1;
  Scalar lambda;
  Scalar compliance;
};

struct BendConstraint {
  Scalar theta;
  int idx0;
  int idx1;
  int idx2;
  int idx3;
  Scalar lambda;
  Scalar compliance;
};

}  // namespace BasicConstraint
}  // namespace XRTailor