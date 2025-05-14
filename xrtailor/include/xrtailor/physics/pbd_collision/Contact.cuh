#pragma once

#include <xrtailor/memory/Face.cuh>

namespace XRTailor {

struct VFContact {
  int index;
  Vector3 qs;
};

struct EEContact {
  Scalar dist;
  Vector3 p2;
  Vector3 q2;
  Scalar s;
  Scalar t;
  Vector3 fn;
};

struct RTContact {
  int index;
  Vector3 qc;
};

}  // namespace XRTailor