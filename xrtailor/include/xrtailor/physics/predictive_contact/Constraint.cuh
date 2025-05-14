#pragma once

#include <xrtailor/core/Scalar.cuh>

namespace XRTailor {
namespace PredictiveContact {

// Describes a contact between two edges of the cloth mesh for self-collision
struct EEContact {
  // Offset normal between closest points on the two lines
  Vector3 normal;
  // Parameter of closest point on line 0
  Scalar s;
  // Parameter of closest point on line 1
  Scalar t;
  // Index of the 0-th point on line 0
  int e0p0;
  // Index of the 1-st point on line 0
  int e0p1;
  // Index of the 0-th point on line 1
  int e1p0;
  // Index of the 1-st point on line 1
  int e1p1;
};

// Describes a contact between a cloth point and a cloth triangle for self-collision
struct VFContact {
  // index of the point;
  int point_index;
  // index of the triangle
  int triangle_index;
  // which side of the triangle the point started the frame on
  int is_back_face;
};

// Describes a contact between two points of the cloth mesh for self-collision
struct VVContact {
  // Offset normal between the points at the beginning of the frame
  Vector3 normal;
  // Index of the 0th point
  int p0;
  // Index of the 1st point
  int p1;
};

}  // namespace PredictiveContact
}  // namespace XRTailor