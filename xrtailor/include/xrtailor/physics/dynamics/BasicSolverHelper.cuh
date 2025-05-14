#pragma once

#include <xrtailor/memory/Node.cuh>
#include <xrtailor/physics/dynamics/BasicConstraint.cuh>

namespace XRTailor {
namespace BasicConstraint {

void SolveBend(BendConstraint* constraints, Node** nodes, uint color, CONST(uint*) colors,
               const Scalar dt, int iter, const uint n_constraints);

void SolveStretch(StretchConstraint* constraints, Node** nodes, Vector3* prev, uint color,
                  uint* colors, const Scalar dt, int iter, Scalar sor, int stretch_size);

}  // namespace BasicConstraint
}  // namespace XRTailor
