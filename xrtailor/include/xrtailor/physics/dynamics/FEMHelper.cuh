#pragma once

#include <xrtailor/core/DeviceHelper.cuh>
#include <xrtailor/core/Common.cuh>
#include <xrtailor/memory/Face.cuh>
#include <xrtailor/physics/dynamics/FEMConstraint.cuh>

namespace XRTailor {
namespace FEM {

void InitStrain(StrainConstraint* constraints, Vector3* restPositions, const Face* const* faces,
                Scalar xx_stiffness, Scalar yy_stiffness, Scalar xy_stiffness, Scalar xy_poisson_ratio,
                Scalar yx_poisson_ratio, uint strain_size);

void SolveStrain(StrainConstraint* constraints, Node** nodes, uint color, uint* colors,
                 uint strain_size);

void InitIsometricBending(IsometricBendingConstraint* constraints, Vector3* restPositions,
                          uint* bend_indices, Scalar stiffness, uint bend_size);

void SolveIsometricBending(IsometricBendingConstraint* constraints, Node** nodes, uint color,
                           uint* colors, int iter, Scalar dt, uint bend_size);

}  // namespace FEM
}  // namespace XRTailor