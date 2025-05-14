#pragma once

#include <xrtailor/physics/graph_coloring/impl/FEMStrainColoring.cuh>
#include <xrtailor/physics/graph_coloring/impl/FEMIsometricBending.cuh>
#include <xrtailor/physics/dynamics/FEMHelper.cuh>
#include <xrtailor/physics/dynamics/FEMConstraint.cuh>
#include <xrtailor/memory/Face.cuh>

namespace XRTailor {
namespace FEM {

class IsometricBendingSolver {
 public:
  IsometricBendingSolver();

  void Init(Vector3* restPositions, Scalar stiffness);

  void AddCloth(const std::vector<uint>& bend_indices, const std::vector<Scalar>& bend_angles);

  void SetupGraphColoring(uint seed);

  void Solve(Node** nodes, uint color, int iter, Scalar dt);

  std::shared_ptr<GraphColoring::FEMIsometricBendingColoring> Coloring();

 private:
  int n_constraints_;
  thrust::device_vector<IsometricBendingConstraint> constraints_;
  std::shared_ptr<GraphColoring::FEMIsometricBendingColoring> coloring_;
  thrust::device_vector<uint> bend_indices_;
};

class StrainSolver {
 public:
  StrainSolver(Face** faces, int n_faces, uint seed);

  void Init(Vector3* restPositions, const Face* const* faces, Scalar xx_stiffness,
            Scalar yy_stiffness, Scalar xy_stiffness, Scalar xy_poisson_ratio, Scalar yx_poisson_ratio);

  void Solve(Node** nodes, uint color);

  std::shared_ptr<GraphColoring::FEMStrainColoring> Coloring();

 private:
  int n_constraints_;
  thrust::device_vector<StrainConstraint> constraints_;
  std::shared_ptr<GraphColoring::FEMStrainColoring> coloring_;
};

}  // namespace FEM
}  // namespace XRTailor