#pragma once

#include <iostream>
#include <vector>

#include <xrtailor/core/DeviceHelper.cuh>
#include <xrtailor/memory/Node.cuh>
#include <xrtailor/physics/dynamics/BasicConstraint.cuh>
#include <xrtailor/physics/graph_coloring/impl/StretchColoring.cuh>
#include <xrtailor/physics/graph_coloring/impl/BendColoring.cuh>

namespace XRTailor {
namespace BasicConstraint {

class BendSolver {
 public:
  BendSolver();

  void SetupGraphColoring(uint seed);

  void AddCloth(const std::vector<uint>& bend_indices, const std::vector<Scalar>& bend_angles,
                const Scalar& bend_compliance);

  void Solve(Node** nodes, uint color, const Scalar dt, int iter);

  std::shared_ptr<GraphColoring::BendColoring> Coloring();

 private:
  int bend_size_;
  thrust::device_vector<BendConstraint> constraints_;
  std::shared_ptr<GraphColoring::BendColoring> coloring_;
  thrust::device_vector<uint> bend_indices_;
};

class StretchSolver {
 public:
  StretchSolver(Scalar sor = static_cast<Scalar>(1));

  void SetupGraphColoring(Edge** edges, int n_edges, uint seed);

  void AddCloth(const std::vector<int>& ev0_indices, const std::vector<int>& ev1_indices,
                const std::vector<Scalar>& e_lengths,
                const Scalar& stretch_compliance = static_cast<Scalar>(0));

  void Solve(Node** nodes, Vector3* prev, uint color, const Scalar dt, int iter);

  std::shared_ptr<GraphColoring::StretchColoring> Coloring();

 private:
  int stretch_size_;
  Scalar sor_;
  thrust::device_vector<StretchConstraint> constraints_;
  std::shared_ptr<GraphColoring::StretchColoring> coloring_;
};

}  // namespace BasicConstraint
}  // namespace XRTailor