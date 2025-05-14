#include <xrtailor/physics/dynamics/FEMSolver.cuh>
#include <xrtailor/math/MathFunctions.cuh>
#include <xrtailor/utils/ObjUtils.hpp>
#include <xrtailor/utils/Timer.hpp>

namespace XRTailor {
namespace FEM {

StrainSolver::StrainSolver(Face** faces, int n_faces, uint seed) {
  n_constraints_ = n_faces;
  constraints_.resize(n_faces);

  coloring_ = std::make_shared<GraphColoring::FEMStrainColoring>();
  printf("Build adj table\n");
  coloring_->BuildAdjacencyTable(faces, n_constraints_);
  printf(
      "Build strain graph, max degree: %d, min degree: %d, avg degree: %.2f, node_size: %d, "
      "table_size: %d\n",
      coloring_->max_degree, coloring_->min_degree, coloring_->avg_degree, coloring_->node_size,
      coloring_->adj_table.size());
  printf("Paint\n");
  coloring_->Paint(seed, static_cast<Scalar>(coloring_->min_degree));
  coloring_->n_colors = coloring_->ColorUsed();
  printf("Paint FEM strain graph, shrinking: %.2f, color used: %d, step took: %d\n",
         static_cast<Scalar>(coloring_->min_degree), coloring_->n_colors, coloring_->n_steps);
}

void StrainSolver::Init(Vector3* rest_positions, const Face* const* faces, Scalar xx_stiffness,
                        Scalar yy_stiffness, Scalar xy_stiffness, Scalar xy_poisson_ratio,
                        Scalar yx_poisson_ratio) {

  InitStrain(pointer(constraints_), rest_positions, faces, xx_stiffness, yy_stiffness, xy_stiffness,
             xy_poisson_ratio, yx_poisson_ratio, n_constraints_);
}

void StrainSolver::Solve(Node** nodes, uint color) {
  SolveStrain(pointer(constraints_), nodes, color, pointer(coloring_->colors), n_constraints_);
}

std::shared_ptr<GraphColoring::FEMStrainColoring> StrainSolver::Coloring() {
  return coloring_;
}

IsometricBendingSolver::IsometricBendingSolver() {}

void IsometricBendingSolver::SetupGraphColoring(uint seed) {
  n_constraints_ = bend_indices_.size() / 4;
  constraints_.resize(n_constraints_);

  coloring_ = std::make_shared<GraphColoring::FEMIsometricBendingColoring>();
  Timer::StartTimerGPU("FEM_COLORING_ADJ_TABLE_BUILD");
  coloring_->BuildAdjacencyTable(pointer(bend_indices_), n_constraints_);
  Timer::EndTimerGPU("FEM_COLORING_ADJ_TABLE_BUILD");
  auto t_adj = Timer::GetTimerGPU("FEM_COLORING_ADJ_TABLE_BUILD");
  printf("Build adj table, took %.2fms\n", t_adj);
  printf(
      "Build bend graph, max degree: %d, min degree: %d, avg degree: %.2f, node_size: %d, "
      "table_size: %d\n",
      coloring_->max_degree, coloring_->min_degree, coloring_->avg_degree, coloring_->node_size,
      coloring_->adj_table.size());

  Timer::StartTimerGPU("FEM_COLORING_PAINT");
  coloring_->Paint(seed, static_cast<Scalar>(coloring_->min_degree));
  Timer::EndTimerGPU("FEM_COLORING_PAINT");
  coloring_->n_colors = coloring_->ColorUsed();
  auto t_paint = Timer::GetTimerGPU("FEM_COLORING_PAINT");

  printf(
      "Paint FEM bend graph, shrinking: %.2f, color used: %d, step took: %d, time took: %.2fms\n",
      static_cast<Scalar>(coloring_->min_degree), coloring_->n_colors, coloring_->n_steps,
      t_paint);
}

void IsometricBendingSolver::Init(Vector3* rest_positions, Scalar stiffness) {
  InitIsometricBending(pointer(constraints_), rest_positions, pointer(bend_indices_), stiffness,
                       n_constraints_);
}

std::shared_ptr<GraphColoring::FEMIsometricBendingColoring> IsometricBendingSolver::Coloring() {
  return coloring_;
}

void IsometricBendingSolver::Solve(Node** nodes, uint color, int iter, Scalar dt) {
  SolveIsometricBending(pointer(constraints_), nodes, color, pointer(coloring_->colors), iter, dt,
                        n_constraints_);
}

void IsometricBendingSolver::AddCloth(const std::vector<uint>& bend_indices,
                                      const std::vector<Scalar>& bend_angles) {
  for (int i = 0; i < bend_angles.size(); i++) {
    auto idx0 = bend_indices[i * 4u + 0u];
    auto idx1 = bend_indices[i * 4u + 1u];
    auto idx2 = bend_indices[i * 4u + 2u];
    auto idx3 = bend_indices[i * 4u + 3u];
    auto theta = bend_angles[i];

    bend_indices_.push_back(idx0);
    bend_indices_.push_back(idx1);
    bend_indices_.push_back(idx2);
    bend_indices_.push_back(idx3);
  }
}

}  // namespace FEM
}  // namespace XRTailor