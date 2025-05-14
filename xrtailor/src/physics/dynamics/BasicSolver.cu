#include <xrtailor/physics/dynamics/BasicSolver.cuh>
#include <xrtailor/physics/dynamics/BasicSolverHelper.cuh>
#include <xrtailor/utils/Timer.hpp>

namespace XRTailor {
namespace BasicConstraint {
BendSolver::BendSolver() {}

void BendSolver::SetupGraphColoring(uint seed) {
  coloring_ = std::make_shared<GraphColoring::BendColoring>();
  bend_size_ = bend_indices_.size() / 4;
  std::cout << "bend_size: " << bend_size_ << std::endl;

  {
    ScopedTimerGPU timer("Solver_bendColoring_buildAdj");
    coloring_->BuildAdjacencyTable(pointer(bend_indices_), bend_size_);
  }
  double buildAdjTime = Timer::GetTimerGPU("Solver_bendColoring_buildAdj");

  std::cout << "Build bend graph, max degree: " << coloring_->max_degree
            << ", min degree: " << coloring_->min_degree
            << ", avg degree: " << coloring_->avg_degree << std::endl;

  {
    ScopedTimerGPU timer("Solver_bendColoring_paint");
    coloring_->Paint(seed, static_cast<Scalar>(coloring_->min_degree));
  }
  coloring_->n_colors = coloring_->ColorUsed();
  double coloringTime = Timer::GetTimerGPU("Solver_bendColoring_paint");
  std::cout << "Paint bend graph, node size: " << bend_size_
            << ", shrinking: " << static_cast<Scalar>(coloring_->min_degree)
            << ", color used: " << coloring_->n_colors << ", build adj " << buildAdjTime
            << "ms, coloring " << coloringTime << "ms, step took: " << coloring_->n_steps
            << std::endl;
}

void BendSolver::AddCloth(const std::vector<uint>& bend_indices,
                          const std::vector<Scalar>& bend_angles, const Scalar& bend_compliance) {
  int bend_size = bend_angles.size();
  thrust::host_vector<BendConstraint> h_constraints(bend_size);
  thrust::host_vector<uint> h_bend_indices(bend_size * 4);
  for (int i = 0; i < bend_angles.size(); i++) {
    auto idx0 = bend_indices[i * 4u + 0u];
    auto idx1 = bend_indices[i * 4u + 1u];
    auto idx2 = bend_indices[i * 4u + 2u];
    auto idx3 = bend_indices[i * 4u + 3u];
    auto theta = bend_angles[i];

    BendConstraint& c = h_constraints[i];
    c.idx0 = idx0;
    c.idx1 = idx1;
    c.idx2 = idx2;
    c.idx3 = idx3;
    c.theta = theta;
    c.lambda = static_cast<Scalar>(0.0);
    c.compliance = bend_compliance;

    h_bend_indices[i * 4] = idx0;
    h_bend_indices[i * 4 + 1] = idx1;
    h_bend_indices[i * 4 + 2] = idx2;
    h_bend_indices[i * 4 + 3] = idx3;
  }

  constraints_ = std::move(h_constraints);
  bend_indices_ = std::move(h_bend_indices);
}

void BendSolver::Solve(Node** nodes, uint color, const Scalar dt, int iter) {
  SolveBend(pointer(constraints_), nodes, color, pointer(coloring_->colors), dt, iter,
            bend_size_);
}

std::shared_ptr<GraphColoring::BendColoring> BendSolver::Coloring() {
  return coloring_;
}

StretchSolver::StretchSolver(Scalar sor) : sor_(sor) {}

void StretchSolver::SetupGraphColoring(Edge** edges, int n_edges, uint seed) {
  stretch_size_ = n_edges;
  constraints_.resize(n_edges);

  coloring_ = std::make_shared<GraphColoring::StretchColoring>();
  checkCudaErrors(cudaDeviceSynchronize());
  {
    ScopedTimerGPU timer("Solver_stretchColoring_buildAdj");
    coloring_->BuildAdjacencyTable(edges, stretch_size_);
  }
  checkCudaErrors(cudaDeviceSynchronize());
  double buildAdjTime = Timer::GetTimerGPU("Solver_stretchColoring_buildAdj");
  std::cout << "[BasicStretchSolver] Build stretch graph, max degree: " << coloring_->max_degree
            << ", min degree: " << coloring_->min_degree
            << ", avg degree: " << coloring_->avg_degree << std::endl;

  {
    ScopedTimerGPU timer("Solver_stretchColoring_paint");
    coloring_->Paint(seed, static_cast<Scalar>(coloring_->min_degree));
  }
  coloring_->n_colors = coloring_->ColorUsed();
  double coloringTime = Timer::GetTimerGPU("Solver_stretchColoring_paint");
  std::cout << "[BasicStretchSolver] Paint stretch graph, node size: " << stretch_size_
            << ", shrinking: " << static_cast<Scalar>(coloring_->min_degree)
            << ", color used: " << coloring_->n_colors << ", build adj " << buildAdjTime
            << "ms, coloring " << coloringTime << "ms, step took: " << coloring_->n_steps
            << std::endl;
}

void StretchSolver::AddCloth(const std::vector<int>& ev0_indices,
                             const std::vector<int>& ev1_indices,
                             const std::vector<Scalar>& e_lengths,
                             const Scalar& stretch_compliance) {
  thrust::host_vector<StretchConstraint> h_constraints;
  int n_edges = ev0_indices.size();
  for (int i = 0; i < n_edges; i++) {
    StretchConstraint c;
    c.idx0 = ev0_indices[i];
    c.idx1 = ev1_indices[i];
    c.rest_length = e_lengths[i];
    c.lambda = static_cast<Scalar>(0);
    c.compliance = stretch_compliance;

    h_constraints.push_back(c);
  }
  this->constraints_.insert(constraints_.end(), h_constraints.begin(), h_constraints.end());
}

std::shared_ptr<GraphColoring::StretchColoring> StretchSolver::Coloring() {
  return coloring_;
}

void StretchSolver::Solve(Node** nodes, Vector3* prev, uint color, const Scalar dt, int iter) {
  SolveStretch(pointer(constraints_), nodes, prev, color, pointer(coloring_->colors), dt, iter,
               sor_, stretch_size_);
}

}  // namespace BasicConstraint
}  // namespace XRTailor