#include <xrtailor/physics/graph_coloring/impl/FEMIsometricBending.cuh>

namespace XRTailor {
namespace GraphColoring {

void FEMIsometricBendingColoring::BuildAdjacencyTable(CONST(uint*) bend_indices,
                                                      uint n_constraints) {
  this->node_size = n_constraints;
  adj_table.resize(node_size, AdjItem());
  BuildBendAdjacencyTable(bend_indices, pointer(adj_table), node_size);

#ifdef PGS_DEBUG
  printf("Build FEM bend adjacency table done.\n");
#endif  // PGS_DEBUG

  colors.resize(node_size, 0u);
  palettes.resize(node_size, Palette());
  is_colored.resize(node_size, 0u);
  rand_state.resize(node_size, curandState());

  thrust::host_vector<AdjItem> h_adj_table = adj_table;
  int total_degree = 0;
  for (auto i = 0u; i < node_size; i++) {
    int degree = h_adj_table[i].valid_size;
    if (degree > max_degree)
      max_degree = degree;
    if (degree < min_degree)
      min_degree = degree;
    total_degree += degree;
  }

  avg_degree = static_cast<Scalar>(total_degree) / node_size;

  checkCudaErrors(cudaDeviceSynchronize());
}

}  // namespace GraphColoring
}  // namespace XRTailor