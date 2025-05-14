#include <xrtailor/physics/graph_coloring/impl/BendColoring.cuh>

namespace XRTailor {
namespace GraphColoring {

void BendColoring::BuildAdjacencyTable(CONST(uint*) bend_indices, uint num_constraints) {
  printf("BendColoring::BuildAdjacencyTable\n");
  this->node_size = num_constraints;
  adj_table.resize(node_size, AdjItem());

  BuildBendAdjacencyTable(bend_indices, pointer(adj_table), node_size);

  cudaError_t kernelError = cudaGetLastError();
  if (kernelError != cudaSuccess) {
    fprintf(stderr, "CUDA Error after VVonstraint::GenerateStackless kernel launch: %s\n",
            cudaGetErrorString(kernelError));
  }

  cudaError_t syncError = cudaDeviceSynchronize();
  if (syncError != cudaSuccess) {
    fprintf(stderr, "CUDA Error after VVConstraint::GenerateStackless cudaDeviceSynchronize: %s\n",
            cudaGetErrorString(syncError));
  }
#ifdef PGS_DEBUG
  printf("Build bend adjacency table done.\n");
#endif  // PGS_DEBUG

  colors.resize(node_size, 0u);
  palettes.resize(node_size, Palette());
  is_colored.resize(node_size, 0u);
  rand_state.resize(node_size, curandState());

  thrust::host_vector<AdjItem> h_adj_table = adj_table;
  int total_degree = 0;
  for (auto i = 0u; i < node_size; i++) {
    int degree = h_adj_table[i].valid_size;
    if (degree > 99) {
      printf("cid: %d, size: %d\n", i, degree);
    }
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