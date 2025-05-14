#include <xrtailor/physics/graph_coloring/impl/StretchColoring.cuh>

namespace XRTailor {
namespace GraphColoring {

__global__ __inline__ void BuildStretchAdjacencyTable_Kernel(const Edge* const* edges,
                                                             AdjItem* adj_cons,
                                                             uint num_constraints) {
  GET_CUDA_ID(cid, num_constraints);
  auto pid0 = edges[cid]->nodes[0]->index;
  auto pid1 = edges[cid]->nodes[1]->index;

  adj_cons[cid].valid_size = 0u;
  for (auto i = 0u; i < num_constraints; i++) {
    int qid0 = edges[i]->nodes[0]->index;
    int qid1 = edges[i]->nodes[1]->index;
    if (cid != i &&  // skip identity
        (            // shared unknowns
            qid0 == pid0 || qid0 == pid1 || qid1 == pid0 || qid1 == pid1)) {
      int pos = atomicAdd(&adj_cons[cid].valid_size, 1);
      adj_cons[cid].adjs[pos] = i;
    }
  }
}

void StretchColoring::BuildAdjacencyTable(const Edge* const* edges, uint num_constraints) {
  this->node_size = num_constraints;
  adj_table.resize(node_size, AdjItem());
  CUDA_CALL(BuildStretchAdjacencyTable_Kernel, node_size)
  (edges, pointer(adj_table), node_size);
  CUDA_CHECK_LAST();
#ifdef PGS_DEBUG
  printf("Build stretch adjacency table done.\n");
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