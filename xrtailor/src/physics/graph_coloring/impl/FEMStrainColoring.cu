#include <xrtailor/physics/graph_coloring/impl/FEMStrainColoring.cuh>

namespace XRTailor {
namespace GraphColoring {

__global__ __inline__ void BuildFEMStrainAdjacencyTable_Kernel(const Face* const* faces,
                                                               AdjItem* adj_cons,
                                                               uint n_constraints) {
  GET_CUDA_ID(cid, n_constraints);

  int pid0 = faces[cid]->nodes[0]->index;
  int pid1 = faces[cid]->nodes[1]->index;
  int pid2 = faces[cid]->nodes[2]->index;

  adj_cons[cid].valid_size = 0u;
  for (auto i = 0u; i < n_constraints; i++) {
    int qid0 = faces[i]->nodes[0]->index;
    int qid1 = faces[i]->nodes[1]->index;
    int qid2 = faces[i]->nodes[2]->index;
    if (cid != i &&  // skip identity
        (            // shared unknowns
            qid0 == pid0 || qid0 == pid1 || qid0 == pid2 || qid1 == pid0 || qid1 == pid1 ||
            qid1 == pid2 || qid2 == pid0 || qid2 == pid1 || qid2 == pid2)) {
      int pos = atomicAdd(&adj_cons[cid].valid_size, 1);
      adj_cons[cid].adjs[pos] = i;
    }
  }
}

void FEMStrainColoring::BuildAdjacencyTable(const Face* const* faces, uint n_constraints) {
  this->node_size = n_constraints;
  adj_table.resize(node_size, AdjItem());

  CUDA_CALL(BuildFEMStrainAdjacencyTable_Kernel, node_size)
  (faces, pointer(adj_table), node_size);
#ifdef PGS_DEBUG
  printf("Build FEM strain adjacency table done.\n");
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