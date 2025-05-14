#include <xrtailor/physics/graph_coloring/GraphColoringHelper.cuh>

namespace XRTailor {
namespace GraphColoring {

__global__ __inline__ void BuildBendAdjacencyTable_Kernel(CONST(uint*) bend_indices, AdjItem* adj_cons,
                                               uint num_constraints) {
  GET_CUDA_ID(cid, num_constraints);

  auto pid0 = bend_indices[cid * 4u];
  auto pid1 = bend_indices[cid * 4u + 1u];
  auto pid2 = bend_indices[cid * 4u + 2u];
  auto pid3 = bend_indices[cid * 4u + 3u];

  adj_cons[cid].valid_size = 0u;
  for (auto i = 0u; i < num_constraints; i++) {
    auto pid0_hat = bend_indices[i * 4u];
    auto pid1_hat = bend_indices[i * 4u + 1u];
    auto pid2_hat = bend_indices[i * 4u + 2u];
    auto pid3_hat = bend_indices[i * 4u + 3u];
    if (cid != i &&  // skip identity
        (            // shared unknowns
            pid0_hat == pid0 || pid0_hat == pid1 || pid0_hat == pid2 || pid0_hat == pid3 ||
            pid1_hat == pid0 || pid1_hat == pid1 || pid1_hat == pid2 || pid1_hat == pid3 ||
            pid2_hat == pid0 || pid2_hat == pid1 || pid2_hat == pid2 || pid2_hat == pid3 ||
            pid3_hat == pid0 || pid3_hat == pid1 || pid3_hat == pid2 || pid3_hat == pid3)) {
      int pos = atomicAdd(&adj_cons[cid].valid_size, 1);
      adj_cons[cid].adjs[pos] = i;
    }
  }
}

void BuildBendAdjacencyTable(CONST(uint*) bend_indices, AdjItem* adj_cons, uint num_constraints) {
  CUDA_CALL(BuildBendAdjacencyTable_Kernel, num_constraints)
  (bend_indices, adj_cons, num_constraints);
  CUDA_CHECK_LAST();
}

}  // namespace GraphColoring
}  // namespace XRTailor
