#include <xrtailor/physics/graph_coloring/impl/MeshColoring.cuh>

namespace XRTailor {
namespace GraphColoring {

void MeshColoring::BuildAdjacencyTable(CONST(int*) stretch_indices, uint n_cons, uint n_verts) {
  this->n_verts = n_verts;
  h_nb_vv.resize(n_verts);
  for (auto i = 0u; i < n_cons; i++) {
    auto idx0 = stretch_indices[i * 2u];
    auto idx1 = stretch_indices[i * 2u + 1u];

    h_nb_vv[idx0].insert(idx1);
    h_nb_vv[idx1].insert(idx0);
  }

  int max_degree = 0;
  int min_degree = 9999;
  int total_degree = 0;
  Scalar avg_degree = static_cast<Scalar>(0.0);
  for (auto i = 0u; i < n_verts; i++) {
    int degree = h_nb_vv[i].size();
    if (degree > max_degree) {
      max_degree = degree;
    }
    if (degree < min_degree) {
      min_degree = degree;
    }
    total_degree += degree;
  }

  avg_degree = total_degree / n_verts;

  printf("max degree: %d, min degree: %d, avg degree: %.2f\n", max_degree, min_degree, avg_degree);

  this->node_size = n_verts;

  for (auto i = 0u; i < n_verts; i++) {
    auto vv_adj = h_nb_vv[i];
    auto adj_item = AdjItem();
    adj_item.valid_size = 0;
    for (auto& vert : vv_adj) {
      adj_item.adjs[adj_item.valid_size++] = vert;
    }
    adj_table.push_back(adj_item);
  }

  colors.resize(node_size, 0u);
  palettes.resize(node_size, Palette());
  is_colored.resize(node_size, 0u);
  rand_state.resize(node_size, curandState());

#ifdef PGS_DEBUG
  printf("Build adjacency table done.\n");
#endif  // PGS_DEBUG

  checkCudaErrors(cudaDeviceSynchronize());
}

void MeshColoring::BuildAdjacencyTableDummy() {
  this->n_verts = 11;
  h_nb_vv.resize(n_verts);
  h_nb_vv[0].insert({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  h_nb_vv[1].insert({0, 2, 6, 7});
  h_nb_vv[2].insert({0, 1, 3, 7});
  h_nb_vv[3].insert({0, 2, 4, 8});
  h_nb_vv[4].insert({0, 3, 5, 8, 9});
  h_nb_vv[5].insert({0, 4, 6, 9, 10});
  h_nb_vv[6].insert({0, 1, 5, 10});
  h_nb_vv[7].insert({0, 1, 2});
  h_nb_vv[8].insert({0, 3, 4});
  h_nb_vv[9].insert({0, 4, 5});
  h_nb_vv[10].insert({0, 5, 6});

  int max_degree = 0;
  int min_degree = 9999;
  int total_degree = 0;
  Scalar avg_degree = static_cast<Scalar>(0.0);
  for (auto i = 0u; i < n_verts; i++) {
    int degree = h_nb_vv[i].size();
    //printf("degree: %d \t", degree);
    if (degree > max_degree) {
      max_degree = degree;
    }
    if (degree < min_degree) {
      min_degree = degree;
    }
    total_degree += degree;
  }

  avg_degree = static_cast<Scalar>(total_degree) / n_verts;

  printf("max degree: %d, min degree: %d, avg degree: %.2f\n", max_degree, min_degree, avg_degree);

  this->node_size = n_verts;

  for (auto i = 0u; i < n_verts; i++) {
    auto vv_adj = h_nb_vv[i];
    auto adj_item = AdjItem();
    adj_item.valid_size = 0;
    for (auto& vert : vv_adj) {
      adj_item.adjs[adj_item.valid_size++] = vert;
    }
    adj_table.push_back(adj_item);
  }

  colors.resize(node_size, 0u);
  palettes.resize(node_size, Palette());
  is_colored.resize(node_size, 0u);
  rand_state.resize(node_size, curandState());

#ifdef PGS_DEBUG
  printf("Build adjacency table done.\n");
#endif  // PGS_DEBUG

  checkCudaErrors(cudaDeviceSynchronize());
}

void MeshColoring::AssignColors() {
#ifdef PGS_DEBUG
  printf("Begin assign vertex colors.\n");
#endif  // PGS_DEBUG
  checkCudaErrors(cudaDeviceSynchronize());
  for (auto i = 0u; i < n_verts; i++) {
    verts_colors.push_back(color_map[colors[i] % color_map.size()]);
  }
#ifdef PGS_DEBUG
  printf("Assign vertex colors done.\n");
#endif  // PGS_DEBUG
}

}  // namespace GraphColoring
}  // namespace XRTailor