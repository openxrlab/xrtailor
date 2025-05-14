#include <xrtailor/physics/ClothSolverHelper.hpp>

namespace XRTailor {
void GenerateBendingElements(std::shared_ptr<Mesh> mesh, const uint& index_offset,
                             std::vector<uint>& bend_indices, std::vector<Scalar>& bend_angles) {
  LOG_TRACE("Generate bending elements");
  int idx1, idx2, idx3, idx4;
  glm::vec3 v1, v2, v3, v4;
  std::vector<TailorFace*> face_vec;
  std::vector<int> ind_ved;

  auto positions = mesh->Positions();
  auto indices = mesh->Indices();
  vcg::face::Pos<TailorFace> he1, he2;
  for (auto eit = mesh->TMesh().edge.begin(); eit != mesh->TMesh().edge.end(); ++eit) {
    if (!eit->IsB()) {
      auto idx1 = Mesh::GetIndex(eit->V(0), mesh->TMesh());
      auto idx2 = Mesh::GetIndex(eit->V(1), mesh->TMesh());
      vcg::face::EFStarFF(eit->EFp(), eit->EFi(), face_vec, ind_ved);
      for (size_t i = 0; i < face_vec.size(); i++) {
        for (size_t j = i + 1; j < face_vec.size(); j++) {
          auto f1 = face_vec[i];
          auto f2 = face_vec[j];

          he1.Set(f1, Mesh::GetEdgeIndex(f1, &(*eit)), eit->V(0));
          he1.FlipE();
          he1.FlipV();
          auto idx3 = Mesh::GetIndex(he1.V(), mesh->TMesh());

          he2.Set(f2, Mesh::GetEdgeIndex(f2, &(*eit)), eit->V(0));
          he2.FlipE();
          he2.FlipV();

          auto idx4 = Mesh::GetIndex(he2.V(), mesh->TMesh());

          int f1_idx = Mesh::GetIndex(f1, mesh->TMesh());
          int f2_idx = Mesh::GetIndex(f2, mesh->TMesh());

          // Let the 'edge' array be [n,m]. We have two triangles with this edge as common. We also have arrays of
          // triangle vertex indices. One of these two arrays contains common edge indices in right order and the other one
          // contains common edge indices in reversed order. We should identify which of triangles contains common edge indices in right order.
          std::string f1_indices_str = std::to_string(indices[f1_idx * 3 + 0]) + "," +
                                       std::to_string(indices[f1_idx * 3 + 1]) + "," +
                                       std::to_string(indices[f1_idx * 3 + 2]);
          f1_indices_str += "," + f1_indices_str;
          const std::string common_edge_indices_str =
              std::to_string(idx1) + "," + std::to_string(idx2);
          const bool right_order =
              (f1_indices_str.find(common_edge_indices_str) != std::string::npos);

          if (!right_order) {
            int tmp = idx1;
            idx1 = idx2;
            idx2 = tmp;
          }

          // Now we can create a right order of triangles indices to put them to the new constraint. What we want to get is:
          // 1) [0] and [1] are indices of common edge
          // 2) [2] and [3] are indices of free vertices of the triangles
          // 3) first triangle vertices are [0], [1], [2]
          // 4) second triangle vertices are [1], [0], [3]
          // 5) both triangles should go around counterclockwise
          v1 = positions[idx1];
          v2 = positions[idx2];
          v3 = positions[idx3];
          v4 = positions[idx4];

          // compute dihedral angle
          glm::vec3 n1 = glm::normalize(glm::cross(v1 - v3, v2 - v3));
          glm::vec3 n2 = glm::normalize(glm::cross(v2 - v4, v1 - v4));
          Scalar d = glm::dot(n1, n2);
          if (d < -1) {
            d = -1;
          } else if (d > 1) {
            d = 1;
          }
          Scalar theta = acos(d);

          bend_indices.push_back(idx1 + index_offset);
          bend_indices.push_back(idx2 + index_offset);
          bend_indices.push_back(idx3 + index_offset);
          bend_indices.push_back(idx4 + index_offset);
          bend_angles.push_back(theta);
        }
      }
    }
  }
}

void PrepareBasicStretchData(std::shared_ptr<Mesh> mesh, std::vector<int>& ev0_indices,
                             std::vector<int>& ev1_indices, std::vector<Scalar>& e_lengths,
                             int prevNumParticles) {
  auto positions = mesh->Positions();
  LOG_TRACE("[BasicStretchSolver] Generate stretch constraints");
  auto distance_between = [&positions](int idx1, int idx2) {
    return glm::length(positions[idx1] - positions[idx2]);
  };
  int idx0, idx1;
  auto edges = mesh->TMesh().edge;
  for (auto& edge : edges) {

    idx0 = Mesh::GetIndex(edge.V(0), mesh->TMesh());
    idx1 = Mesh::GetIndex(edge.V(1), mesh->TMesh());

    ev0_indices.push_back(idx0 + prevNumParticles);
    ev1_indices.push_back(idx1 + prevNumParticles);
    e_lengths.push_back(distance_between(idx0, idx1));
  }
}

void BuildEFAdjacency(std::shared_ptr<Mesh> mesh, std::vector<std::set<uint>>& nb_ef,
                      int& prev_num_edges, int& prev_num_faces) {
  LOG_INFO("Build EF adjacency");
  auto n_edges = mesh->TMesh().edge.size();
  nb_ef.resize(n_edges + prev_num_edges);
  for (int i = 0; i < n_edges; i++) {
    auto eit = &mesh->TMesh().edge[i];
    std::vector<TailorFace*> face_vec;
    std::vector<int> ind_ved;
    vcg::face::EFStarFF(eit->EFp(), eit->EFi(), face_vec, ind_ved);
    for (size_t fidx = 0; fidx < face_vec.size(); fidx++) {
      nb_ef[i + prev_num_edges].insert(Mesh::GetIndex(face_vec[fidx], mesh->TMesh()) + prev_num_faces);
    }
  }
}
}  // namespace XRTailor