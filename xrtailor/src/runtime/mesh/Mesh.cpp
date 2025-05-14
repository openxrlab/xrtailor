#include <xrtailor/core/Precompiled.h>
#include <xrtailor/runtime/mesh/Mesh.hpp>

#include <xrtailor/core/Global.hpp>
#include <xrtailor/utils/ObjUtils.hpp>

namespace XRTailor {
void Mesh::ParseBoundaries() {
  vcg::face::Pos<TailorMesh::FaceType> he, hei;

  UnMarkAll(tmesh_);

  // simple search and trace of all the borders of the mesh
  int boundary_num = 0;
  for (auto fi = tmesh_.face.begin(); fi != tmesh_.face.end(); ++fi) {
    if (!(*fi).IsD()) {
      for (int j = 0; j < 3; j++) {
        if (vcg::face::IsBorder(*fi, j) && !vcg::tri::IsMarked(tmesh_, &*fi)) {
          vcg::tri::Mark(tmesh_, &*fi);
          hei.Set(&*fi, j, fi->V(j));
          he = hei;
          std::vector<TailorVertex*> boundary_vertices;
          do {
            he.NextB();  // next pos along a border
            auto e = he.f->FEp(he.E());
            boundary_vertices.push_back(e->V(0));
            boundary_vertices.push_back(e->V(1));

            vcg::tri::Mark(tmesh_, he.f);
          } while (he.f != hei.f);

          boundary_num++;
          boundaries_.push_back(boundary_vertices);
        }
      }
    }
  }
}

int Mesh::GetIndex(TailorVertex* v, TailorMesh& m) {
  return v - &m.vert[0];
}

int Mesh::GetIndex(TailorEdge* e, TailorMesh& m) {
  return e - &m.edge[0];
}

int Mesh::GetIndex(TailorFace* f, TailorMesh& m) {
  return f - &m.face[0];
}

int Mesh::GetEdgeIndex(TailorFace* f, TailorEdge* e) {
  for (size_t i = 0; i < 3; i++) {
    if (f->FEp(i) == e) {
      return i;
    }
  }

  // edge does not belong to face
  return -1;
}

void Mesh::GetNeighbors(TailorVertex* v, std::vector<TailorVertex*>& neighbors) {
  vcg::face::VVStarVF<TailorFace>(v, neighbors);
}

bool IsManifold(TailorFace* f0, int e) {
  return (f0 == f0->FFp(e)->FFp(f0->FFi(e)));
}

void FindNonManifoldEdgeVerticesByVertex(TailorMesh& m, TailorVertex* v,
                                         std::vector<TailorVertex*>& res_vertices) {
  /*
		* Algorithm. FindNonManifoldVerticesByVertex
		*  1. Given a vertex, find its adjacent non-manifold edges and push them to stack
		*  2. For each non-manifold edge on stack top:
		*  3.	visited ? pop the stack top : mark edge as visited
		*  4.	add the edge to non_manifold_edges list
		*  5.	find its non-manifold edges
		*  6.	for each non-manifold edge:
		*  7.		if not visited => push
		*  8. For edge in non_manifold_edges:
		*  9.	get adjacent vertices of e;
		*  10.	add adjacent vertices into non_manifold_vertices
		*  11.Remove duplicated vertices into non_manifold_vertices
		*/
  UnMarkAll(m);

  std::vector<TailorEdge*> res_edges;

  std::stack<TailorEdge*> e_stack;

  std::vector<TailorEdge*> adj_edges;
  vcg::edge::VEStarVE(v, adj_edges);

  for (auto eit = adj_edges.begin(); eit != adj_edges.end(); ++eit) {
    if (!IsManifold((*eit)->EFp(), (*eit)->EFi())) {
      e_stack.push((*eit));
    }
  }

  std::vector<TailorEdge*> adj_v0_e;
  std::vector<TailorEdge*> adj_non_manifold_v0_e;
  std::vector<TailorEdge*> adj_v1_e;
  std::vector<TailorEdge*> adj_non_manifold_v1_e;
  while (!e_stack.empty()) {
    auto e = e_stack.top();
    e_stack.pop();
    vcg::tri::Mark(m, e);

    auto v0 = e->V(0);
    auto v1 = e->V(1);
    adj_non_manifold_v0_e.clear();
    adj_non_manifold_v1_e.clear();
    vcg::edge::VEStarVE(v0, adj_v0_e);
    vcg::edge::VEStarVE(v1, adj_v1_e);
    res_edges.push_back(e);
    adj_v1_e.insert(adj_v1_e.end(), adj_v0_e.begin(), adj_v0_e.end());

    for (auto eit = adj_v1_e.begin(); eit != adj_v1_e.end(); ++eit) {
      if (!IsManifold((*eit)->EFp(), (*eit)->EFi()) && !vcg::tri::IsMarked(m, (*eit)))
      {
        e_stack.push((*eit));
      }
    }
  }
  res_vertices.clear();
  for (auto eit = res_edges.begin(); eit != res_edges.end(); ++eit) {
    res_vertices.push_back((*eit)->V(0));
    res_vertices.push_back((*eit)->V(1));
  }

  // remove duplicated vertices
  std::sort(res_vertices.begin(), res_vertices.end());
  res_vertices.erase(std::unique(res_vertices.begin(), res_vertices.end()), res_vertices.end());
}

void Mesh::AddVertices(const int& extend_mode, std::vector<unsigned int>& markers,
                       std::vector<unsigned int>& target) {
  switch (extend_mode) {
    case EXTEND_MODE::NEIGHBOR: {
      std::set<unsigned int> extended_indices;
      std::vector<TailorVertex*> neighbors;
      for (auto& marker : markers) {
        extended_indices.insert(marker);
        vcg::face::VVStarVF<TailorFace>(&tmesh_.vert[marker], neighbors);
        for (auto neighbor : neighbors) {
          extended_indices.insert(GetIndex(neighbor, tmesh_));
        }
      }

      for (auto& index : extended_indices) {
        target.push_back(index);
      }

      break;
    }
    case EXTEND_MODE::BOUNDARY: {
      std::set<unsigned int> boundary_indices;
      for (auto marker : markers) {
        for (auto boundary : boundaries_) {
          auto vit = std::find(boundary.begin(), boundary.end(), &tmesh_.vert[marker]);
          if (vit != boundary.end()) {
            // marker is on the boundary, add all vertices to the boundary
            for (auto v : boundary) {
              boundary_indices.insert(GetIndex(v, tmesh_));
            }
          }
        }
      }
      for (auto& index : boundary_indices) {
        target.push_back(index);
      }
      break;
    }
    case EXTEND_MODE::NONMANIFOLD_EDGES: {
      for (auto marker : markers) {
        std::vector<TailorVertex*> verts;
        FindNonManifoldEdgeVerticesByVertex(tmesh_, &tmesh_.vert[marker], verts);
        for (auto v : verts) {
          target.push_back(GetIndex(v, tmesh_));
        }
      }
      break;
    }
    default:
      LOG_WARN("Invalid vertex extend mode.");
      break;
  }
}

void UVIndexInFace(TailorFace* f, TailorEdge* e, int& uv0_idx, int& uv1_idx) {
  auto ev0 = e->V(0);
  auto ev1 = e->V(1);

  for (int i = 0; i < 3; i++) {
    if (f->V(i) == ev0) {
      uv0_idx = f->WT(i).N();
    }
    if (f->V(i) == ev1) {
      uv1_idx = f->WT(i).N();
    }
  }
}

void Mesh::AddBindedVertices(const int& extend_mode, std::vector<BindingParam>& binding_params) {
  switch (extend_mode) {
    case EXTEND_MODE::NEIGHBOR: {
      std::vector<TailorVertex*> neighbors;

      for (int i = 0; i < binding_params.size(); i++) {
        auto marker = binding_params[i].idx;
        auto stiffness = binding_params[i].stiffness;
        auto distance = binding_params[i].distance;
        this->extended_bindings_.insert(binding_params[i]);
        vcg::face::VVStarVF<TailorFace>(&tmesh_.vert[marker], neighbors);
        for (auto neighbor : neighbors) {
          BindingParam cfg = binding_params[i];
          cfg.idx = GetIndex(neighbor, tmesh_);
          this->extended_bindings_.insert(cfg);
        }
      }
      break;
    }
    case EXTEND_MODE::BOUNDARY: {
      std::vector<TailorVertex*> neighbors;

      for (int i = 0; i < binding_params.size(); i++) {
        auto marker = binding_params[i].idx;

        for (auto boundary : boundaries_) {
          auto vit = std::find(boundary.begin(), boundary.end(), &tmesh_.vert[marker]);
          if (vit != boundary.end()) {
            // marker is on the boundary, add all vertices to the boundary
            for (auto v : boundary) {
              BindingParam cfg = binding_params[i];
              cfg.idx = GetIndex(v, tmesh_);
              this->extended_bindings_.insert(cfg);
            }
          }
        }
      }

      break;
    }
    case EXTEND_MODE::NONMANIFOLD_EDGES: {
      for (int i = 0; i < binding_params.size(); i++) {
        auto marker = binding_params[i].idx;
        auto stiffness = binding_params[i].stiffness;
        auto distance = binding_params[i].distance;

        std::vector<TailorVertex*> verts;
        FindNonManifoldEdgeVerticesByVertex(tmesh_, &tmesh_.vert[marker], verts);
        for (auto v : verts) {
          BindingParam cfg = binding_params[i];
          cfg.idx = GetIndex(v, tmesh_);
          this->extended_bindings_.insert(cfg);
        }
      }
      break;
    }
    case EXTEND_MODE::UV_ISLAND: {
      for (int i = 0; i < binding_params.size(); i++) {
        auto marker = binding_params[i].idx;
        std::stack<TailorFace*> f_stack;
        f_stack.push(&tmesh_.face[marker]);
        vcg::tri::UnMarkAll(tmesh_);
        while (!f_stack.empty()) {
          auto f = f_stack.top();
          f_stack.pop();
          std::vector<TailorFace*> nb_ff;
          vcg::face::Pos<TailorFace> he;
          vcg::face::FFExtendedStarFF<TailorFace>(f, 1, nb_ff);
          for (auto fit = nb_ff.begin(); fit != nb_ff.end(); ++fit) {
            if (*fit == f) {
              continue;
            }
            TailorEdge* e = &tmesh_.edge[0];
            for (int i = 0; i < 3; i++) {
              for (int j = 0; j < 3; j++) {
                if (f->FFp(i) == *fit) {
                  e = f->FEp(i);
                  break;
                }
              }
            }
            int e0_uv0_idx, e0_uv1_idx, e1_uv0_idx, e1_uv1_idx;
            UVIndexInFace(f, e, e0_uv0_idx, e0_uv1_idx);
            UVIndexInFace((*fit), e, e1_uv0_idx, e1_uv1_idx);
            if (!vcg::tri::IsMarked(tmesh_, f) && (e0_uv0_idx == e1_uv0_idx) &&
                (e0_uv1_idx == e1_uv1_idx)) {
              BindingParam cfg0, cfg1 = binding_params[i];
              cfg0.idx = GetIndex(e->V(0), tmesh_);
              cfg1.idx = GetIndex(e->V(1), tmesh_);
              this->extended_bindings_.insert(cfg0);
              this->extended_bindings_.insert(cfg1);
              f_stack.push(*fit);
            }
          }
          vcg::tri::Mark(tmesh_, f);
        }
      }
      break;
    }
    default:
      LOG_WARN("Invalid vertex extend mode.");
      break;
  }
}

void Mesh::ApplyBindings() {
  for (auto& binding : extended_bindings_) {
    this->binded_indices_.push_back(binding.idx);
    this->bind_stiffnesses_.push_back(binding.stiffness);
    this->bind_distances_.push_back(binding.distance);
  }
}

void Mesh::AddFixedVertices(const std::vector<unsigned int>& indices) {
  fixed_indices_.insert(fixed_indices_.end(), indices.begin(), indices.end());
}

void Mesh::AddAttachedVertices(std::vector<unsigned int>& indices) {
  for (auto& idx : indices) {
    attached_indices_.push_back(idx);
  }
}

void Mesh::BuildGeodesic() {
  Geodesic geodesic_calculator(this);
  for (auto& idx : attached_indices_) {
    geodesic_calculator.sources.emplace_back(idx);
  }

  for (unsigned int i = 0; i < geodesic_calculator.sources.size(); i++) {
    geodesic_calculator.ComputeGeodesicDistance(i);

    std::vector<unsigned int> previous = geodesic_calculator.vec_previous[i];
    for (size_t i = 0; i < previous.size(); i++) {
      geodesic_previous_.push_back(previous[i]);
    }
  }
  geodesic_distances_ = geodesic_calculator.distances;
}

thrust::host_vector<uint> Mesh::FaceEdgeIndices() {
  thrust::host_vector<uint> res;
  for (int i = 0; i < tmesh_.face.size(); i++) {
    auto f = &tmesh_.face[i];
    for (int j = 0; j < 3; j++) {
      res.push_back(GetIndex(f->FEp(j), tmesh_));
    }
  }

  return res;
}
}  // namespace XRTailor