#pragma once

#include <thrust/host_vector.h>
#include <vcg/complex/complex.h>
#include <vcg/complex/algorithms/create/platonic.h>

#include <xrtailor/core/Common.hpp>
#include <xrtailor/memory/RenderableVertex.cuh>
#include <xrtailor/runtime/mesh/Types.hpp>
#include <xrtailor/runtime/mesh/MeshIO.hpp>
#include <xrtailor/utils/Geodesic.hpp>
#include <xrtailor/utils/Logger.hpp>

namespace XRTailor {

class Mesh {
 public:
  Mesh(const MeshData& data, uint id) {
    LOG_TRACE("Construct mesh");
    std::vector<Vector3> vertices;
    std::vector<Vector3> normals;
    std::vector<Vector2> texCoords;
    std::vector<uint> indices;

    auto num_vertices = data.positions.size();
    // walk through each of the mesh's vertices
    for (uint i = 0; i < num_vertices; i++) {
      // positions
      vertices.push_back(data.positions[i]);
      // normals
      if (data.normals.size()) {
        normals.push_back(data.normals[i]);
      } else {
        LOG_WARN("Normals not found");
      }

      texCoords.push_back(Vector2(0.0f, 0.0f));
    }

    auto num_faces = data.indices.size();

    bool hasUV = true;

    if (data.uvs.size() == 0)
      hasUV = false;

    if (hasUV) {
      for (uint i = 0; i < data.uvs.size(); i++) {
        uvs_.push_back(Vector2(data.uvs[i].x, data.uvs[i].y));
      }
    }
    LOG_DEBUG(
        "Walk through each of the mesh's faces and retrieve the corresponding vertex indices.");
    for (uint i = 0; i < num_faces; i++) {
      std::vector<XRTailor::Index> face = data.indices[i];
      for (uint j = 0; j < 3; j++) {
        // retrieve all indices of the face and store them in the indices vector
        indices.push_back(face[j].position);
        if (hasUV)
          uv_indices_.push_back(face[j].uv);
        normal_indices_.push_back(face[j].normal);
      }
    }
    Initialize(vertices, normals, texCoords, indices);

    // makes the mesh build format compatible with vcg
    std::vector<vcg::Point3f> coordVec;
    std::vector<vcg::Point3i> indexVec;

    for (size_t i = 0; i < num_vertices; i++) {
      Vector3 pos = positions_[i];
      coordVec.push_back(vcg::Point3f(static_cast<float>(pos.x), static_cast<float>(pos.y),
                                      static_cast<float>(pos.z)));
    }

    for (size_t i = 0; i < num_faces; i++) {
      uint idx1 = indices_[i * 3];
      uint idx2 = indices_[i * 3 + 1];
      uint idx3 = indices_[i * 3 + 2];
      indexVec.push_back(vcg::Point3i(idx1, idx2, idx3));
    }

    vcg::tri::BuildMeshFromCoordVectorIndexVector(tmesh_, coordVec, indexVec);

    for (size_t i = 0; i < num_faces; i++) {
      auto f = &tmesh_.face[i];
      for (int j = 0; j < 3; ++j) {
        uint normal_idx = normal_indices_[i * 3 + j];
        Vector3 normal = normals_[normal_idx];
        f->V(j)->N() =
            TailorFace::NormalType(static_cast<float>(normal.x), static_cast<float>(normal.y),
                                   static_cast<float>(normal.z));
        if (hasUV) {
          uint uv_idx = uv_indices_[i * 3 + j];
          Vector2 uv = uvs_[uv_idx];
          f->WT(j) = TailorFace::TexCoordType(static_cast<float>(uv.x), static_cast<float>(uv.y));
          f->WT(j).N() = uv_idx;
        }
      }
    }

    // some cleaning to get rid of bad file formats like stl that duplicate vertexes..
    int dup = vcg::tri::Clean<TailorMesh>::RemoveDuplicateVertex(tmesh_);
    int unref = vcg::tri::Clean<TailorMesh>::RemoveUnreferencedVertex(tmesh_);
    LOG_INFO("Removed {} duplicate and {} unreferenced vertices", dup, unref);

    vcg::tri::UpdateTopology<TailorMesh>::FaceFace(tmesh_);
    vcg::tri::UpdateTopology<TailorMesh>::AllocateEdge(tmesh_);
    vcg::tri::UpdateTopology<TailorMesh>::VertexFace(tmesh_);
    vcg::tri::UpdateTopology<TailorMesh>::VertexEdge(tmesh_);
    vcg::tri::UpdateTopology<TailorMesh>::EdgeEdge(tmesh_);

    ParseBoundaries();
    LOG_INFO("Imported mesh: #VERTS: {}, #FACES: {}, #EDGES: {}", tmesh_.VN(), tmesh_.FN(),
             tmesh_.EN());
    id_ = id;
  }

  /*
		Construct mesh utilizing packed attributes(positions, normals, uvs are combined together)

		arrtibute_sizes: dimension of each attribute, e.g., positions has 3 dimensions, texture coordinates has 2 dimensions
		packed_vertices: vectorized vertices, which contains vertex postions, normals and texture coordinates, etc.
		indices: vectorized triangle indices
	*/
  Mesh(std::vector<uint> attribute_sizes, std::vector<Scalar> packed_vertices,
       std::vector<uint> indices = std::vector<uint>()) {
    LOG_DEBUG("Construct mesh utilizing packed attributes");
    std::vector<Vector3> positions;
    std::vector<Vector3> normals;
    std::vector<Vector2> tex_coords;
    uint stride = 0;  // total dimensions of attributes
    for (int i = 0; i < attribute_sizes.size(); i++) {
      stride += attribute_sizes[i];
    }
    uint num_vertices = static_cast<uint>(packed_vertices.size()) / stride;

    for (uint i = 0; i < num_vertices; i++) {
      uint base_v = stride * i;
      uint base_n = (stride >= 6) ? base_v + 3 : base_v;
      uint base_t = base_n + 3;

      positions.push_back(Vector3(packed_vertices[base_v + 0], packed_vertices[base_v + 1],
                                  packed_vertices[base_v + 2]));
      if (stride >= 6) {
        normals_.push_back(Vector3(packed_vertices[base_n + 0], packed_vertices[base_n + 1],
                                   packed_vertices[base_n + 2]));
      }
      tex_coords.push_back(Vector2(packed_vertices[base_t + 0], packed_vertices[base_t + 1]));
    }
    Initialize(positions, normals, tex_coords, indices);
  }

  Mesh(const std::vector<Vector3>& vertices,
       const std::vector<Vector3>& normals = std::vector<Vector3>(),
       const std::vector<Vector2>& tex_coords = std::vector<Vector2>(),
       const std::vector<uint>& indices = std::vector<uint>()) {
    Initialize(vertices, normals, tex_coords, indices);
  }

  Mesh(const Mesh&) = delete;

  ~Mesh() {
    if (ebo_ > 0) {
      glDeleteBuffers(1, &ebo_);
    }
    if (vbo_ > 0) {
      glDeleteBuffers(1, &vbo_);
    }
    if (vao_ > 0) {
      glDeleteVertexArrays(1, &vao_);
    }
  }

  uint Index() { return id_; }

  uint VAO() const {
    if (vao_ == 0) {
      LOG_ERROR("Access VAO of 0. Possibly uninitialized.");
    }
    return vao_;
  }

  bool UseIndices() const { return indices_.size() > 0; }

  uint DrawCount() const {
    if (UseIndices()) {
      return (uint)indices_.size();
    } else {
      return (uint)tmesh_.vert.size();
    }
  }

  const std::vector<Vector3>& Positions() const { return positions_; }

  const std::vector<Vector3>& Normals() const { return normals_; }

  const std::vector<Vector2>& UVs() const { return uvs_; }

  const std::vector<uint>& Indices() const { return indices_; }

  const std::vector<uint>& NormalIndices() const { return normal_indices_; }

  const std::vector<uint>& UVIndices() const { return uv_indices_; }

  const std::vector<uint>& AttachedIndices() const { return attached_indices_; }

  const std::vector<Scalar>& BindDistances() const { return bind_distances_; }

  const std::vector<Scalar>& BindStiffnesses() const { return bind_stiffnesses_; }

  const std::vector<uint>& BindedIndices() const { return binded_indices_; }

  const std::vector<uint>& FixedIndices() const { return fixed_indices_; }

  const std::vector<std::vector<Scalar>>& GeodesicDistances() const { return geodesic_distances_; }

  TailorMesh& TMesh() { return tmesh_; }

  const GLuint VBO() const { return vbo_; }

  std::vector<Vector3> ComputeNormals(const std::vector<Vector3> positions) {
    std::vector<Vector3> normals(positions.size());
    for (int i = 0; i < indices_.size(); i += 3) {
      auto idx1 = indices_[i];
      auto idx2 = indices_[i + 1];
      auto idx3 = indices_[i + 2];

      auto p1 = positions[idx1];
      auto p2 = positions[idx2];
      auto p3 = positions[idx3];

      auto normal = glm::cross(p2 - p1, p3 - p1);
      normals[idx1] += normal;
      normals[idx2] += normal;
      normals[idx3] += normal;
    }
    for (int i = 0; i < normals.size(); i++) {
      normals[i] = glm::normalize(normals[i]);
    }
    return normals;
  }

  struct BindingComp {
    bool operator()(const BindingParam& A, const BindingParam& B) const { return A.idx < B.idx; }
  };

  void ParseBoundaries();

  static int GetIndex(TailorVertex* v, TailorMesh& m);

  static int GetIndex(TailorEdge* e, TailorMesh& m);

  static int GetIndex(TailorFace* f, TailorMesh& m);

  static int GetEdgeIndex(TailorFace* f, TailorEdge* e);

  static void GetNeighbors(TailorVertex* v, std::vector<TailorVertex*>& neighbors);

  void AddAttachedVertices(std::vector<uint>& indices);

  void AddVertices(const int& EXTEND_MODE, std::vector<uint>& markers, std::vector<uint>& target);

  void AddBindedVertices(const int& EXTEND_MODE, std::vector<BindingParam>& binding_params);

  void ApplyBindings();

  void AddFixedVertices(const std::vector<uint>& indices);

  void BuildGeodesic();

  thrust::host_vector<uint> FaceEdgeIndices();

 private:
  std::vector<Vector3> positions_;
  std::vector<Vector3> normals_;
  std::vector<Vector2> tex_coords_;
  std::vector<uint> indices_;

  std::vector<Vector2> uvs_;
  std::vector<uint> uv_indices_;
  std::vector<uint> normal_indices_;

  TailorMesh tmesh_;
  std::vector<std::vector<TailorVertex*>> boundaries_;

  // attachments
  std::vector<uint> attached_indices_;
  std::vector<uint> fixed_indices_;
  std::set<BindingParam, BindingComp> extended_bindings_;
  std::vector<uint> binded_indices_;
  std::vector<Scalar> bind_stiffnesses_;
  std::vector<Scalar> bind_distances_;

  std::vector<uint> geodesic_previous_;
  std::vector<std::vector<Scalar>> geodesic_distances_;

  uint id_;
  GLuint vao_ = 0;  // Vertex Attribute Object
  GLuint ebo_ = 0;  // Element Buffer Object
  GLuint vbo_;      // Vertex Buffer Object

  void Initialize(const std::vector<Vector3>& vertices, const std::vector<Vector3>& normals,
                  const std::vector<Vector2>& tex_coords, const std::vector<uint>& indices) {
    positions_ = vertices;
    normals_ = normals;
    tex_coords_ = tex_coords;
    indices_ = indices;

    LOG_INFO("Construct render proxy");
    size_t n_normals = normals.size();
    size_t n_uvs = tex_coords_.size();
    std::vector<RenderableVertex> renderable_vertices(positions_.size());
    for (size_t i = 0; i < positions_.size(); i++) {
      RenderableVertex& rv = renderable_vertices[i];
      rv.x = positions_[i];
      if (i < n_normals)
        rv.n = normals_[i];
      if (i < n_uvs)
        rv.uv = tex_coords_[i];
    }
    LOG_INFO("Bind Vertex Attribute Object");
    glGenVertexArrays(1, &vao_);
    glBindVertexArray(vao_);

    LOG_INFO("Copy vertices array in a buffer for OpenGL to use");
    glGenBuffers(1, &vbo_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, renderable_vertices.size() * sizeof(RenderableVertex),
                 renderable_vertices.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(RenderableVertex),
                          reinterpret_cast<void*>(offsetof(RenderableVertex, x)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(RenderableVertex),
                          reinterpret_cast<void*>(offsetof(RenderableVertex, n)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(RenderableVertex),
                          reinterpret_cast<void*>(offsetof(RenderableVertex, uv)));

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    LOG_INFO("Copy index array in a element buffer for OpenGL to use");
    if (UseIndices()) {
      glGenBuffers(1, &ebo_);
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_);
      glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(uint), indices.data(),
                   GL_STATIC_DRAW);
    }
    glBindVertexArray(0);
    LOG_INFO("Mesh initialized");
  }
};
}  // namespace XRTailor