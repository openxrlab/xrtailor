#include <xrtailor/physics/DebugDrawingHelper.hpp>
#include <glm/glm.hpp>

namespace XRTailor {
namespace Debug {
void DrawEEProximities(std::shared_ptr<LineRenderer> line_renderer,
                       std::shared_ptr<PhysicsMesh> physics_mesh,
                       const thrust::host_vector<QuadInd>& quad_inds) {
  line_renderer->Reset();

  std::vector<glm::vec3> line_points;

  thrust::host_vector<Vector3> h_positions = physics_mesh->HostPositions();

  size_t n_quads = quad_inds.size();

  for (size_t i = 0; i < n_quads; i++) {
    auto quad_ind = quad_inds[i];
    auto inds = quad_ind.ids;
    line_points.push_back(h_positions[inds[0]]);
    line_points.push_back(h_positions[inds[1]]);
    line_points.push_back(h_positions[inds[2]]);
    line_points.push_back(h_positions[inds[3]]);
  }

  line_renderer->SetVertices(line_points);
}

void DrawICMIntersections(
    std::shared_ptr<LineRenderer> line_renderer, std::shared_ptr<LineRenderer> line_renderer_B,
    std::shared_ptr<PointRenderer> point_renderer, std::shared_ptr<ArrowRenderer> arrow_renderer,
    std::shared_ptr<PhysicsMesh> physics_mesh,
    const thrust::host_vector<Untangling::IntersectionWithGradient>& intersections) {
  line_renderer->Reset();
  line_renderer_B->Reset();
  point_renderer->Reset();
  arrow_renderer->Reset();

  std::vector<glm::vec3> edges;
  std::vector<glm::vec3> faces;
  std::vector<glm::vec3> projections;
  std::vector<Arrow> arrows;

  thrust::host_vector<Vector3> h_positions = physics_mesh->HostPositions();
  thrust::host_vector<unsigned int> h_indices = physics_mesh->HostIndices();

  size_t n_intersections = intersections.size();

  for (size_t i = 0; i < n_intersections; i++) {
    auto& intersection = intersections[i];
    auto ev0_idx = intersection.ev0_idx;
    auto ev1_idx = intersection.ev1_idx;
    auto p = intersection.p;

    edges.push_back(static_cast<glm::vec3>(intersection.ev0));
    edges.push_back(static_cast<glm::vec3>(intersection.ev1));

    projections.push_back(static_cast<glm::vec3>(p));

    faces.push_back(static_cast<glm::vec3>(intersection.v0));
    faces.push_back(static_cast<glm::vec3>(intersection.v1));
    faces.push_back(static_cast<glm::vec3>(intersection.v0));
    faces.push_back(static_cast<glm::vec3>(intersection.v2));
    faces.push_back(static_cast<glm::vec3>(intersection.v1));
    faces.push_back(static_cast<glm::vec3>(intersection.v2));

    glm::vec3 start = intersection.p;
    glm::vec3 end = intersection.p + intersection.G;
    arrows.emplace_back(start, end);
  }

  line_renderer->SetVertices(edges);
  line_renderer_B->SetVertices(faces);
  point_renderer->SetVertices(projections);
  arrow_renderer->SetArrows(arrows);
}

void DrawGIAContours(std::shared_ptr<PointRenderer> point_renderer_A,
                     std::shared_ptr<PointRenderer> point_renderer_B,
                     std::shared_ptr<PointRenderer> point_renderer_C,
                     std::shared_ptr<PointRenderer> point_renderer_D,
                     std::shared_ptr<PointRenderer> point_renderer_E,
                     const thrust::host_vector<Untangling::EFIntersection>& intersections,
                     const thrust::host_vector<Untangling::IntersectionState>& intersection_states,
                     std::shared_ptr<PhysicsMesh> physics_mesh) {
  point_renderer_A->Reset();
  point_renderer_B->Reset();
  point_renderer_C->Reset();
  point_renderer_D->Reset();
  point_renderer_E->Reset();

  thrust::host_vector<Vector3> h_positions = physics_mesh->HostPositions();
  thrust::host_vector<unsigned int> h_indices = physics_mesh->HostIndices();

  size_t n_intersections = intersections.size();

  std::vector<glm::vec3> projections_a;
  std::vector<glm::vec3> projections_b;
  std::vector<glm::vec3> projections_c;
  std::vector<glm::vec3> projections_d;
  std::vector<glm::vec3> projections_e;

  for (size_t i = 0; i < n_intersections; i++) {
    auto& intersection = intersections[i];
    auto state = intersection_states[i];
    auto p = intersection.p;
    auto color = state.color;
    if (color == 0) {
      projections_a.push_back(p);
    } else if (color == 1) {
      projections_b.push_back(p);
    } else if (color == 2) {
      projections_c.push_back(p);
    } else if (color == 3) {
      projections_d.push_back(p);
    } else if (color == 4) {
      projections_e.push_back(p);
    } else if (color == -1) {
      LOG_ERROR("Intersection {} is uncolored", i);
    } else {
      LOG_WARN("Color {} excceeds the drawing range", color);
      projections_a.push_back(p);
    }
  }
  point_renderer_A->SetVertices(projections_a);
  point_renderer_B->SetVertices(projections_b);
  point_renderer_C->SetVertices(projections_c);
  point_renderer_D->SetVertices(projections_d);
  point_renderer_E->SetVertices(projections_e);
}

}  // namespace Debug
}  // namespace XRTailor