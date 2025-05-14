#include <xrtailor/runtime/rendering/AABBRenderer.hpp>

namespace XRTailor {
AABBRenderer::AABBRenderer() {
  SET_COMPONENT_NAME;
}

void AABBRenderer::SetDefaultAABBConfig() {
  vertices.clear();
  AddAABB(AABB(glm::vec3(1.0, 0.0, 0.0), glm::vec3(2.0, 1.0, 1.0)));
  RebindBuffer();

  //ShowDebugInfo();
}

void AABBRenderer::SetCustomedAABBConfig() {
  vertices.clear();
  AddAABB(AABB(glm::vec3(1.0, 0.0, 0.0), glm::vec3(2.0, 1.0, 1.0)));
  AddAABB(AABB(glm::vec3(-1.0, 0.0, 0.0), glm::vec3(0.0, 1.0, 1.0)));
  RebindBuffer();

  //ShowDebugInfo();
}

void AABBRenderer::ShowDebugInfo() {
  printf("buffer has %d float attributes, %d vertices, %d edges\n", vertices.size(),
         vertices.size() / 3, vertices.size() / 6);
  for (size_t i = 0; i < vertices.size(); i++) {
    printf("%f ", vertices[i]);
  }
}

void AABBRenderer::AddAABB(const AABB& aabb) {
  glm::vec3 lower = aabb.lower;
  glm::vec3 upper = aabb.upper;
  glm::vec3 extent = aabb.extent;
  float x = extent.x;
  float y = extent.y;
  float z = extent.z;

  /*
        * Coordinates:
        *
        *                | +y
        *                |
        *
        *                v4--------v5
        *               /|        /|
        *              / |       / |
        *            v7--+------v6 |        +x
        *            |   v0-----|--v1  ------
        *            |  /       | /
        *            | /        |/
        *            v3--------v2
        *
        *           /
        *          / +z
        *
        *   Edges:
        *       (v0, v1) (v0, v3) (v0, v4)
        *       (v1, v2) (v1, v5)
        *       (v2, v3) (v2, v6)
        *       (v3, v7)
        *       (v4, v5) (v4, v7)
        *       (v5, v6)
        *       (v6, v7)
        *
        */

  glm::vec3 v0(lower.x, lower.y, lower.z);
  glm::vec3 v1(lower.x + x, lower.y, lower.z);
  glm::vec3 v2(lower.x + x, lower.y, lower.z + z);
  glm::vec3 v3(lower.x, lower.y, lower.z + z);
  glm::vec3 v4(lower.x, lower.y + y, lower.z);
  glm::vec3 v5(lower.x + x, lower.y + y, lower.z);
  glm::vec3 v6(lower.x + x, lower.y + y, lower.z + z);
  glm::vec3 v7(lower.x, lower.y + y, lower.z + z);

  std::vector<glm::vec3> aabb_verts = {v0, v1, v0, v3, v0, v4, v1, v2, v1, v5, v2, v3,
                                       v2, v6, v3, v7, v4, v5, v4, v7, v5, v6, v6, v7};

  AddVertices(aabb_verts);
}

void AABBRenderer::SetAABBs(std::vector<AABB> aabbs) {
  vertices.clear();
  for (auto aabb_iter = aabbs.begin(); aabb_iter != aabbs.end(); aabb_iter++) {
    AddAABB(*aabb_iter);
  }
  RebindBuffer();
}

AABBRenderer::~AABBRenderer() = default;
}  // namespace XRTailor