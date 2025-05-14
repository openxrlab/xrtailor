#pragma once

#include <xrtailor/runtime/rendering/MeshRenderer.hpp>

namespace XRTailor {
// Render particles using geometry shader
class ParticleGeometryRenderer : public MeshRenderer {
 public:
  ParticleGeometryRenderer()
      : MeshRenderer(nullptr, Resource::LoadMaterial("_InstancedParticle", true)),
        particle_diameter_(0.0f) {
    SET_COMPONENT_NAME;
    material_->double_sided = true;
    material_->SetVec3("material.tint", glm::vec3(0.2, 0.3, 0.6));
    material_->specular = 0.0f;
    material_->SetBool("material.useTexture", false);
  }

  void Start() override {
    LOG_TRACE("Starting {0}'s component: {1}", actor->name, name);
    mesh_ = CustomMesh();
  }

  std::shared_ptr<Mesh> CustomMesh() {
    // placeholder mesh
    std::vector<Vector3> points = {
        Vector3(0),
    };
    auto mesh = std::make_shared<Mesh>(points);
    glBindVertexArray(mesh->VAO());
    auto clothMesh = actor->GetComponent<MeshRenderer>()->mesh();
    num_particles_ = static_cast<int>(clothMesh->Positions().size());
    glBindBuffer(GL_ARRAY_BUFFER, clothMesh->VBO());

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(RenderableVertex),
                          reinterpret_cast<void*>(offsetof(RenderableVertex, x)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(RenderableVertex),
                          reinterpret_cast<void*>(offsetof(RenderableVertex, n)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(RenderableVertex),
                          reinterpret_cast<void*>(offsetof(RenderableVertex, uv)));
    glBindVertexArray(0);

    return mesh;
  }

  void DrawCall() override {
    if (!Global::game_state.draw_particles) {
      return;
    }

    material_->Use();
    material_->SetFloat("_ParticleRadius", particle_diameter_ * 0.5f);
    glBindVertexArray(mesh_->VAO());
    glDrawArrays(GL_POINTS, 0, num_particles_);
  }

  void SetParticleDiameter(float _particle_diameter) { particle_diameter_ = _particle_diameter; }

 private:
  int num_particles_;
  float particle_diameter_;
};
}  // namespace XRTailor