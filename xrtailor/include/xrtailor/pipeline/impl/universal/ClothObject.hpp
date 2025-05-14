#pragma once
#include <xrtailor/pipeline/base/ClothObjectBase.hpp>
#include <xrtailor/pipeline/impl/universal/ClothSolver.hpp>
#include <xrtailor/pipeline/impl/universal/ObstacleObject.hpp>
#include <xrtailor/runtime/rendering/ParticleGeometryRenderer.hpp>

namespace XRTailor {
class ClothObjectUniversal : public ClothObjectBase {
 public:
  ClothObjectUniversal(const uint cloth_id, const uint obstacle_id)
      : ClothObjectBase(cloth_id, obstacle_id) {
    SET_COMPONENT_NAME;
  }

 public:
  void Start() override {
    ClothObjectBase::Start();

    LOG_TRACE("Starting {0}'s component: {1}", actor->name, name);
    LOG_DEBUG("Garment ID: {}, Obstacle ID: {}", cloth_id_, obstacle_id_);

    auto solver_actor = Global::game->FindActor("ClothSolver");
    if (solver_actor == nullptr) {
      LOG_ERROR("Cloth solver not found");
      exit(TAILOR_EXIT::SUCCESS);
    }
    solver_ = solver_actor->GetComponent<ClothSolverUniversal>();
    ClothObjectBase::solver_ = solver_;

    std::shared_ptr<Mesh> mesh = actor->GetComponent<MeshRenderer>()->mesh();
    auto transform_matrix = actor->transform->matrix();
    auto positions = mesh->Positions();
    auto indices = mesh->Indices();
    auto edges = mesh->TMesh().edge;

    particle_diameter_ = glm::length(positions[indices[0]] - positions[indices[1]]) *
                         Global::sim_config.swift.collision.self_contact.particle_diameter;

    index_offset_ = solver_->AddCloth(mesh, transform_matrix, particle_diameter_);

    actor->transform->Reset();

    ApplyTransform(positions, transform_matrix);

    GenerateLongRangeConstraints(mesh);

    Global::sim_params.cloth_exported[cloth_id_] = false;

    ParticleGeometryRenderer* p_renderer = actor->GetComponent<ParticleGeometryRenderer>();
    if (p_renderer != nullptr) {
      p_renderer->SetParticleDiameter(static_cast<float>(particle_diameter_));
    }
  }

  void FixedUpdate() override { ClothObjectBase::FixedUpdate(); }

  ClothSolverUniversal* solver_;
};
}  // namespace XRTailor