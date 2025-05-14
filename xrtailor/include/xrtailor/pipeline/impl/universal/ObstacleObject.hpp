#pragma once

#include <xrtailor/pipeline/impl/universal/ClothSolver.hpp>
#include <xrtailor/pipeline/base/ObstacleObjectBase.hpp>

namespace XRTailor {

class ObstacleObjectUniversal : public ObstacleObjectBase {
 public:
  ObstacleObjectUniversal(uint obstacle_id) : ObstacleObjectBase(obstacle_id) { SET_COMPONENT_NAME; }

  void Start() override {
    LOG_TRACE("Start Start obstacle object universal");

    auto solver_actor = Global::game->FindActor("ClothSolver");
    if (solver_actor == nullptr) {
      LOG_ERROR("Cloth solver not found");
      exit(TAILOR_EXIT::SUCCESS);
    }
    solver_ = solver_actor->GetComponent<ClothSolverUniversal>();
    ObstacleObjectBase::solver_ = solver_;
    auto mesh = actor->GetComponent<MeshRenderer>()->mesh();
    auto positions = mesh->Positions();
    auto transformMatrix = actor->transform->matrix();

    index_offset_ = solver_->AddObstacle(mesh, transformMatrix, obstacle_id_);
    solver_->SetupInternalDynamics();
    actor->transform->Reset();

    ApplyTransform(positions, transformMatrix);

    Global::sim_params.obstacle_exported[obstacle_id_] = false;
  }

  void FixedUpdate() override { CheckStatus(); }

 private:
  ClothSolverUniversal* solver_;
};

}  // namespace XRTailor