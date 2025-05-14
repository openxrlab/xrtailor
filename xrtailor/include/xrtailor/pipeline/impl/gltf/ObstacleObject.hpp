#pragma once

#include <xrtailor/pipeline/base/ObstacleObjectBase.hpp>
#include <xrtailor/pipeline/impl/gltf/ClothSolver.hpp>
#include <xrtailor/runtime/rag_doll/gltf/GLTFLoader.cuh>

namespace XRTailor {

class ObstacleObjectGLTF : public ObstacleObjectBase {
 public:
  ObstacleObjectGLTF(uint obstacle_id) : ObstacleObjectBase(obstacle_id) {
    SET_COMPONENT_NAME;
  }

  void Start() override {
    LOG_TRACE("Start Start obstacle object GLTF");

    auto solver_actor = Global::game->FindActor("ClothSolverGLTF");
    if (solver_actor == nullptr) {
      LOG_ERROR("Cloth solver not found");
      exit(TAILOR_EXIT::SUCCESS);
    }
    solver_ = solver_actor->GetComponent<ClothSolverGLTF>();

    ObstacleObjectBase::solver_ = solver_;

    solver_->SetGltfLoader(this->gltf_loader_);
    auto mesh = actor->GetComponent<MeshRenderer>()->mesh();
    auto positions = mesh->Positions();
    auto transform_matrix = actor->transform->matrix();
    checkCudaErrors(cudaDeviceSynchronize());
    index_offset_ = solver_->AddObstacle(mesh, transform_matrix, obstacle_id_);
    checkCudaErrors(cudaDeviceSynchronize());

    actor->transform->Reset();

    ApplyTransform(positions, transform_matrix);

    Global::sim_params.obstacle_exported[obstacle_id_] = false;
  }

  void FixedUpdate() override { CheckStatus(); }

  void SetGltfLoader(std::shared_ptr<GLTFLoader> loader) { gltf_loader_ = loader; }

 private:
  ClothSolverGLTF* solver_;
  std::shared_ptr<GLTFLoader> gltf_loader_;
};

}  // namespace XRTailor