#pragma once

#include <xrtailor/pipeline/base/ObstacleObjectBase.hpp>
#include <xrtailor/pipeline/impl/smpl/ClothSolver.hpp>
#include <xrtailor/runtime/rag_doll/smpl/sequence.hpp>
#include <xrtailor/runtime/rag_doll/smpl/smplx.hpp>
#include <xrtailor/runtime/rag_doll/smpl/util.hpp>

namespace XRTailor {
template <class ModelConfig, class SequenceConfig>
class ObstacleObjectSMPL : public ObstacleObjectBase {
 public:
  ObstacleObjectSMPL(uint obstacle_id) : ObstacleObjectBase(obstacle_id) {
    SET_COMPONENT_NAME;
  }

  void Start() {
    LOG_TRACE("Start obstacle object");

    auto solver_actor = Global::game->FindActor("ClothSolver");
    if (solver_actor == nullptr) {
      LOG_ERROR("Cloth solver not found");
      exit(TAILOR_EXIT::SUCCESS);
    }
    solver_ = solver_actor->GetComponent<ClothSolverSMPL<ModelConfig, SequenceConfig>>();
    ObstacleObjectBase::solver_ = solver_;

    auto mesh = actor->GetComponent<MeshRenderer>()->mesh();
    auto positions = mesh->Positions();
    auto transform_matrix = actor->transform->matrix();

    smplx::Gender gender = sequence_->gender;
    model_ = std::make_shared<smplx::Model<ModelConfig>>(gender);
    body_ = std::make_shared<smplx::Body<ModelConfig>>(*model_);

    std::vector<uint> masked_indices = ComputeMaskedIndices();

    index_offset_ =
        solver_->AddObstacle(mesh, transform_matrix, obstacle_id_, body_, sequence_, masked_indices);

    actor->transform->Reset();

    ApplyTransform(positions, transform_matrix);

    Global::sim_params.obstacle_exported[obstacle_id_] = false;
  }

  std::string GetGender() {
    std::string gender;
    switch (sequence_->gender) {
      case smplx::Gender::female:
        gender = "female";
        break;
      case smplx::Gender::male:
        gender = "male";
        break;
      case smplx::Gender::neutral:
        gender = "neutral";
        break;
      default:
        gender = "unknown";
    }

    return gender;
  }

  void FixedUpdate() { CheckStatus(); }

  void SetSequence(std::shared_ptr<smplx::Sequence<SequenceConfig>> _sequence) {
    sequence_ = _sequence;
  }

  std::shared_ptr<smplx::Sequence<SequenceConfig>> GetSequence() { return sequence_; }

  void SetMask(const std::vector<uint>& mask) { mask_ = mask; }

  std::vector<uint> ComputeMaskedIndices() {
    std::vector<uint> masked_indices;
    auto mesh = actor->GetComponent<MeshRenderer>()->mesh();
    auto indices = mesh->Indices();
    uint num_primitives = indices.size() / 3;
    for (int i = 0; i < num_primitives; i++) {
      uint idx1 = indices[i * 3u];
      uint idx2 = indices[i * 3u + 1u];
      uint idx3 = indices[i * 3u + 2u];
      if (mask_[idx1] & mask_[idx2] & mask_[idx3]) {
        masked_indices.push_back(idx1);
        masked_indices.push_back(idx2);
        masked_indices.push_back(idx3);
      }
    }

    return masked_indices;
  }

 private:
  ClothSolverSMPL<ModelConfig, SequenceConfig>* solver_;
  std::shared_ptr<smplx::Model<ModelConfig>> model_;
  std::shared_ptr<smplx::Body<ModelConfig>> body_;
  std::shared_ptr<smplx::Sequence<SequenceConfig>> sequence_;
};
}  // namespace XRTailor