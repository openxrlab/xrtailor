#pragma once

#include <xrtailor/pipeline/base/ClothSolverBase.hpp>
#include <xrtailor/runtime/rag_doll/smpl/sequence.hpp>
#include <xrtailor/runtime/rag_doll/smpl/smplx.hpp>
#include <xrtailor/runtime/rag_doll/smpl/util.hpp>

namespace XRTailor {

template <class ModelConfig, class SequenceConfig>
class ClothSolverSMPL : public ClothSolverBase {
 public:
  ClothSolverSMPL();

  void Start();

  void Update();

  void FixedUpdate();

  void OnDestroy();

  bool UpdateObstacleAnimationFrame() override;

 public:
  void SaveCache() override;

  void InstantSkinning();

  int AddObstacle(std::shared_ptr<Mesh> mesh, Mat4 model_matrix, uint obstacle_id,
                  std::shared_ptr<smplx::Body<ModelConfig>> body,
                  std::shared_ptr<smplx::Sequence<SequenceConfig>> sequence,
                  const std::vector<uint>& masked_indices);

  int AddCloth(std::shared_ptr<Mesh> mesh, Mat4 model_matrix, Scalar particle_diameter);

 public:
  std::shared_ptr<smplx::Body<ModelConfig>> body_;
  std::shared_ptr<smplx::Sequence<SequenceConfig>> sequence_;

 private:
  Scalar amass_x_rotation_;
  bool pose_blendshape_ = Global::sim_config.smpl.enable_pose_blendshape;
};
}  // namespace XRTailor