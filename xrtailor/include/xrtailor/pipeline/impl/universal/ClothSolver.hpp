#pragma once

#include <xrtailor/pipeline/base/ClothSolverBase.hpp>

namespace XRTailor {

class ClothSolverUniversal : public ClothSolverBase {
 public:
  ClothSolverUniversal();

  void Start();

  void Update();

  void FixedUpdate();

  void OnDestroy();

  bool UpdateObstacleAnimationFrame() override;

 public:
  void SaveCache() override;

  int AddCloth(std::shared_ptr<Mesh> mesh, Mat4 model_matrix, Scalar particle_diameter);

  int AddObstacle(std::shared_ptr<Mesh> mesh, Mat4 model_matrix, uint obstacle_id);
};
}  // namespace XRTailor