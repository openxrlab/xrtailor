#pragma once

#include <xrtailor/pipeline/base/ClothSolverBase.hpp>
#include <xrtailor/runtime/rag_doll/gltf/GLTFLoader.cuh>

namespace XRTailor {

class ClothSolverGLTF : public ClothSolverBase {
 public:
  ClothSolverGLTF();

  void Start();

  void Update();

  void FixedUpdate();

  void OnDestroy();

  bool UpdateObstacleAnimationFrame() override;

 public:
  void SaveCache() override;

  void SetGltfLoader(std::shared_ptr<GLTFLoader> loader);

  int AddCloth(std::shared_ptr<Mesh> mesh, Mat4 model_matrix, Scalar particle_diameter);

  int AddObstacle(std::shared_ptr<Mesh> mesh, Mat4 model_matrix, uint bodyID);

 private:
  std::shared_ptr<GLTFLoader> gltf_loader_;
};
}  // namespace XRTailor