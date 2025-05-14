#pragma once

#include <xrtailor/runtime/scene/Component.hpp>
#include <xrtailor/pipeline/base/ClothSolverBase.hpp>

#include <unordered_map>
#include <numeric>
#include <vector>
#include <memory>
#include <string>

#if defined(_WIN64) || defined(WIN32) || defined(_WIN32)
#include <filesystem>
namespace filesystem = std::filesystem;
#else
#include <experimental/filesystem>
namespace filesystem = std::experimental::filesystem;
#endif

namespace XRTailor {

// 前置声明
class AlembicExporter;
class ObjSequenceExporter;
class Actor;
class MeshRenderer;
class Mesh;
class ObstacleObjectBase;

class ClothObjectBase : public Component {
 public:
  ClothObjectBase() {}

  ClothObjectBase(const uint cloth_id, const uint obstacle_id) {
    SET_COMPONENT_NAME;

    cloth_id_ = cloth_id;
    obstacle_id_ = obstacle_id;
  }

  Scalar particle_diameter() const { return particle_diameter_; }

 public:
  void Start() override {}

  void FixedUpdate() override { CheckStatus(); }

  void CheckStatus();
  void ExportAlembic();
  void ExportObjSequence();
  void ExportLastFrame();

 protected:
  ClothSolverBase* solver_;
  int index_offset_ = 0;
  uint cloth_id_ = 0;
  uint obstacle_id_ = 0;
  Scalar particle_diameter_ = 0;

  void ApplyTransform(std::vector<Vector3>& positions, Mat4 transform);
  void GenerateLongRangeConstraints(std::shared_ptr<Mesh> mesh);
  void GenerateGeodesicLongRangeConstraints(const std::vector<Vector3>& positions,
                                            const std::vector<uint>& attached_indices,
                                            std::vector<std::vector<Scalar>> geodesic_distances);
  void GenerateEuclideanLongRangeConstraints(const std::vector<Vector3>& positions,
                                             const std::vector<uint>& attached_indices);
};

}  // namespace XRTailor