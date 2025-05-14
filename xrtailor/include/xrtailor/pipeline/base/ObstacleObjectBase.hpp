#pragma once

// ------------------------------------------
// WARNING: DO NOT CHANGE THE INCLUDE ORDER !
// AlembicExporter should be included first
// ------------------------------------------
#include <xrtailor/runtime/export/AlembicExporter.hpp>
#include <xrtailor/runtime/export/ObjSequenceExporter.hpp>

#include <iostream>
#include <xrtailor/runtime/scene/Component.hpp>
#include <xrtailor/runtime/rendering/MeshRenderer.hpp>
#include <xrtailor/pipeline/base/ClothSolverBase.hpp>

namespace XRTailor {
class ObstacleObjectBase : public Component {
 public:
  ObstacleObjectBase() {}

  ObstacleObjectBase(uint obstacle_id) : obstacle_id_(obstacle_id) { SET_COMPONENT_NAME; }

  void Start() override {}

  void FixedUpdate() override {}

  void CheckStatus() {
    if (!Global::sim_params.record_obstacle || Global::sim_params.obstacle_exported[obstacle_id_])
      return;

    if (solver_ == nullptr) {
      LOG_ERROR("Solver not found");
    }

    if (solver_->simulation_finished) {
      const int& export_format = Global::sim_config.animation.export_format;
      if (export_format == EXPORT_FORMAT::ALEMBIC) {
        ExportAlembic();
      } else if (export_format == EXPORT_FORMAT::OBJ_SEQUENCE) {
        ExportObjSequence();
      } else {
        LOG_WARN("Invalid export format, failed to export body");
      }
      Global::sim_params.obstacle_exported[obstacle_id_] = true;
    }
  }

  void ExportAlembic() {
    auto alembic_exporter = actor->GetComponent<AlembicExporter>();
    auto mesh = actor->GetComponent<MeshRenderer>()->mesh();
    filesystem::path export_path(Global::sim_config.animation.export_directory);
    export_path.append(solver_->GetIdentifier() + "+" + this->actor->name + ".abc");
    alembic_exporter->ExportAlembic(export_path.string(), mesh, solver_->h_obstacle_positions_cache,
                                   solver_->h_obstacle_normals_cache, index_offset_,
                                   Global::sim_config.animation.target_frame_rate);
  }

  void ExportObjSequence() {
    auto obj_sequence_exporter = actor->GetComponent<ObjSequenceExporter>();
    auto mesh = actor->GetComponent<MeshRenderer>()->mesh();
    filesystem::path export_path(Global::sim_config.animation.export_directory);
    export_path.append(solver_->GetIdentifier() + "_obstacle");
    obj_sequence_exporter->ExportObjSequence(export_path.string(), mesh,
                                           solver_->h_obstacle_positions_cache,
                                           solver_->h_obstacle_normals_cache, index_offset_);
  }

  void ApplyTransform(std::vector<Vector3>& positions, Mat4 transform) {
    for (int i = 0; i < positions.size(); i++) {
      positions[i] = transform * Vector4(positions[i], 1);
    }
  }

 protected:
  ClothSolverBase* solver_;

  uint index_offset_ = 0;

  uint obstacle_id_ = 0;

  std::vector<uint> mask_;
};
}  // namespace XRTailor