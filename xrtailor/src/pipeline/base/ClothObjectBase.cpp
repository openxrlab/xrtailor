#include <xrtailor/core/Precompiled.h>
#include <xrtailor/runtime/export/AlembicExporter.hpp>
#include <xrtailor/runtime/export/ObjSequenceExporter.hpp>
#include <xrtailor/runtime/scene/Actor.hpp>
#include <xrtailor/runtime/rendering/MeshRenderer.hpp>
#include <xrtailor/pipeline/base/ObstacleObjectBase.hpp>
#include <xrtailor/pipeline/base/ClothObjectBase.hpp>

namespace XRTailor {

void ClothObjectBase::CheckStatus() {
  if (!Global::sim_params.record_cloth || Global::sim_params.cloth_exported[cloth_id_])
    return;

  if (solver_->simulation_finished) {
    const int& export_format = Global::sim_config.animation.export_format;
    if (export_format == EXPORT_FORMAT::ALEMBIC) {
      ExportAlembic();
    } else if (export_format == EXPORT_FORMAT::OBJ_SEQUENCE) {
      ExportObjSequence();
    } else {
      LOG_WARN("Invalid export format, failed to export garment");
    }
    Global::sim_params.cloth_exported[cloth_id_] = true;
  }
}

void ClothObjectBase::ExportAlembic() {
  auto alembic_exporter = actor->GetComponent<AlembicExporter>();
  auto mesh = actor->GetComponent<MeshRenderer>()->mesh();
  filesystem::path export_path(Global::sim_config.animation.export_directory);
  export_path.append(solver_->GetIdentifier() + "+" + this->actor->name + ".abc");
  checkCudaErrors(cudaDeviceSynchronize());
  LOG_DEBUG("Export abc: {}, {}, {}", export_path.string(), index_offset_,
            Global::sim_config.animation.target_frame_rate);
  alembic_exporter->ExportAlembic(export_path.string(), mesh, solver_->h_cloth_positions_cache,
                                  solver_->h_cloth_normals_cache, index_offset_,
                                  Global::sim_config.animation.target_frame_rate);
  checkCudaErrors(cudaDeviceSynchronize());
  LOG_DEBUG("Export done.");
}

void ClothObjectBase::ExportObjSequence() {
  auto obj_sequence_exporter = actor->GetComponent<ObjSequenceExporter>();
  auto mesh = actor->GetComponent<MeshRenderer>()->mesh();
  filesystem::path export_path(Global::sim_config.animation.export_directory);
  export_path.append(solver_->GetIdentifier() + "_cloth");
  obj_sequence_exporter->ExportObjSequence(export_path, mesh, solver_->h_cloth_positions_cache,
                                           solver_->h_cloth_normals_cache, index_offset_);
}

void ClothObjectBase::ExportLastFrame() {
  auto obj_sequence_exporter = actor->GetComponent<ObjSequenceExporter>();
  auto mesh = actor->GetComponent<MeshRenderer>()->mesh();
  filesystem::path export_path(Global::sim_config.animation.export_directory);
  std::string export_filename = solver_->GetIdentifier() + "_cloth.obj";
  std::string prefix = "cloth_";
  auto n_frames = solver_->h_cloth_positions_cache->size();
  obj_sequence_exporter->ExportFrame(export_path, export_filename, mesh, n_frames - 1, prefix,
                                     solver_->h_cloth_positions_cache,
                                     solver_->h_cloth_normals_cache, index_offset_);
}

void ClothObjectBase::ApplyTransform(std::vector<Vector3>& positions, Mat4 transform) {
  for (int i = 0; i < positions.size(); i++) {
    positions[i] = transform * Vector4(positions[i], 1);
  }
}

void ClothObjectBase::GenerateLongRangeConstraints(std::shared_ptr<Mesh> mesh) {
  if (Global::sim_params.geodesic_LRA) {
    mesh->BuildGeodesic();
    GenerateGeodesicLongRangeConstraints(mesh->Positions(), mesh->AttachedIndices(),
                                         mesh->GeodesicDistances());
  } else {
    GenerateEuclideanLongRangeConstraints(mesh->Positions(), mesh->AttachedIndices());
  }
}

void ClothObjectBase::GenerateGeodesicLongRangeConstraints(
    const std::vector<Vector3>& positions, const std::vector<uint>& attached_indices,
    std::vector<std::vector<Scalar>> geodesic_distances) {
  LOG_TRACE("Generate Geodesic LRA constraints");
  for (int slot_idx = 0; slot_idx < attached_indices.size(); slot_idx++) {
    LOG_DEBUG("slot vert id: {}, prev slots: {}", attached_indices[slot_idx],
              solver_->prev_num_attached_slots);
    for (int vert_idx = 0; vert_idx < positions.size(); vert_idx++) {
      int particle_id = attached_indices[slot_idx];
      Vector3 slotPos = positions[particle_id];
      Scalar rest_length = geodesic_distances[slot_idx][vert_idx];
      solver_->AddGeodesic(particle_id + index_offset_, vert_idx + index_offset_, rest_length);
    }
  }
}

void ClothObjectBase::GenerateEuclideanLongRangeConstraints(
    const std::vector<Vector3>& positions, const std::vector<uint>& attached_indices) {
  LOG_TRACE("Generate Euclidean LRA constraints");

  for (int slot_idx = 0; slot_idx < attached_indices.size(); slot_idx++) {
    for (int vert_idx = 0; vert_idx < positions.size(); vert_idx++) {
      int particle_id = attached_indices[slot_idx];
      Vector3 slot_pos = positions[particle_id];
      Scalar rest_distance = glm::length(slot_pos - positions[vert_idx]);
      solver_->AddEuclidean(vert_idx + index_offset_, slot_idx + solver_->prev_num_attached_slots,
                            rest_distance);
    }
  }
}

}  // namespace XRTailor