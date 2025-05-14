#pragma once

#include <xrtailor/runtime/scene/Component.hpp>
#include <xrtailor/runtime/mesh/Mesh.hpp>
#include <xrtailor/utils/FileSystemUtils.hpp>

namespace XRTailor {
class ObjSequenceExporter : public Component {
 public:
  ObjSequenceExporter() { name = __func__; }

  void ExportObjSequence(filesystem::path exportPath, std::shared_ptr<Mesh> mesh,
                         std::shared_ptr<std::vector<Vector3*>> positions_cache,
                         std::shared_ptr<std::vector<Vector3*>> normals_cache, uint index_offset);

  void ExportObjSequence(filesystem::path exportPath, std::shared_ptr<Mesh> mesh,
                         std::shared_ptr<std::vector<Vector3*>> positions_cache, uint index_offset);

  void ExportFrame(filesystem::path export_path, std::string export_filename,
                   std::shared_ptr<Mesh> mesh, uint frame_index, std::string prefix,
                   std::shared_ptr<std::vector<Vector3*>> positions_cache,
                   std::shared_ptr<std::vector<Vector3*>> normals_cache, uint index_offset);
};
}  // namespace XRTailor