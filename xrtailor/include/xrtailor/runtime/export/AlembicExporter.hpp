#pragma once

#include <Alembic/AbcGeom/All.h>
#include <Alembic/AbcCoreOgawa/All.h>

#include <xrtailor/core/Scalar.hpp>
#include <xrtailor/runtime/scene/Component.hpp>

namespace XRTailor {
using namespace Alembic::AbcGeom;  // Contains Abc, AbcCoreAbstract

class Mesh;

class AlembicExporter : public Component {
 public:
  AlembicExporter() { name = __func__; }

  void ExportAlembic(const std::string& name, std::shared_ptr<Mesh> _mesh);

  void ExportAlembic(const std::string& name, std::shared_ptr<Mesh> _mesh,
                     std::shared_ptr<std::vector<Vector3*>> raw_verts_cache,
                     std::shared_ptr<std::vector<Vector3*>> raw_normals_cache, uint index_offset,
                     uint frame_rate);
};
}  // namespace XRTailor