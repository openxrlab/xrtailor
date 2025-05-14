#include <xrtailor/runtime/export/AlembicExporter.hpp>

#include <vector>
#include <glm/glm.hpp>

#include <xrtailor/runtime/mesh/Mesh.hpp>
#include <xrtailor/utils/Logger.hpp>

namespace XRTailor {
namespace Abc = Alembic::Abc;

void AlembicExporter::ExportAlembic(const std::string& name,
                                    std::shared_ptr<Mesh> _mesh) {

  std::vector<Abc::float32_t> verts;
  std::vector<Abc::int32_t> indices;
  std::vector<Abc::float32_t> normals;

  const std::vector<Vector3>& raw_verts = _mesh->Positions();
  const std::vector<uint>& raw_indices = _mesh->Indices();
  const std::vector<Vector3>& raw_normals = _mesh->Normals();

  size_t num_verts = raw_verts.size();
  size_t num_indices = raw_indices.size();
  size_t num_faces = num_indices / 3;

  size_t num_normals = raw_normals.size();

  size_t num_counts = num_faces;
  // "Face Counts" - number of vertices in each face.
  std::vector<Abc::int32_t> counts(num_counts, 3);

  for (auto v = raw_verts.begin(); v != raw_verts.end(); v++) {
    verts.emplace_back(static_cast<Abc::float32_t>(v->x));
    verts.emplace_back(static_cast<Abc::float32_t>(v->y));
    verts.emplace_back(static_cast<Abc::float32_t>(v->z));
  }

  for (auto idx = raw_indices.begin(); idx != raw_indices.end(); idx++) {
    indices.emplace_back(static_cast<Abc::int32_t>(*idx));
  }

  for (auto n = raw_normals.begin(); n != raw_normals.end(); n++) {
    normals.emplace_back(static_cast<Abc::float32_t>(n->x));
    normals.emplace_back(static_cast<Abc::float32_t>(n->y));
    normals.emplace_back(static_cast<Abc::float32_t>(n->z));
  }

  //TODO : add UV
  //size_t num_uvs;
  //vector<Abc::float32_t> uvs;

  // Create an OArchive.
  // Like std::iostreams, we have a completely separate-but-parallel class
  // hierarchy for output and for input (OArchive, IArchive, and so on). This
  // maintains the important abstraction that Alembic is for storage,
  // representation, and archival. (as opposed to being a dynamic scene
  // manipulation framework).
  OArchive archive(

      // The hard link to the implementation.
      Alembic::AbcCoreOgawa::WriteArchive(),

      // The file name.
      // Because we're an OArchive, this is creating (or clobbering)
      // the archive with this filename.
      name);

  // Create a PolyMesh class.
  OPolyMesh meshy_obj(OObject(archive, kTop), "meshy");
  OPolyMeshSchema& mesh = meshy_obj.getSchema();

  // UVs and Normals use GeomParams, which can be written or read
  // as indexed or not, as you'd like.

  // indexed normals
  ON3fGeomParam::Sample nsamp(
      N3fArraySample(reinterpret_cast<const N3f*>(normals.data()), num_normals), kFacevaryingScope);

  // Set a mesh sample.
  // We're creating the sample inline here,
  // but we could create a static sample and leave it around,
  // only modifying the parts that have changed.
  OPolyMeshSchema::Sample mesh_samp(
      V3fArraySample(reinterpret_cast<const V3f*>(verts.data()), num_verts),
      Int32ArraySample(indices.data(), num_indices), Int32ArraySample(counts.data(), num_counts),
      OV2fGeomParam::Sample(), nsamp);

  mesh.set(mesh_samp);

  // Alembic objects close themselves automatically when they go out
  // of scope. So - we don't have to do anything to finish
  // them off!
  LOG_INFO("Writing: {}", archive.getName());
}

void ApplyAlembicConvention(const std::vector<uint>& vert_indices,
                            const std::vector<uint>& uv_indices,
                            const std::vector<uint>& normal_indices,
                            std::vector<uint>& vert_indices_flipped,
                            std::vector<uint>& uv_indices_flipped,
                            std::vector<uint>& normal_indices_flipped) {
  /*
            There exists a polygon winding order gap between OpenGL and Alembic. For arbitrary face 'f' in mesh:

                   o  V1(#VERT1/#UV1/#NORMAL1)

                           V3(#VERT3/#UV3/#NORMAL3)
             o            o
               V2(#VERT2/#UV2/#NORMAL2)
        
            OpenGL uses counter-clockwise(CCW) polygon winding order:
                f #VERT1/#UV1/#NORMAL1 #VERT2/#UV2/#NORMAL2 #VERT3/#UV3/#NORMAL3

            Whereas the Alembic uses clockwise(CW) polygon winding order:
                f #VERT1/#UV1/#NORMAL1 #VERT3/#UV3/#NORMAL3 #VERT2/#UV2/#NORMAL2

            See https://github.com/alembic/alembic/wiki/Alembic-Conventions-and-Schema-Types for more details.
  */

  uint num_faces = vert_indices.size() / 3;
  for (size_t i = 0; i < num_faces; i++) {
    uint vert_idx1 = vert_indices[i * 3];
    uint vert_idx2 = vert_indices[i * 3 + 1];
    uint vert_idx3 = vert_indices[i * 3 + 2];

    uint uv_idx1 = uv_indices[i * 3];
    uint uv_idx2 = uv_indices[i * 3 + 1];
    uint uv_idx3 = uv_indices[i * 3 + 2];

    uint normal_idx1 = normal_indices[i * 3];
    uint normal_idx2 = normal_indices[i * 3 + 1];
    uint normal_idx3 = normal_indices[i * 3 + 2];

    vert_indices_flipped.push_back(vert_idx1);
    vert_indices_flipped.push_back(vert_idx3);
    vert_indices_flipped.push_back(vert_idx2);

    uv_indices_flipped.push_back(uv_idx1);
    uv_indices_flipped.push_back(uv_idx3);
    uv_indices_flipped.push_back(uv_idx2);

    normal_indices_flipped.push_back(normal_idx1);
    normal_indices_flipped.push_back(normal_idx3);
    normal_indices_flipped.push_back(normal_idx2);
  }
}

void AlembicExporter::ExportAlembic(const std::string& name, std::shared_ptr<Mesh> _mesh,
                                    std::shared_ptr<std::vector<Vector3*>> raw_verts_cache,
                                    std::shared_ptr<std::vector<Vector3*>> raw_normals_cache,
                                    uint index_offset, uint frame_rate) {
  size_t num_verts = _mesh->Positions().size();
  size_t num_frames = raw_verts_cache->size();

  std::vector<Abc::int32_t> indices;
  const std::vector<uint>& raw_vert_indices = _mesh->Indices();
  const std::vector<uint>& raw_uv_indices = _mesh->UVIndices();
  const std::vector<uint>& raw_normal_indices = _mesh->NormalIndices();
  std::vector<uint> vert_indices, uv_indices, normal_indices;
  ApplyAlembicConvention(raw_vert_indices, raw_uv_indices, raw_normal_indices, vert_indices,
                         uv_indices, normal_indices);

  size_t num_indices = vert_indices.size();
  size_t num_faces = num_indices / 3;
  for (auto idx = vert_indices.begin(); idx != vert_indices.end(); idx++) {
    indices.emplace_back(*idx);
  }
  size_t num_counts = num_faces;
  // "Face Counts" - number of vertices in each face.
  std::vector<Abc::int32_t> counts(num_counts, 3);

  // Create an OArchive.
  // Like std::iostreams, we have a completely separate-but-parallel class
  // hierarchy for output and for input (OArchive, IArchive, and so on). This
  // maintains the important abstraction that Alembic is for storage,
  // representation, and archival. (as opposed to being a dynamic scene
  // manipulation framework).
  OArchive archive(
      // The hard link to the implementation.
      Alembic::AbcCoreOgawa::WriteArchive(),
      // The file name.
      // Because we're an OArchive, this is creating (or clobbering)
      // the archive with this filename.
      name);

  TimeSamplingPtr ts(new TimeSampling(1.0f / float(frame_rate), 0.0f));

  // Create a PolyMesh class.
  OPolyMesh meshy_obj(OObject(archive, kTop), "mesh", ts);
  OPolyMeshSchema& mesh_schema = meshy_obj.getSchema();
  // some apps can arbitrarily name their primary UVs, this function allows
  // you to do that, and must be done before the first time you set UVs
  // on the schema
  mesh_schema.setUVSourceName("TextureCoordinates");

  std::vector<std::vector<Abc::float32_t>> verts_cache;
  std::vector<std::vector<Abc::float32_t>> normals_cache;

  std::vector<Abc::float32_t> verts;
  std::vector<Abc::float32_t> uvs;

  Vector3* raw_verts = (*raw_verts_cache)[0];
  Vector3* raw_normals = (*raw_normals_cache)[0];

  for (size_t i = index_offset; i < index_offset + num_verts; i++) {
    verts.emplace_back(static_cast<Abc::float32_t>(raw_verts[i].x));
    verts.emplace_back(static_cast<Abc::float32_t>(raw_verts[i].y));
    verts.emplace_back(static_cast<Abc::float32_t>(raw_verts[i].z));
  }

  auto* out_normals = new N3f[normal_indices.size()];
  for (size_t i = 0; i < normal_indices.size(); i++) {
    glm::vec3 inv_normal = static_cast<glm::vec3>(raw_normals[index_offset + normal_indices[i]]) *
                           glm::vec3(-1, -1, -1);

    out_normals[i] = N3f(inv_normal.x, inv_normal.y, inv_normal.z);
  }
  LOG_DEBUG("name: {}, num normal indices: {}", name, normal_indices.size());

  const std::vector<Vector2>& raw_uvs = _mesh->UVs();
  for (size_t i = 0; i < indices.size(); i++) {
    uvs.emplace_back(static_cast<Abc::float32_t>(raw_uvs[uv_indices[i]].x));
    uvs.emplace_back(static_cast<Abc::float32_t>(raw_uvs[uv_indices[i]].y));
  }

  // UVs and Normals use GeomParams, which can be written or read
  // as indexed or not, as you'd like.

  // indexed normals
  ON3fGeomParam::Sample nsamp(N3fArraySample((const N3f*)out_normals, normal_indices.size() / 3),
                              kFacevaryingScope);

  // indexed uvs
  OV2fGeomParam::Sample uvsamp(
      V2fArraySample(reinterpret_cast<const V2f*>(uvs.data()), num_indices), kFacevaryingScope);

  // Set a mesh sample.
  // We're creating the sample inline here,
  // but we could create a static sample and leave it around,
  // only modifying the parts that have changed.
  OPolyMeshSchema::Sample mesh_samp(
      V3fArraySample(reinterpret_cast<const V3f*>(verts.data()), num_verts),
      Int32ArraySample(indices.data(), num_indices), Int32ArraySample(counts.data(), num_counts),
      uvsamp, nsamp);

  mesh_schema.set(mesh_samp);

  for (size_t frame = 0; frame < num_frames; frame++) {
    std::vector<Abc::float32_t> verts;
    std::vector<Abc::float32_t> normals;
    Vector3* raw_verts = (*raw_verts_cache)[frame];
    Vector3* raw_normals = (*raw_normals_cache)[frame];
    for (size_t i = index_offset; i < index_offset + num_verts; i++) {
      verts.emplace_back(static_cast<Abc::float32_t>(raw_verts[i].x));
      verts.emplace_back(static_cast<Abc::float32_t>(raw_verts[i].y));
      verts.emplace_back(static_cast<Abc::float32_t>(raw_verts[i].z));

      normals.emplace_back(static_cast<Abc::float32_t>(raw_normals[i].x));
      normals.emplace_back(static_cast<Abc::float32_t>(raw_normals[i].y));
      normals.emplace_back(static_cast<Abc::float32_t>(raw_normals[i].z));
    }

    mesh_samp.setPositions(
        P3fArraySample(reinterpret_cast<const V3f*>(&verts.front()), verts.size() / 3));
    mesh_samp.setNormals(ON3fGeomParam::Sample(
        N3fArraySample(reinterpret_cast<const N3f*>(normals.data()), num_verts),
        kFacevaryingScope));
    mesh_schema.set(mesh_samp);
  }

  // Alembic objects close themselves automatically when they go out
  // of scope. So - we don't have to do anything to finish
  // them off!
  LOG_INFO("Writing alembic: {}", archive.getName());
}
}  // namespace XRTailor