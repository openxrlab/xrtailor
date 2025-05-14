#include <xrtailor/runtime/mesh/MeshIO.hpp>
#include <set>
#include <map>
#include <sstream>

#include <xrtailor/utils/Logger.hpp>

namespace XRTailor {
Index ParseFaceIndex(const std::string& token) {
  std::stringstream in(token);
  std::string index_string;
  int indices[3] = {-1, -1, -1};

  int i = 0;
  while (getline(in, index_string, '/')) {
    if (index_string != "\\") {
      std::stringstream ss(index_string);
      ss >> indices[i++];
    }
  }

  // decrement since indices in OBJ files are 1-based
  return Index(indices[0] - 1, indices[1] - 1, indices[2] - 1);
}

MeshData MeshIO::Read(std::ifstream& in)
{
  MeshData data;

  // parse obj format
  std::string line;
  while (getline(in, line)) {
    std::stringstream ss(line);
    std::string token;

    ss >> token;

    if (token == "v") {
      Scalar x, y, z;
      ss >> x >> y >> z;

      data.positions.emplace_back(x, y, z);

    } else if (token == "vt") {
      Scalar u, v;
      ss >> u >> v;

      data.uvs.emplace_back(u, v, 0);

    } else if (token == "vn") {
      Scalar x, y, z;
      ss >> x >> y >> z;

      data.normals.emplace_back(x, y, z);

    } else if (token == "f") {
      std::vector<Index> face_indices;

      while (ss >> token) {
        Index index = ParseFaceIndex(token);
        if (index.position < 0) {
          getline(in, line);
          size_t i = line.find_first_not_of("\t\n\v\f\r ");
          index = ParseFaceIndex(line.substr(i));
        }

        face_indices.push_back(index);
      }

      data.indices.push_back(face_indices);
    }
  }
  LOG_TRACE("#VERTS={}, #FACES={}, #UVS={}, #NORMALS={}", data.positions.size(),
            data.indices.size(), data.uvs.size(), data.normals.size());
  return data;
}
}  // namespace XRTailor