#pragma once

#include <vector>

#include <xrtailor/core/Scalar.hpp>

namespace XRTailor {
class Index {
 public:
  Index() {}

  Index(int v, int vt, int vn) : position(v), uv(vt), normal(vn) {}

  bool operator<(const Index& i) const {
    if (position < i.position)
      return true;
    if (position > i.position)
      return false;
    if (uv < i.uv)
      return true;
    if (uv > i.uv)
      return false;
    if (normal < i.normal)
      return true;
    if (normal > i.normal)
      return false;

    return false;
  }

  bool operator==(const Index&) const = delete;
  bool operator!=(const Index&) const = delete;
  bool operator>(const Index&) const = delete;

  int position;
  int uv;
  int normal;
};

class MeshData {
 public:
  std::vector<Vector3> positions;
  std::vector<Vector3> uvs;
  std::vector<Vector3> normals;
  std::vector<std::vector<Index>> indices;
};

class MeshIO {
 public:
  // reads data from obj file
  static MeshData Read(std::ifstream& in);
};

}  // namespace XRTailor
