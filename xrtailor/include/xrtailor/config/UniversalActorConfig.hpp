#pragma once

#include <string>
#include <vector>

#include <xrtailor/core/Scalar.hpp>

namespace XRTailor {

enum class UniversalActorType { CLOTH, OBSTACLE };

class UniversalActorConfig {
 public:
  UniversalActorConfig();

  UniversalActorConfig(UniversalActorType _type, std::string _name, std::string _mesh_path,
                       std::vector<uint> _fixed_nodes = std::vector<uint>(),
                       Vector3 _position = Vector3(0.0f), Vector3 _scale = Vector3(1.0f),
                       Vector3 _rotation = Vector3(0.0f));

  ~UniversalActorConfig();

  UniversalActorType type;
  std::string name;
  std::string mesh_path;
  Vector3 position;
  Vector3 scale;
  Vector3 rotation;
  std::vector<uint> fixed_nodes;
};
}  // namespace XRTailor