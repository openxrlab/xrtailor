#include <xrtailor/config/UniversalActorConfig.hpp>

namespace XRTailor {

UniversalActorConfig::UniversalActorConfig() = default;

UniversalActorConfig::UniversalActorConfig(UniversalActorType _type, std::string _name,
                                           std::string _mesh_path,
                                           std::vector<unsigned int> _fixed_nodes,
                                           glm::vec3 _position, glm::vec3 _scale,
                                           glm::vec3 _rotation)
    : type(_type),
      name(_name),
      mesh_path(_mesh_path),
      fixed_nodes(_fixed_nodes),
      position(_position),
      scale(_scale),
      rotation(_rotation) {}

UniversalActorConfig::~UniversalActorConfig() = default;

}  // namespace XRTailor