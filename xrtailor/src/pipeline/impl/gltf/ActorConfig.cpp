#include <xrtailor/pipeline/impl/gltf/ActorConfig.hpp>

namespace XRTailor {

GLTFActorConfig::GLTFActorConfig() = default;

GLTFActorConfig::GLTFActorConfig(GLTFActorType _type, std::string _name, std::string _mesh_path,
                                 std::vector<unsigned int> _fixed_nodes, glm::vec3 _position,
                                 glm::vec3 _scale, glm::vec3 _rotation)
    : type(_type),
      name(_name),
      mesh_path(_mesh_path),
      fixed_nodes(_fixed_nodes),
      position(_position),
      scale(_scale),
      rotation(_rotation) {}

GLTFActorConfig::~GLTFActorConfig() = default;

}  // namespace XRTailor