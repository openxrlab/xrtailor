#pragma once

#include <xrtailor/runtime/scene/Scene.hpp>
#include <xrtailor/config/SimulationConfigParser.hpp>
#include <xrtailor/config/ClothConfigParser.hpp>
#include <xrtailor/pipeline/impl/gltf/ClothObject.hpp>
#include <xrtailor/pipeline/impl/gltf/ObstacleObject.hpp>

namespace XRTailor {

class SceneGLTF : public Scene {
 public:
  SceneGLTF(std::shared_ptr<SimulationConfigParser> simulation_config_parser);

  ~SceneGLTF();

  void PopulateActors(GameInstance* game);

  std::shared_ptr<Actor> SpawnCloth(GameInstance* game, std::string character_id,
                                      std::shared_ptr<ClothConfigParser> config_parser);

  std::shared_ptr<Actor> SpawnObstacle(GameInstance* game, const std::string& path);

  void SpawnObjectsFromSimulationConfig(GameInstance* game);
};

}  // namespace XRTailor