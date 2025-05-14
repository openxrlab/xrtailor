#pragma once

#include <xrtailor/runtime/scene/Scene.hpp>
#include <xrtailor/pipeline/impl/universal/ClothObject.hpp>
#include <xrtailor/pipeline/impl/universal/ObstacleObject.hpp>
#include <xrtailor/config/UniversalActorConfig.hpp>

namespace XRTailor {

class SceneUniversal : public Scene {
 public:
  SceneUniversal(std::shared_ptr<SimulationConfigParser> simulation_config_parser);

  ~SceneUniversal();

  void PopulateActors(GameInstance* game);

  std::shared_ptr<Actor> SpawnCloth(GameInstance* game, const UniversalActorConfig& config);

  std::shared_ptr<Actor> SpawnObstacle(GameInstance* game, const UniversalActorConfig& config);

  void AddActor(const UniversalActorConfig& config);

  void AddActors(const std::vector<UniversalActorConfig>& configs);

  void SpawnObjects(GameInstance* game);

  void SpawnObjectsFromSimulationConfig(GameInstance* game);

 private:
  std::vector<UniversalActorConfig> configs_;
};

}  // namespace XRTailor