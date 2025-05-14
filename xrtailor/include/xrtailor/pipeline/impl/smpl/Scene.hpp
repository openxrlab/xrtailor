#pragma once

#include <xrtailor/runtime/scene/Scene.hpp>
#include <xrtailor/runtime/scene/WatchDog.hpp>
#include <xrtailor/config/SimulationConfigParser.hpp>
#include <xrtailor/config/ClothConfigParser.hpp>
#include <xrtailor/pipeline/impl/smpl/ClothObject.hpp>
#include <xrtailor/pipeline/impl/smpl/ObstacleObject.hpp>
#include <xrtailor/runtime/rendering/ParticleGeometryRenderer.hpp>

namespace XRTailor {
using SMPL = smplx::model_config::SMPL;
using SMPLH = smplx::model_config::SMPLH;
using SMPLX = smplx::model_config::SMPLX;
using AMASS_SMPLH_G = smplx::sequence_config::AMASS_SMPLH_G;
using AMASS_SMPLX_G = smplx::sequence_config::AMASS_SMPLX_G;

class SceneSMPL : public Scene {
 public:
  SceneSMPL(std::shared_ptr<SimulationConfigParser> simulation_config_parser);

  void PopulateActors(GameInstance* game);

  template <class ModelConfig, class SequenceConfig>
  std::shared_ptr<Actor> SpawnCloth(GameInstance* game,
                                      std::shared_ptr<ClothConfigParser> config_parser);

  template <class ModelConfig, class SequenceConfig>
  std::shared_ptr<Actor> SpawnObstacle(GameInstance* game, const std::string& path);

  template <class ModelConfig, class SequenceConfig>
  void SpawnObjectsFromSimulationConfig(GameInstance* game, std::string& sequence_label);
};

}  // namespace XRTailor