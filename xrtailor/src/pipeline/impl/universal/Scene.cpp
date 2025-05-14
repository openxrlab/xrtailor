#include <xrtailor/pipeline/impl/universal/Scene.hpp>
#include <xrtailor/runtime/rendering/ParticleGeometryRenderer.hpp>

namespace XRTailor {

SceneUniversal::SceneUniversal(std::shared_ptr<SimulationConfigParser> simulation_config_parser) {
  simulation_config_parser_ = simulation_config_parser;
  name = "XRTailor / Universal";
}

SceneUniversal::~SceneUniversal() = default;

void SceneUniversal::PopulateActors(GameInstance* game) {
  LOG_TRACE("Populate actors");

  SpawnCameraAndLight(game);
  LOG_TRACE("Spawn camera and light done");
  SpawnInfinitePlane(game);
  SpawnDebugger(game);

  SpawnObjectsFromSimulationConfig(game);

  // create solver
  auto solver_actor = game->CreateActor("ClothSolver");
  auto solver = std::make_shared<ClothSolverUniversal>();
  solver_actor->AddComponent(solver);

  auto watch_dog_actor = SpawnWatchDogActor(game);

  std::string time = GetFormattedTime();
  solver->SetIdentifier(time);
}

void SceneUniversal::SpawnObjectsFromSimulationConfig(GameInstance* game) {
  simulation_config_parser_->Apply();
  const auto& clothes = simulation_config_parser_->universal_pipeline_settings.clothes;
  const auto& obstacles = simulation_config_parser_->universal_pipeline_settings.obstacles;
  for (auto& cloth : clothes) {
    SpawnCloth(game, cloth);
  }
  for (auto& obstacle : obstacles) {
    SpawnObstacle(game, obstacle);
  }
}

std::shared_ptr<Actor> SceneUniversal::SpawnCloth(GameInstance* game,
                                                  const UniversalActorConfig& config) {
  LOG_TRACE("Spawn cloth actor_: {}", config.name);
  const unsigned int garment_id = Global::sim_params.num_clothes;

  const std::string actor_name = "cloth_" + config.name + "_" + std::to_string(garment_id);
  auto cloth = game->CreateActor(actor_name);

  auto material = Resource::LoadMaterial("_Default");
  material->Use();
  material->double_sided = true;

  MaterialProperty material_property;

  int n_colors = front_face_colors.size();
  int c_idx = garment_id % n_colors;

  material_property.pre_rendering = [c_idx](Material* mat) {
    mat->SetBool("material.useTexture", false);
    mat->SetVec3("material.frontFaceColor", front_face_colors[c_idx]);
    mat->SetVec3("material.backFaceColor", back_face_colors[c_idx]);
    mat->specular = 0.01f;
  };
  LOG_TRACE("Create mesh renderer component");

  auto mesh = Resource::LoadMeshDataAndBuildStructure(config.mesh_path, garment_id);

  mesh->AddFixedVertices(config.fixed_nodes);

  auto renderer = std::make_shared<MeshRenderer>(mesh, material, true);
  renderer->SetMaterialProperty(material_property);

  LOG_TRACE("Create particle geometry renderer component");
  auto prenderer = std::make_shared<ParticleGeometryRenderer>();

  LOG_TRACE("Create cloth object component");
  auto cloth_obj = std::make_shared<ClothObjectUniversal>(
      Global::sim_params.num_clothes,
      ((Global::sim_params.num_obstacles > 0) ? Global::sim_params.num_obstacles - 1 : 0));

  cloth->AddComponents({renderer, cloth_obj, prenderer});
  Global::sim_params.num_clothes++;

  cloth->Initialize(config.position, config.scale, config.rotation);

  return cloth;
}

std::shared_ptr<Actor> SceneUniversal::SpawnObstacle(GameInstance* game,
                                                     const UniversalActorConfig& config) {
  const unsigned int id = Global::sim_params.num_obstacles;
  const std::string name = "obstacle_" + config.name + "_" + std::to_string(id);

  auto obstacle = game->CreateActor(name);
  MaterialProperty material_property;
  material_property.pre_rendering = [](Material* mat) {
    mat->SetBool("material.useTexture", false);
    mat->SetVec3("material.frontFaceColor", skin_color);
    mat->SetVec3("material.backFaceColor", skin_color);
  };

  auto material = Resource::LoadMaterial("_Default");

  auto mesh = Resource::LoadMeshDataAndBuildStructure(config.mesh_path, id);

  auto renderer = std::make_shared<MeshRenderer>(mesh, material, true);
  renderer->SetMaterialProperty(material_property);

  auto obstacle_object = std::make_shared<ObstacleObjectUniversal>(id);

  obstacle->AddComponents({renderer, obstacle_object});
  Global::sim_params.num_obstacles++;

  obstacle->Initialize(config.position, config.scale, config.rotation);

  return obstacle;
}

void SceneUniversal::SpawnObjects(GameInstance* game) {
  for (auto iter = configs_.begin(); iter != configs_.end(); iter++) {
    switch (iter->type) {
      case UniversalActorType::CLOTH: {
        SpawnCloth(game, *iter);
        break;
      }
      case UniversalActorType::OBSTACLE: {
        SpawnObstacle(game, *iter);
        break;
      }
      default:
        break;
    }
  }
}

void SceneUniversal::AddActor(const UniversalActorConfig& config) {
  configs_.push_back(config);
}

void SceneUniversal::AddActors(const std::vector<UniversalActorConfig>& configs) {
  configs_.insert(configs_.end(), configs.begin(), configs.end());
}

}  // namespace XRTailor