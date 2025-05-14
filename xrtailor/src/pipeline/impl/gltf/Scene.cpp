#include <xrtailor/pipeline/impl/gltf/Scene.hpp>

#include <xrtailor/runtime/rag_doll/gltf/GLTFLoader.cuh>
#include <xrtailor/runtime/rendering/ParticleGeometryRenderer.hpp>

namespace XRTailor {

SceneGLTF::SceneGLTF(std::shared_ptr<SimulationConfigParser> simulation_config_parser) {
  simulation_config_parser_ = simulation_config_parser;
  name = "XRTailor / " + simulation_config_parser_->name;
}

SceneGLTF::~SceneGLTF() = default;

void SceneGLTF::PopulateActors(GameInstance* game) {
  LOG_TRACE("Populate actors");

  simulation_config_parser_->Apply();

  SpawnCameraAndLight(game);
  LOG_TRACE("Spawn camera and light done");
  SpawnInfinitePlane(game);
  SpawnDebugger(game);

  SpawnObjectsFromSimulationConfig(game);
  auto solver_actor = game->CreateActor("ClothSolverGLTF");
  auto solver = std::make_shared<ClothSolverGLTF>();
  solver_actor->AddComponent(solver);
  std::string time = GetFormattedTime();
  solver->SetIdentifier(time);
  LOG_DEBUG("time: {}", time);

  auto watch_dog_actor = SpawnWatchDogActor(game);
}

void SceneGLTF::SpawnObjectsFromSimulationConfig(GameInstance* game) {
  std::string character_id = simulation_config_parser_->gltf_pipeline_settings.character_name;

  filesystem::path cloth_config_dir = GetClothConfigDirectoryGLTF(character_id);
  auto styles = simulation_config_parser_->gltf_pipeline_settings.cloth_styles;

  for (int i = 0; i < styles.size(); i++) {
    // load cloth config
    std::shared_ptr<ClothConfigParser> parser = std::make_shared<ClothConfigParser>();
    filesystem::path cloth_config_path(cloth_config_dir);
    cloth_config_path.append(styles[i] + ".json");
    if (!parser->LoadFromJson(cloth_config_path.string())) {
      LOG_ERROR("An error occured when reading {}, exit engine.", cloth_config_path.string());
      exit(TAILOR_EXIT::INVALID_CLOTH_CONFIG);
    };

    // add cloth into scene
    std::shared_ptr<Actor> cloth = SpawnCloth(game, character_id, parser);
    cloth->Initialize(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0), glm::vec3(0, 0, 0));
    auto mesh_renderer = cloth->GetComponent<MeshRenderer>();
    auto cloth_mesh = mesh_renderer->mesh();

    cloth_mesh->AddAttachedVertices(parser->attached_indices);
    cloth_mesh->AddBindedVertices(EXTEND_MODE::BOUNDARY, parser->binding_boundary);
    cloth_mesh->AddBindedVertices(EXTEND_MODE::NEIGHBOR, parser->binding_neighbor);
    cloth_mesh->AddBindedVertices(EXTEND_MODE::NONMANIFOLD_EDGES,
                                  parser->binding_nonmanifold_edges);
    cloth_mesh->AddBindedVertices(EXTEND_MODE::UV_ISLAND, parser->binding_uv_island);
    cloth_mesh->ApplyBindings();
  }

  std::string gltf_path = simulation_config_parser_->gltf_pipeline_settings.gltf_path;
  std::shared_ptr<Actor> obstacle = SpawnObstacle(game, gltf_path);
  auto obstacle_object = obstacle->GetComponent<ObstacleObjectGLTF>();

  obstacle->Initialize(glm::vec3(0.0f, 0.00, 0.0f), glm::vec3(1.0), glm::vec3(0, 0, 0));
}

std::shared_ptr<Actor> SceneGLTF::SpawnCloth(
    GameInstance* game, std::string character_id,
    std::shared_ptr<ClothConfigParser> cloth_config_parser) {
  std::string style = cloth_config_parser->style;
  const unsigned int cloth_id = Global::sim_params.num_clothes;
  const std::string actor_name = "cloth_" + style + "_" + std::to_string(cloth_id);
  auto cloth = game->CreateActor(actor_name);

  auto material = Resource::LoadMaterial("_Default");
  material->Use();
  material->double_sided = true;

  MaterialProperty material_property;
  material_property.pre_rendering = [cloth_id](Material* mat) {
    //mat->SetVec3("material.tint", glm::vec3(0.0f, 0.5f, 1.0f));
    mat->SetBool("material.useTexture", false);
    //mat->SetTexture("material.diffuse", texture);
    mat->SetVec3("material.frontFaceColor", front_face_colors[cloth_id]);  // Lenurple
    mat->SetVec3("material.backFaceColor", back_face_colors[cloth_id]);
    mat->specular = 0.01f;
  };

  filesystem::path cloth_template_dir = GetClothTemplateDirectoryGLTF(character_id);
  filesystem::path mesh_path = cloth_template_dir.append(style + ".obj");
  auto mesh = Resource::LoadMeshDataAndBuildStructure(mesh_path.string(), cloth_id);

  auto renderer = std::make_shared<MeshRenderer>(mesh, material, true);
  renderer->SetMaterialProperty(material_property);

  auto prenderer = std::make_shared<ParticleGeometryRenderer>();

  auto cloth_obj = std::make_shared<ClothObjectGLTF>(
      Global::sim_params.num_clothes,
      ((Global::sim_params.num_obstacles > 0) ? Global::sim_params.num_obstacles - 1 : 0));

  cloth->AddComponents({renderer, cloth_obj, prenderer});

  Global::sim_params.num_clothes += 1;
  LOG_DEBUG("Num clothes: {}", Global::sim_params.num_clothes);
  return cloth;
}

std::shared_ptr<Actor> SceneGLTF::SpawnObstacle(GameInstance* game, const std::string& path) {
  const unsigned int id = Global::sim_params.num_obstacles;
  const std::string name = "obstacle_" + std::to_string(id);

  auto obstacle = game->CreateActor(name);
  MaterialProperty material_property;
  material_property.pre_rendering = [](Material* mat) {
    //mat->SetVec3("material.tint", glm::vec3(1.0));
    mat->SetBool("material.useTexture", false);
    mat->SetVec3("material.frontFaceColor", skin_color);
    mat->SetVec3("material.backFaceColor", skin_color);
  };

  auto material = Resource::LoadMaterial("_Default");

  std::shared_ptr<GLTFLoader> gltf_loader = std::make_shared<GLTFLoader>();

  auto mesh = Resource::LoadGLBAndBuildStructure(gltf_loader, path, id);

  auto renderer = std::make_shared<MeshRenderer>(mesh, material, true);
  renderer->SetMaterialProperty(material_property);

  auto obstacle_object = std::make_shared<ObstacleObjectGLTF>(id);

  obstacle_object->SetGltfLoader(gltf_loader);

  obstacle->AddComponents({renderer, obstacle_object});
  Global::sim_params.num_obstacles++;

  return obstacle;
}

}  // namespace XRTailor