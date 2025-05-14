#include <xrtailor/pipeline/impl/smpl/Scene.hpp>
#include <xrtailor/runtime/rendering/ParticleGeometryRenderer.hpp>

namespace XRTailor {

SceneSMPL::SceneSMPL(std::shared_ptr<SimulationConfigParser> simulation_config_parser) {
  simulation_config_parser_ = simulation_config_parser;
  name = "XRTailor / " + simulation_config_parser_->name;
}

void SceneSMPL::PopulateActors(GameInstance* game) {
  LOG_TRACE("Populate actors");

  simulation_config_parser_->Apply();

  SpawnCameraAndLight(game);
  LOG_TRACE("Spawn camera and light done");
  SpawnInfinitePlane(game);
  SpawnDebugger(game);

  std::string sequence_label;
  const std::string& body_model = Global::sim_config.smpl.body_model;
  if (body_model == "SMPL") {
    SpawnObjectsFromSimulationConfig<SMPL, AMASS_SMPLH_G>(game, sequence_label);
  } else if (body_model == "SMPLH") {
    SpawnObjectsFromSimulationConfig<SMPLH, AMASS_SMPLH_G>(game, sequence_label);
  } else if (body_model == "SMPLX") {
    SpawnObjectsFromSimulationConfig<SMPLX, AMASS_SMPLX_G>(game, sequence_label);
  }

  std::string time = GetFormattedTime();
  // create solver
  auto solver_actor = game->CreateActor("ClothSolver");
  if (body_model == "SMPL") {
    auto solver = std::make_shared<ClothSolverSMPL<SMPL, AMASS_SMPLH_G>>();
    solver_actor->AddComponent(solver);
    solver->SetIdentifier(time);
    solver->selected_sequence_label = sequence_label;
  } else if (body_model == "SMPLH") {
    auto solver = std::make_shared<ClothSolverSMPL<SMPLH, AMASS_SMPLH_G>>();
    solver_actor->AddComponent(solver);
    solver->SetIdentifier(time);
    solver->selected_sequence_label = sequence_label;
  } else if (body_model == "SMPLX") {
    auto solver = std::make_shared<ClothSolverSMPL<SMPLX, AMASS_SMPLX_G>>();
    solver_actor->AddComponent(solver);
    solver->SetIdentifier(time);
    solver->selected_sequence_label = sequence_label;
  }

  auto watch_dog_actor = SpawnWatchDogActor(game);
}

template <class ModelConfig, class SequenceConfig>
std::shared_ptr<Actor> SceneSMPL::SpawnCloth(GameInstance* game, 
    std::shared_ptr<ClothConfigParser> cloth_config_parser) {
  std::string style = cloth_config_parser->style;
  const unsigned int garment_id = Global::sim_params.num_clothes;

  const std::string actor_name =
      "cloth_" + style + "_" + std::to_string(garment_id);
  auto cloth = game->CreateActor(actor_name);

  auto material = Resource::LoadMaterial("_Default");
  material->Use();
  material->double_sided = true;

  MaterialProperty material_property;
  material_property.pre_rendering = [garment_id](Material* mat) {
    mat->SetBool("material.useTexture", false);
    mat->SetVec3("material.frontFaceColor", front_face_colors[garment_id]);
    mat->SetVec3("material.backFaceColor", back_face_colors[garment_id]);
    mat->specular = 0.01f;
  };

  filesystem::path cloth_template_dir = GetClothTemplateDirectorySMPL();
  filesystem::path mesh_path = cloth_template_dir.append(style + ".obj");
  auto mesh = Resource::LoadMeshDataAndBuildStructure(mesh_path.string(), garment_id);

  auto renderer = std::make_shared<MeshRenderer>(mesh, material, true);
  renderer->SetMaterialProperty(material_property);

  auto prenderer = std::make_shared<ParticleGeometryRenderer>();

  auto cloth_obj = std::make_shared<ClothObjectSMPL<ModelConfig, SequenceConfig>>(
      Global::sim_params.num_clothes,
      ((Global::sim_params.num_obstacles > 0) ? Global::sim_params.num_obstacles - 1 : 0));

  cloth->AddComponents({renderer, cloth_obj, prenderer});
  Global::sim_params.num_clothes++;

  return cloth;
}

template <class ModelConfig, class SequenceConfig>
std::shared_ptr<Actor> SceneSMPL::SpawnObstacle(GameInstance* game, const std::string& path) {
  const unsigned int id = Global::sim_params.num_obstacles;
  const std::string name = "obstacle_" + std::to_string(id);

  auto obstacle = game->CreateActor(name);
  LOG_DEBUG("CreateActor");
  MaterialProperty material_property;
  material_property.pre_rendering = [](Material* mat) {
    mat->SetBool("material.useTexture", false);
    mat->SetVec3("material.frontFaceColor", skin_color);
    mat->SetVec3("material.backFaceColor", skin_color);
  };

  auto material = Resource::LoadMaterial("_Default");

  auto mesh = Resource::LoadMeshDataAndBuildStructure(path, id);

  auto renderer = std::make_shared<MeshRenderer>(mesh, material, true);
  renderer->SetMaterialProperty(material_property);

  auto obstacle_object = std::make_shared<ObstacleObjectSMPL<ModelConfig, SequenceConfig>>(id);
  std::vector<unsigned int> mask = Resource::LoadMask(Global::sim_config.smpl.body_model);
  obstacle_object->SetMask(mask);

  obstacle->AddComponents({renderer, obstacle_object});
  Global::sim_params.num_obstacles++;

  return obstacle;
}

template <class ModelConfig, class SequenceConfig>
void SceneSMPL::SpawnObjectsFromSimulationConfig(GameInstance* game, std::string& sequence_label) {
  // init motion sequence
  filesystem::path npz_path(simulation_config_parser_->smpl_pipeline_settings.npz_path);
  std::shared_ptr<smplx::Sequence<SequenceConfig>> sequence =
      std::make_shared<smplx::Sequence<SequenceConfig>>(npz_path.string());
  filesystem::path seq_path(npz_path.string());
  filesystem::path filename = seq_path.stem();
  sequence_label = filename.string();
  LOG_INFO("Sequence label: {}", sequence_label);

  std::string gender;
  switch (sequence->gender) {
    case smplx::Gender::female:
      gender = "female";
      break;
    case smplx::Gender::male:
      gender = "male";
      break;
    case smplx::Gender::neutral:
      gender = "neutral";
      break;
    default:
      gender = "unknown";
  }

  std::vector<std::shared_ptr<ClothConfigParser>> cloth_configs;
  filesystem::path cloth_config_dir = GetClothConfigDirectorySMPL();
  const std::vector<std::string>& styles =
      simulation_config_parser_->smpl_pipeline_settings.cloth_styles;

  for (int i = 0; i < styles.size(); i++) {
    // load garment config
    std::shared_ptr<ClothConfigParser> cloth_config_parser = std::make_shared<ClothConfigParser>();
    filesystem::path cloth_config_path(cloth_config_dir);
    cloth_config_path.append(styles[i] + "_" + gender + ".json");
    if (!cloth_config_parser->LoadFromJson(cloth_config_path.string())) {
      LOG_ERROR("An error occured when reading {}, exit engine.", cloth_config_path.string());
      exit(TAILOR_EXIT::INVALID_CLOTH_CONFIG);
    };
    cloth_configs.push_back(cloth_config_parser);

    // add cloth into scene
    std::shared_ptr<Actor> cloth = SpawnCloth<ModelConfig, SequenceConfig>(game, cloth_config_parser);
    cloth->Initialize(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0), glm::vec3(0, 0, 0));
    auto mesh_renderer = cloth->GetComponent<MeshRenderer>();
    auto cloth_mesh = mesh_renderer->mesh();

    cloth_mesh->AddAttachedVertices(cloth_config_parser->attached_indices);
    cloth_mesh->AddBindedVertices(EXTEND_MODE::BOUNDARY, cloth_config_parser->binding_boundary);
    cloth_mesh->AddBindedVertices(EXTEND_MODE::NEIGHBOR, cloth_config_parser->binding_neighbor);
    cloth_mesh->AddBindedVertices(EXTEND_MODE::NONMANIFOLD_EDGES,
                                  cloth_config_parser->binding_nonmanifold_edges);
    cloth_mesh->AddBindedVertices(EXTEND_MODE::UV_ISLAND, cloth_config_parser->binding_uv_island);
    cloth_mesh->ApplyBindings();
  }

  // add obstacle
  filesystem::path obstacle_template_path(GetBodyTemplateDirectory());
  if (simulation_config_parser_->smpl_pipeline_settings.body_model == "SMPL" ||
      simulation_config_parser_->smpl_pipeline_settings.body_model == "SMPLH") {
      // smpl and smplh shares the same template
      obstacle_template_path.append("SMPLH_" + gender + ".obj"); 
  } else {
    obstacle_template_path.append(simulation_config_parser_->smpl_pipeline_settings.body_model + "_" + gender + ".obj");
  }
  std::shared_ptr<Actor> obstacle =
      SpawnObstacle<ModelConfig, SequenceConfig>(game, obstacle_template_path.string());
  auto obstacle_object = obstacle->GetComponent<ObstacleObjectSMPL<ModelConfig, SequenceConfig>>();

  obstacle_object->SetSequence(sequence);

  obstacle->Initialize(glm::vec3(0.0f, 0.00, 0.0f), glm::vec3(1.0), glm::vec3(0, 0, 0));
}

}  // namespace XRTailor