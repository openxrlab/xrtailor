#include <xrtailor/config/SimulationConfigParser.hpp>

#include <iostream>
#include <fstream>

#include <xrtailor/core/Scalar.hpp>
#include <xrtailor/utils/FileSystemUtils.hpp>
#include <xrtailor/utils/Logger.hpp>

namespace XRTailor {
SimulationConfigParser::SimulationConfigParser() = default;

int ParsePipelineType(const Json::Value& root) {
  if (!root["PIPELINE_SMPL"].isNull())
    return PIPELINE::PIPELINE_SMPL;
  if (!root["PIPELINE_GLTF"].isNull())
    return PIPELINE::PIPELINE_GLTF;
  if (!root["PIPELINE_UNIVERSAL"].isNull())
    return PIPELINE::PIPELINE_UNIVERSAL;

  return PIPELINE::PIPELINE_UNKNOWN;
}

int ParseSolverModeType(const Json::Value& root) {
  if (!root["MODE_SWIFT"].isNull())
    return SOLVER_MODE::SWIFT;
  if (!root["MODE_QUALITY"].isNull())
    return SOLVER_MODE::QUALITY;

  return SOLVER_MODE::UNKNOWN;
}

bool SimulationConfigParser::ParsePipelineSMPL(const Json::Value& root) {
  Json::Value scope_smpl = root["PIPELINE_SMPL"];
  Json::Value cloth_styles = scope_smpl["CLOTH_STYLES"];
  if (!cloth_styles.isArray() || cloth_styles.empty()) {
    LOG_ERROR("Cloth styles must be an non-empty vector of type string.");
    return false;
  }
  for (int i = 0; i < cloth_styles.size(); i++) {
    this->smpl_pipeline_settings.cloth_styles.push_back(cloth_styles[i].asString());
  }
  this->smpl_pipeline_settings.npz_path = scope_smpl["NPZ_PATH"].asString();
  this->smpl_pipeline_settings.body_model = scope_smpl["BODY_MODEL"].asString();
  this->smpl_pipeline_settings.num_lerped_frames = scope_smpl["NUM_LERPED_FRAMES"].asInt();
  if (!scope_smpl["ENABLE_POSE_BLENDSHAPE"].isNull()) {
    this->smpl_pipeline_settings.enable_pose_blendshape =
        scope_smpl["ENABLE_POSE_BLENDSHAPE"].asBool();
  }

  if (!scope_smpl["ENABLE_COLLISION_FILTER"].isNull()) {
    this->smpl_pipeline_settings.enable_collision_filter =
        scope_smpl["ENABLE_COLLISION_FILTER"].asBool();
  }

  return true;
}


bool SimulationConfigParser::ParsePipelineGLTF(const Json::Value& root) {
  Json::Value scope_gltf = root["PIPELINE_GLTF"];
  Json::Value cloth_styles = scope_gltf["CLOTH_STYLES"];
  if (!cloth_styles.isArray() || cloth_styles.empty()) {
    LOG_ERROR("Cloth styles must be an non-empty vector of type string.");
    return false;
  }

  this->gltf_pipeline_settings.character_name = scope_gltf["CHARACTER_NAME"].asString();
  this->gltf_pipeline_settings.gltf_path = scope_gltf["GLTF_PATH"].asString();
  this->gltf_pipeline_settings.num_lerped_frames = scope_gltf["NUM_LERPED_FRAMES"].asInt();
  for (int i = 0; i < cloth_styles.size(); i++) {
    this->gltf_pipeline_settings.cloth_styles.push_back(cloth_styles[i].asString());
  }

  return true;
}

bool SimulationConfigParser::ParsePipelineUniversal(const Json::Value& root) {
  Json::Value scope_universal = root["PIPELINE_UNIVERSAL"];
  if (!scope_universal["CLOTHES"] || !scope_universal["OBSTACLES"]) {
    LOG_ERROR("Property CLOTHES/OBSTACLES not found.");
    return false;
  }
  Json::Value scope_clothes = scope_universal["CLOTHES"];
  Json::Value scope_obstacles = scope_universal["OBSTACLES"];

  for (int i = 0; i < scope_clothes.size(); i++) {
    Json::Value cloth = scope_clothes[i];
    const std::string& obj_path = cloth["OBJ_PATH"].asString();
    const std::string& name = GetFileNameFromPath(obj_path);
    UniversalActorConfig config{UniversalActorType::CLOTH, name, obj_path};
    if (!cloth["ATTACHED_INDICES"].isNull()) {
      Json::Value raw_attached_indices = cloth["ATTACHED_INDICES"];
      std::vector<unsigned int> attached_indices;
      for (int j = 0; j < raw_attached_indices.size(); j++) {
        attached_indices.push_back(raw_attached_indices[j].asUInt());
      }
      config.fixed_nodes = attached_indices;
    }
    if (!cloth["POSITION"].isNull()) {
      Json::Value raw_position = cloth["POSITION"];
      config.position = 
          Vector3(raw_position[0].asFloat(), raw_position[1].asFloat(), raw_position[2].asFloat());
    }
    if (!cloth["SCALE"].isNull()) {
      Json::Value raw_scale = cloth["SCALE"];
      config.scale =
          Vector3(raw_scale[0].asFloat(), raw_scale[1].asFloat(), raw_scale[2].asFloat());
    }
    if (!cloth["ROTATION"].isNull()) {
      Json::Value raw_rotation = cloth["ROTATION"];
      config.rotation =
          Vector3(raw_rotation[0].asFloat(), raw_rotation[1].asFloat(), raw_rotation[2].asFloat());
    }
    this->universal_pipeline_settings.clothes.push_back(config);
  }

  for (int i = 0; i < scope_obstacles.size(); i++) {
    Json::Value obstacle = scope_obstacles[i];
    const std::string& obj_path = obstacle["OBJ_PATH"].asString();
    UniversalActorConfig config{UniversalActorType::CLOTH, obj_path, obj_path};
    if (!obstacle["POSITION"].isNull()) {
      Json::Value raw_position = obstacle["POSITION"];
      config.position =
          Vector3(raw_position[0].asFloat(), raw_position[1].asFloat(), raw_position[2].asFloat());
    }
    if (!obstacle["SCALE"].isNull()) {
      Json::Value raw_scale = obstacle["SCALE"];
      config.scale =
          Vector3(raw_scale[0].asFloat(), raw_scale[1].asFloat(), raw_scale[2].asFloat());
    }
    if (!obstacle["ROTATION"].isNull()) {
      Json::Value raw_rotation = obstacle["ROTATION"];
      config.rotation =
          Vector3(raw_rotation[0].asFloat(), raw_rotation[1].asFloat(), raw_rotation[2].asFloat());
    }
    this->universal_pipeline_settings.obstacles.push_back(config);
  }

  if (!scope_universal["NUM_FRAMES"].isNull()) {
    this->universal_pipeline_settings.num_frames = scope_universal["NUM_FRAMES"].asInt();
  }

  return true;
}

bool SimulationConfigParser::ParseSwift(const Json::Value& root) {
  Json::Value scope_swift_mode = root["MODE_SWIFT"];

  if (scope_swift_mode.isNull()) {
    LOG_ERROR("Property MODE_SWIFT not found");
    return false;
  }

  Json::Value scope_solver = scope_swift_mode["SOLVER"];
  if (!scope_solver.isNull()) {
    if (!scope_solver["NUM_SUBSTEPS"].isNull()) {
      this->swift_mode_settings.solver.num_substeps = scope_solver["NUM_SUBSTEPS"].asInt();
    }
    if (!scope_solver["NUM_ITERATIONS"].isNull()) {
      this->swift_mode_settings.solver.num_iterations = scope_solver["NUM_ITERATIONS"].asInt();
    }
    if (!scope_solver["MAX_SPEED"].isNull()) {
      this->swift_mode_settings.solver.max_speed = scope_solver["MAX_SPEED"].asFloat();
    }
    Json::Value gravity = scope_solver["GRAVITY"];
    if (!gravity.isNull()) {
      this->swift_mode_settings.solver.gravity =
          glm::vec3(gravity[0].asFloat(), gravity[1].asFloat(), gravity[2].asFloat());
    }
    if (!scope_solver["DAMPING"].isNull()) {
      this->swift_mode_settings.solver.damping = scope_solver["DAMPING"].asFloat();
    }
  }

  Json::Value scope_fabric = scope_swift_mode["FABRIC"];
  if (!scope_fabric.isNull()) {
    if (!scope_fabric["STRETCH_COMPLIANCE"].isNull()) {
      this->swift_mode_settings.fabric.stretch_compliance =
          scope_fabric["STRETCH_COMPLIANCE"].asFloat();
    }
    if (!scope_fabric["BEND_COMPLIANCE"].isNull()) {
      this->swift_mode_settings.fabric.bend_compliance = scope_fabric["BEND_COMPLIANCE"].asFloat();
    }
    if (!scope_fabric["RELAXATION_FACTOR"].isNull()) {
      this->swift_mode_settings.fabric.relaxation_factor =
          scope_fabric["RELAXATION_FACTOR"].asFloat();
    }
    if (!scope_fabric["LONG_RANGE_STRETCHINESS"].isNull()) {
      this->swift_mode_settings.fabric.long_range_stretchiness =
          scope_fabric["LONG_RANGE_STRETCHINESS"].asFloat();
    }
    if (!scope_fabric["GEODESIC_LRA"].isNull()) {
      this->swift_mode_settings.fabric.geodesic_LRA = scope_fabric["GEODESIC_LRA"].asBool();
    }
    if (!scope_fabric["SOLVE_BENDING"].isNull()) {
      this->swift_mode_settings.fabric.solve_bending = scope_fabric["SOLVE_BENDING"].asBool();
    }
  }

  Json::Value scope_collision = scope_swift_mode["COLLISION"];
  if (!scope_collision.isNull()) {
    if (!scope_collision["NUM_COLLISION_PASSES"].isNull()) {
      this->swift_mode_settings.collision.num_collision_passes =
          scope_collision["NUM_COLLISION_PASSES"].asInt();
    }
    if (!scope_collision["SDF_COLLISION_MARGIN"].isNull()) {
      this->swift_mode_settings.collision.sdf_collision_margin =
          scope_collision["SDF_COLLISION_MARGIN"].asFloat();
    }
    if (!scope_collision["BVH_TOLERANCE"].isNull()) {
      this->swift_mode_settings.collision.bvh_tolerance = scope_collision["BVH_TOLERANCE"].asFloat();
    }

    Json::Value scope_self_contact = scope_collision["PARTICLE"];
    if (!scope_self_contact.isNull()) {
      if (!scope_self_contact["FRICTION"].isNull()) {
        this->swift_mode_settings.collision.self_contact.friction =
            scope_self_contact["FRICTION"].asFloat();
      }
      if (!scope_self_contact["MAX_NEIGHBOR_SIZE"].isNull()) {
        this->swift_mode_settings.collision.self_contact.max_neighbor_size =
            scope_self_contact["MAX_NEIGHBOR_SIZE"].asInt();
      }
      if (!scope_self_contact["INTER_LEAVED_HASH"].isNull()) {
        this->swift_mode_settings.collision.self_contact.inter_leaved_hash =
            scope_self_contact["INTER_LEAVED_HASH"].asInt();
      }
      if (!scope_self_contact["ENABLE_SELF_COLLISION"].isNull()) {
        this->swift_mode_settings.collision.self_contact.enable_self_collision =
            scope_self_contact["ENABLE_SELF_COLLISION"].asBool();
      }
      if (!scope_self_contact["PARTICLE_DIAMETER"].isNull()) {
        this->swift_mode_settings.collision.self_contact.particle_diameter =
            scope_self_contact["PARTICLE_DIAMETER"].asFloat();
      }
      if (!scope_self_contact["HASH_CELL_SIZE"].isNull()) {
        this->swift_mode_settings.collision.self_contact.hash_cell_size =
            scope_self_contact["HASH_CELL_SIZE"].asFloat();
      }
    }
  }

  return true;
}

bool SimulationConfigParser::ParseQuality(const Json::Value& root) {
  Json::Value scope_quality_mode = root["MODE_QUALITY"];
  if (scope_quality_mode.isNull()) {
    LOG_ERROR("Property MODE_QUALITY not found");
    return false;
  }

  Json::Value scope_solver = scope_quality_mode["SOLVER"];
  if (!scope_solver.isNull()) {
    if (!scope_solver["NUM_SUBSTEPS"].isNull()) {
      this->quality_mode_settings.solver.num_substeps = scope_solver["NUM_SUBSTEPS"].asInt();
    }
    if (!scope_solver["NUM_ITERATIONS"].isNull()) {
      this->quality_mode_settings.solver.num_iterations = scope_solver["NUM_ITERATIONS"].asInt();
    }
    if (!scope_solver["MAX_SPEED"].isNull()) {
      this->quality_mode_settings.solver.max_speed = scope_solver["MAX_SPEED"].asFloat();
    }
    Json::Value gravity = scope_solver["GRAVITY"];
    if (!gravity.isNull()) {
      this->quality_mode_settings.solver.gravity =
          glm::vec3(gravity[0].asFloat(), gravity[1].asFloat(), gravity[2].asFloat());
      this->quality_mode_settings.solver.damping = scope_solver["DAMPING"].asFloat();
    }
  }
  
  Json::Value scope_fabric = scope_quality_mode["FABRIC"];
  if (!scope_fabric.isNull()) {
    if (!scope_fabric["XX_STIFFNESS"].isNull()) {
      this->quality_mode_settings.fabric.xx_stiffness = scope_fabric["XX_STIFFNESS"].asFloat();
    }
    if (!scope_fabric["XY_STIFFNESS"].isNull()) {
      this->quality_mode_settings.fabric.xy_stiffness = scope_fabric["XY_STIFFNESS"].asFloat();
    }
    if (!scope_fabric["YY_STIFFNESS"].isNull()) {
      this->quality_mode_settings.fabric.yy_stiffness = scope_fabric["YY_STIFFNESS"].asFloat();
    }
    if (!scope_fabric["XY_POISSION_RATIO"].isNull()) {
      this->quality_mode_settings.fabric.xy_poisson_ratio =
          scope_fabric["XY_POISSION_RATIO"].asFloat();
    }
    if (!scope_fabric["YX_POISSION_RATIO"].isNull()) {
      this->quality_mode_settings.fabric.yx_poisson_ratio =
          scope_fabric["YX_POISSION_RATIO"].asFloat();
    }
    if (!scope_fabric["SOLVE_BENDING"].isNull()) {
      this->quality_mode_settings.fabric.solve_bending = scope_fabric["SOLVE_BENDING"].asBool();
    }
    if (!scope_fabric["BENDING_STIFFNESS"].isNull()) {
      this->quality_mode_settings.fabric.bending_stiffness =
          scope_fabric["BENDING_STIFFNESS"].asFloat();
    }
    if (!scope_fabric["LONG_RANGE_STRETCHINESS"].isNull()) {
      this->quality_mode_settings.fabric.long_range_stretchiness =
          scope_fabric["LONG_RANGE_STRETCHINESS"].asFloat();
    }
    if (!scope_fabric["GEODESIC_LRA"].isNull()) {
      this->quality_mode_settings.fabric.geodesic_LRA = scope_fabric["GEODESIC_LRA"].asBool();
    }
  }

  if (!scope_quality_mode["REPULSION"].isNull()) {
    Json::Value scope_repulsion = scope_quality_mode["REPULSION"];
    if (!scope_repulsion["ENABLE_IMMINENT_REPULSION"].isNull()) {
      this->quality_mode_settings.repulsion.enable_imminent_repulsion =
          scope_repulsion["ENABLE_IMMINENT_REPULSION"].asBool();
    }
    if (!scope_repulsion["IMMINENT_THICKNESS"].isNull()) {
      this->quality_mode_settings.repulsion.imminent_thickness =
          scope_repulsion["IMMINENT_THICKNESS"].asFloat();
    }
    if (!scope_repulsion["RELAXATION_RATE"].isNull()) {
      this->quality_mode_settings.repulsion.relaxation_rate =
          scope_repulsion["RELAXATION_RATE"].asFloat();
    }
    if (!scope_repulsion["ENABLE_PBD_REPULSION"].isNull()) {
      this->quality_mode_settings.repulsion.enable_pbd_repulsion =
          scope_repulsion["ENABLE_PBD_REPULSION"].asBool();
    }
    if (!scope_repulsion["PBD_THICKNESS"].isNull()) {
      this->quality_mode_settings.repulsion.pbd_thickness =
          scope_repulsion["PBD_THICKNESS"].asFloat();
    }
  }

  if (!scope_quality_mode["IMPACT_ZONE"].isNull()) {
    Json::Value scope_impact_zone = scope_quality_mode["IMPACT_ZONE"];
    if (!scope_impact_zone["OBSTACLE_MASS"].isNull()) {
      this->quality_mode_settings.impact_zone.obstacle_mass =
          scope_impact_zone["OBSTACLE_MASS"].asFloat();
    }
    if (!scope_impact_zone["THICKNESS"].isNull()) {
      this->quality_mode_settings.impact_zone.thickness = scope_impact_zone["THICKNESS"].asFloat();
    }
  }

  return true;
}

bool SimulationConfigParser::LoadFromJson(const std::string& path) {
  LOG_INFO("Reading simulation config from: {}", path);

  std::ifstream file(path, std::ios::binary);

  if (!file.is_open()) {
    LOG_ERROR("Error opening file: {}", path);
    return false;
  }

  this->name = GetFileNameFromPath(path);

  Json::Reader reader;
  Json::Value root;

  if (!reader.parse(file, root, false)) {
    LOG_ERROR("Failed to parse file: {}", path);
    return false;
  }

  pipeline_type = ParsePipelineType(root);
  LOG_DEBUG("{}", pipeline_type);
  switch (pipeline_type) {
    case PIPELINE::PIPELINE_SMPL: {
      if (!ParsePipelineSMPL(root))
        return false;
      break;
    }
    case PIPELINE::PIPELINE_GLTF: {
      if (!ParsePipelineGLTF(root))
        return false;
      break;
    }
    case PIPELINE::PIPELINE_UNIVERSAL: {
      if (!ParsePipelineUniversal(root))
        return false;
      break;
    }
    default: {
      LOG_ERROR("Failed to parse pipeline type.");
      return false;
    }
  }

  if (!root["ANIMATION"].isNull()) {
    Json::Value scope_animation = root["ANIMATION"];
    this->animation_settings.num_pre_simulation_frames =
        scope_animation["NUM_PRE_SIMULATION_FRAMES"].asInt();
    this->animation_settings.record_obstacle = scope_animation["RECORD_OBSTACLE"].asBool();
    this->animation_settings.record_cloth = scope_animation["RECORD_CLOTH"].asBool();
    this->animation_settings.export_format = scope_animation["EXPORT_FORMAT"].asInt();
    this->animation_settings.export_directory = scope_animation["EXPORT_DIRECTORY"].asString();
    this->animation_settings.target_frame_rate = scope_animation["TARGET_FRAME_RATE"].asInt();
    LOG_DEBUG("{}", this->animation_settings.record_obstacle);
  }

  this->solver_mode = ParseSolverModeType(root);
  switch (this->solver_mode) {
    case SOLVER_MODE::QUALITY: {
      if (!ParseQuality(root))
        return false;
      break;
    }
    case SOLVER_MODE::SWIFT: {
      if (!ParseSwift(root))
        return false;
      break;
    }
    default:
      break;
  }

  return true;
}

void SimulationConfigParser::Apply() {
  switch (this->pipeline_type) {
    case PIPELINE::PIPELINE_SMPL: {
      Global::sim_config.smpl = smpl_pipeline_settings;
      Global::sim_params.num_lerped_frames = smpl_pipeline_settings.num_lerped_frames;
      break;
    }
    case PIPELINE::PIPELINE_GLTF: {
      Global::sim_config.gltf = gltf_pipeline_settings;
      Global::sim_params.num_lerped_frames = gltf_pipeline_settings.num_lerped_frames;
      break;
    }
    case PIPELINE::PIPELINE_UNIVERSAL: {
      Global::sim_config.universal = universal_pipeline_settings;
      Global::sim_params.num_frames = universal_pipeline_settings.num_frames;
      break;
    }
    default:
      break;
  }

  // animation
  Global::sim_config.animation = this->animation_settings;
  Global::sim_params.pre_simulation_frames = this->animation_settings.num_pre_simulation_frames;
  Global::sim_params.record_obstacle = this->animation_settings.record_obstacle;
  Global::sim_params.record_cloth = this->animation_settings.record_cloth;
  
  if (!filesystem::exists(this->animation_settings.export_directory)) {
    filesystem::create_directory(this->animation_settings.export_directory);
  }

  switch (this->solver_mode) {
    case SOLVER_MODE::QUALITY: {
      Global::sim_config.quality = this->quality_mode_settings;

      SolverSettings* solver = &this->quality_mode_settings.solver;
      Global::sim_params.num_substeps = solver->num_substeps;
      Global::sim_params.num_iterations = solver->num_iterations;
      Global::sim_params.max_speed = solver->max_speed;
      Global::sim_params.gravity = solver->gravity;
      Global::sim_params.damping = solver->damping;
      Global::sim_params.solver_mode = this->solver_mode;

      Global::sim_params.imminent_repulsion =
          this->quality_mode_settings.repulsion.enable_imminent_repulsion;
      Global::sim_params.pbd_repulsion = 
          this->quality_mode_settings.repulsion.enable_pbd_repulsion;
      break;
    }
    case SOLVER_MODE::SWIFT: {
      Global::sim_config.swift = this->swift_mode_settings;
      
      SolverSettings* solver = &this->swift_mode_settings.solver;
      Global::sim_params.num_substeps = solver->num_substeps;
      Global::sim_params.num_iterations = solver->num_iterations;
      Global::sim_params.max_speed = solver->max_speed;
      Global::sim_params.gravity = solver->gravity;
      Global::sim_params.damping = solver->damping;
      Global::sim_params.solver_mode = this->solver_mode;

      Global::sim_params.solve_bending = this->swift_mode_settings.fabric.solve_bending;
      Global::sim_params.long_range_stretchiness =
          this->swift_mode_settings.fabric.long_range_stretchiness;

      SwiftModeCollisionSettings* collision = &this->swift_mode_settings.collision;
      Global::sim_params.sdf_friction = collision->self_contact.friction;
      Global::sim_params.max_num_neighbors = collision->self_contact.max_neighbor_size;
      Global::sim_params.enable_self_collision = collision->self_contact.enable_self_collision;

      Global::sim_params.collision_margin = collision->sdf_collision_margin;
      Global::sim_params.bvh_tolerance = collision->bvh_tolerance;
      Global::sim_params.num_collision_passes = collision->num_collision_passes;
      break;
    }
    default:
      break;
  }
}
}  // namespace XRTailor