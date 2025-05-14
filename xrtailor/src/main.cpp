#include <iostream>
#include <random>
#include <vector>
#include <string>

#include <xrtailor/runtime/engine/Engine.hpp>
#include <xrtailor/runtime/engine/GameInstance.hpp>

#include <xrtailor/utils/FileSystemUtils.hpp>
#include <xrtailor/utils/Logger.hpp>

#include <xrtailor/config/SimulationConfigParser.hpp>
#include <xrtailor/config/EngineConfigParser.hpp>
#include <xrtailor/pipeline/impl/smpl/Scene.hpp>
#include <xrtailor/pipeline/impl/universal/Scene.hpp>

#include <xrtailor/pipeline/impl/gltf/Scene.hpp>
#include <xrtailor/pipeline/impl/gltf/ActorConfig.hpp>

#include "cxxopts.hpp"

using namespace XRTailor;

void PrintUsage() {
  std::cout << "\nUsage: "
            << "Tailor"
            << " [options]" << std::endl
            << "Options:" << std::endl
            << "  -s, --simulation_config        path to simulation json config file\n"
            << "  -e, --engine_config            path to engine json config file\n"
            << "  -h, --help                     show help info\n"
            << "\n"
            << "Example usage:\n"
            << "  (Use customed engine settings):\n"
            << "    .\\Tailor.exe --engine_config \".\\engine_conf.json\" --simulation_config "
               "\".\\SimConfig\\simulation_conf1.json\"\n"
            << "  (Use default engine settings):\n"
            << R"(    .\Tailor.exe --simulation_config ".\SimConfig\simulation_conf1.json")"
            << std::endl;
}

int main(int argc, char* argv[]) {
  cxxopts::Options options("XRTailor",
                           "GPU Cloth Simulator for Efficient Large-Scale Data Synthesis");

  options.add_options()
      ("e,engine_config", "Engine config", cxxopts::value<string>()->default_value("./engine_conf.json"))
      ("s,simulation_config", "Simulation config", cxxopts::value<string>())
      ("h,help", "Usage");

  cxxopts::ParseResult result;
  try {
    result = options.parse(argc, argv);
  } catch (cxxopts::exceptions::parsing e) {
    std::cout << e.what() << std::endl;
    return TAILOR_EXIT::ARGUMENT_PARSING_ERROR;
  }

  if (result.count("help")) {
    PrintUsage();
    return EXIT_FAILURE;
  }

  if (result.count("engine_config") == 0) {
    std::cout << "No engine config file specified, using default config: "
              << result["engine_config"].as<string>() << std::endl;
  }
  string engine_config_path = result["engine_config"].as<string>();

  string simulation_config_path;
  if (result.count("simulation_config") == 1) {
    simulation_config_path = result["simulation_config"].as<string>();
  } else if (result.count("simulation_config") > 1) {
    std::cout << "Only one simulation config is allowed, exit engine." << std::endl;
    return EXIT_FAILURE;
  } else {
    std::cout << "Missing argument: \"--simulation_config\", use \"--help\" for example usage. "
              << std::endl;
    return TAILOR_EXIT::MISSING_ARGUMENT;
  }

  EngineConfigParser engine_config;
  if (!engine_config.LoadFromJson(engine_config_path)) {
    return TAILOR_EXIT::INVALID_ENGINE_CONFIG;
  }
  engine_config.Apply();

  //=====================================
  // 1. Create graphics
  //=====================================
  auto engine = std::make_shared<Engine>();

  //=====================================
  // 2. Instantiate actors
  //=====================================
  std::vector<std::shared_ptr<Scene>> scenes;
  std::vector<std::shared_ptr<SimulationConfigParser>> simulation_config_parsers;
  LOG_DEBUG("Simulation config path: {}", simulation_config_path);
  std::shared_ptr<SimulationConfigParser> simulation_config_parser =
      std::make_shared<SimulationConfigParser>();
  if (!simulation_config_parser->LoadFromJson(simulation_config_path)) {
    LOG_ERROR("An error occured when reading {}, exit engine.", simulation_config_path);
    exit(TAILOR_EXIT::INVALID_SIMULATION_CONFIG);
  }
  simulation_config_parsers.push_back(simulation_config_parser);
  switch (simulation_config_parser->pipeline_type) {
    case PIPELINE::PIPELINE_SMPL: {
      scenes.push_back(std::make_shared<SceneSMPL>(simulation_config_parser));
      break;
    }
    case PIPELINE::PIPELINE_GLTF: {
      scenes.push_back(std::make_shared<SceneGLTF>(simulation_config_parser));
      break;
    }
    case PIPELINE::PIPELINE_UNIVERSAL: {
      scenes.push_back(std::make_shared<SceneUniversal>(simulation_config_parser));
      break;
    }
    default:
      break;
  }

  engine->SetScenes(scenes);

  //=====================================
  // 3. Run graphics
  //=====================================
  return engine->Run();
}