#include <xrtailor/config/EngineConfigParser.hpp>

#include <fstream>
#include <json/json.h>

#include "xrtailor/core/Common.hpp"
#include "xrtailor/core/Global.hpp"

namespace XRTailor {
EngineConfigParser::EngineConfigParser() = default;

bool EngineConfigParser::LoadFromJson(const std::string& path) {
  std::cout << "Reading engine config from: " << path << std::endl;

  std::ifstream file(path, std::ios::binary);

  if (!file.is_open()) {
    std::cerr << "Error opening file:" << path << std::endl;
    return false;
  }

  Json::Reader reader;
  Json::Value root;

  if (reader.parse(file, root, false)) {
    this->log_path_ = root["LOG_PATH"].asString();
    this->log_level_ = root["LOG_LEVEL"].asInt();
    this->headless_simulation_ = root["HEADLESS_SIMULATION"].asBool();
    this->asset_directory_ = root["ASSET_DIRECTORY"].asString();
  } else {
    std::cerr << "Failed to parse file." << std::endl;
    return false;
  }
  return true;
}

void EngineConfigParser::Apply() {
  Global::engine_config.log_path = this->log_path_;
  Global::engine_config.log_level = this->log_level_;
  Global::engine_config.headless_simulation = this->headless_simulation_;
  Global::engine_config.asset_directory = this->asset_directory_;

  if (!filesystem::exists(this->log_path_)) {
    filesystem::create_directory(this->log_path_);
  }
}
}  // namespace XRTailor