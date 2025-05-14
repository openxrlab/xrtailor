#pragma once

#include <string>

#include "json/json.h"

#include <xrtailor/config/Base.hpp>

namespace XRTailor {
class SimulationConfigParser {
 public:
  SimulationConfigParser();

  bool LoadFromJson(const std::string& path);

  void Apply();

  std::string name;

  int pipeline_type;
  int solver_mode;

  SMPLPipelineSettings smpl_pipeline_settings;
  GLTFPipelineSettings gltf_pipeline_settings;
  UniversalPipelineSettings universal_pipeline_settings;

  AnimationSettings animation_settings;

  SwiftModeSettings swift_mode_settings;
  QualityModeSettings quality_mode_settings;

 private:
  bool ParsePipelineSMPL(const Json::Value& root);
  bool ParsePipelineGLTF(const Json::Value& root);
  bool ParsePipelineUniversal(const Json::Value& root);
  bool ParseSwift(const Json::Value& root);
  bool ParseQuality(const Json::Value& root);
};
}  // namespace XRTailor