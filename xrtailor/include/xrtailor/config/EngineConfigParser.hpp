#pragma once

#include <string>

namespace XRTailor {
class EngineConfigParser {
 public:
  EngineConfigParser();

  bool LoadFromJson(const std::string& path);

  void Apply();

 private:
  std::string log_path_;
  int log_level_;
  bool headless_simulation_;
  std::string asset_directory_;
};
}  // namespace XRTailor