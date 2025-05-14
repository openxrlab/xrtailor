#include <xrtailor/utils/FileSystemUtils.hpp>
#include <xrtailor/core/Global.hpp>
#include <random>

#include <iostream>
#include <ctime>

#if defined(_WIN64) || defined(WIN32) || defined(_WIN32)
#include <direct.h>
#else
#include <unistd.h>
#endif  //

namespace XRTailor {
filesystem::path GetClothTemplateDirectorySMPL() {
  filesystem::path res(Global::engine_config.asset_directory);
  const std::string& body_model = Global::sim_config.smpl.body_model;
  res.append("Garment");
  if (body_model == "SMPL" || body_model == "SMPLH") {
    res.append("SMPLH");  // smpl and smplh shares the same template
  } else {
    res.append(body_model);
  }

  res.append("Template");

  return filesystem::absolute(res);
}

filesystem::path GetClothTemplateDirectoryGLTF(std::string character_id) {
  filesystem::path res(Global::engine_config.asset_directory);
  res.append("Garment");
  res.append(character_id);
  res.append("Template");

  return filesystem::absolute(res);
}

filesystem::path GetClothConfigDirectorySMPL() {
  filesystem::path res(Global::engine_config.asset_directory);
  res.append("Garment");
  const std::string& body_model = Global::sim_config.smpl.body_model;
  if (body_model == "SMPL" || body_model == "SMPLH") {
    res.append("SMPLH");
  } else {
    res.append(body_model);
  }

  res.append("Config");

  return filesystem::absolute(res);
}

filesystem::path GetClothConfigDirectoryGLTF(std::string character_id) {
  filesystem::path res(Global::engine_config.asset_directory);
  res.append("Garment");
  res.append(character_id);
  res.append("Config");

  return filesystem::absolute(res);
}

filesystem::path GetBodyTemplateDirectory() {
  filesystem::path res(Global::engine_config.asset_directory);
  res.append("Body");
  res.append("Template");

  return filesystem::absolute(res);
}

filesystem::path GetBodyModelDirectory() {
  filesystem::path res(Global::engine_config.asset_directory);
  res.append("Body");
  res.append("Model");

  return filesystem::absolute(res);
}

filesystem::path GetHelperDirectory() {
  filesystem::path res(Global::engine_config.asset_directory);
  res.append("Helper");

  return filesystem::absolute(res);
}

filesystem::path GetTextureDirectory() {
  filesystem::path res(Global::engine_config.asset_directory);
  res.append("Texture");

  return filesystem::absolute(res);
}

filesystem::path GetShaderDirectory() {
  filesystem::path res(Global::engine_config.asset_directory);
  res.append("Shader");

  return filesystem::absolute(res);
}

std::string GetFileNameFromPath(std::string path) {
  filesystem::path p(path);
  return p.stem().string();
}

static constexpr char kCch[] = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

std::string MakeRandomStr(int sz, bool printable) {
  std::string ret;
  ret.resize(sz);
  std::mt19937 rng(std::random_device{}());
  for (int i = 0; i < sz; ++i) {
    if (printable) {
      uint32_t x = rng() % (sizeof(kCch) - 1);
      ret[i] = kCch[x];
    } else {
      ret[i] = rng() % 0xFF;
    }
  }

  return ret;
}

std::string GetFormattedTime() {
  std::time_t t = time(nullptr);
  char tmp[128] = {NULL};
  std::strftime(tmp, sizeof(tmp), "%Y-%m-%d-%H-%M-%S", std::localtime(&t));

  return std::string(tmp);
}

void DeleteFolderContents(const filesystem::path& folder_path) {
  try {
    if (filesystem::exists(folder_path) && filesystem::is_directory(folder_path)) {
      for (const auto& entry : filesystem::directory_iterator(folder_path)) {
        filesystem::remove_all(entry.path());  // Recursively delete each item
      }
      std::cout << "All contents deleted from: " << folder_path << std::endl;
    } else {
      std::cerr << "Provided path is not a directory or does not exist." << std::endl;
    }
  } catch (const filesystem::filesystem_error& e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
}

void DeleteFilesOnly(const filesystem::path& folder_path) {
  try {
    if (filesystem::exists(folder_path) && filesystem::is_directory(folder_path)) {
      for (const auto& entry : filesystem::directory_iterator(folder_path)) {
        if (filesystem::is_regular_file(entry.path())) {
          filesystem::remove(entry.path());  // Delete only files
        }
      }
      std::cout << "All files deleted from: " << folder_path << std::endl;
    } else {
      std::cerr << "Provided path is not a directory or does not exist." << std::endl;
    }
  } catch (const filesystem::filesystem_error& e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
}

}  // namespace XRTailor