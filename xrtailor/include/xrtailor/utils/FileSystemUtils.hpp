#pragma once

#if defined(_WIN64) || defined(WIN32) || defined(_WIN32)
#include <filesystem>
namespace filesystem = std::filesystem;
#else
#include <experimental/filesystem>
namespace filesystem = std::experimental::filesystem;
#endif

namespace XRTailor {
filesystem::path GetClothConfigDirectorySMPL();

filesystem::path GetClothConfigDirectoryGLTF(std::string character_id);

filesystem::path GetClothTemplateDirectorySMPL();

filesystem::path GetClothTemplateDirectoryGLTF(std::string character_id);

filesystem::path GetBodyTemplateDirectory();

filesystem::path GetBodyModelDirectory();

filesystem::path GetHelperDirectory();

filesystem::path GetTextureDirectory();

filesystem::path GetShaderDirectory();

std::string GetFileNameFromPath(std::string path);

std::string MakeRandomStr(int sz, bool printable = true);

std::string GetFormattedTime();

void DeleteFolderContents(const filesystem::path& folder_path);

void DeleteFilesOnly(const filesystem::path& folder_path);
}  // namespace XRTailor