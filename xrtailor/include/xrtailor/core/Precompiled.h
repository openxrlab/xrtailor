#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <queue>
#include <algorithm>
#include <numeric>
#include <functional>
#include <memory>
#include <tuple>
#include <chrono>
#include <random>
#include <limits>

// 文件系统支持（跨平台）
#if defined(_WIN64) || defined(WIN32) || defined(_WIN32)
#include <filesystem>
namespace filesystem = std::filesystem;
#else
#include <experimental/filesystem>
namespace filesystem = std::experimental::filesystem;
#endif