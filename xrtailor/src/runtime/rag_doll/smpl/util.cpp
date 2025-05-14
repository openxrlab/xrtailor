#include <fstream>
#include <iostream>

#include <xrtailor/runtime/rag_doll/smpl/util.hpp>
#include <xrtailor/runtime/rag_doll/smpl/util_cnpy.hpp>
#include <xrtailor/core/ErrorDefs.hpp>

namespace smplx {
namespace util {

const char* GenderToStr(Gender gender) {
  switch (gender) {
    case Gender::neutral:
      return "NEUTRAL";
    case Gender::male:
      return "MALE";
    case Gender::female:
      return "FEMALE";
    default:
      return "UNKNOWN";
  }
}

Gender ParseGender(std::string str) {
  for (auto& c : str) {
    c = std::toupper(c);
  }
  if (str == "NEUTRAL") {
    return Gender::neutral;
  }
  if (str == "MALE") {
    return Gender::male;
  }
  if (str == "FEMALE") {
    return Gender::female;
  }
  std::cerr << "WARNING: Gender '" << str << "' could not be parsed\n";
  return Gender::unknown;
}

std::string FindDataFile(const std::string& data_path) {
  static const std::string kTestPath = "data/models/smplx/uv.txt";
  static const int kMaxLevels = 3;
  static std::string data_dir_saved = "../XRTailor/";
  if (data_dir_saved == "\n") {
    data_dir_saved.clear();
    const char* env = std::getenv("SMPLX_DIR");
    if (env != nullptr) {
      // use environmental variable if exists and works
      data_dir_saved = env;

      // auto append slash
      if (!data_dir_saved.empty() && data_dir_saved.back() != '/' &&
          data_dir_saved.back() != '\\') {
        data_dir_saved.push_back('/');
      }

      std::ifstream test_ifs(data_dir_saved + kTestPath);
      if (!test_ifs) {
        data_dir_saved.clear();
      }
    }

    // else check current directory and parents
    if (data_dir_saved.empty()) {
      for (int i = 0; i < kMaxLevels; ++i) {
        std::ifstream test_ifs(data_dir_saved + kTestPath);
        if (test_ifs) {
          break;
        }
        data_dir_saved.append("../");
      }
    }

    data_dir_saved.append("data/");
  }
  return data_dir_saved + data_path;
}

Eigen::Vector3f AutoColor(size_t color_index) {
  static const Eigen::Vector3f kPalette[] = {
      Eigen::Vector3f{1.f, 0.2f, 0.3f},   Eigen::Vector3f{0.3f, 0.2f, 1.f},
      Eigen::Vector3f{0.3f, 1.2f, 0.2f},  Eigen::Vector3f{0.8f, 0.2f, 1.f},
      Eigen::Vector3f{0.7f, 0.7f, 0.7f},  Eigen::Vector3f{1.f, 0.45f, 0.f},
      Eigen::Vector3f{1.f, 0.17f, 0.54f}, Eigen::Vector3f{0.133f, 1.f, 0.37f},
      Eigen::Vector3f{1.f, 0.25, 0.21},   Eigen::Vector3f{1.f, 1.f, 0.25},
      Eigen::Vector3f{0.f, 0.45, 0.9},    Eigen::Vector3f{0.105, 0.522, 1.f},
      Eigen::Vector3f{0.9f, 0.5f, 0.7f},  Eigen::Vector3f{1.f, 0.522, 0.7f},
      Eigen::Vector3f{0.f, 1.0f, 0.8f},   Eigen::Vector3f{0.9f, 0.7f, 0.9f},
  };
  return kPalette[color_index % (sizeof kPalette / sizeof kPalette[0])];
}

Points AutoColorTable(size_t num_colors) {
  Points colors(num_colors, 3);
  for (size_t i = 0; i < num_colors; ++i) {
    colors.row(i) = util::AutoColor(i).transpose();
  }
  return colors;
}

Matrix LoadFloatMatrix(const cnpy::NpyArray& raw, size_t r, size_t c) {
  size_t dwidth = raw.word_size;
  _SMPLX_ASSERT(dwidth == 4 || dwidth == 8);
  if (raw.fortran_order) {
    if (dwidth == 4) {
      return Eigen::template Map<
                 const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(
                 raw.data<float>(), r, c)
          .template cast<Scalar>();
    }
    return Eigen::template Map<
               const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(
               raw.data<double>(), r, c)
        .template cast<Scalar>();
  }
  if (dwidth == 4) {
    return Eigen::template Map<
               const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
               raw.data<float>(), r, c)
        .template cast<Scalar>();
  }
  return Eigen::template Map<
             const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
             raw.data<double>(), r, c)
      .template cast<Scalar>();
}
Eigen::Matrix<Index, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> LoadUintMatrix(
    const cnpy::NpyArray& raw, size_t r, size_t c) {
  size_t dwidth = raw.word_size;
  _SMPLX_ASSERT(dwidth == 4 || dwidth == 8);
  if (raw.fortran_order) {
    if (dwidth == 4) {
      return Eigen::template Map<
                 const Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(
                 raw.data<uint32_t>(), r, c)
          .template cast<Index>();
    }
    return Eigen::template Map<
               const Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(
               raw.data<uint64_t>(), r, c)
        .template cast<Index>();
  }
  if (dwidth == 4) {
    return Eigen::template Map<
               const Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
               raw.data<uint32_t>(), r, c)
        .template cast<Index>();
  }
  return Eigen::template Map<
             const Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
             raw.data<uint64_t>(), r, c)
      .template cast<Index>();
}

void AssertShape(const cnpy::NpyArray& m, std::initializer_list<size_t> shape) {
  _SMPLX_ASSERT_EQ(m.shape.size(), shape.size());
  size_t idx = 0;
  for (auto& dim : shape) {
    if (dim != ANY_SHAPE) {
      _SMPLX_ASSERT_EQ(m.shape[idx], dim);
    }
    ++idx;
  }
}

}  // namespace util
}  // namespace smplx
