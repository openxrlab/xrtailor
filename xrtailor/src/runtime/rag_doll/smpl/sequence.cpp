#include <xrtailor/runtime/rag_doll/smpl/sequence.hpp>

#include <fstream>
#include <iostream>
#include <cnpy.h>

#include <xrtailor/runtime/rag_doll/smpl/util.hpp>
#include <xrtailor/runtime/rag_doll/smpl/util_cnpy.hpp>
#include <xrtailor/core/ErrorDefs.hpp>
namespace smplx {

namespace {
using util::AssertShape;
}  // namespace

// AMASS SMPLH_G npz structure
// 'trans':           (#frames, 3)
// 'gender':          str
// 'mocap_framerate': float
// 'betas':           (16)              first 10 are from the usual SMPL
// 'dmpls':           (#frames, 8)      soft tissue
// 'poses':           (#frames, 156)    first 66 are SMPL joint parameters
// excluding hand.
//                                      last 90 are MANO joint parameters, which
//                                          correspond to last 90 joints in
//                                          SMPL-X (NOT hand PCA)
//
template <class SequenceConfig>
Sequence<SequenceConfig>::Sequence(const std::string& path) {
  if (static_cast<unsigned int>(!path.empty()) != 0u) {
    Load(path);
  } else {
    n_frames = 0;
    gender = Gender::neutral;
  }
}

template <class SequenceConfig>
void Sequence<SequenceConfig>::Load(const std::string& path) {
  if (!std::ifstream(path)) {
    n_frames = 0;
    gender = Gender::unknown;
    std::cerr << "Sequence '" << path << "' does not exist.\n";
    exit(TAILOR_EXIT::SEQUENCE_NOT_FOUND);
  }
  // ** READ NPZ **
  cnpy::npz_t npz = cnpy::npz_load(path);

  cnpy::NpyArray trans_raw = cnpy::npz_load(path, "trans");

  AssertShape(trans_raw, {util::ANY_SHAPE, 3});
  n_frames = trans_raw.shape[0];
  trans = util::LoadFloatMatrix(trans_raw, n_frames, 3);  // (N, 3)

  cnpy::NpyArray poses_raw = cnpy::npz_load(path, "poses");
  AssertShape(poses_raw, {n_frames, SequenceConfig::n_pose_params()});
  pose = util::LoadFloatMatrix(poses_raw, n_frames, SequenceConfig::n_pose_params());

  cnpy::NpyArray shape_raw = cnpy::npz_load(path, "betas");
  AssertShape(shape_raw, {SequenceConfig::n_shape_params()});
  shape = util::LoadFloatMatrix(shape_raw, SequenceConfig::n_shape_params(), 1);

  if (SequenceConfig::n_dmpls() && npz.count("dmpls") == 1) {
    cnpy::NpyArray dmpls_raw = npz["dmpls"];
    AssertShape(dmpls_raw, {n_frames, SequenceConfig::n_dmpls()});
    dmpls = util::LoadFloatMatrix(poses_raw, n_frames, SequenceConfig::n_dmpls());
  }

  if (npz.count("gender") != 0u) {
    char gender_spec = npz["gender"].data_holder->at(0);
    gender = gender_spec == 'f'   ? Gender::female
             : gender_spec == 'm' ? Gender::male
             : gender_spec == 'n' ? Gender::neutral
                                  : Gender::unknown;
  } else {
    // Default to neutral
    std::cerr << "WARNING: gender not present in '" << path << "', using neutral\n";
    gender = Gender::neutral;
  }

  if (npz.count("mocap_framerate") != 0u) {
    auto& mocap_frate_raw = npz["mocap_framerate"];
    if (mocap_frate_raw.word_size == 8) {
      frame_rate = *mocap_frate_raw.data<double>();
    } else if (mocap_frate_raw.word_size == 4) {
      frame_rate = *mocap_frate_raw.data<float>();
    }
  } else {
    // Reasonable default
    std::cerr << "WARNING: mocap_framerate not present in '" << path << "', assuming 120 FPS\n";
    frame_rate = 120.f;
  }
}

Gender ParseGender(const std::string& path) {
  Gender gender = Gender::unknown;
  if (!std::ifstream(path)) {
    gender = Gender::unknown;
    std::cerr << "Sequence '" << path << "' does not exist.\n";
    exit(TAILOR_EXIT::SEQUENCE_NOT_FOUND);
  }
  // ** READ NPZ **
  cnpy::npz_t npz = cnpy::npz_load(path);

  if (npz.count("gender") != 0u) {
    char gender_spec = npz["gender"].data_holder->at(0);
    gender = gender_spec == 'f'   ? Gender::female
             : gender_spec == 'm' ? Gender::male
             : gender_spec == 'n' ? Gender::neutral
                                  : Gender::unknown;
  } else {
    // Default to neutral
    std::cerr << "WARNING: gender not present in '" << path << "', using neutral\n";
    gender = Gender::neutral;
  }

  return gender;
}

// Instantiation
template class Sequence<sequence_config::AMASS_SMPLH_G>;
template class Sequence<sequence_config::AMASS_SMPLX_G>;

}  // namespace smplx
