#pragma once
#ifndef SMPLX_SEQUENCE_9512D947_1D9B_478F_AAB5_6E6A846A6828
#define SMPLX_SEQUENCE_9512D947_1D9B_478F_AAB5_6E6A846A6828

#include <xrtailor/runtime/rag_doll/smpl/internal/sequence_model_spec.hpp>
#include <xrtailor/runtime/rag_doll/smpl/sequence_config.hpp>
#include <xrtailor/runtime/rag_doll/smpl/smplx.hpp>
namespace smplx {

// Note: SequenceModelSpec is in smplx/internal/sequence_model_spec.hpp

// An AMASS-compatible body pose+translation[+DMPL] sequence
// with overall shape and gender information
// SequenceConfig: pick from smplx::sequence_config::*
// (currently only AMASS available)
template <class SequenceConfig>
class Sequence {
 public:
  // Create sequence and load from AMASS-like .npz, with fields:
  // trans, gender (optional), mocap_framerate (optional),
  // betas, dmpls, poses.
  // If path is empty, constructs empty sequence (n_frames = 0).
  explicit Sequence(const std::string& path = "");

  // Load sequence from AMASS-like .npz, with fields:
  // trans, gender (optional), mocap_framerate (optional),
  // betas, dmpls, poses
  void Load(const std::string& path);

  // Set body shape
  template <class ModelConfig>
  inline void set_shape(Body<ModelConfig>& body) {
    internal::SequenceModelSpec<SequenceConfig, ModelConfig>::set_shape(*this, body);
  }

  template <class ModelConfig>
  inline void set_shape(float pc0, float pc1, Body<ModelConfig>& body) {
    internal::SequenceModelSpec<SequenceConfig, ModelConfig>::set_shape(pc0, pc1, body);
  }
  // Set body pose
  template <class ModelConfig>
  inline void set_pose(Body<ModelConfig>& body, size_t frame) {
    internal::SequenceModelSpec<SequenceConfig, ModelConfig>::set_pose(*this, body, frame);
  }

  // * METADATA
  // Number of frames in sequence
  size_t n_frames;

  // Mocap frame rate
  double frame_rate;
  // Gender, may be unknown
  Gender gender = Gender::unknown;

  // * BODY DATA
  // Extended shape parameters (betas)
  Eigen::Matrix<Scalar, SequenceConfig::n_shape_params(), 1> shape;

  // Root translations
  Eigen::Matrix<Scalar, Eigen::Dynamic, 3, Eigen::RowMajor> trans;

  // Pose parameters
  Eigen::Matrix<Scalar, Eigen::Dynamic, SequenceConfig::n_pose_params(), Eigen::RowMajor> pose;

  // DMPLs
  Eigen::Matrix<Scalar, Eigen::Dynamic, SequenceConfig::n_dmpls(), Eigen::RowMajor> dmpls;
};

Gender ParseGender(const std::string& path);

// An AMASS sequence
//using SequenceAMASS = Sequence<sequence_config::AMASS>;
using SequenceAMASS_SMPLH_G = Sequence<sequence_config::AMASS_SMPLH_G>;
using SequenceAMASS_SMPLX_G = Sequence<sequence_config::AMASS_SMPLX_G>;
}  // namespace smplx

#endif  // ifndef SMPLX_SEQUENCE_9512D947_1D9B_478F_AAB5_6E6A846A6828
