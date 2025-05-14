#pragma once

#include <vector>
#include <string>

#include <xrtailor/core/Scalar.hpp>
#include <xrtailor/config/UniversalActorConfig.hpp>

namespace XRTailor {

enum EXTEND_MODE : int { BOUNDARY, NEIGHBOR, NONMANIFOLD_EDGES, UV_ISLAND };

enum SOLVER_MODE : int { QUALITY, SWIFT, UNKNOWN };

enum PIPELINE : int { PIPELINE_SMPL, PIPELINE_GLTF, PIPELINE_UNIVERSAL, PIPELINE_UNKNOWN };

enum EXPORT_FORMAT : int { ALEMBIC, OBJ_SEQUENCE };

enum STRETCH_MODE : int { BASIC_STRETCH, FEM_STRAIN };

enum BENDING_MODE : int { BASIC_BENDING, FEM_ISOMETRIC };

struct SkinParam {
  int idx0;
  int idx1;
  int idx2;
  float u;
  float v;
  float w;
};

struct BindingParam {
  uint idx;
  float stiffness;
  float distance;
};

struct BasicFabricSettings {
  float stretch_compliance{0.0f};
  float bend_compliance{1e2f};
  float relaxation_factor{0.25f};
  float long_range_stretchiness{1.1f};
  bool geodesic_LRA{false};
  bool solve_bending{true};
};

struct FEMFabricSettings {
  float xx_stiffness{1.0f};
  float xy_stiffness{1.0f};
  float yy_stiffness{1.0f};
  float xy_poisson_ratio{0.3f};
  float yx_poisson_ratio{0.3f};
  bool solve_bending{true};
  float bending_stiffness{1e0f};
  float long_range_stretchiness{1.1f};
  bool geodesic_LRA{true};
};

struct ParticleCollisionSettings {
  float friction{0.1f};
  int max_neighbor_size{64};
  int inter_leaved_hash{3};
  bool enable_self_collision{true};
  float particle_diameter{1.3f};
  float hash_cell_size{1.5f};
};

struct SMPLPipelineSettings {
  std::vector<std::string> cloth_styles;
  std::string npz_path;
  std::string body_model;
  int num_lerped_frames;
  bool enable_pose_blendshape{false};
  bool enable_collision_filter{false};
  float amass_x_rotation{-90.0f};
};

struct GLTFPipelineSettings {
  std::vector<std::string> cloth_styles;
  std::string character_name;
  std::string gltf_path;
  int num_lerped_frames;
};

struct UniversalPipelineSettings {
  std::vector<UniversalActorConfig> clothes;
  std::vector<UniversalActorConfig> obstacles;
  int num_frames{static_cast<int>(1e5)};
};

struct SolverSettings {
  int num_substeps{1};
  int num_iterations{200};
  float max_speed{1e6f};
  Vector3 gravity{0.0f, -9.8f, 0.0f};
  float damping{0.98f};
};

struct AnimationSettings {
  int num_pre_simulation_frames;
  bool record_obstacle;
  bool record_cloth;
  int export_format;
  std::string export_directory;
  int target_frame_rate;
};

struct RepulsionSettings {
  bool enable_imminent_repulsion{true};
  float imminent_thickness{1e-3f};
  float relaxation_rate{0.25f};
  bool enable_pbd_repulsion{true};
  float pbd_thickness{1e-4f};
};

struct ImpactZoneSettings {
  float obstacle_mass{1e3f};
  float thickness{1e-4f};
};

struct QualityModeSettings {
  SolverSettings solver;
  FEMFabricSettings fabric;
  RepulsionSettings repulsion;
  ImpactZoneSettings impact_zone;
};

struct SwiftModeCollisionSettings {
  int num_collision_passes{10};
  float sdf_collision_margin{1e-2};
  float bvh_tolerance{0.003};
  ParticleCollisionSettings self_contact;
};

struct SwiftModeSettings {
  SolverSettings solver;
  BasicFabricSettings fabric;
  SwiftModeCollisionSettings collision;
};

}  // namespace XRTailor