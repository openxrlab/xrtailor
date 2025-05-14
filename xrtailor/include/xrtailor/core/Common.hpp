#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "imgui.h"
#include <functional>
#include <vector>

#include <xrtailor/core/Scalar.hpp>
#include <xrtailor/config/Base.hpp>
#include <xrtailor/core/ErrorDefs.hpp>

#define IMGUI_LEFT_LABEL(func, label, ...) \
  (ImGui::TextUnformatted(label), ImGui::SameLine(), func("##" label, __VA_ARGS__))

// Only initialize value on host.
// Since CUDA doesn't allow dynamics initialization,
// we use this macro to ignore initialization when compiling with NVCC.
#ifdef __CUDA_ARCH__
#define HOST_INIT(val)
#else
#define HOST_INIT(val) = val
#endif

//#define DEBUG_MODE

namespace XRTailor {

// may change on each frame update
struct SimParams {
  // Number of steps that execute within a frame
  int num_substeps HOST_INIT(1);
  // Number of solver iterations to perform per-substep
  int num_iterations HOST_INIT(100);
  int max_num_neighbors HOST_INIT(64);
  // The magnitude of particle velocity will be clamped to this value at the end of each step
  float max_speed HOST_INIT(999.0f);
  int num_lerped_frames HOST_INIT(0);
  // Constant acceleration applied to all particles
  Vector3 gravity HOST_INIT(Vector3(0, -9.8f, 0));
  // Viscous drag force, applies a force proportional, and opposite to the particle velocity
  float damping HOST_INIT(0.98f);
  // Control the convergence rate of the Jacobi solver, default: 1, values greater than 1 may lead to instability
  float relaxation_factor HOST_INIT(0.25f);
  float long_range_stretchiness HOST_INIT(1.00f);

  // Distance of particles maintain against shapes, should be greater than 0
  float collision_margin HOST_INIT(0.01f);
  // Coefficient of friction used when colliding against shapes
  float sdf_friction HOST_INIT(0.1f);
  bool enable_self_collision HOST_INIT(true);

  // Total number of particles
  uint num_particles HOST_INIT(0u);
  uint num_edges HOST_INIT(0u);
  uint num_faces HOST_INIT(0u);
  uint num_attached_slots HOST_INIT(0u);
  uint num_skinned_slots HOST_INIT(0u);
  // The maximum interaction radius for particle
  float particle_diameter HOST_INIT(0.0f);

  float delta_time HOST_INIT(0.0f);

  uint num_clothes HOST_INIT(0u);
  uint num_obstacles HOST_INIT(0u);
  uint num_obstacle_vertices HOST_INIT(0u);
  uint num_obstacle_edges HOST_INIT(0u);
  uint num_obstacle_faces HOST_INIT(0u);
  uint num_overall_particles HOST_INIT(0);

  float player_speed HOST_INIT(0.5f);

  int pre_simulation_frame_index HOST_INIT(0);
  int pre_simulation_frames HOST_INIT(120);
  int frame_index HOST_INIT(0);
  int num_frames HOST_INIT(0);

  bool record_obstacle HOST_INIT(false);
  bool record_cloth HOST_INIT(false);
  bool update_obstacle_animation HOST_INIT(true);
  bool imminent_repulsion HOST_INIT(true);
  bool pbd_repulsion HOST_INIT(true);

  bool geodesic_LRA HOST_INIT(true);

  bool solve_bending HOST_INIT(true);

  float bvh_tolerance HOST_INIT(0.001f);

  bool draw_obstacle_normals HOST_INIT(true);
  bool draw_cloth_normals HOST_INIT(true);
  bool draw_obstacle_aabbs HOST_INIT(true);
  bool draw_internal_nodes HOST_INIT(true);
  bool draw_external_nodes HOST_INIT(true);

  int solver_mode HOST_INIT(0);
  int pipeline HOST_INIT(0);

  int num_collision_passes HOST_INIT(10);

  bool icm_enable HOST_INIT(false);
  float icm_h0 HOST_INIT(1e-4f);
  float icm_g0 HOST_INIT(10.5f);
  int icm_iters HOST_INIT(10);

  // whether the clothes have been exported
  bool cloth_exported[5];
  // whether the obstacles have been exported
  bool obstacle_exported[5];

  // predictive contact
  float scr HOST_INIT(0.002f);
  float radius HOST_INIT(0.002f);

  uint frame_rate HOST_INIT(60);

  void OnDebuggerGUI() {
    IMGUI_LEFT_LABEL(ImGui::SliderFloat, "Player Speed", &player_speed, 0.01f, 1.0f);

    IMGUI_LEFT_LABEL(ImGui::InputFloat, "BVH torlence", &bvh_tolerance, 0.0f, 0.01f);

    IMGUI_LEFT_LABEL(ImGui::Checkbox, "ICM Enable", &icm_enable);
    IMGUI_LEFT_LABEL(ImGui::InputFloat, "ICM h0", &icm_h0, 0.0f, 0.01f);
    IMGUI_LEFT_LABEL(ImGui::InputFloat, "ICM g0", &icm_g0, 0.0f, 1.0f);
    IMGUI_LEFT_LABEL(ImGui::InputInt, "ICM iters", &icm_iters, 1, 100);

    IMGUI_LEFT_LABEL(ImGui::Checkbox, "Obstacle Normals", &draw_obstacle_normals);
    IMGUI_LEFT_LABEL(ImGui::Checkbox, "Obstacle AABBs", &draw_obstacle_aabbs);
    IMGUI_LEFT_LABEL(ImGui::Checkbox, "Internal Nodes", &draw_internal_nodes);
    IMGUI_LEFT_LABEL(ImGui::Checkbox, "External Nodes", &draw_external_nodes);
    IMGUI_LEFT_LABEL(ImGui::Checkbox, "Cloth Normals", &draw_cloth_normals);
  }

  void OnAnimationGUI() {
    int idx = frame_index > 0 ? frame_index - 1 : 0;
    ImGui::Text("Frame %i (%i total)", idx, num_frames);
    IMGUI_LEFT_LABEL(ImGui::SliderInt, "Frame", &idx, 0, num_frames - 1);
    IMGUI_LEFT_LABEL(ImGui::Checkbox, "Update Obstacle Animation", &update_obstacle_animation);
    IMGUI_LEFT_LABEL(ImGui::Checkbox, "Record Obstacle", &record_obstacle);
    IMGUI_LEFT_LABEL(ImGui::Checkbox, "Record Cloth", &record_cloth);
  }

  void OnShared() {
    IMGUI_LEFT_LABEL(ImGui::SliderInt, "Num Substeps", &num_substeps, 1, 20);
    IMGUI_LEFT_LABEL(ImGui::SliderInt, "Num Iterations", &num_iterations, 1, 200);
    ImGui::Separator();
    IMGUI_LEFT_LABEL(ImGui::SliderFloat3, "Gravity", (float*)&gravity, -50, 50);

    IMGUI_LEFT_LABEL(ImGui::SliderFloat, "Damping", &damping, 0, 10.0f);

    ImGui::Separator();
    IMGUI_LEFT_LABEL(ImGui::Checkbox, "Enable Bending", &solve_bending);
    IMGUI_LEFT_LABEL(ImGui::SliderFloat, "Long Range Stretch", &long_range_stretchiness, 1.0, 2.0,
                     "%.3f");
  }

  void OnSwift() {
    IMGUI_LEFT_LABEL(ImGui::SliderFloat, "Collision Margin", &collision_margin, 0, 0.5);
    IMGUI_LEFT_LABEL(ImGui::Checkbox, "Enable Self Collision", &enable_self_collision);
    IMGUI_LEFT_LABEL(ImGui::SliderFloat, "Relaxation Factor", &relaxation_factor, 0, 2.0);
  }

  void OnQuality() {
    IMGUI_LEFT_LABEL(ImGui::Checkbox, "Imminent Repulsion", &imminent_repulsion);
    IMGUI_LEFT_LABEL(ImGui::Checkbox, "PBD Repulsion", &pbd_repulsion);
  }
};

struct GameState {
  bool step = false;  // execute one step
  bool pause = false;  // pause simulation
  bool render_wireframe = false;
  bool render_obstacle = true;
  bool draw_particles = false;
  bool hide_gui = false;
  bool detail_timer = false;

  bool actor_visibilities[99];
};

// the value of configs remain unchanged once the simulation config has been loaded
struct EngineConfig {
  std::string log_path HOST_INIT("./log/");
  int log_level HOST_INIT(0);
  bool headless_simulation HOST_INIT(false);
  std::string asset_directory HOST_INIT("./Assets/");
  int max_frame_rate HOST_INIT(240);
};

struct SimConfig {
  SMPLPipelineSettings smpl;
  GLTFPipelineSettings gltf;
  UniversalPipelineSettings universal;
  AnimationSettings animation;
  QualityModeSettings quality;
  SwiftModeSettings swift;
};

} // namespace XRTailor

