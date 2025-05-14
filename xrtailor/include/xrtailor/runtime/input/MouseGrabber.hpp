#pragma once

#include "glm/glm.hpp"

#include <xrtailor/runtime/scene/Component.hpp>
#include <xrtailor/utils/Helper.hpp>
#include <xrtailor/core/Global.hpp>
#include <xrtailor/runtime/input/Input.hpp>
#include <xrtailor/runtime/scene/Actor.hpp>
#include <xrtailor/utils/Timer.hpp>
#include <xrtailor/runtime/engine/Engine.hpp>
#include <xrtailor/runtime/scene/Camera.hpp>

#include <xrtailor/physics/PhysicsMesh.cuh>

namespace XRTailor {
struct Ray {
  glm::vec3 origin;
  glm::vec3 direction;
};

struct RaycastCollision {
  bool collide = false;
  int object_index;
  float distance_to_origin;
};

class MouseGrabber {
 public:
  void Initialize(std::shared_ptr<PhysicsMesh> physics_mesh) { physics_mesh_ = physics_mesh; }

  void HandleMouseInteraction() {
    bool should_pick_object = Global::input->GetMouseDown(GLFW_MOUSE_BUTTON_LEFT);
    if (should_pick_object) {
      Ray ray = GetMouseRay();
      ray_collision_ = FindClosestVertexToRay(ray);

      if (ray_collision_.collide) {
        is_grabbing_ = true;
        grabbed_vertex_mass_ =
            static_cast<float>(physics_mesh_->GetInvMass(ray_collision_.object_index));
        physics_mesh_->SetInvMass(0, ray_collision_.object_index);
      }
    }

    bool should_release_object = Global::input->GetMouseUp(GLFW_MOUSE_BUTTON_LEFT);
    if (should_release_object && is_grabbing_) {
      is_grabbing_ = false;
      physics_mesh_->SetInvMass(grabbed_vertex_mass_, ray_collision_.object_index);
      checkCudaErrors(cudaDeviceSynchronize());
    }
  }

  void UpdateGrappedVertex() {
    if (is_grabbing_) {
      Ray ray = GetMouseRay();
      glm::vec3 mouse_pos = ray.origin + ray.direction * ray_collision_.distance_to_origin;
      int id = ray_collision_.object_index;
      auto cur_pos = static_cast<glm::vec3>(physics_mesh_->GetX0(id));
      glm::vec3 target = Helper::Lerp(mouse_pos, cur_pos, 0.8f);
      physics_mesh_->SetX0(target, id);
      physics_mesh_->SetVelocity(static_cast<Vector3>((target - cur_pos) / Timer::FixedDeltaTime()),
                                 id);
      checkCudaErrors(cudaDeviceSynchronize());
    }
  }

 private:
  bool is_grabbing_ = false;
  float grabbed_vertex_mass_ = 0;
  RaycastCollision ray_collision_;

  std::shared_ptr<PhysicsMesh> physics_mesh_;

  RaycastCollision FindClosestVertexToRay(Ray ray) {
    int result = -1;
    float min_distance_to_view = FLT_MAX;

    thrust::host_vector<Vector3> h_positions = physics_mesh_->HostPositions();
    for (int i = 0; i < h_positions.size(); i++) {
      const auto& position = static_cast<glm::vec3>(h_positions[i]);
      float distance_to_view = glm::dot(ray.direction, position - ray.origin);
      float distance_to_ray = glm::length(glm::cross(ray.direction, position - ray.origin));

      if (distance_to_ray < Global::sim_params.particle_diameter &&
          distance_to_view < min_distance_to_view) {
        result = i;
        min_distance_to_view = distance_to_view;
      }
    }
    return RaycastCollision{result >= 0, result, min_distance_to_view};
  }

  Ray GetMouseRay() {
    glm::vec2 screen_pos = Global::input->GetMousePos();
    // [0, 1]
    auto window_size = Global::engine->WindowSize();
    auto normalized_screen_pos = 2.0f * screen_pos / glm::vec2(window_size.x, window_size.y) - 1.0f;
    normalized_screen_pos.y = -normalized_screen_pos.y;

    glm::mat4 inv_vp = glm::inverse(Global::camera->Projection() * Global::camera->View());
    glm::vec4 near_point_raw = inv_vp * glm::vec4(normalized_screen_pos, 0, 1);
    glm::vec4 far_point_raw = inv_vp * glm::vec4(normalized_screen_pos, 1, 1);

    glm::vec3 near_point =
        glm::vec3(near_point_raw.x, near_point_raw.y, near_point_raw.z) / near_point_raw.w;
    glm::vec3 far_point =
        glm::vec3(far_point_raw.x, far_point_raw.y, far_point_raw.z) / far_point_raw.w;
    glm::vec3 direction = glm::normalize(far_point - near_point);

    return Ray{near_point, direction};
  }
};
}  // namespace XRTailor