#pragma once

#include <xrtailor/core/Global.hpp>

#include <functional>

#include <xrtailor/runtime/engine/GameInstance.hpp>
#include <xrtailor/runtime/scene/Component.hpp>
#include <xrtailor/runtime/rendering/Light.hpp>
#include <xrtailor/runtime/input/Input.hpp>
#include <xrtailor/runtime/scene/Camera.hpp>
#include <xrtailor/utils/Helper.hpp>
#include <xrtailor/utils/Timer.hpp>

#include "helper_math.h"

namespace XRTailor {

class PlayerController : public Component {
 public:
  PlayerController() { name = __func__; }

  void Start() override {
    LOG_TRACE("Starting {0}'s component: {1}", actor->name, name);
    Global::game->on_mouse_move.Register(on_mouse_move);
    Global::game->on_god_update.Register(on_god_update);
  }

  static void on_god_update() {
    const auto& camera = Global::camera;

    if (camera) {
      const auto& trans = camera->transform();
      const float speed_scalar = Global::Config::camera_translate_speed;

      static glm::vec3 current_speed(0);
      glm::vec3 target_speed(0);

      if (Global::input->GetKey(GLFW_KEY_W))
        target_speed += camera->Front();
      else if (Global::input->GetKey(GLFW_KEY_S))
        target_speed -= camera->Front();

      if (Global::input->GetKey(GLFW_KEY_A))
        target_speed -= glm::normalize(glm::cross(camera->Front(), camera->Up()));
      else if (Global::input->GetKey(GLFW_KEY_D))
        target_speed += glm::normalize(glm::cross(camera->Front(), camera->Up()));

      if (Global::input->GetKey(GLFW_KEY_Q))
        target_speed += camera->Up();
      else if (Global::input->GetKey(GLFW_KEY_E))
        target_speed -= camera->Up();

      current_speed = Helper::Lerp(current_speed, target_speed, Timer::DeltaTime() * 10);
      trans->position +=
          current_speed * speed_scalar * Timer::DeltaTime() * Global::sim_params.player_speed;
    } else {
      LOG_ERROR("Camera not found.");
    }
  }

  static void on_mouse_scroll(double x_offset, double y_offset) {
    auto camera = Global::camera;
    camera->zoom -= static_cast<float>(y_offset);
    if (camera->zoom < 1.0f)
      camera->zoom = 1.0f;
    if (camera->zoom > 45.0f)
      camera->zoom = 45.0f;
  }

  static void on_mouse_move(double x_pos, double y_pos) {
    static float last_x = Global::Config::screen_width / 2,
                 last_y = Global::Config::screen_height / 2;

    bool should_rotate = Global::input->GetMouse(GLFW_MOUSE_BUTTON_RIGHT);

    if (should_rotate) {
      auto rot = Global::camera->transform()->rotation;
      float yaw = -rot.y, pitch = rot.x;

      float x_offset = static_cast<float>(x_pos) - last_x;
      float y_offset = last_y - static_cast<float>(y_pos);
      x_offset *= Global::Config::camera_rotate_sensitivity;
      y_offset *= Global::Config::camera_rotate_sensitivity;
      yaw += x_offset;
      pitch = clamp(pitch + y_offset, -89.0f, 89.0f);

      Global::camera->transform()->rotation = glm::vec3(pitch, -yaw, 0);
    }
    last_x = static_cast<float>(x_pos);
    last_y = static_cast<float>(y_pos);
  }
};
}  // namespace XRTailor