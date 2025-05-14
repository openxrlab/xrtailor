#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <memory>
#include <chrono>

#include <xrtailor/runtime/scene/Component.hpp>
#include <xrtailor/runtime/scene/Actor.hpp>
#include <xrtailor/utils/Callback.hpp>
#include <xrtailor/core/Common.hpp>

namespace XRTailor {
class Light;
class RenderPipeline;
class GUI;
class Timer;

class GameInstance {
 public:
  GameInstance(GLFWwindow* window, std::shared_ptr<GUI> gui);
  GameInstance(const GameInstance&) = delete;

  std::shared_ptr<Actor> AddActor(std::shared_ptr<Actor> actor);
  std::shared_ptr<Actor> CreateActor(const std::string& name);

  int Run();

  void ProcessMouse(GLFWwindow* window_, double xpos, double ypos);
  void ProcessScroll(GLFWwindow* window_, double xoffset, double yoffset);
  void ProcessKeyboard(GLFWwindow* window_);

  template <class T>
  std::enable_if_t<std::is_base_of<Component, T>::value, std::vector<T*>> FindComponents() {
    std::vector<T*> result;

    for (auto actor : actors_) {
      auto component = actor->template GetComponents<T>();
      if (component.size() > 0) {
        result.insert(result.end(), component.begin(), component.end());
      }
    }
    return result;
  }

  std::shared_ptr<Actor> FindActor(const std::string& name);

  std::vector<std::shared_ptr<Actor>> GetActors();

 public:
  uint DepthFrameBuffer();
  glm::ivec2 WindowSize();
  bool WindowMinimized();

  Callback<void(double, double)> on_mouse_scroll;
  Callback<void(double, double)> on_mouse_move;
  Callback<void()> on_animation_update;
  Callback<void()> on_god_update;  // invoke when main logic is paused (for debugging purpose)
  Callback<void()> on_finalize;

  bool pending_reset = false;
  glm::vec4 sky_color = glm::vec4(0.0f);

 private:
  void Initialize();
  void MainLoop();
  void Finalize();

 private:
  GLFWwindow* window_ = nullptr;
  std::shared_ptr<GUI> gui_;
  std::shared_ptr<Timer> timer_;

  std::vector<std::shared_ptr<Actor>> actors_;
  std::shared_ptr<RenderPipeline> render_pipeline_;
};
}  // namespace XRTailor