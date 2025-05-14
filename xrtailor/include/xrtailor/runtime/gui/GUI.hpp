#pragma once

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <functional>

#include <xrtailor/utils/Callback.hpp>
#include <xrtailor/utils/Timer.hpp>
#include <xrtailor/core/Common.hpp>

#define IMGUI_LEFT_LABEL(func, label, ...) \
  (ImGui::TextUnformatted(label), ImGui::SameLine(), func("##" label, __VA_ARGS__))

namespace XRTailor {
class GUI {
 public:
  static void RegisterDebug(std::function<void()> callback);

  static void RegisterDebugOnce(std::function<void()> callback);

  static void RegisterDebugOnce(const std::string& debug_message);

 public:
  GUI(GLFWwindow* window);

  void OnUpdate();

  void Render();

  void ShutDown();

  void ClearCallback();

 private:
  void CustomizeStyle();

  void ShowSceneWindow();

  void ShowOptionWindow();

  void ShowStatWindow();

  void ShowAnimationWindow();

  void ShowSequenceWindow();

  const ImGuiWindowFlags k_windowFlags_ =
      ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings |
      ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoResize |
      ImGuiWindowFlags_NoCollapse;

  Callback<void()> on_show_debug_info_;
  Callback<void()> on_show_debug_info_once_;

  GLFWwindow* window_ = nullptr;
  int canvas_width_ = 0;
  int canvas_height_ = 0;
  std::string device_name_;
  bool actor_visibilities_[99];
};
}  // namespace XRTailor