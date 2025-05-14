#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <memory>

#include <xrtailor/core/Common.hpp>

namespace XRTailor {
class Scene;
class GUI;
class GameInstance;
class Input;

class Engine {
 public:
  Engine();
  ~Engine();

  int Run();

  void Reset();
  void SwitchScene(uint _scene_index);
  void SetScenes(const std::vector<std::shared_ptr<Scene>>& initializers);

  glm::ivec2 WindowSize();

  std::vector<std::shared_ptr<Scene>> scenes;
  uint scene_index = 0;  // current scene index
 private:
  uint next_scene_index_ = 0;
  GLFWwindow* window_ = nullptr;
  std::shared_ptr<GUI> gui_;
  std::shared_ptr<GameInstance> game_;
  std::shared_ptr<Input> input_;
};
}  // namespace XRTailor