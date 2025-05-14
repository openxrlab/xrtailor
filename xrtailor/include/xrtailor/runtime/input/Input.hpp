#pragma once

#include <xrtailor/core/Global.hpp>

namespace XRTailor {
class Input {
 public:
  Input(GLFWwindow* window);

  void OnUpdate();

  // Returns true while the user holds down the key.
  bool GetKey(int key);

  // Returns true during the frame the user starts pressing down the key.
  bool GetKeyDown(int key);

  void ToggleOnKeyDown(int key, bool& variable);

  // Returns true during the frame the user releases the key.
  bool GetKeyUp(int key);

  bool GetMouse(int button);

  bool GetMouseDown(int button);

  bool GetMouseUp(int button);

  glm::vec2 GetMousePos() {
    double x, y;
    glfwGetCursorPos(window_, &x, &y);
    return glm::vec2(x, y);
  }

 private:
  GLFWwindow* window_;
  char key_once_[GLFW_KEY_LAST + 1];
  char key_now_[GLFW_KEY_LAST + 1];
};
}  // namespace XRTailor