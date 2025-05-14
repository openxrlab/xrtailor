#include <xrtailor/runtime/input/Input.hpp>

using namespace XRTailor;

Input::Input(GLFWwindow* window) {
  window_ = window;
  Global::input = this;
  memset(key_once_, 0, sizeof(key_once_));
  memset(key_now_, 0, sizeof(key_now_));
}

void Input::OnUpdate() {
  std::swap(key_once_, key_now_);
}

// Returns true while the user holds down the key.
bool Input::GetKey(int key) {
  return (glfwGetKey(window_, key) == GLFW_PRESS);
}

// Returns true during the frame the user starts pressing down the key.
bool Input::GetKeyDown(int key) {
  key_now_[key] = static_cast<char>(GetKey(key));
  return (key_once_[key] == 0) && (key_now_[key] != 0);
}

void Input::ToggleOnKeyDown(int key, bool& variable) {
  if (GetKeyDown(key)) {
    variable = !variable;
  }
}

// Returns true during the frame the user releases the key.
bool Input::GetKeyUp(int key) {
  key_now_[key] = static_cast<char>(GetKey(key));
  return (key_once_[key] != 0) && (key_now_[key] == 0);
}

bool Input::GetMouse(int button) {
  int state = glfwGetMouseButton(window_, button);
  return state == GLFW_PRESS;
}

bool Input::GetMouseDown(int button) {
  key_now_[button] = static_cast<char>(GetMouse(button));
  return (key_once_[button] == 0) && (key_now_[button] != 0);
}

bool Input::GetMouseUp(int button) {
  key_now_[button] = static_cast<char>(GetMouse(button));
  return (key_once_[button] != 0) && (key_now_[button] == 0);
}
