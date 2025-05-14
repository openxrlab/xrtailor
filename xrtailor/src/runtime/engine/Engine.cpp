#include <xrtailor/runtime/engine/Engine.hpp>

#include <algorithm>

#include <xrtailor/runtime/scene/Scene.hpp>
#include <xrtailor/runtime/gui/GUI.hpp>
#include <xrtailor/runtime/engine/GameInstance.hpp>
#include <xrtailor/runtime/input/Input.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


using namespace XRTailor;

void PrintGlfwError(int error, const char* description) {
  LOG_ERROR("Error(Glfw): Code({}), {}", error, description);
}

Engine::Engine() {
  Global::engine = this;
  // setup glfw
  glfwInit();
  if (Global::engine_config.headless_simulation) {
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
  }
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  // Multi-sample Anti-aliasing
  glfwWindowHint(GLFW_SAMPLES, 4);

  window_ = glfwCreateWindow(Global::Config::screen_width, Global::Config::screen_height, "Tailor",
                              nullptr, nullptr);

  if (window_ == nullptr) {
    LOG_ERROR("Failed to create GLFW window");
    glfwTerminate();
    return;
  }
  glfwMakeContextCurrent(window_);

  // set the swap interval for the current OpenGL context
  // https://www.glfw.org/docs/3.3/group__context.html#ga6d4e0cdf151b5e579bd67f13202994ed
  glfwSwapInterval(1);
  //glfwSwapInterval(0);

  glfwSetFramebufferSizeCallback(window_, [](GLFWwindow* window_, int width, int height) {
    glViewport(0, 0, width, height);
  });

  //glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
  glfwSetCursorPosCallback(window_, [](GLFWwindow* window_, double xpos, double ypos) {
    Global::game->ProcessMouse(window_, xpos, ypos);
  });
  glfwSetScrollCallback(window_, [](GLFWwindow* window_, double xoffset, double yoffset) {
    Global::game->ProcessScroll(window_, xoffset, yoffset);
  });
  glfwSetErrorCallback(PrintGlfwError);

  // setup opengl
  if (gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)) == 0) {
    LOG_ERROR("Failed to initialize GLAD");
    return;
  }
  glViewport(0, 0, Global::Config::screen_width, Global::Config::screen_height);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK);

  // setup stbi
  stbi_set_flip_vertically_on_load(1);

  // setup members
  gui_ = std::make_shared<GUI>(window_);
  input_ = std::make_shared<Input>(window_);
}

Engine::~Engine() {
  gui_->ShutDown();
  glfwTerminate();
}

void Engine::SetScenes(const std::vector<std::shared_ptr<Scene>>& initializers) {
  scenes = initializers;
}

int Engine::Run() {
  do {
#pragma warning(push)
#pragma warning(disable : 4129)
    LOG_TRACE("Run Engine");
#pragma warning(pop)

    game_ = std::make_shared<GameInstance>(window_, gui_);
    scene_index = next_scene_index_;
    scenes[scene_index]->PopulateActors(game_.get());
    scenes[scene_index]->on_enter.Invoke();
    game_->Run();
    scenes[scene_index]->on_exit.Invoke();
    scenes[scene_index]->ClearCallbacks();

    Resource::ClearCache();
    gui_->ClearCallback();
  } while (game_->pending_reset);

  return EXIT_FAILURE;
}

void Engine::Reset() {
  game_->pending_reset = true;
}

void Engine::SwitchScene(unsigned int scene_index) {
  next_scene_index_ = std::clamp(scene_index, 0u, static_cast<uint>(scenes.size()) - 1);
  game_->pending_reset = true;
}

glm::ivec2 Engine::WindowSize() {
  glm::ivec2 result;
  glfwGetWindowSize(window_, &result.x, &result.y);
  return result;
}
