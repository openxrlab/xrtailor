#include <xrtailor/runtime/engine/GameInstance.hpp>

#include <memory>
#include <functional>
#include <ctime>

#include <xrtailor/utils/Helper.hpp>
#include <xrtailor/runtime/scene/Camera.hpp>
#include <xrtailor/runtime/input/Input.hpp>
#include <xrtailor/runtime/rendering/RenderPipeline.hpp>
#include <xrtailor/runtime/gui/GUI.hpp>
#include <xrtailor/utils/Timer.hpp>
#include <xrtailor/runtime/engine/Engine.hpp>
#include <xrtailor/runtime/resources/Resource.hpp>
#include <xrtailor/runtime/rendering/Light.hpp>
#include <xrtailor/runtime/scene/Actor.hpp>

namespace XRTailor {

GameInstance::GameInstance(GLFWwindow* window, std::shared_ptr<GUI> gui) {
  Global::game = this;

  // setup members
  window_ = window;
  gui_ = gui;
  render_pipeline_ = std::make_shared<RenderPipeline>();
  timer_ = std::make_shared<Timer>();

  Timer::StartTimer("GAME_INSTANCE_INIT");
}

std::shared_ptr<Actor> GameInstance::AddActor(std::shared_ptr<Actor> actor) {
  actors_.push_back(actor);
  return actor;
}

std::shared_ptr<Actor> GameInstance::CreateActor(const std::string& name) {
  auto actor = std::make_shared<Actor>(name);
  return AddActor(actor);
}

std::shared_ptr<Actor> GameInstance::FindActor(const std::string& name) {
  for (auto actor : actors_) {
    if (actor->GetName() == name) {
      return actor;
    }
  }
  return nullptr;
}

std::vector<std::shared_ptr<Actor>> GameInstance::GetActors() {
  return actors_;
}

int GameInstance::Run() {
  LOG_TRACE("Total actors: {}", actors_.size());
  LOG_TRACE(" - XRTailor");
  for (auto actor : actors_) {
    LOG_TRACE("   + {}", actor->name);
    for (auto component : actor->components) {
      LOG_TRACE("   |-- {}", component->name);
    }
  }

  Initialize();
  MainLoop();
  Finalize();

  return 0;
}

unsigned int GameInstance::DepthFrameBuffer() {
  return render_pipeline_->depth_tex;
}

glm::ivec2 XRTailor::GameInstance::WindowSize() {
  glm::ivec2 result;
  glfwGetWindowSize(window_, &result.x, &result.y);
  return result;
}

bool XRTailor::GameInstance::WindowMinimized() {
  auto size = WindowSize();
  return (size.x < 1 || size.y < 1);
}

void GameInstance::ProcessMouse(GLFWwindow* window_, double xpos, double ypos) {
  on_mouse_move.Invoke(xpos, ypos);
}

void GameInstance::ProcessScroll(GLFWwindow* window_, double x_offset, double y_offset) {
  on_mouse_scroll.Invoke(x_offset, y_offset);
}

void GameInstance::ProcessKeyboard(GLFWwindow* window_) {
  Global::input->ToggleOnKeyDown(GLFW_KEY_H, Global::game_state.hide_gui);

  if (Global::input->GetKey(GLFW_KEY_ESCAPE)) {
    glfwSetWindowShouldClose(window_, 1);
  }
  if (Global::input->GetKeyDown(GLFW_KEY_N)) {
    Global::game_state.step = true;
    Global::game_state.pause = false;
  }

  if (Global::input->GetKeyDown(GLFW_KEY_R)) {
    Global::engine->Reset();
  }
}

void GameInstance::Initialize() {
  for (const auto& go : actors_) {
    go->Start();
  }
}

void GameInstance::MainLoop() {
  double init_time = Timer::EndTimer("GAME_INSTANCE_INIT") * 1000;
  LOG_TRACE("Initialization success within {:.2f} ms. Enter main loop.", init_time);

  // render loop
  while ((glfwWindowShouldClose(window_) == 0) && !pending_reset) {
    if (WindowMinimized()) {
      glfwPollEvents();
      continue;
    }
    // Input
    ProcessKeyboard(window_);

    // Init
    glClearColor(sky_color.x, sky_color.y, sky_color.z, sky_color.w);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glPolygonMode(GL_FRONT_AND_BACK, Global::game_state.render_wireframe ? GL_LINE : GL_FILL);

    Timer::StartTimer("CPU_TIME");
    Timer::UpdateDeltaTime();

    // Logic Updates
    if (!Global::game_state.hide_gui) {
      gui_->OnUpdate();
    }

    if (!Global::game_state.pause) {
      //printf("[GameInstance] MainLoop, %d\n", Global::game_state.pause);
      Timer::NextFrame();
      // execute one physics frame
      if (Timer::NextFixedFrame()) {
        for (const auto& go : actors_) {
          go->FixedUpdate();
        }

        on_animation_update.Invoke();

        if (Global::game_state.step) {
          Global::game_state.pause = true;
          Global::game_state.step = false;
        }
      }

      for (const auto& go : actors_) {
        go->Update();
      }
    }

    Global::input->OnUpdate();

    on_god_update.Invoke();

    Timer::EndTimer("CPU_TIME");

    // Render
    render_pipeline_->Render();
    if (!Global::game_state.hide_gui) {
      gui_->Render();
    }

    // Check and call events and swap the buffers
    glfwSwapBuffers(window_);
    glfwPollEvents();
  }
}

void GameInstance::Finalize() {
  for (const auto& go : actors_) {
    go->OnDestroy();
  }
}

}  // namespace XRTailor