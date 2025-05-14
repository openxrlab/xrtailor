#include <xrtailor/runtime/gui/GUI.hpp>

#include <xrtailor/runtime/engine/Engine.hpp>
#include <xrtailor/runtime/scene/Scene.hpp>

#include <xrtailor/utils/FileSystemUtils.hpp>
#include <sstream>

using namespace XRTailor;

#define SHORTCUT_BOOL(key, variable)  \
  if (Global::input->GetKeyDown(key)) \
  variable = !variable

inline GUI* g_gui;
const float kLeftWindowWidth = 250.0f;
const float kRightWindowWidth = 330.0f;

template <typename T>
auto ToStringWithPrecision(T val, const int n = 6) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(n) << val;
  return oss.str();
}

void HelpMarker(const char* desc) {
  ImGui::SameLine();
  ImGui::TextDisabled("(?)");
  if (ImGui::IsItemHovered()) {
    ImGui::BeginTooltip();
    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
    ImGui::TextUnformatted(desc);
    ImGui::PopTextWrapPos();
    ImGui::EndTooltip();
  }
}

struct SolverTiming {
  int count = 0;

  std::vector<std::string> labels = {
      "Total",
  };

  std::unordered_map<std::string, double> label_to_time;
  std::unordered_map<std::string, double> label_to_avg_time;

  ImVec4 color_high = ImVec4(1.000f, 0.244f, 0.000f, 1.000f);
  ImVec4 color_mid = ImVec4(1.000f, 0.602f, 0.000f, 1.000f);
  ImVec4 color_low = ImVec4(1.000f, 0.889f, 0.000f, 1.000f);
  ImVec4 color_disabled = ImVec4(0.5f, 0.5f, 0.5f, 1.000f);

  void DisplayKernelTiming(const std::string name, bool auto_color = true) {
    double total = label_to_avg_time["Total"];
    bool should_pop = false;
    float percentage = (total > 0) ? float(label_to_avg_time[name] / total * 100) : 0.0f;
    if (auto_color) {
      if (percentage > 10 || percentage == 0.0f) {
        ImVec4 text_color = color_low;
        if (percentage > 30) {
          text_color = color_high;
        } else if (percentage > 10) {
          text_color = color_mid;
        } else if (percentage == 0.0f) {
          text_color = color_disabled;
        }
        ImGui::PushStyleColor(ImGuiCol_Text, text_color);
        should_pop = true;
      }
    }

    ImGui::TableNextColumn();
    ImGui::Text(name.c_str());
    ImGui::TableNextColumn();
    ImGui::Text("%.2f ms", label_to_time[name]);
    ImGui::TableNextColumn();
    ImGui::Text("%.2f ms", label_to_avg_time[name] / count);
    ImGui::TableNextColumn();
    ImGui::Text("%.2f %%", percentage);

    if (should_pop) {
      ImGui::PopStyleColor();
    }
  }

  void Update() {
    if (Timer::PeriodicUpdate("GUI_SOLVER", 0.2f)) {
      if (Timer::FrameCount() < 2) {
        count = 0;
        for (const auto& label : labels) {
          label_to_avg_time[label] = 0;
        }
        label_to_avg_time["KernelSum"] = 0;
      }

      label_to_time["KernelSum"] = 0;
      for (const auto& label : labels) {
        label_to_time[label] = Timer::GetTimerGPU("Solver_" + label);
        label_to_avg_time[label] += label_to_time[label];

        if (label != "Total" && label != "Initialize") {
          label_to_time["KernelSum"] += label_to_time[label];
        }
      }

      label_to_avg_time["KernelSum"] += label_to_time["KernelSum"];
      count++;
    }
  }

  void OnGUI() {
    // Solver timing statistics panel
    if (!ImGui::CollapsingHeader("Solver timing", ImGuiTreeNodeFlags_DefaultOpen)) {
      Global::game_state.detail_timer = false;
      return;
    }
    Global::game_state.detail_timer = true;

    HelpMarker("solver_total = kernel_sum + cuda_dispatch_time");

    if (ImGui::BeginTable("timing", 4))
    {
      ImGui::TableSetupColumn("Kernel");
      ImGui::TableSetupColumn("Time (ms)");
      ImGui::TableSetupColumn("Avg (ms)");
      ImGui::TableSetupColumn("%");
      ImGui::TableHeadersRow();

      for (int i = 0; i < labels.size() - 1; i++) {
        auto& label = labels[i];
        DisplayKernelTiming(label);
      }
      //DisplayKernelTiming(std::string("KernelSum"), false);
      DisplayKernelTiming(labels[labels.size() - 1], false);

      ImGui::EndTable();
    }
  }
};

struct PerformanceStat {
  float delta_time = 0;
  int frame_rate = 0;
  int frameCount = 0;
  int physics_frame_count = 0;

  float graph_values[180] = {};
  int graph_index = 0;
  float graph_average = 0.0f;

  double cpu_time = 0;
  double gpu_time = 0;
  double solver_time = 0;

  void Update() {
    if (Global::game_state.pause) {
      return;
    }

    const auto& game = Global::game;
    float elapsed_time = Timer::ElapsedTime();
    float delta_time_miliseconds = Timer::DeltaTime() * 1000;

    frameCount = Timer::FrameCount();
    physics_frame_count = Timer::PhysicsFrameCount();

    if (Timer::PeriodicUpdate("GUI_FAST", Timer::FixedDeltaTime())) {
      graph_values[graph_index] = static_cast<float>(Timer::GetTimerGPU("Solver_Total"));
      graph_index = (graph_index + 1) % IM_ARRAYSIZE(graph_values);
    }

    if (Timer::PeriodicUpdate("GUI_SLOW", 0.3f)) {
      delta_time = delta_time_miliseconds;
      frame_rate = elapsed_time > 0 ? static_cast<int>(frameCount / elapsed_time) : 0;
      cpu_time = Timer::GetTimer("CPU_TIME") * 1000;
      gpu_time = Timer::GetTimer("GPU_TIME") * 1000;
      solver_time = Timer::GetTimerGPU("Solver_Total");

      for (int n = 0; n < IM_ARRAYSIZE(graph_values); n++) {
        graph_average += graph_values[n];
      }
      graph_average /= static_cast<float> IM_ARRAYSIZE(graph_values);
    }
  }

  void OnGUI() {
    if (ImGui::BeginTable("stat", 2, ImGuiTableFlags_SizingStretchProp)) {
      ImGui::TableNextColumn();
      ImGui::Text("Render Frame: ");
      ImGui::TableNextColumn();
      ImGui::Text("%d", frameCount);
      ImGui::TableNextColumn();
      ImGui::Text("Physics Frame: ");
      ImGui::TableNextColumn();
      ImGui::Text("%d", physics_frame_count);
      ImGui::TableNextColumn();
      ImGui::Text("Render FrameRate: ");
      ImGui::TableNextColumn();
      ImGui::Text("%d FPS", frame_rate);
      ImGui::TableNextColumn();
      ImGui::Text("CPU time: ");
      ImGui::TableNextColumn();
      ImGui::Text("%.2f ms", cpu_time);
      ImGui::TableNextColumn();
      ImGui::Text("GPU time: ");
      ImGui::TableNextColumn();
      ImGui::Text("%.2f ms", gpu_time);
      HelpMarker("gpu_time = solver_time + cuda_synchronize_time");
      ImGui::TableNextColumn();
      ImGui::Text("Num Particles: ");
      ImGui::TableNextColumn();
      ImGui::Text("%d", Global::sim_params.num_overall_particles);
      ImGui::EndTable();
    }

    ImGui::Dummy(ImVec2(0, 5));
    ImGui::PushItemWidth(-FLT_MIN);
    std::string overlay =
        "Solver: " + ToStringWithPrecision<double>(solver_time, 2) + " ms(" +
        ToStringWithPrecision<double>(solver_time > 0 ? (1000.0 / solver_time) : 0, 2) + " FPS)";

    ImGui::PlotLines("##", graph_values, IM_ARRAYSIZE(graph_values), graph_index, overlay.c_str(), 0,
                     graph_average * 2.0f, ImVec2(0, 80.0f));
    ImGui::Dummy(ImVec2(0, 5));
  }
};

void GUI::RegisterDebug(std::function<void()> callback) {
  g_gui->on_show_debug_info_.Register(callback);
}

void GUI::RegisterDebugOnce(std::function<void()> callback) {
  g_gui->on_show_debug_info_once_.Register(callback);
}

void GUI::RegisterDebugOnce(const std::string& debug_message) {
  g_gui->on_show_debug_info_once_.Register([debug_message]() { ImGui::Text(debug_message.c_str()); });
}

GUI::GUI(GLFWwindow* window) {
  g_gui = this;
  window_ = window;

  // Setup Dear ImGui context
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  (void)io;
  io.IniFilename = nullptr;

  filesystem::path font_path(Global::engine_config.asset_directory);
  font_path.append("DroidSans.ttf");
  io.Fonts->AddFontFromFileTTF(font_path.string().c_str(), 19);

  // Setup Dear ImGui style
  CustomizeStyle();

  // Setup Platform/Renderer backends
  const char* glsl_version = "#version 330";
  ImGui_ImplGlfw_InitForOpenGL(window_, true);
  ImGui_ImplOpenGL3_Init(glsl_version);

  device_name_ = std::string((char*)glGetString(GL_RENDERER));
  device_name_ = device_name_.substr(0, device_name_.find("/"));

  for (int i = 0; i < 99; i++) {
    actor_visibilities_[i] = true;
  }
}

void GUI::OnUpdate() {
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  glfwGetWindowSize(window_, &canvas_width_, &canvas_height_);

  ShowSceneWindow();

  ShowOptionWindow();
  ShowAnimationWindow();
  ShowStatWindow();

  ShowSequenceWindow();
}

void GUI::Render() {
  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void GUI::ClearCallback() {
  on_show_debug_info_.Clear();
  on_show_debug_info_once_.Clear();
}

void GUI::ShutDown() {
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
}

void GUI::CustomizeStyle() {
  ImGui::StyleColorsDark();

  auto style = &ImGui::GetStyle();
  style->SelectableTextAlign = ImVec2(0, 0.5);
  style->WindowPadding = ImVec2(10, 12);
  style->WindowRounding = 6;
  style->GrabRounding = 8;
  style->FrameRounding = 6;
  style->WindowTitleAlign = ImVec2(0.5, 0.5);

  style->Colors[ImGuiCol_WindowBg] = ImVec4(0.06f, 0.06f, 0.06f, 0.7f);
  style->Colors[ImGuiCol_TitleBg] = style->Colors[ImGuiCol_WindowBg];
  style->Colors[ImGuiCol_TitleBgActive] = style->Colors[ImGuiCol_TitleBg];
  style->Colors[ImGuiCol_SliderGrab] = ImVec4(0.325f, 0.325f, 0.325f, 1.0f);
  style->Colors[ImGuiCol_FrameBg] = ImVec4(0.114f, 0.114f, 0.114f, 1.0f);
  style->Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.2f, 0.2f, 0.2f, 1.0f);
  style->Colors[ImGuiCol_Button] = ImVec4(0.46f, 0.46f, 0.46f, 0.46f);
  style->Colors[ImGuiCol_CheckMark] = ImVec4(0.851f, 0.851f, 0.851f, 1.0f);

  style->Colors[ImGuiCol_TableBorderLight] = ImVec4(1.0f, 1.0f, 1.0f, 0.3f);
  style->Colors[ImGuiCol_TableBorderStrong] = ImVec4(1.0f, 1.0f, 1.0f, 0.6f);
}

void GUI::ShowSceneWindow() {
  ImGui::SetNextWindowSize(ImVec2(kLeftWindowWidth, (canvas_height_ - 60.0f) * 0.4f));
  ImGui::SetNextWindowPos(ImVec2(20, 20));
  ImGui::Begin("Scene", nullptr, k_windowFlags_);

  const auto& scenes = Global::engine->scenes;

  for (unsigned int i = 0; i < scenes.size(); i++) {
    auto scene = scenes[i];
    auto label = scene->name;
    if (ImGui::Selectable(label.c_str(), Global::engine->scene_index == i, 0, ImVec2(0, 28))) {
      Global::engine->SwitchScene(i);
    }
  }

  ImGui::End();
}

void GUI::ShowOptionWindow() {
  ImGui::SetNextWindowSize(ImVec2(kLeftWindowWidth, (canvas_height_ - 60.0f) * 0.6f));
  ImGui::SetNextWindowPos(ImVec2(20, 40 + (canvas_height_ - 60.0f) * 0.4f));

  ImGui::Begin("Options", nullptr, k_windowFlags_);

  ImGui::PushItemWidth(-FLT_MIN);

  if (ImGui::Button("Reset (R)", ImVec2(-FLT_MIN, 0))) {
    Global::engine->Reset();
  }
  ImGui::Dummy(ImVec2(0.0f, 5.0f));

  {
    static bool radio = false;
    ImGui::Checkbox("Pause (P, O)", &Global::game_state.pause);
    Global::input->ToggleOnKeyDown(GLFW_KEY_P, Global::game_state.pause);
    ImGui::Checkbox("Draw Particles (K)", &Global::game_state.draw_particles);
    Global::input->ToggleOnKeyDown(GLFW_KEY_K, Global::game_state.draw_particles);
    ImGui::Checkbox("Draw Wireframe (L)", &Global::game_state.render_wireframe);
    Global::input->ToggleOnKeyDown(GLFW_KEY_L, Global::game_state.render_wireframe);
    ImGui::Dummy(ImVec2(0.0f, 10.0f));
  }

  if (ImGui::CollapsingHeader("Common", ImGuiTreeNodeFlags_DefaultOpen)) {
    Global::sim_params.OnShared();
  }

  if (Global::sim_params.solver_mode == SOLVER_MODE::SWIFT &&
      ImGui::CollapsingHeader("Swift", ImGuiTreeNodeFlags_DefaultOpen)) {
    Global::sim_params.OnSwift();
  }

  if (Global::sim_params.solver_mode == SOLVER_MODE::QUALITY &&
      ImGui::CollapsingHeader("Quality", ImGuiTreeNodeFlags_DefaultOpen)) {
    Global::sim_params.OnQuality();
  }

  ImGui::End();
}

void GUI::ShowSequenceWindow() {
  ImGui::SetNextWindowSize(ImVec2(kRightWindowWidth * 1.1f, (canvas_height_ - 60.0f) * 0.4f));
  ImGui::SetNextWindowPos(ImVec2(20 + kLeftWindowWidth * 1.1f, 20));
  ImGui::Begin("Objects", nullptr, k_windowFlags_);

  auto actors = Global::game->GetActors();
  int n_actor = actors.size();
  for (int i = 0; i < n_actor; i++)
  {
    auto actor = actors[i];
    auto mesh_renderer = actor->GetComponent<MeshRenderer>();
    ImGui::Dummy(ImVec2(0, 2));
    ImGui::PushItemWidth(-FLT_MIN);
    if (mesh_renderer != nullptr) {
      if (ImGui::Checkbox(actor->name.c_str(), &actor_visibilities_[i])) {
        mesh_renderer->enabled = actor_visibilities_[i];
      }
    } else {
      ImGui::Text(actor->name.c_str());
    }

    auto components = actor->components;
    for (auto& component : components) {
      ImGui::Text((std::string(" |-- ") + component->name).c_str());
    }
    ImGui::Dummy(ImVec2(0, 2));
  }

  ImGui::End();
}

void GUI::ShowAnimationWindow() {
  ImGui::SetNextWindowSize(ImVec2(kRightWindowWidth * 1.1f, (canvas_height_ - 60.0f) * 0.2f));
  ImGui::SetNextWindowPos(
      ImVec2(20 + kLeftWindowWidth * 1.1f, 40 + (canvas_height_ - 60.0f) * 0.4f));
  ImGui::Begin("Animation", nullptr, k_windowFlags_);

  ImGui::PushItemWidth(-FLT_MIN);

  Global::sim_params.OnAnimationGUI();
}

void GUI::ShowStatWindow() {
  ImGui::SetNextWindowSize(ImVec2(kRightWindowWidth * 1.1f, 0));
  ImGui::SetNextWindowPos(ImVec2(canvas_width_ - kRightWindowWidth * 1.1f - 20, 20.0f));
  ImGui::Begin("Statistics", nullptr, k_windowFlags_);
  ImGui::Text("Device:  %s", device_name_.c_str());

  static PerformanceStat stat;
  stat.Update();
  stat.OnGUI();

  static SolverTiming solver_timing;
  solver_timing.Update();
  solver_timing.OnGUI();

  if (!on_show_debug_info_.Empty() || !on_show_debug_info_once_.Empty()) {
    if (ImGui::CollapsingHeader("Debug", ImGuiTreeNodeFlags_DefaultOpen)) {
      on_show_debug_info_.Invoke();
      on_show_debug_info_once_.Invoke();

      if (!Global::game_state.pause) {
        on_show_debug_info_once_.Clear();
      }
    }
  }

  ImGui::End();
}