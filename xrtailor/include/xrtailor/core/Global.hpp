#pragma once

#include <vector>
#include <string>

#include <xrtailor/core/Common.hpp>

namespace XRTailor {
class GameInstance;
class Camera;
class Light;
class Input;
class Engine;

namespace Global {
inline Engine* engine;
inline GameInstance* game;
inline Camera* camera;
inline Input* input;
inline std::vector<Light*> lights;

inline GameState game_state;
inline SimParams sim_params;
inline EngineConfig engine_config;
inline SimConfig sim_config;

namespace Config {
// Controls how fast the camera moves
const float camera_translate_speed = 2.5f;
const float camera_rotate_sensitivity = 0.15f;

// 1k
const uint screen_width = 1920;
const uint screen_height = 1080;

// 2k
//const uint screen_width = 2560;
//const uint screen_height = 1440;

// 4k
//const uint screen_width = 3840;
//const uint screen_height = 2160;

const uint shadow_width = 1024;
const uint shadow_height = 1024;
}  // namespace Config
}  // namespace Global
}  // namespace XRTailor