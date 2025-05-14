#pragma once

#include <string>
#include <vector>
#include <functional>
#if defined(_WIN64) || defined(WIN32) || defined(_WIN32)
#include <memory>
#endif

#include "glm/glm.hpp"

#include <xrtailor/runtime/engine/GameInstance.hpp>
#include <xrtailor/runtime/input/Input.hpp>
#include <xrtailor/runtime/resources/Resource.hpp>
#include <xrtailor/runtime/scene/Actor.hpp>
#include <xrtailor/runtime/input/PlayerController.hpp>
#include <xrtailor/runtime/resources/MaterialProperty.hpp>
#include <xrtailor/runtime/rendering/MeshRenderer.hpp>
#include <xrtailor/runtime/scene/Constants.hpp>
#include <xrtailor/utils/Callback.hpp>
#include <xrtailor/config/SimulationConfigParser.hpp>
#include <xrtailor/utils/FileSystemUtils.hpp>

namespace XRTailor {

class Scene {
 public:
  std::string name = "BaseScene";

  virtual void PopulateActors(GameInstance* game) = 0;

  void ClearCallbacks();

  Callback<void()> on_enter;
  Callback<void()> on_exit;

 protected:
  template <class T>
  void ModifyParameter(T* ptr, T value) {
    on_enter.Register([this, ptr, value]() {
      T prev = *ptr;
      *ptr = value;
      on_exit.Register([ptr, prev, value]() { *ptr = prev; });
    });
  }

  void SpawnCameraAndLight(GameInstance* game);

  void SpawnDebug(GameInstance* game);

  std::shared_ptr<Actor> SpawnDebugger(GameInstance* game);

  std::shared_ptr<Actor> SpawnWatchDogActor(GameInstance* game);

  std::shared_ptr<Actor> SpawnSphere(GameInstance* game, const std::string& name,
                                     glm::vec3 color = glm::vec3(1.0f));

  std::shared_ptr<Actor> SpawnLight(GameInstance* game);

  std::shared_ptr<Actor> SpawnCamera(GameInstance* game);

  std::shared_ptr<Actor> SpawnInfinitePlane(GameInstance* game);

  std::shared_ptr<Actor> SpawnColoredCube(GameInstance* game, glm::vec3 color = glm::vec3(1.0f));

 protected:
  std::shared_ptr<SimulationConfigParser> simulation_config_parser_;
};
}  // namespace XRTailor