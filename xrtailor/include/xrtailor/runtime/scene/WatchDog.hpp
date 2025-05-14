#pragma once

#include <iostream>
#include <memory>
#include <string>

#include <xrtailor/runtime/scene/Component.hpp>

namespace XRTailor {

class WatchDog : public Component {
 public:
  WatchDog() { SET_COMPONENT_NAME; }

  void FixedUpdate() override {
    bool should_exit = true;

    for (int i = 0; i < Global::sim_params.num_clothes; i++) {
      if (Global::sim_params.record_cloth) {
        should_exit &= Global::sim_params.cloth_exported[i];
      }
    }

    for (int i = 0; i < Global::sim_params.num_obstacles; i++) {
      if (Global::sim_params.record_obstacle) {
        should_exit &= Global::sim_params.obstacle_exported[i];
      }
    }

    // when running universal pipeline, prevent early exit even if the cloth/obstacle is not recorded
    if (Global::sim_params.pipeline == PIPELINE::PIPELINE_UNIVERSAL) {
      if (!Global::sim_params.record_cloth && !Global::sim_params.record_obstacle)
        should_exit = false;
    }

    if (should_exit) {
      LOG_INFO("Simulation done, exit engine.");
      exit(TAILOR_EXIT::SUCCESS);
    }
  }
};

}  // namespace XRTailor