#pragma once

#include <iostream>
#include <unordered_map>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <xrtailor/core/Common.hpp>


namespace XRTailor {

class Timer {
 public:
  Timer() {
    s_timer_ = this;

    last_update_time_ = static_cast<float>(CurrentTime());
    fixed_update_timer_ = static_cast<float>(CurrentTime());
  }

  ~Timer() {
    for (const auto& label2events : cuda_events_) {
      for (auto& e : label2events.second) {
        cudaEventDestroy(e);
      }
    }
  }

  static void StartTimer(const std::string& label) { s_timer_->times_[label] = CurrentTime(); }

  // Returns elapsed time from StartTimer in seconds.
  // When called multiple time during one frame, result gets accumulated.
  static double EndTimer(const std::string& label, int frame = -1) {
    double time = CurrentTime() - s_timer_->times_[label];
    if (frame == -1) {
      frame = s_timer_->frame_count_;
    }

    if (s_timer_->times_.count(label)) {
      if (frame > s_timer_->frames_[label]) {
        s_timer_->history_[label] = time;
      } else {
        s_timer_->history_[label] += time;
      }
      s_timer_->frames_[label] = frame;
      return s_timer_->history_[label];
    } else {
      return -1;
    }
  }

  // returns time in seconds
  static double GetTimer(const std::string& label) {
    if (s_timer_->history_.count(label)) {
      return s_timer_->history_[label];
    } else {
      return 0;
    }
  }

  static double CurrentTime() { return glfwGetTime(); }

 public:
  static void StartTimerGPU(const std::string& label) {
    int frame = s_timer_->frame_count_;

    if (s_timer_->frames_.count(label) && s_timer_->frames_[label] != frame) {
      GetTimerGPU(label);
    }
    s_timer_->frames_[label] = frame;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    auto& events = s_timer_->cuda_events_[label];
    events.push_back(start);
    events.push_back(end);
    cudaEventRecord(start);
  }

  static void EndTimerGPU(const std::string& label) {
    const auto& events = s_timer_->cuda_events_[label];
    auto stop = events[events.size() - 1];
    cudaEventRecord(stop);
  }

  // return time in mili seconds
  static double GetTimerGPU(const std::string& label) {
    auto& events = s_timer_->cuda_events_[label];
    if (events.size() > 0) {
      auto lastEvent = events[events.size() - 1];
      cudaEventSynchronize(lastEvent);

      float total_time = 0.0f;
      for (int i = 0; i < events.size(); i += 2) {
        float time;
        cudaEventElapsedTime(&time, events[i], events[i + 1]);
        total_time += time;
        cudaEventDestroy(events[i]);
        cudaEventDestroy(events[i + 1]);
      }
      events.clear();
      s_timer_->history_[label] = total_time;
    }

    if (s_timer_->history_.count(label)) {
      return s_timer_->history_[label];
    } else {
      return 0;
    }
  }

 public:
  static void UpdateDeltaTime() {
    float current = (float)CurrentTime();
    s_timer_->delta_time_ = std::min(current - s_timer_->last_update_time_, 0.2f);
    s_timer_->last_update_time_ = current;
  }

  static void NextFrame() {
    s_timer_->frame_count_++;
    s_timer_->elapsed_time_ += s_timer_->delta_time_;
  }

  // Return true when fixed update should be executed
  static bool NextFixedFrame() {
    s_timer_->fixed_update_timer_ += s_timer_->delta_time_;

    if (s_timer_->fixed_update_timer_ > s_timer_->fixed_delta_time_) {
      s_timer_->fixed_update_timer_ = 0;
      s_timer_->physics_frame_count_++;
      return true;
    }
    return false;
  }

  static bool PeriodicUpdate(const std::string& label, float interval,
                             bool allow_repetition = true) {
    auto& l2t = s_timer_->label2accumulated_time;
    if (!l2t.count(label)) {
      l2t[label] = 0;
    }

    if (l2t[label] < s_timer_->elapsed_time_) {
      l2t[label] = allow_repetition ? l2t[label] + interval : s_timer_->elapsed_time_ + interval;
      return true;
    }
    return false;
  }

  static auto FrameCount() { return s_timer_->frame_count_; }

  static auto PhysicsFrameCount() { return s_timer_->physics_frame_count_; }

  static auto ElapsedTime() { return s_timer_->elapsed_time_; }

  static auto DeltaTime() { return s_timer_->delta_time_; }

  static auto FixedDeltaTime() { return s_timer_->fixed_delta_time_; }

 private:
  static Timer* s_timer_;

  std::unordered_map<std::string, double> times_;
  std::unordered_map<std::string, double> history_;
  std::unordered_map<std::string, int> frames_;
  std::unordered_map<std::string, std::vector<cudaEvent_t>> cuda_events_;
  std::unordered_map<std::string, float> label2accumulated_time;

  int frame_count_ = 0;
  int physics_frame_count_ = 0;
  float elapsed_time_ = 0.0f;
  float delta_time_ = 0.0f;
  const float fixed_delta_time_ = 1.0f / 60.0f;

  float last_update_time_ = 0.0f;
  float fixed_update_timer_ = 0.0f;
};

class ScopedTimerGPU {
 public:
  ScopedTimerGPU(const std::string&& _label) {
    label_ = _label;
    Timer::StartTimerGPU(_label);
  }

  ~ScopedTimerGPU() {
    Timer::EndTimerGPU(label_);
  }

 private:
  std::string label_;
};
}  // namespace XRTailor