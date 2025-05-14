#pragma once

namespace XRTailor {
template <class T, class... TArgs>
class Callback {
 public:
  void Register(const std::function<T>& func) { funcs_.push_back(func); }

  template <class... _TArgs>
  void Invoke(_TArgs... args) {
    for (const auto& func : funcs_) {
      func(std::forward<_TArgs>(args)...);
    }
  }

  void Clear() { funcs_.clear(); }

  bool Empty() { return funcs_.size() == 0; }

 private:
  std::vector<std::function<T>> funcs_;
};
} // namespace XRTailor