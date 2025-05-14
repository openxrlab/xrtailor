#pragma once
#include <memory>
#include <iostream>
#include <vector>

#include <xrtailor/runtime/scene/Component.hpp>
#include <xrtailor/runtime/scene/Transform.hpp>

namespace XRTailor {
class Actor {
 public:
  Actor();

  Actor(std::string name);

  void Initialize(glm::vec3 position, glm::vec3 scale = glm::vec3(1),
                  glm::vec3 rotation = glm::vec3(0));

  void Translate(glm::vec3 position);

  void Rotate(glm::vec3 rotation);

  void Scale(glm::vec3 scale);

  void Start();

  void Update();

  void FixedUpdate();

  void OnDestroy();

  void AddComponent(std::shared_ptr<Component> component);

  void AddComponents(const std::initializer_list<std::shared_ptr<Component>>& new_components);

  template <typename T>
  std::enable_if_t<std::is_base_of<Component, T>::value, T*> GetComponent() {
    T* result = nullptr;
    for (auto c : components) {
      result = dynamic_cast<T*>(c.get());
      if (result)
        return result;
    }
    return result;
  }

  template <typename T>
  std::enable_if_t<std::is_base_of<Component, T>::value, std::vector<T*>> GetComponents() {
    std::vector<T*> result;
    for (auto c : components) {
      auto item = dynamic_cast<T*>(c.get());
      if (item) {
        result.push_back(item);
      }
    }
    return result;
  }

  const std::string GetName();

 public:
  std::shared_ptr<Transform> transform = std::make_shared<Transform>(Transform(this));
  std::vector<std::shared_ptr<Component>> components;
  std::string name;
};
}  // namespace XRTailor