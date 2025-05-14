#include <xrtailor/runtime/scene/Actor.hpp>

namespace XRTailor {
Actor::Actor() = default;

Actor::Actor(std::string name) : name(name) {}

void Actor::Initialize(glm::vec3 position, glm::vec3 scale, glm::vec3 rotation) {
  transform->position = position;
  transform->scale = scale;
  transform->rotation = rotation;
}

void Actor::Translate(glm::vec3 position) {
  transform->position = position;
}

void Actor::Rotate(glm::vec3 rotation) {
  transform->rotation = rotation;
}

void Actor::Scale(glm::vec3 scale) {
  transform->scale = scale;
}

void Actor::Start() {
  for (const auto& c : components) {
    c->Start();
  }
}

void Actor::AddComponent(std::shared_ptr<Component> component) {
  component->actor = this;
  components.push_back(component);
}

void Actor::AddComponents(const std::initializer_list<std::shared_ptr<Component>>& new_components) {
  for (const auto& c : new_components) {
    AddComponent(c);
  }
}

void Actor::OnDestroy() {
  for (const auto& c : components) {
    c->OnDestroy();
  }
}

void Actor::Update() {
  for (const auto& c : components) {
    c->Update();
  }
}

void Actor::FixedUpdate() {
  for (const auto& c : components) {
    if (c->enabled) {
      c->FixedUpdate();
    }
  }
}

const std::string Actor::GetName() {
  return name;
}
}  // namespace XRTailor