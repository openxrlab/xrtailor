#pragma once

#include <xrtailor/runtime/scene/Component.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <xrtailor/runtime/mesh/Mesh.hpp>
#include <xrtailor/runtime/resources/Material.hpp>
#include <xrtailor/runtime/resources/MaterialProperty.hpp>
#include <xrtailor/runtime/scene/Actor.hpp>
#include <xrtailor/utils/Logger.hpp>

namespace XRTailor {
class MeshRenderer : public Component {
 public:
  MeshRenderer(std::shared_ptr<Mesh> mesh, std::shared_ptr<Material> material,
               bool cast_shadow = false);

  void SetMaterialProperty(const MaterialProperty& material_property);

  virtual void Render(glm::mat4 light_matrix);

  virtual void RenderShadow(glm::mat4 light_matrix);

  virtual void DrawCall();

  std::shared_ptr<Material> material() const;

  std::shared_ptr<Mesh> mesh() const;

  void Start();

 protected:
  void SetupLighting(std::shared_ptr<Material> material);

  int num_instances_ = 0;
  std::shared_ptr<Mesh> mesh_;
  std::shared_ptr<Material> material_;
  std::shared_ptr<Material> shadow_material_;
  MaterialProperty material_property_;
};
}  // namespace XRTailor