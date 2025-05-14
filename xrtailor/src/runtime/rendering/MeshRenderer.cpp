#include <xrtailor/runtime/rendering/MeshRenderer.hpp>

#include <xrtailor/core/Global.hpp>
#include <xrtailor/runtime/engine/GameInstance.hpp>
#include <xrtailor/runtime/scene/Camera.hpp>
#include <xrtailor/runtime/scene/Actor.hpp>
#include <xrtailor/runtime/rendering/Light.hpp>
#include <xrtailor/runtime/resources/Resource.hpp>
#include <xrtailor/utils/Helper.hpp>

namespace XRTailor {
MeshRenderer::MeshRenderer(std::shared_ptr<Mesh> mesh, std::shared_ptr<Material> material,
                           bool cast_shadow)
    : mesh_(mesh), material_(material) {
  SET_COMPONENT_NAME;

  if (cast_shadow) {
    shadow_material_ = Resource::LoadMaterial("_ShadowDepth");
  }
}

void MeshRenderer::SetMaterialProperty(const MaterialProperty& material_property) {
  material_property_ = material_property;
}

// Only support spot light for now
void MeshRenderer::SetupLighting(std::shared_ptr<Material> material) {
  if (Global::lights.empty()) {
    return;
  }
  auto light = Global::lights[0];

  std::string prefix = "spotLight.";
  auto front = Helper::RotateWithDegree(glm::vec3(0, -1, 0), light->transform()->rotation);

  material->SetVec3(prefix + "position", light->position());
  material->SetVec3(prefix + "direction", front);
  material->SetFloat(prefix + "cutOff", glm::cos(glm::radians(light->inner_cutoff)));
  material->SetFloat(prefix + "outerCutOff", glm::cos(glm::radians(light->outer_cutoff)));

  material->SetFloat(prefix + "constant", light->constant);
  material->SetFloat(prefix + "linear", light->linear);
  material->SetFloat(prefix + "quadratic", light->quadratic);

  material->SetVec3(prefix + "color", light->color);
  material->SetFloat(prefix + "ambient", light->ambient);
}

void MeshRenderer::Render(glm::mat4 lightMatrix) {
  if (material_->no_wireframe && Global::game_state.render_wireframe) {
    return;
  }

  material_->Use();

  // material
  material_->SetFloat("material.specular", material_->specular);
  material_->SetFloat("material.smoothness", material_->smoothness);
  material_->SetTexture("_ShadowTex", Global::game->DepthFrameBuffer());

  if (material_property_.pre_rendering) {
    material_property_.pre_rendering(material_.get());
  }

  // camera param
  material_->SetVec3("_CameraPos", Global::camera->transform()->position);

  // light params
  SetupLighting(material_);

  // texture
  int i = 0;
  for (auto tex : material_->textures) {
    glActiveTexture(GL_TEXTURE0 + i);
    glBindTexture(GL_TEXTURE_2D, tex.second);
    material_->SetInt(tex.first, i);
    i++;
  }

  // matrices
  auto model = actor->transform->matrix();
  auto view = Global::camera->View();
  auto projection = Global::camera->Projection();

  material_->SetMat4("_Model", model);
  material_->SetMat4("_View", view);
  material_->SetMat4("_Projection", projection);

  material_->SetMat4("_MVP", projection * view * model);
  material_->SetMat4("_InvView", glm::inverse(view));
  material_->SetMat3("_Normalmatrix", glm::mat3(glm::transpose(glm::inverse(model))));

  material_->SetMat4("_WorldToLight", lightMatrix);

  DrawCall();
}

void MeshRenderer::RenderShadow(glm::mat4 lightMatrix) {
  if (shadow_material_ == nullptr) {
    return;
  }

  shadow_material_->Use();
  shadow_material_->SetMat4("_Model", actor->transform->matrix());
  shadow_material_->SetMat4("_WorldToLight", lightMatrix);

  DrawCall();
}

void MeshRenderer::DrawCall() {
  if (material_->double_sided) {
    glDisable(GL_CULL_FACE);
  }

  glBindVertexArray(mesh_->VAO());
  if (mesh_->UseIndices()) {
    if (num_instances_ > 0) {
      glDrawElementsInstanced(GL_TRIANGLES, mesh_->DrawCount(), GL_UNSIGNED_INT, nullptr,
                              num_instances_);
    } else {
      glDrawElements(GL_TRIANGLES, mesh_->DrawCount(), GL_UNSIGNED_INT, nullptr);
    }
  } else {
    if (num_instances_ > 0) {
      glDrawArraysInstanced(GL_TRIANGLES, 0, mesh_->DrawCount(), num_instances_);
    } else {
      glDrawArrays(GL_TRIANGLES, 0, mesh_->DrawCount());
    }
  }

  if (material_->double_sided) {
    glEnable(GL_CULL_FACE);
  }
}

std::shared_ptr<Material> MeshRenderer::material() const {
  return material_;
}

std::shared_ptr<Mesh> MeshRenderer::mesh() const {
  return mesh_;
}

void MeshRenderer::Start() {
  LOG_TRACE("Starting {0}'s component: {1}", actor->name, name);
}
}  // namespace XRTailor