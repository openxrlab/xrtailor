#pragma once

#include <xrtailor/runtime/engine/GameInstance.hpp>
#include <xrtailor/runtime/rendering/Light.hpp>
#include <xrtailor/runtime/rendering/MeshRenderer.hpp>
#include <xrtailor/runtime/rendering/LineRenderer.hpp>
#include <xrtailor/runtime/rendering/AABBRenderer.hpp>
#include <xrtailor/runtime/rendering/PointRenderer.hpp>
#include <xrtailor/runtime/rendering/ArrowRenderer.hpp>

namespace XRTailor {
class RenderPipeline {
 public:
  RenderPipeline() {
    // configure depth map FBO
    // -----------------------
    const uint SHADOW_WIDTH = 1024, SHADOW_HEIGHT = 1024;
    glGenFramebuffers(1, &depth_frame_buffer);
    // create depth texture
    glGenTextures(1, &depth_tex);
    glBindTexture(GL_TEXTURE_2D, depth_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, SHADOW_WIDTH, SHADOW_HEIGHT, 0,
                 GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    float border_color[] = {1.0, 1.0, 1.0, 1.0};
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border_color);
    // attach depth texture as FBO's depth buffer
    glBindFramebuffer(GL_FRAMEBUFFER, depth_frame_buffer);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_tex, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
  }

  RenderPipeline(const RenderPipeline&) = delete;

  ~RenderPipeline() {
    if (depth_frame_buffer > 0) {
      glDeleteFramebuffers(1, &depth_frame_buffer);
    }
    if (depth_tex > 0) {
      glDeleteTextures(1, &depth_tex);
    }
  }

  void Render() {
    std::vector<MeshRenderer*> renderers = Global::game->FindComponents<MeshRenderer>();

    RenderShadow(renderers);
    RenderObjects(renderers);

    std::vector<LineRenderer*> line_renderers = Global::game->FindComponents<LineRenderer>();
    RenderLines(line_renderers);

    std::vector<AABBRenderer*> aabb_renderers = Global::game->FindComponents<AABBRenderer>();
    RenderAABBs(aabb_renderers);

    std::vector<PointRenderer*> point_renderers = Global::game->FindComponents<PointRenderer>();
    RenderPoints(point_renderers);

    std::vector<ArrowRenderer*> arrow_renderers = Global::game->FindComponents<ArrowRenderer>();
    RenderArrows(arrow_renderers);
  }

  uint depth_frame_buffer = 0;
  uint depth_tex = 0;

 private:
  glm::mat4 ComputeLightMatrix() {
    if (Global::lights.size() == 0) {
      return glm::mat4(1);
    }
    // 1. render depth of scene to texture (from light's perspective)
    // --------------------------------------------------------------
    glm::mat4 light_projection, light_view;
    glm::mat4 light_space_matrix;
    float near_plane = 1.0f, far_plane = 20.0f;
    auto light = Global::lights[0];
    glm::vec3 light_pos = light->position();
    if (light->type == LightType::SpotLight) {
      // note that if you use a perspective projection matrix you'll have to change the light position as the current light position isn't enough to reflect the whole scene
      light_projection = glm::perspective(
          glm::radians(90.0f),
          (GLfloat)Global::Config::shadow_width / (GLfloat)Global::Config::shadow_height,
          near_plane, far_plane);
    } else {
      light_projection = glm::ortho(-10.0f, 10.0f, -10.0f, 10.0f, near_plane, far_plane);
    }
    light_view = glm::lookAt(light_pos, glm::vec3(0.0f), glm::vec3(0.0, 1.0, 0.0));
    light_space_matrix = light_projection * light_view;
    return light_space_matrix;
  }

  void RenderShadow(std::vector<MeshRenderer*> renderers) {
    if (Global::lights.size() == 0)
      return;

    auto original_window_size = Global::game->WindowSize();
    glViewport(0, 0, Global::Config::shadow_width, Global::Config::shadow_height);
    glBindFramebuffer(GL_FRAMEBUFFER, depth_frame_buffer);
    glClear(GL_DEPTH_BUFFER_BIT);

    glCullFace(GL_FRONT);

    auto light_space_matrix = ComputeLightMatrix();

    for (auto r : renderers) {
      if (r->enabled) {
        r->RenderShadow(light_space_matrix);
      }
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, original_window_size.x, original_window_size.y);
  }

  void RenderLines(std::vector<LineRenderer*> renderers) {
    for (auto r : renderers) {
      if (r->enabled) {
        r->Render();
      }
    }
  }

  void RenderAABBs(std::vector<AABBRenderer*> renderers) {
    for (auto r : renderers) {
      if (r->enabled) {
        r->Render();
      }
    }
  }

  void RenderPoints(std::vector<PointRenderer*> renderers) {
    for (auto r : renderers) {
      if (r->enabled) {
        r->Render();
      }
    }
  }

  void RenderArrows(std::vector<ArrowRenderer*> renderers) {
    for (auto r : renderers) {
      if (r->enabled) {
        r->Render();
      }
    }
  }

  void RenderObjects(std::vector<MeshRenderer*> renderers) {
    // reset viewport
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glCullFace(GL_BACK);
    glLineWidth(2.0f);

    auto light_space_matrix = ComputeLightMatrix();

    for (auto r : renderers) {
      if (r->enabled) {
        r->Render(light_space_matrix);
      }
    }
  }
};
}  // namespace XRTailor