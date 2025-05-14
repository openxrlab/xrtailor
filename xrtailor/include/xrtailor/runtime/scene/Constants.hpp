#pragma once

#include <vector>
#include <glm/glm.hpp>

namespace XRTailor {
// colors from https://colorswall.com/palettes
static std::vector<glm::vec3> front_face_colors = {
    glm::vec3(186.0f / 255.0f, 147.0f / 255.0f, 216.0f / 255.0f),  // Lenurple
    glm::vec3(237.0f / 255.0f, 79.0f / 255.0f, 109.0f / 255.0f),   // Kiss of Death
    glm::vec3(95.0f / 255.0f, 178.0f / 255.0f, 110.0f / 255.0f),   // Forest Green
    glm::vec3(0.94, 0.64, 1.00),
    glm::vec3(0.00, 0.46, 0.86),
    glm::vec3(0.60, 0.25, 0.00),
    glm::vec3(0.10, 0.10, 0.10),

    glm::vec3(0.17, 0.81, 0.28),
    glm::vec3(1.00, 0.80, 0.60),
    glm::vec3(1.00, 0.00, 0.06),
    glm::vec3(1.00, 0.64, 0.02),
};

static std::vector<glm::vec3> back_face_colors = {
    glm::vec3(254.0f / 255.0f, 207.0f / 255.0f, 91.0f / 255.0f),  // Kourtney's Kandy
    glm::vec3(255.0f / 255.0f, 100.0f / 255.0f, 60.0f / 255.0f),  // Smashed Pumpkin
    glm::vec3(99.0f / 255.0f, 80.0f / 255.0f, 254.0f / 255.0f),   // Illusion Neon Blue
    glm::vec3(0.76, 0.00, 0.53),
    glm::vec3(0.37, 0.95, 0.95),
    glm::vec3(0.00, 0.60, 0.56),
    glm::vec3(0.88, 1.00, 0.40),
    glm::vec3(0.45, 0.04, 1.00),
    glm::vec3(1.00, 0.31, 0.02),
    glm::vec3(0.00, 0.36, 0.19),
    glm::vec3(0.50, 0.50, 0.50)};

static glm::vec3 skin_color = glm::vec3(236.0f / 255.0f, 188.0f / 255.0f, 180.0f / 255.0f);

}  // namespace XRTailor