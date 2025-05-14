#include <xrtailor/config/ClothConfigParser.hpp>

#include <ios>
#include <iosfwd>
#include <fstream>

#include "json/reader.h"
#include "json/value.h"

#include <xrtailor/utils/Logger.hpp>


namespace XRTailor {
ClothConfigParser::ClothConfigParser() = default;

bool ClothConfigParser::LoadFromJson(const std::string& path) {
  LOG_TRACE("Reading garment config from: {}", path);

  std::ifstream file(path, std::ios::binary);

  if (!file.is_open()) {
    LOG_ERROR("Error opening file: {}", path);
    return false;
  }

  Json::Reader reader;
  Json::Value root;

  if (reader.parse(file, root, false)) {
    this->style = root["STYLE"].asString();
    this->mass = root["MASS"].asFloat();
    auto attached_indices = root["ATTACHED_INDICES"];
    for (int i = 0; i < attached_indices.size(); i++) {
      this->attached_indices.push_back(attached_indices[i].asInt());
    }
    Json::Value scope_binding = root["BINDING"];
    Json::Value scope_uv_island = scope_binding["UV_ISLAND"];
    auto uv_island_indices = scope_uv_island["INDICES"];
    auto uv_island_stiffness = scope_uv_island["STIFFNESS"];
    auto uv_island_distance = scope_uv_island["DISTANCE"];
    for (int i = 0; i < uv_island_indices.size(); i++) {
      this->binding_uv_island.push_back({uv_island_indices[i].asUInt(),
                                         uv_island_stiffness[i].asFloat(),
                                         uv_island_distance[i].asFloat()});
    }

    Json::Value scope_neighbor = scope_binding["NEIGHBOR"];
    auto neighbor_indices = scope_neighbor["INDICES"];
    auto neighbor_stiffness = scope_neighbor["STIFFNESS"];
    auto neighbor_distance = scope_neighbor["DISTANCE"];
    for (int i = 0; i < neighbor_indices.size(); i++) {
      this->binding_neighbor.push_back(BindingParam{neighbor_indices[i].asUInt(),
                                                    neighbor_stiffness[i].asFloat(),
                                                    neighbor_distance[i].asFloat()});
    }

    Json::Value scope_boundary = scope_binding["BOUNDARY"];
    auto boundary_indices = scope_boundary["INDICES"];
    auto boundary_stiffness = scope_boundary["STIFFNESS"];
    auto boundary_distance = scope_boundary["DISTANCE"];
    for (int i = 0; i < boundary_indices.size(); i++) {
      this->binding_boundary.push_back(BindingParam{boundary_indices[i].asUInt(),
                                                    boundary_stiffness[i].asFloat(),
                                                    boundary_distance[i].asFloat()});
    }

    Json::Value scope_nonmanifold_edges = scope_binding["NONMANIFOLD_EDGES"];
    auto nonmanifold_indices = scope_nonmanifold_edges["INDICES"];
    auto nonmanifold_stiffness = scope_nonmanifold_edges["STIFFNESS"];
    auto nonmanifold_distance = scope_nonmanifold_edges["DISTANCE"];
    for (int i = 0; i < nonmanifold_indices.size(); i++) {
      this->binding_nonmanifold_edges.push_back(BindingParam{nonmanifold_indices[i].asUInt(),
                                                             nonmanifold_stiffness[i].asFloat(),
                                                             nonmanifold_distance[i].asFloat()});
    }
  } else {
    LOG_ERROR("Failed to parse file {}", path);
    return false;
  }
  return true;
}
}  // namespace XRTailor