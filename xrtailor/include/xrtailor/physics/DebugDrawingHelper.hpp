#pragma once

#include <memory>
#include <thrust/host_vector.h>

#include <xrtailor/utils/Logger.hpp>
#include <xrtailor/runtime/rendering/AABB.hpp>
#include <xrtailor/runtime/rendering/AABBRenderer.hpp>
#include <xrtailor/runtime/rendering/LineRenderer.hpp>
#include <xrtailor/runtime/rendering/PointRenderer.hpp>
#include <xrtailor/runtime/rendering/ArrowRenderer.hpp>
#include <xrtailor/physics/PhysicsMesh.cuh>
#include <xrtailor/physics/repulsion/ImminentRepulsion.cuh>
#include <xrtailor/physics/repulsion/PBDRepulsion.cuh>
#include <xrtailor/physics/icm/IntersectionContourMinimizationHelper.cuh>
#include <xrtailor/physics/icm/GlobalIntersectionAnalysis.cuh>


namespace XRTailor {
namespace Debug {

void DrawEEProximities(std::shared_ptr<LineRenderer> line_renderer,
                       std::shared_ptr<PhysicsMesh> physics_mesh,
                       const thrust::host_vector<QuadInd>& quadInds);

void DrawICMIntersections(
    std::shared_ptr<LineRenderer> line_renderer, std::shared_ptr<LineRenderer> line_renderer_B,
    std::shared_ptr<PointRenderer> point_renderer, std::shared_ptr<ArrowRenderer> arrow_renderer,
    std::shared_ptr<PhysicsMesh> physics_mesh,
    const thrust::host_vector<Untangling::IntersectionWithGradient>& intersections);

void DrawGIAContours(std::shared_ptr<PointRenderer> point_renderer_A,
                     std::shared_ptr<PointRenderer> point_renderer_B,
                     std::shared_ptr<PointRenderer> point_renderer_C,
                     std::shared_ptr<PointRenderer> point_renderer_D,
                     std::shared_ptr<PointRenderer> point_renderer_E,
                     const thrust::host_vector<Untangling::EFIntersection>& intersections,
                     const thrust::host_vector<Untangling::IntersectionState>& intersection_states,
                     std::shared_ptr<PhysicsMesh> physics_mesh);

}  // namespace Debug
}  // namespace XRTailor