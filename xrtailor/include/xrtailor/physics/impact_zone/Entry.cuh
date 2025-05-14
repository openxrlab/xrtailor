#pragma once

#include <xrtailor/physics/impact_zone/Defs.hpp>
#include <xrtailor/physics/PhysicsMesh.cuh>
#include <xrtailor/core/DeviceHelper.cuh>

namespace XRTailor {
namespace ImpactZoneOptimization {

void CollisionStep(std::shared_ptr<PhysicsMesh> cloth, std::shared_ptr<PhysicsMesh> obstacle,
                   std::shared_ptr<MemoryPool> memory_pool, int frame_index, Scalar dt,
                   const Scalar& init_obstacle_mass = static_cast<Scalar>(1e3f),
                   const Scalar& collision_thickness = static_cast<Scalar>(1e-4f));

}  // namespace ImpactZoneOptimization
}  // namespace XRTailor