#pragma once

#include <xrtailor/math/MathFunctions.cuh>
#include <xrtailor/core/DeviceHelper.cuh>
#include <xrtailor/physics/impact_zone/ParallelOptimizer.cuh>
#include <xrtailor/memory/Pair.cuh>
#include <xrtailor/memory/Node.cuh>
#include <xrtailor/physics/impact_zone/Impact.cuh>
#include <xrtailor/physics/impact_zone/ImpactZoneHelper.cuh>

namespace XRTailor {
namespace ImpactZoneOptimization {

class ParallelImpactZoneOptimizer : public ParallelOptimizer {
 public:
  ParallelImpactZoneOptimizer(Scalar thickness, int deform, Scalar obstacle_mass,
                              const ImpactZones& zones);

  ~ParallelImpactZoneOptimizer();

 protected:
  void Objective(const thrust::device_vector<Vector3>& x,
                 thrust::device_vector<Scalar>& objectives) const override;

  void ObjectiveGradient(const thrust::device_vector<Vector3>& x,
                         thrust::device_vector<Vector3>& gradient) const override;

  void Constraint(const thrust::device_vector<Vector3>& x,
                  thrust::device_vector<Scalar>& constraints,
                  thrust::device_vector<int>& signs) const override;

  void ConstraintGradient(const thrust::device_vector<Vector3>& x,
                          const thrust::device_vector<Scalar>& coefficients, Scalar mu,
                          thrust::device_vector<Vector3>& gradient) const override;

  void LineSearchStep(thrust::device_vector<Scalar>& objectives) override;

  void UpdateConvergency() override;

 private:
  Scalar inv_mass, thickness, obstacle_mass_;
  thrust::device_vector<Impact> impacts_;
  thrust::device_vector<int> indices_;
};

}  // namespace ImpactZoneOptimization
}  // namespace XRTailor