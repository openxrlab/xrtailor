#pragma once

#include <xrtailor/math/MathFunctions.cuh>
#include <xrtailor/core/DeviceHelper.cuh>
#include <xrtailor/physics/impact_zone/Optimizer.cuh>
#include <xrtailor/memory/Pair.cuh>
#include <xrtailor/memory/Node.cuh>
#include <xrtailor/physics/impact_zone/Impact.cuh>

namespace XRTailor {

class ImpactZoneOptimizer : public Optimizer {
 public:
  ImpactZoneOptimizer(const thrust::device_vector<Impact>& impacts, Scalar thickness, int deform,
                      Scalar obstacleMass);

  ~ImpactZoneOptimizer();

 protected:
  Scalar Objective(const thrust::device_vector<Vector3>& x, Scalar& O) const override;

  void ObjectiveGradient(const thrust::device_vector<Vector3>& x,
                         thrust::device_vector<Vector3>& gradient) const override;

  void Constraint(const thrust::device_vector<Vector3>& x,
                  thrust::device_vector<Scalar>& constraints,
                  thrust::device_vector<int>& signs) const override;

  void ConstraintGradient(const thrust::device_vector<Vector3>& x,
                          const thrust::device_vector<Scalar>& coefficients, Scalar mu,
                          thrust::device_vector<Vector3>& gradient) const override;

 private:
  Scalar inv_mass_, thickness_, obstacle_mass_;
  thrust::device_vector<Impact> impacts_;
  // map from global node indices to local node indices.
  // Global: contains duplicated nodes;
  // Local: unique nodes
  thrust::device_vector<int> indices_;
};

}  // namespace XRTailor