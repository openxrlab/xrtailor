#pragma once

#include <xrtailor/math/MathFunctions.cuh>
#include <xrtailor/core/DeviceHelper.cuh>
#include <xrtailor/memory/Node.cuh>

namespace XRTailor {

const int MAX_ITERATIONS = 100;  // maximum number of interations
const Scalar EPSILON_S = 1e-12f;
const Scalar EPSILON_F = 1e-10f;
const Scalar RHO2 = MathFunctions::sqr(static_cast<Scalar>(0.9992f));

class Optimizer {
 public:
  Optimizer();

  virtual ~Optimizer();

  void Solve(int frame_index, int global_iter);

 protected:
  int n_nodes_, n_constraints_;           // number of unique nodes/constranits
  Scalar mu_;                             // augmented Lagrangian multiplier control parameter
  thrust::device_vector<Node*> nodes_;    // independent nodes that related to the impact zone
  thrust::device_vector<Scalar> lambda_;  // Lagrangian multipliers

  void Initialize(thrust::device_vector<Vector3>& x) const;

  void Finalize(const thrust::device_vector<Vector3>& x);

  virtual Scalar Objective(const thrust::device_vector<Vector3>& x, Scalar& O) const = 0;

  virtual void ObjectiveGradient(const thrust::device_vector<Vector3>& x,
                                 thrust::device_vector<Vector3>& gradient) const = 0;

  virtual void Constraint(const thrust::device_vector<Vector3>& x,
                          thrust::device_vector<Scalar>& constraints,
                          thrust::device_vector<int>& signs) const = 0;

  virtual void ConstraintGradient(const thrust::device_vector<Vector3>& x,
                                  const thrust::device_vector<Scalar>& coefficients, Scalar mu,
                                  thrust::device_vector<Vector3>& gradient) const = 0;

  Scalar Value(const thrust::device_vector<Vector3>& x, Scalar& O2, Scalar& C2, Scalar& L2);

  void ValueAndGradient(const thrust::device_vector<Vector3>& x, Scalar& value,
                        thrust::device_vector<Vector3>& gradient, Scalar& O2, Scalar& C2,
                        Scalar& L2);

  void UpdateMultiplier(const thrust::device_vector<Vector3>& x);
};

}  // namespace XRTailor