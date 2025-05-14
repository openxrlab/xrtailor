#pragma once

#include <xrtailor/math/MathFunctions.cuh>
#include <xrtailor/core/DeviceHelper.cuh>
#include <xrtailor/memory/Node.cuh>
#include <xrtailor/physics/impact_zone/ImpactZoneHelper.cuh>

namespace XRTailor {
namespace ImpactZoneOptimization {
//#define IZO_DEBUG

#define IZO_GRID_SIZE 64
#define IZO_BLOCK_SIZE 256
#define STEP_LENGTH_COEFF_SCALAR 1e-3f
#define ALPHA 0.5f  // control paramter in the Wolfe condition
#define EPSILON_S 1e-12f
#define EPSILON_F 1e-10f
#define RHO2 MathFunctions::sqr(static_cast<Scalar>(0.9992f))
#define MAX_ITERATIONS 100  // maximum number of interations

#define REDUCE_NODE 0
#define REDUCE_IMPACT 1

#define REDUCE_OBJECTIVE 0
#define REDUCE_CONSTRAINT 1
#define REDUCE_CONSTRAINT_FT 2
#define REDUCE_GRADIENT 3
#define REDUCE_LAMBDA 4

#define USE_BASELINE_REDUCE true

constexpr int kBlockSize = 256;
constexpr int kNumWaves = 1;

int64_t GetNumBlocks(int64_t n);

class ParallelOptimizer {
 public:
  ParallelOptimizer();
  virtual ~ParallelOptimizer();
  void Solve(int frame_index, int iteration);

 protected:
  int n_nodes_, n_constraints_;            // number of unique nodes/constranits
  Scalar mu_;                              // augmented Lagrangian multiplier control parameter
  thrust::device_vector<Node*> nodes_;     // independent nodes that related to the impact zone
  thrust::device_vector<int> nodes_color_; // colors of nodes
  thrust::device_vector<Scalar> lambda_;   // Lagrangian multipliers
  ImpactZones zones_;
  thrust::device_vector<ZoneAttribute> zone_attributes_;

  thrust::device_vector<int> impact_block_indices_, node_block_indices_;
  thrust::device_vector<int> impact_block_local_indices_, node_block_local_indices_;
  thrust::device_vector<int> impact_block_offsets_, node_block_offsets_;
  int n_total_impact_blocks_, n_total_node_blocks_;

  void Initialize(thrust::device_vector<Vector3>& x) const;

  void Finalize(const thrust::device_vector<Vector3>& x);

  virtual void Objective(const thrust::device_vector<Vector3>& x,
                         thrust::device_vector<Scalar>& objectives) const = 0;

  virtual void ObjectiveGradient(const thrust::device_vector<Vector3>& x,
                                 thrust::device_vector<Vector3>& gradient) const = 0;

  virtual void Constraint(const thrust::device_vector<Vector3>& x,
                          thrust::device_vector<Scalar>& constraints,
                          thrust::device_vector<int>& signs) const = 0;

  virtual void ConstraintGradient(const thrust::device_vector<Vector3>& x,
                                  const thrust::device_vector<Scalar>& coefficients, Scalar mu,
                                  thrust::device_vector<Vector3>& gradient) const = 0;

  virtual void LineSearchStep(thrust::device_vector<Scalar>& objectives) = 0;

  virtual void UpdateConvergency() = 0;

  template <typename InType, int reduce_type, bool norm2>
  void ReduceLocal(const thrust::device_vector<InType>& in, thrust::device_vector<Scalar>& out);

  template <typename InType, int reduce_type, int reduce_object, bool norm2>
  void Reduce(const thrust::device_vector<InType>& in);

  void TotalGradient(const thrust::device_vector<Vector3>& x,
                      thrust::device_vector<Vector3>& gradient,
                      thrust::device_vector<Scalar>& constraint, thrust::device_vector<int>& signs);

  void UpdateMultiplier(const thrust::device_vector<Vector3>& x);
};

}  // namespace ImpactZoneOptimization
}  // namespace XRTailor