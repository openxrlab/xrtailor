#pragma once

#include <vector>

#include <xrtailor/core/Common.hpp>
#include <xrtailor/core/DeviceHelper.cuh>
#include <xrtailor/utils/Timer.hpp>

namespace XRTailor {

struct BindingItem {
  int cloth_vert_idx;
  Scalar u;
  Scalar v;
  Scalar w;
  int u_idx;
  int v_idx;
  int w_idx;
  Scalar L;
  Scalar stretch_stiffness;
};

class BindingConstraint {
 public:
  BindingConstraint();

  ~BindingConstraint();

  void Add(const std::vector<uint>& indices, const std::vector<Scalar>& stiffnesses,
           const std::vector<Scalar>& distances, const uint& prevNumParticles);

  void Generate(Node** cloth_nodes, Node** obstacle_nodes, SkinParam* skin_params);

  void Solve(Node** cloth_nodes, Node** obstacle_nodes);

 public:
  thrust::device_vector<uint> binded_indices;
  thrust::device_vector<Scalar> bind_stiffnesses;
  thrust::device_vector<Scalar> bind_distances;

  thrust::device_vector<BindingItem> bindings;
  int n_bindings;
};
}  // namespace XRTailor