#include <xrtailor/physics/impact_zone/ParallelImpactZoneOptimizer.cuh>

#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/unique.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/count.h>

namespace XRTailor {
namespace ImpactZoneOptimization {

__global__ void CollectCollisionNodes_Kernel(int n_constraints, const Impact* impacts,
                                             CONST(int*) impact_colors, int deform, int* indices,
                                             Node** nodes, int* node_colors) {
  GET_CUDA_ID(i, n_constraints);

  const Impact& impact = impacts[i];
  const int& impact_color = impact_colors[i];
  for (int j = 0; j < 4; j++) {
    int index = 4 * i + j;
    Node* node = impact.nodes[j];
    if (deform == 1 || (node->is_cloth && node->is_free)) {
      indices[index] = index;
      nodes[index] = node;
      node_colors[index] = impact_color;
    } else {
      indices[index] = -1;
      nodes[index] = nullptr;
      node_colors[index] = -1;
    }
  }
}

__global__ void SetDiff_Kernel(int n_nodes, const Node* const* nodes, int* diff) {
  GET_CUDA_ID(idx, n_nodes);

  diff[idx] = (idx > 0 && nodes[idx] != nodes[idx - 1]);
}

__global__ void SetIndices_Kernel(int n_nodes, const int* node_indices, const int* diff,
                                  int* indices) {
  GET_CUDA_ID(idx, n_nodes);

  indices[node_indices[idx]] = diff[idx];
}

__global__ void CollisionInv_Kernel(int n_nodes, const Node* const* nodes, Scalar obstacle_mass,
                                    Scalar* inv) {
  GET_CUDA_ID(idx, n_nodes);

  const Node* node = nodes[idx];
  Scalar inv_mass = node->is_cloth ? node->inv_mass : (static_cast<Scalar>(1.0) / obstacle_mass);
  inv[idx] = inv_mass;
}

__global__ void CollisionObjective_Kernel(int n_nodes, const Node* const* nodes,
                                          Scalar obstacle_mass, const Vector3* x,
                                          Scalar* objectives) {
  GET_CUDA_ID(idx, n_nodes);

  const Node* node = nodes[idx];
  Scalar mass = node->is_cloth ? (static_cast<Scalar>(1.0) / node->inv_mass) : obstacle_mass;
  objectives[idx] = mass * glm::dot(x[idx] - node->x1, x[idx] - node->x1);
}

/**
 * @brief Evaluate descent direction
*/
__global__ void CollisionObjectiveGradient_Kernel(int n_nodes, const Node* const* nodes,
                                                  Scalar inv_mass, Scalar obstacle_mass,
                                                  const Vector3* x, Vector3* gradient) {
  GET_CUDA_ID(idx, n_nodes);

  const Node* node = nodes[idx];
  Scalar mass = node->is_cloth ? (1 / node->inv_mass) : obstacle_mass;
  gradient[idx] = inv_mass * mass * (x[idx] - node->x1);
}

__global__ void CollisionConstraint_Kernel(int n_constraints, const Impact* impacts,
                                           const int* indices, Scalar thickness, const Vector3* x,
                                           Scalar* constraints, int* signs) {
  GET_CUDA_ID(i, n_constraints);

  Scalar c = -thickness;
  const Impact& impact = impacts[i];
  for (int j = 0; j < 4; j++) {
    int k = indices[4 * i + j];
    if (k > -1)
      c += impact.w[j] * glm::dot(impact.n, x[k]);
    else
      c += impact.w[j] * glm::dot(impact.n, impact.nodes[j]->x);
  }
  constraints[i] = c;
  signs[i] = 1;
}

__global__ void CollectCollisionConstraintGradient_Kernel(int n_constraints, const Impact* impacts,
                                                          const Scalar* coefficients, Scalar mu,
                                                          Vector3* grad) {
  GET_CUDA_ID(i, n_constraints);

  const Impact& impact = impacts[i];

  for (int j = 0; j < 4; j++)
    grad[4 * i + j] = mu * coefficients[i] * impact.w[j] * impact.n;
}

__global__ void AddConstraintGradient_Kernel(int n_indices, const int* indices, const Vector3* grad,
                                             Vector3* gradients) {
  GET_CUDA_ID(i, n_indices);

  AtomicAdd(gradients, indices[i], grad[i]);
}

__global__ void PaintNodeWithImpact_Kernel(CONST(Impact*) impacts, CONST(int*) impact_colors,
                                           int* node_colors, int n_impacts) {
  GET_CUDA_ID(id, n_impacts);

  const Impact& impact = impacts[id];
  const int& impact_color = impact_colors[id];
  for (int i = 0; i < 4; i++) {
    node_colors[id * 4 + i] = impact_color;
  }
}

struct NodeComp {
  __device__ bool operator()(const thrust::tuple<int, Node*>& lhs,
                             const thrust::tuple<int, Node*>& rhs) {
    int lhs_color = thrust::get<0>(lhs);
    int rhs_color = thrust::get<0>(rhs);

    if (lhs_color != rhs_color)
      return lhs_color < rhs_color;

    Node* lhs_node = thrust::get<1>(lhs);
    Node* rhs_node = thrust::get<1>(rhs);

    return lhs_node < rhs_node;
  };
};

__global__ void SetupZoneAttribute_Kernel(int n_zones, int* node_offsets, int* impact_offsets,
                                          int* zone_actives, ZoneAttribute* zone_attributes) {
  GET_CUDA_ID(zid, n_zones);

  ZoneAttribute& zone_attribute = zone_attributes[zid];
  zone_attribute.impact_offset = impact_offsets[zid];
  zone_attribute.n_impacts = impact_offsets[zid + 1] - impact_offsets[zid];

  zone_attribute.node_offset = node_offsets[zid];
  zone_attribute.n_nodes = node_offsets[zid + 1] - node_offsets[zid];

  zone_attribute.f = 0.0f;
  zone_attribute.ft = 0.0f;
  zone_attribute.s = 0.0f;
  if (zone_actives[zid]) {
    zone_attribute.wolfe_condition_satisfied = false;
    zone_attribute.converged = false;
  } else {
    zone_attribute.wolfe_condition_satisfied = true;
    zone_attribute.converged = true;
  }
}

ParallelImpactZoneOptimizer::ParallelImpactZoneOptimizer(Scalar thickness, int deform,
                                                         Scalar obstacle_mass,
                                                         const ImpactZones& zones)
    : impacts_(zones.impacts), thickness(thickness), obstacle_mass_(obstacle_mass) {
#if 0
	printf("Init Solver\n");
	checkCudaErrors(cudaDeviceSynchronize());
#endif  // 1

  this->zones_ = zones;
  int n_zones = zones.n_colors;
  n_constraints_ = impacts_.size();

  // initialize collision nodes and corresponding indices
  nodes_.resize(4 * n_constraints_);
  nodes_color_.resize(4 * n_constraints_);

  Node** nodes_pointer = pointer(nodes_);
  int* nodes_color_pointer = pointer(nodes_color_);

  thrust::device_vector<int> node_indices(
      4 * n_constraints_);  // indices are represented as thread index offset
  int* node_indices_pointer = pointer(node_indices);
  // filter non-deformable nodes
  CUDA_CALL(CollectCollisionNodes_Kernel, n_constraints_)
  (n_constraints_, pointer(impacts_), pointer(zones.colors), deform, node_indices_pointer,
   nodes_pointer, nodes_color_pointer);
  CUDA_CHECK_LAST();
  nodes_.erase(thrust::remove(nodes_.begin(), nodes_.end(), nullptr), nodes_.end());
  node_indices.erase(thrust::remove(node_indices.begin(), node_indices.end(), -1),
                     node_indices.end());
  nodes_color_.erase(thrust::remove(nodes_color_.begin(), nodes_color_.end(), -1),
                     nodes_color_.end());

  // sort the indices via node memory address
  // duplicated nodes will be next to each other
  thrust::sort_by_key(
      thrust::make_zip_iterator(thrust::make_tuple(nodes_color_.begin(), nodes_.begin())),
      thrust::make_zip_iterator(thrust::make_tuple(nodes_color_.end(), nodes_.end())),
      node_indices.begin(), NodeComp());

  thrust::device_vector<int> diff(nodes_.size());
  CUDA_CALL(SetDiff_Kernel, nodes_.size())
  (nodes_.size(), nodes_pointer, pointer(diff));
  CUDA_CHECK_LAST();
  // do prefix sum for diff
  thrust::inclusive_scan(diff.begin(), diff.end(), diff.begin());
  indices_.assign(4 * n_constraints_, -1);

  // setup mapping from global node indices to local node indices
  CUDA_CALL(SetIndices_Kernel, nodes_.size())
  (nodes_.size(), node_indices_pointer, pointer(diff), pointer(indices_));
  CUDA_CHECK_LAST();

  // remove duplicated nodes, keeping ascending order
  auto new_end = thrust::unique_by_key(nodes_.begin(), nodes_.end(), nodes_color_.begin());
  nodes_.erase(new_end.first, nodes_.end());
  nodes_color_.erase(new_end.second, nodes_color_.end());

  this->zones_.node_offsets.push_back(0);
  for (int color = 0; color < this->zones_.n_colors; color++) {
    this->zones_.node_offsets.push_back(
        thrust::count(nodes_color_.begin(), nodes_color_.end(), color));
  }
  thrust::inclusive_scan(this->zones_.node_offsets.begin(), this->zones_.node_offsets.end(),
                         this->zones_.node_offsets.begin());

  n_nodes_ = nodes_.size();
  thrust::device_vector<Scalar> inv(n_nodes_);
  CUDA_CALL(CollisionInv_Kernel, n_nodes_)
  (n_nodes_, nodes_pointer, obstacle_mass, pointer(inv));
  CUDA_CHECK_LAST();

  inv_mass = thrust::reduce(inv.begin(), inv.end()) / n_nodes_;
#if 0
	printf("zone size: %d\n", zones_.n_colors);
	thrust::host_vector<int> h_offsets = this->zones_.node_offsets;
	for (int i = 0; i < zones_.n_colors; i++)
	{
		printf("  - color %d has %d nodes, range [%d, %d]\n", i, h_offsets[i + 1] - h_offsets[i], h_offsets[i], h_offsets[i + 1] - 1);
	}

	checkCudaErrors(cudaDeviceSynchronize());
	thrust::host_vector<int> h_dbg_indices = node_indices;
	thrust::host_vector<int> h_dbg_colors = nodes_color_;
	printf("indices: %d\n", h_dbg_indices.size());
	for (int i = 0; i < h_dbg_indices.size(); i++)
	{
		printf("%d ", h_dbg_indices[i]);
	}
	printf("\n");
	printf("colors: %d\n", h_dbg_colors.size());
	for (int i = 0; i < h_dbg_colors.size(); i++)
	{
		printf("%d ", h_dbg_colors[i]);
	}
	printf("\n");
#endif

  zone_attributes_.resize(n_zones);

  CUDA_CALL(SetupZoneAttribute_Kernel, n_zones)
  (n_zones, pointer(this->zones_.node_offsets), pointer(this->zones_.impact_offsets),
   pointer(this->zones_.active), pointer(zone_attributes_));
  CUDA_CHECK_LAST();
#if 0
	checkCudaErrors(cudaDeviceSynchronize());
	printf("Solver init done:\n");
	thrust::host_vector<ZoneAttribute> zone_attributes = zone_attributes_;
	for (int i = 0; i < zone_attributes.size(); i++)
	{
		auto z = zone_attributes[i];
		printf("zone %d | n_impacts: %d, n_nodes: %d, impact_offset: %d, node_offset: %d, active: %d\n", i, z.n_impacts, z.n_nodes, z.impact_offset, z.node_offset, !z.converged);
	}
#endif  // 1
}

ParallelImpactZoneOptimizer::~ParallelImpactZoneOptimizer() {}

void ParallelImpactZoneOptimizer::Objective(const thrust::device_vector<Vector3>& x,
                                            thrust::device_vector<Scalar>& objectives) const {
  CUDA_CALL(CollisionObjective_Kernel, n_nodes_)
  (n_nodes_, pointer(nodes_), obstacle_mass_, pointer(x), pointer(objectives));
  CUDA_CHECK_LAST();
}

void ParallelImpactZoneOptimizer::ObjectiveGradient(
    const thrust::device_vector<Vector3>& x, thrust::device_vector<Vector3>& gradient) const {
  CUDA_CALL(CollisionObjectiveGradient_Kernel, n_nodes_)
  (n_nodes_, pointer(nodes_), inv_mass, obstacle_mass_, pointer(x), pointer(gradient));
  CUDA_CHECK_LAST();
}

void ParallelImpactZoneOptimizer::Constraint(const thrust::device_vector<Vector3>& x,
                                             thrust::device_vector<Scalar>& constraints,
                                             thrust::device_vector<int>& signs) const {
  CUDA_CALL(CollisionConstraint_Kernel, n_constraints_)
  (n_constraints_, pointer(impacts_), pointer(indices_), thickness, pointer(x),
   pointer(constraints), pointer(signs));
  CUDA_CHECK_LAST();
}

void ParallelImpactZoneOptimizer::ConstraintGradient(
    const thrust::device_vector<Vector3>& x, const thrust::device_vector<Scalar>& coefficients,
    Scalar mu, thrust::device_vector<Vector3>& gradient) const {
  thrust::device_vector<int> grad_indices = indices_;
  thrust::device_vector<Vector3> grad(4 * n_constraints_);

  // constraint gradients
  CUDA_CALL(CollectCollisionConstraintGradient_Kernel, n_constraints_)
  (n_constraints_, pointer(impacts_), pointer(coefficients), mu, pointer(grad));
  CUDA_CHECK_LAST();

  grad.erase(thrust::remove_if(grad.begin(), grad.end(), grad_indices.begin(), IsNull()),
             grad.end());
  grad_indices.erase(thrust::remove(grad_indices.begin(), grad_indices.end(), -1),
                     grad_indices.end());
  thrust::sort_by_key(grad_indices.begin(), grad_indices.end(), grad.begin());

  thrust::device_vector<int> output_grad_indices(4 * n_constraints_);
  thrust::device_vector<Vector3> output_grad(4 * n_constraints_);
  auto iter = thrust::reduce_by_key(grad_indices.begin(), grad_indices.end(), grad.begin(),
                                    output_grad_indices.begin(), output_grad.begin());
  CUDA_CALL(AddConstraintGradient_Kernel, iter.first - output_grad_indices.begin())
  (iter.first - output_grad_indices.begin(), pointer(output_grad_indices), pointer(output_grad),
   pointer(gradient));
  CUDA_CHECK_LAST();
}

__global__ void BacktrackingLineSearchTrivial_Kernel(int n_zones, Scalar* objectives,
                                                     ZoneAttribute* zone_attributes, Scalar mu,
                                                     Scalar inv_masses) {
  int tid = threadIdx.x;
  int zid = blockIdx.x;

  if (zid >= n_zones)
    return;

  ZoneAttribute& z = zone_attributes[zid];

  if (z.converged || z.wolfe_condition_satisfied)
    return;

  const int& n_nodes = z.n_nodes;
  const int& node_idx = z.node_offset + tid;

  __shared__ Scalar s_objectives[IZO_BLOCK_SIZE];

  s_objectives[tid] = tid < n_nodes ? objectives[node_idx] : 0;

  __syncthreads();  // wait for objectives/gradients load

  for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
    if (tid % (2 * stride) == 0) {
      s_objectives[tid] += s_objectives[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    const Scalar& O = s_objectives[0];
    const Scalar& G2 = z.G2;
    const Scalar& LAMBDA2 = z.lambda2;

    // backtracking line search
    Scalar f =
        0.5f * inv_masses * z.O + 0.5f * mu * z.C2 - 0.5f * LAMBDA2 / mu;  //objective function
    Scalar ft = 0.5f * inv_masses * z.Ot + 0.5f * mu * z.C2t -
                0.5f * LAMBDA2 / mu;  //advanced objective function
    Scalar s = z.s;
    if (s < EPSILON_S || ft < (f - ALPHA * s * G2)) {
      z.wolfe_condition_satisfied = true;
      if (s < EPSILON_S) {
        z.line_search_reason = 1;
      } else if (ft < (f - ALPHA * s * G2)) {
        z.line_search_reason = 2;
      } else if (MathFunctions::abs(f - ft) < EPSILON_F) {
        z.line_search_reason = 3;
      }
#ifdef IZO_DEBUG
      printf(
          "zone %d wolfe condition satisfied, s: %.12f, objective: %.12f, constraints2: %.12f, "
          "gradients2: %.12f\n",
          zid, s, O, C2, G2);
#endif  // IZO_DEBUG
    }
    z.O = O;
    z.ft = ft;
  }
}

template <uint block_size>
__device__ void WarpReduce(volatile Scalar* cache, int tid) {
  if (block_size >= 64)
    cache[tid] += cache[tid + 32];
  if (block_size >= 32)
    cache[tid] += cache[tid + 16];
  if (block_size >= 16)
    cache[tid] += cache[tid + 8];
  if (block_size >= 8)
    cache[tid] += cache[tid + 4];
  if (block_size >= 4)
    cache[tid] += cache[tid + 2];
  if (block_size >= 2)
    cache[tid] += cache[tid + 1];
}

template <unsigned int blockSize>
__global__ void BacktrackingLineSearchAdvanced_Kernel(CONST(Scalar*) in, int* z_block_offsets,
                                                      ZoneAttribute* zs, Scalar mu,
                                                      Scalar invMasses) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  ZoneAttribute& z = zs[bid];

  if (z.converged)
    return;

  const int& n_z_blocks = z_block_offsets[bid + 1] - z_block_offsets[bid];
  const int& offset = z_block_offsets[bid];

  __shared__ Scalar sdata[IZO_BLOCK_SIZE];

  sdata[tid] = tid < n_z_blocks ? in[offset + tid] : 0;

  __syncthreads();

#if USE_BASELINE_REDUCE
  for (unsigned int stride = 1; stride < IZO_BLOCK_SIZE; stride *= 2) {
    if (tid % (2 * stride) == 0) {
      sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
  }
#else
  if (blockSize >= 512) {
    if (tid < 256) {
      sdata[tid] += sdata[tid + 256];
    }
    __syncthreads();
  }
  if (blockSize >= 256) {
    if (tid < 128) {
      sdata[tid] += sdata[tid + 128];
    }
    __syncthreads();
  }
  if (blockSize >= 128) {
    if (tid < 64) {
      sdata[tid] += sdata[tid + 64];
    }
    __syncthreads();
  }

  if (tid < 32)
    WarpReduce<blockSize>(sdata, tid);
#endif  // USE_BASELINE_REDUCE

  if (tid == 0) {
    const Scalar& Ot = sdata[0];
    const Scalar& G2 = z.G2;
    const Scalar& LAMBDA2 = z.lambda2;

    // backtracking line search
    Scalar f =
        0.5f * invMasses * z.O + 0.5f * mu * z.C2 - 0.5f * LAMBDA2 / mu;  //objective function
    Scalar ft = 0.5f * invMasses * Ot + 0.5f * mu * z.C2t -
                0.5f * LAMBDA2 / mu;  //advanced objective function
    Scalar s = z.s;
    if (s < EPSILON_S || ft < (f - ALPHA * s * G2) || MathFunctions::abs(f - ft) < EPSILON_F) {
      z.wolfe_condition_satisfied = true;
      if (s < EPSILON_S) {
        z.line_search_reason = 1;
      } else if (ft < (f - ALPHA * s * G2)) {
        z.line_search_reason = 2;
      } else if (MathFunctions::abs(f - ft) < EPSILON_F) {
        z.line_search_reason = 3;
      }
#ifdef IZO_DEBUG
      printf(
          "zone %d wolfe condition satisfied, s: %.12f, objective: %.12f, constraints2: %.12f, "
          "gradients2: %.12f\n",
          zid, s, O, C2, G2);
#endif  // IZO_DEBUG
    }
    z.Ot = Ot;
    z.f = f;
    z.ft = ft;
  }
}

void ParallelImpactZoneOptimizer::LineSearchStep(thrust::device_vector<Scalar>& objectives) {
  thrust::device_vector<Scalar> partial_sum_objectives(n_total_node_blocks_);
  ReduceLocal<Scalar, REDUCE_NODE, false>(objectives, partial_sum_objectives);

  BacktrackingLineSearchAdvanced_Kernel<IZO_BLOCK_SIZE><<<zones_.n_colors, IZO_BLOCK_SIZE>>>(
      pointer(partial_sum_objectives), pointer(node_block_offsets_), pointer(zone_attributes_), mu_,
      inv_mass);
  CUDA_CHECK_LAST();
}

__global__ void UpdateConvergency_Kernel(int n_zones, ZoneAttribute* zone_attributes) {
  GET_CUDA_ID(zid, n_zones);

  ZoneAttribute& z = zone_attributes[zid];

  if (z.s < EPSILON_S  // step length less than the prescribed threshold
      //|| MathFunctions::abs(z.f - z.ft) < EPSILON_F		// augmented objective function not descending
  ) {
    if (z.s < EPSILON_S) {
      z.converge_reason = 1;
    }
    z.converged = true;
  }
}

void ParallelImpactZoneOptimizer::UpdateConvergency() {
  CUDA_CALL(UpdateConvergency_Kernel, zones_.n_colors)
  (zones_.n_colors, pointer(zone_attributes_));
  CUDA_CHECK_LAST();
}

}  // namespace ImpactZoneOptimization
}  // namespace XRTailor