#include <xrtailor/physics/impact_zone/ImpactZoneOptimizer.cuh>

#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/unique.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>

namespace XRTailor {

__global__ void CollectCollisionNodes_Kernel(int n_constraints, const Impact* impacts, int deform,
                                             int* indices, Node** nodes) {
  GET_CUDA_ID(i, n_constraints);

  const Impact& impact = impacts[i];
  for (int j = 0; j < 4; j++) {
    int index = 4 * i + j;
    Node* node = impact.nodes[j];
    if (deform == 1 || node->is_free) {
      indices[index] = index;
      nodes[index] = node;
    } else {
      indices[index] = -1;
      nodes[index] = nullptr;
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
  Scalar inv_mass = node->is_free ? node->inv_mass : (static_cast<Scalar>(1.0) / obstacle_mass);
  inv[idx] = inv_mass;
}

__global__ void CollisionObjective_Kernel(int n_nodes, const Node* const* nodes,
                                          Scalar obstacle_mass, const Vector3* x,
                                          Scalar* objectives) {
  GET_CUDA_ID(idx, n_nodes);

  const Node* node = nodes[idx];
  Scalar mass = node->is_free ? (1 / node->inv_mass) : obstacle_mass;
  objectives[idx] = mass * glm::dot(x[idx] - node->x1, x[idx] - node->x1);
}

/**
 * @brief Evaluate descent direction
 * @param n_nodes Number of independent nodes involved into the optimization
 * @param nodes independent nodes involved into the optimization
 * @param inv_mass Sum of inverse masses of all nodes 
 * @param obstacle_mass Mass of the obstacle
 * @param x Current position
 * @param gradient The computed gradient
 * @return 
*/
__global__ void CollisionObjectiveGradient_Kernel(int n_nodes, const Node* const* nodes,
                                                  Scalar inv_mass, Scalar obstacle_mass,
                                                  const Vector3* x, Vector3* gradient) {
  GET_CUDA_ID(idx, n_nodes);

  const Node* node = nodes[idx];
  Scalar mass = node->is_free ? (1 / node->inv_mass) : obstacle_mass;
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

ImpactZoneOptimizer::ImpactZoneOptimizer(const thrust::device_vector<Impact>& impacts,
                                         Scalar thickness, int deform, Scalar obstacle_mass)
    : impacts_(impacts), thickness_(thickness), obstacle_mass_(obstacle_mass) {
  n_constraints_ = impacts.size();

  // initialize collision nodes and corresponding indices
  nodes_.resize(4 * n_constraints_);
  Node** nodes_pointer = pointer(nodes_);
  thrust::device_vector<int> node_indices(
      4 * n_constraints_);  // indices are represented as thread index offset
  int* node_indices_pointer = pointer(node_indices);
  // filter non-deformable nodes
  CUDA_CALL(CollectCollisionNodes_Kernel, n_constraints_)
  (n_constraints_, pointer(impacts), deform, node_indices_pointer, nodes_pointer);
  CUDA_CHECK_LAST();
  nodes_.erase(thrust::remove(nodes_.begin(), nodes_.end(), nullptr), nodes_.end());
  node_indices.erase(thrust::remove(node_indices.begin(), node_indices.end(), -1),
                     node_indices.end());
  // sort the indices via node memory address
  // duplicated nodes will be next to each other
  thrust::sort_by_key(nodes_.begin(), nodes_.end(), node_indices.begin());

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
  nodes_.erase(thrust::unique(nodes_.begin(), nodes_.end()), nodes_.end());
  n_nodes_ = nodes_.size();
  thrust::device_vector<Scalar> inv(n_nodes_);
  CUDA_CALL(CollisionInv_Kernel, n_nodes_)
  (n_nodes_, nodes_pointer, obstacle_mass, pointer(inv));
  CUDA_CHECK_LAST();

  inv_mass_ = thrust::reduce(inv.begin(), inv.end()) / n_nodes_;
}

ImpactZoneOptimizer::~ImpactZoneOptimizer() {}

Scalar ImpactZoneOptimizer::Objective(const thrust::device_vector<Vector3>& x, Scalar& O2) const {
  thrust::device_vector<Scalar> objectives(n_nodes_);
  CUDA_CALL(CollisionObjective_Kernel, n_nodes_)
  (n_nodes_, pointer(nodes_), obstacle_mass_, pointer(x), pointer(objectives));
  CUDA_CHECK_LAST();

  O2 = thrust::reduce(objectives.begin(), objectives.end());

  return 0.5f * inv_mass_ * O2;
}

void ImpactZoneOptimizer::ObjectiveGradient(const thrust::device_vector<Vector3>& x,
                                            thrust::device_vector<Vector3>& gradient) const {
  CUDA_CALL(CollisionObjectiveGradient_Kernel, n_nodes_)
  (n_nodes_, pointer(nodes_), inv_mass_, obstacle_mass_, pointer(x), pointer(gradient));
  CUDA_CHECK_LAST();
}

void ImpactZoneOptimizer::Constraint(const thrust::device_vector<Vector3>& x,
                                     thrust::device_vector<Scalar>& constraints,
                                     thrust::device_vector<int>& signs) const {
  CUDA_CALL(CollisionConstraint_Kernel, n_constraints_)
  (n_constraints_, pointer(impacts_), pointer(indices_), thickness_, pointer(x),
   pointer(constraints), pointer(signs));
  CUDA_CHECK_LAST();
}

void ImpactZoneOptimizer::ConstraintGradient(const thrust::device_vector<Vector3>& x,
                                             const thrust::device_vector<Scalar>& coefficients,
                                             Scalar mu,
                                             thrust::device_vector<Vector3>& gradient) const {
  thrust::device_vector<int> grad_indices = indices_;
  thrust::device_vector<Vector3> grad(4 * n_constraints_);
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

}  // namespace XRTailor