#include <xrtailor/physics/dynamics/BindingConstraint.cuh>

#include <xrtailor/core/DeviceHelper.cuh>

namespace XRTailor {
BindingConstraint::BindingConstraint() {}

BindingConstraint::~BindingConstraint() {}

void BindingConstraint::Add(const std::vector<uint>& indices,
                            const std::vector<Scalar>& stiffnesses,
                            const std::vector<Scalar>& distances, const uint& prev_num_particles) {
  for (unsigned int i = 0; i < indices.size(); i++) {
    this->binded_indices.push_back(indices[i] + prev_num_particles);
    this->bind_stiffnesses.push_back(stiffnesses[i]);
    this->bind_distances.push_back(distances[i]);
  }
}

__global__ void GenerateBinding_Kernel(Node** nodes, Node** body_nodes, uint* binded_indices,
                                       Scalar* stretch_stiffnesses, Scalar* stretch_distances,
                                       BindingItem* bindings, SkinParam* skin_params, int n_verts) {
  GET_CUDA_ID(idx, n_verts);

  int cloth_vert_idx = binded_indices[idx];

  Vector3 p_cloth = nodes[cloth_vert_idx]->x0;
  Scalar u = static_cast<Scalar>(skin_params[cloth_vert_idx].u);
  Scalar v = static_cast<Scalar>(skin_params[cloth_vert_idx].v);
  Scalar w = static_cast<Scalar>(skin_params[cloth_vert_idx].w);
  int u_idx = skin_params[cloth_vert_idx].idx0;
  int v_idx = skin_params[cloth_vert_idx].idx1;
  int w_idx = skin_params[cloth_vert_idx].idx2;
  Vector3 pu = body_nodes[u_idx]->x0;
  Vector3 pv = body_nodes[v_idx]->x0;
  Vector3 pw = body_nodes[w_idx]->x0;

  Vector3 p_body = u * pu + v * pv + w * pw;

  Scalar d = stretch_distances[idx];
  if (d < EPSILON)
    d = glm::length(p_cloth - p_body);

  bindings[idx] = {cloth_vert_idx, u, v, w, u_idx, v_idx, w_idx, d, stretch_stiffnesses[idx]};
}

void BindingConstraint::Generate(Node** nodes, Node** body_nodes, SkinParam* skin_params) {
  this->n_bindings = this->binded_indices.size();
  this->bindings.resize(n_bindings, BindingItem());

  CUDA_CALL(GenerateBinding_Kernel, n_bindings)
  (nodes, body_nodes, pointer(this->binded_indices), pointer(this->bind_stiffnesses),
   pointer(this->bind_distances), pointer(bindings), skin_params, n_bindings);
}

__global__ void SolveBinding_Kernel(Node** cloth_nodes, Node** obstacle_nodes,
                                    BindingItem* bindings, int n_bindings) {
  GET_CUDA_ID(idx, n_bindings);

  BindingItem binding = bindings[idx];

  Vector3 p_cloth = cloth_nodes[binding.cloth_vert_idx]->x;
  Scalar inv_mass = cloth_nodes[binding.cloth_vert_idx]->inv_mass;

  if (inv_mass < EPSILON)
    return;

  Vector3 pu = obstacle_nodes[binding.u_idx]->x;
  Vector3 pv = obstacle_nodes[binding.v_idx]->x;
  Vector3 pw = obstacle_nodes[binding.w_idx]->x;

  Vector3 fn = MathFunctions::FaceNormal(pu, pv, pw);

  Vector3 p_obstacle = binding.u * pu + binding.v * pv + binding.w * pw;

  Scalar C = glm::length(p_cloth - p_obstacle) - binding.L;

  Vector3 n = glm::normalize(p_cloth - p_obstacle);

  //float s = 1.0f + 1.0f * binding.u * binding.u + 1.0f * binding.v * binding.v + 1.0f * binding.w * binding.w;
  Scalar s = inv_mass;

  s = C / s;

  if (C < static_cast<Scalar>(0))
    return;
  s *= binding.stretch_stiffness;

  Vector3 delta_cloth = -s * inv_mass * n;

  cloth_nodes[binding.cloth_vert_idx]->x += delta_cloth;
}

void BindingConstraint::Solve(Node** cloth_nodes, Node** obstacle_nodes) {
  if (obstacle_nodes == NULL)
    return;
  CUDA_CALL(SolveBinding_Kernel, n_bindings)
  (cloth_nodes, obstacle_nodes, pointer(bindings), n_bindings);
  CUDA_CHECK_LAST();
}
}  // namespace XRTailor