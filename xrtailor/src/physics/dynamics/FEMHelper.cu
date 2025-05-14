#include <xrtailor/physics/dynamics/FEMHelper.cuh>
#include <xrtailor/math/MathFunctions.cuh>

//#define FEM_STRAIN_DEBUG
//#define FEM_BEND_DEBUG

namespace XRTailor {
namespace FEM {

__device__ bool ComputeStrainMeta(Vector3 p0, Vector3 p1, Vector3 p2, Scalar& area,
                                  Mat2x2& inv_rest_mat) {
  Vector3 normal0 = glm::cross(p1 - p0, p2 - p0);
  area = static_cast<Scalar>(0.5) * glm::length(normal0);

  Vector3 axis0_1 = glm::normalize(p1 - p0);
  Vector3 axis0_2 = glm::normalize(glm::cross(normal0, axis0_1));

  Mat3x2 p;
  p[0] = Vector2(glm::dot(p0, axis0_2), glm::dot(p0, axis0_1));
  p[1] = Vector2(glm::dot(p1, axis0_2), glm::dot(p1, axis0_1));
  p[2] = Vector2(glm::dot(p2, axis0_2), glm::dot(p2, axis0_1));

  Mat2x2 P;
  P[0][0] = p[0][0] - p[2][0];
  P[1][0] = p[0][1] - p[2][1];
  P[0][1] = p[1][0] - p[2][0];
  P[1][1] = p[1][1] - p[2][1];

  Scalar det = glm::determinant(P);
#ifdef FEM_STRAIN_DEBUG
  printf("P (%.4f, %.4f, %.4f, %.4f), det: %.4f\n", P[0][0], P[1][0], P[0][1], P[1][1], det);
#endif  // FEM_STRAIN_DEBUG

  if (fabs(det) > EPSILON) {
    inv_rest_mat = glm::inverse(P);
    return true;
  }

  return false;
}

__global__ void InitStrain_Kernel(StrainConstraint* elements, Vector3* positions,
                                  const Face* const* faces, Scalar xx_stiffness, Scalar yy_stiffness,
                                  Scalar xy_stiffness, Scalar xy_poisson_ratio, Scalar yx_poisson_ratio,
                                  int n_faces) {
  GET_CUDA_ID(id, n_faces);

  int idx0 = faces[id]->nodes[0]->index;
  int idx1 = faces[id]->nodes[1]->index;
  int idx2 = faces[id]->nodes[2]->index;

  Vector3 pos0 = positions[idx0];
  Vector3 pos1 = positions[idx1];
  Vector3 pos2 = positions[idx2];

  Scalar area;
  Mat2x2 inv_rest_mat;
  ComputeStrainMeta(pos0, pos1, pos2, area, inv_rest_mat);

  StrainConstraint element = StrainConstraint();
  element.xx_stiffness = xx_stiffness;
  element.yy_stiffness = yy_stiffness;
  element.xy_stiffness = xy_stiffness;
  element.xy_poisson_ratio = xy_poisson_ratio;
  element.yx_poisson_ratio = yx_poisson_ratio;
  element.idx0 = idx0;
  element.idx1 = idx1;
  element.idx2 = idx2;
  element.area = area;
  element.inv_rest_mat = inv_rest_mat;

#ifdef FEM_STRAIN_DEBUG
  printf(
      "face %d: [%d(%.4f, %.4f, %.4f), %d(%.4f, %.4f, %.4f) ,%d(%.4f, %.4f, %.4f)], area: %.6f, "
      "inv_rest_mat: [%.2f, %.2f, %.2f, %.2f] material:[%.2f, %.2f, %.2f, %.2f, %.2f]\n",
      id, idx0, pos0.x, pos0.y, pos0.z, idx1, pos1.x, pos1.y, pos1.z, idx2, pos2.x, pos2.y, pos2.z,
      area, inv_rest_mat[0][0], inv_rest_mat[0][1], inv_rest_mat[1][0], inv_rest_mat[1][1], xx_stiffness,
      yy_stiffness, xy_stiffness, xy_poisson_ratio, yx_poisson_ratio);
#endif  // FEM_STRAIN_DEBUG

  elements[id] = element;
}

void InitStrain(StrainConstraint* constraints, Vector3* rest_positions, const Face* const* faces,
                Scalar xx_stiffness, Scalar yy_stiffness, Scalar xy_stiffness, Scalar xy_poisson_ratio,
                Scalar yx_poisson_ratio, uint strain_size) {
  CUDA_CALL(InitStrain_Kernel, strain_size)
  (constraints, rest_positions, faces, xx_stiffness, yy_stiffness, xy_stiffness, xy_poisson_ratio,
   yx_poisson_ratio, strain_size);

  checkCudaErrors(cudaDeviceSynchronize());
}

__device__ __inline__ bool ComputeStrainCorrection(Vector3& p0, Vector3& p1, Vector3& p2,
                                                   Scalar& inv_mass0, Scalar& inv_mass1,
                                                   Scalar& inv_mass2, Scalar& area,
                                                   Mat2x2& inv_rest_mat, Scalar youngs_modulus_x,
                                                   Scalar youngs_modulus_y, Scalar youngs_modulus_shear,
                                                   Scalar poisson_ratio_xy, Scalar poisson_ratio_yx,
                                                   Vector3& corr0, Vector3& corr1, Vector3& corr2) {
  // Orthotropic elasticity tensor
  Mat3x3 C = Mat3(0.0);
  C[0][0] = youngs_modulus_x / (static_cast<Scalar>(1.0) - poisson_ratio_xy * poisson_ratio_yx);
  C[0][1] = youngs_modulus_x * poisson_ratio_yx /
            (static_cast<Scalar>(1.0) - poisson_ratio_xy * poisson_ratio_yx);
  C[1][1] = youngs_modulus_y / (static_cast<Scalar>(1.0) - poisson_ratio_xy * poisson_ratio_yx);
  C[1][0] = youngs_modulus_y * poisson_ratio_xy /
            (static_cast<Scalar>(1.0) - poisson_ratio_xy * poisson_ratio_yx);
  C[2][2] = youngs_modulus_shear;

  // Determine \partial x/\partial m_i
  Mat3x2 F;
  const Vector3 p13 = p0 - p2;
  const Vector3 p23 = p1 - p2;
  F[0][0] = p13[0] * inv_rest_mat[0][0] + p23[0] * inv_rest_mat[1][0];
  F[0][1] = p13[0] * inv_rest_mat[0][1] + p23[0] * inv_rest_mat[1][1];
  F[1][0] = p13[1] * inv_rest_mat[0][0] + p23[1] * inv_rest_mat[1][0];
  F[1][1] = p13[1] * inv_rest_mat[0][1] + p23[1] * inv_rest_mat[1][1];
  F[2][0] = p13[2] * inv_rest_mat[0][0] + p23[2] * inv_rest_mat[1][0];
  F[2][1] = p13[2] * inv_rest_mat[0][1] + p23[2] * inv_rest_mat[1][1];

  // epsilon = 0.5(F^T * F - I)
  Mat2x2 epsilon;
  epsilon[0][0] = 0.5f * (F[0][0] * F[0][0] + F[1][0] * F[1][0] + F[2][0] * F[2][0] - 1.0f);  // xx
  epsilon[1][1] = 0.5f * (F[0][1] * F[0][1] + F[1][1] * F[1][1] + F[2][1] * F[2][1] - 1.0f);  // yy
  epsilon[0][1] = 0.5f * (F[0][0] * F[0][1] + F[1][0] * F[1][1] + F[2][0] * F[2][1]);         // xy
  epsilon[1][0] = epsilon[0][1];

  // P(F) = det(F) * C*E * F^-T => E = green strain
  Mat2x2 stress;
  stress[0][0] = C[0][0] * epsilon[0][0] + C[0][1] * epsilon[1][1] + C[0][2] * epsilon[0][1];
  stress[1][1] = C[1][0] * epsilon[0][0] + C[1][1] * epsilon[1][1] + C[1][2] * epsilon[0][1];
  stress[0][1] = C[2][0] * epsilon[0][0] + C[2][1] * epsilon[1][1] + C[2][2] * epsilon[0][1];
  stress[1][0] = stress[0][1];

  Mat3x2 piola_kirchhoff_stress = stress * F;

  Scalar psi = static_cast<Scalar>(0.0);
  for (int j = 0; j < 2; j++)
    for (int k = 0; k < 2; k++)
      psi += epsilon[j][k] * stress[j][k];
  psi = static_cast<Scalar>(0.5) * psi;
  Scalar energy = area * psi;

  Mat3x2 H = glm::transpose(inv_rest_mat) * piola_kirchhoff_stress * area;
  Vector3 grad_c[3];

  for (int j = 0; j < 3; j++) {
    grad_c[0][j] = H[j][0];
    grad_c[1][j] = H[j][1];
  }

  grad_c[2] = -grad_c[0] - grad_c[1];

  Scalar sum_norm_grad_c = inv_mass0 * glm::dot(grad_c[0], grad_c[0]);
  sum_norm_grad_c += inv_mass1 * glm::dot(grad_c[1], grad_c[1]);
  sum_norm_grad_c += inv_mass2 * glm::dot(grad_c[2], grad_c[2]);

  // exit early if required
  if (fabs(sum_norm_grad_c) > EPSILON) {
    // compute scaling factor
    const Scalar s = energy / sum_norm_grad_c;

    // update positions
    corr0 = -(s * inv_mass0) * grad_c[0];
    corr1 = -(s * inv_mass1) * grad_c[1];
    corr2 = -(s * inv_mass2) * grad_c[2];

    return true;
  }

  return false;
}

__global__ void SolveStrain_Kernel(StrainConstraint* elements, Node** nodes, uint color,
                                   uint* colors, int n_faces) {
  GET_CUDA_ID(id, n_faces);

  if (colors[id] != color)
    return;

  auto element = elements[id];

  int idx0 = element.idx0;
  int idx1 = element.idx1;
  int idx2 = element.idx2;

  Vector3 pred0 = nodes[idx0]->x;
  Vector3 pred1 = nodes[idx1]->x;
  Vector3 pred2 = nodes[idx2]->x;

  Scalar inv_mass0 = nodes[idx0]->inv_mass;
  Scalar inv_mass1 = nodes[idx1]->inv_mass;
  Scalar inv_mass2 = nodes[idx2]->inv_mass;

  Vector3 corr0, corr1, corr2;

  const bool res = ComputeStrainCorrection(
      pred0, pred1, pred2, inv_mass0, inv_mass1, inv_mass2, element.area, element.inv_rest_mat,
      element.xx_stiffness, element.yy_stiffness, element.xy_stiffness, element.xy_poisson_ratio,
      element.yx_poisson_ratio, corr0, corr1, corr2);

  if (res) {
    int reorder = idx0 + idx1 + idx2;
    if (inv_mass0 > EPSILON)
      AtomicAddX(nodes, idx0, corr0, reorder);
    if (inv_mass1 > EPSILON)
      AtomicAddX(nodes, idx1, corr1, reorder);
    if (inv_mass2 > EPSILON)
      AtomicAddX(nodes, idx2, corr2, reorder);
  }
}

void SolveStrain(StrainConstraint* constraints, Node** nodes, uint color, uint* colors,
                 uint strain_size) {
  CUDA_CALL(SolveStrain_Kernel, strain_size)
  (constraints, nodes, color, colors, strain_size);
  CUDA_CHECK_LAST();
}

__device__ __inline__ void ComputeHessianEnergy(const Vector3& p0, const Vector3& p1,
                                                const Vector3& p2, const Vector3& p3, Mat4x4& Q) {
  const Vector3 e0 = p1 - p0;
  const Vector3 e1 = p2 - p0;
  const Vector3 e2 = p3 - p0;
  const Vector3 e3 = p2 - p1;
  const Vector3 e4 = p3 - p1;

  const Scalar c01 = MathFunctions::CotTheta(e0, e1);
  const Scalar c02 = MathFunctions::CotTheta(e0, e2);
  const Scalar c03 = MathFunctions::CotTheta(-e0, e3);
  const Scalar c04 = MathFunctions::CotTheta(-e0, e4);

  const Scalar A0 = MathFunctions::TriangleArea(p0, p1, p2);
  const Scalar A1 = MathFunctions::TriangleArea(p1, p0, p3);

  const Scalar coeff = static_cast<Scalar>(3.0) / (A0 + A1);
  const Vector4 K = {c03 + c04, c01 + c02, -c01 - c03, -c02 - c04};

  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      Q[j][i] = K[i] * K[j];

  Q = coeff * Q;
}

__device__ __inline__ void ComputeHessian(const Vector3& p0, const Vector3& p1, const Vector3& p2,
                                          const Vector3& p3, Mat4x4& Q) {
  const Vector3 x[4] = {p2, p3, p0, p1};

  const Vector3 e0 = x[1] - x[0];
  const Vector3 e1 = x[2] - x[0];
  const Vector3 e2 = x[3] - x[0];
  const Vector3 e3 = x[2] - x[1];
  const Vector3 e4 = x[3] - x[1];

  const Scalar c01 = MathFunctions::CotTheta(e0, e1);
  const Scalar c02 = MathFunctions::CotTheta(e0, e2);
  const Scalar c03 = MathFunctions::CotTheta(-e0, e3);
  const Scalar c04 = MathFunctions::CotTheta(-e0, e4);

  const Scalar A0 = static_cast<Scalar>(0.5) * glm::length(glm::cross(e0, e1));
  const Scalar A1 = static_cast<Scalar>(0.5) * glm::length(glm::cross(e0, e2));

  const Scalar coeff = static_cast<Scalar>(-3.0) / (static_cast<Scalar>(2.0) * (A0 + A1));
  const Scalar K[4] = {c03 + c04, c01 + c02, -c01 - c03, -c02 - c04};
  const Scalar K2[4] = {coeff * K[0], coeff * K[1], coeff * K[2], coeff * K[3]};

  for (int j = 0; j < 4; j++) {
    for (int k = 0; k < j; k++) {
      Q[j][k] = Q[k][j] = K[j] * K2[k];
    }
    Q[j][j] = K[j] * K2[j];
  }
}

__global__ void InitIsometricBending_Kernel(IsometricBendingConstraint* constraints,
                                            Vector3* rest_positions, uint* bend_indices,
                                            Scalar stiffness, int n_constraints) {
  GET_CUDA_ID(id, n_constraints);

  int idx0 = bend_indices[id * 4u + 0u];
  int idx1 = bend_indices[id * 4u + 1u];
  int idx2 = bend_indices[id * 4u + 2u];
  int idx3 = bend_indices[id * 4u + 3u];

  Vector3 pos0 = rest_positions[idx0];
  Vector3 pos1 = rest_positions[idx1];
  Vector3 pos2 = rest_positions[idx2];
  Vector3 pos3 = rest_positions[idx3];

  Mat4x4 Q;
  ComputeHessianEnergy(pos0, pos1, pos2, pos3, Q);

  IsometricBendingConstraint c;
  c.idx0 = idx0;
  c.idx1 = idx1;
  c.idx2 = idx2;
  c.idx3 = idx3;
  c.lambda = static_cast<Scalar>(0.0);
  c.Q = Q;
  c.stiffness = stiffness;

  constraints[id] = c;
}

void InitIsometricBending(IsometricBendingConstraint* constraints, Vector3* rest_positions,
                          uint* bend_indices, Scalar stiffness, uint bend_size) {
  checkCudaErrors(cudaDeviceSynchronize());

  CUDA_CALL(InitIsometricBending_Kernel, bend_size)
  (constraints, rest_positions, bend_indices, stiffness, bend_size);

  checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void SolveIsometricBending_Kernel(IsometricBendingConstraint* constraints, Node** nodes,
                                             uint color, uint* colors, int iter, Scalar dt,
                                             int n_constraints) {
  GET_CUDA_ID(id, n_constraints);

  if (colors[id] != color)
    return;

  if (iter == 0)
    constraints[id].lambda = static_cast<Scalar>(0.0);

  const IsometricBendingConstraint c = constraints[id];

  const int i0 = c.idx0;
  const int i1 = c.idx1;
  const int i2 = c.idx2;
  const int i3 = c.idx3;

  const Vector3 x[4] = {nodes[i0]->x, nodes[i1]->x, nodes[i2]->x, nodes[i3]->x};
  const Vector4 inv_mass = {nodes[i0]->inv_mass, nodes[i1]->inv_mass, nodes[i2]->inv_mass,
                           nodes[i3]->inv_mass};

  Scalar energy = static_cast<Scalar>(0.0);
  const Mat4x4 Q = c.Q;
  for (int k = 0; k < 4; k++)
    for (int j = 0; j < 4; j++)
      energy += Q[j][k] * glm::dot(x[k], x[j]);
  energy *= static_cast<Scalar>(0.5);

  if (fabs(energy) < EPSILON)
    return;

  Mat4x3 derivatives(0);
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      derivatives[j] += Q[j][i] * x[i];

  const Vector4 derivative_norms = {
      glm::dot(derivatives[0], derivatives[0]), glm::dot(derivatives[1], derivatives[1]),
      glm::dot(derivatives[2], derivatives[2]), glm::dot(derivatives[3], derivatives[3])};

  const Scalar division_coeff = glm::dot(inv_mass, derivative_norms);
  const Scalar alpha_tilde = 1.0f / (c.stiffness * dt * dt);
  const Scalar delta_lambda = (-energy - alpha_tilde * c.lambda) / (division_coeff + alpha_tilde);


  Vector3 corr0 = delta_lambda * inv_mass[0] * derivatives[0];
  Vector3 corr1 = delta_lambda * inv_mass[1] * derivatives[1];
  Vector3 corr2 = delta_lambda * inv_mass[2] * derivatives[2];
  Vector3 corr3 = delta_lambda * inv_mass[3] * derivatives[3];

#ifdef FEM_BEND_DEBUG
  printf("c%d (%d, %d, %d, %d), "
  	"corr0(%.4f, %.4f, %.4f), corr1(%.4f, %.4f, %.4f), "
  	"corr2(%.4f, %.4f, %.4f), corr3(%.4f, %.4f, %.4f)\n",
  	id, i0, i1, i2, i3,
  	corr0.x, corr0.y, corr0.z,
  	corr1.x, corr1.y, corr1.z,
  	corr2.x, corr2.y, corr2.z,
  	corr3.x, corr3.y, corr3.z
  );
#endif

  for (int j = 0; j < 3; j++) {
    atomicAdd(&nodes[i0]->x[j], corr0[j]);
    atomicAdd(&nodes[i1]->x[j], corr1[j]);
    atomicAdd(&nodes[i2]->x[j], corr2[j]);
    atomicAdd(&nodes[i3]->x[j], corr3[j]);
  }

  constraints[id].lambda += delta_lambda;
}

void SolveIsometricBending(IsometricBendingConstraint* constraints, Node** nodes, uint color,
                           uint* colors, int iter, Scalar dt, uint bend_size) {
  CUDA_CALL(SolveIsometricBending_Kernel, bend_size)
  (constraints, nodes, color, colors, iter, dt, bend_size);
}

}  // namespace FEM
}  // namespace XRTailor