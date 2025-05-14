#include <xrtailor/physics/dynamics/BasicSolverHelper.cuh>

#include <xrtailor/core/DeviceHelper.cuh>

namespace XRTailor {
namespace BasicConstraint {

__global__ void SolveBend_Kernel(BendConstraint* constraints, Node** nodes, uint color,
                                 CONST(uint*) colors, const Scalar dt, int iter,
                                 const uint n_constraints) {
  GET_CUDA_ID(id, n_constraints);

  if (color != colors[id])
    return;

  BendConstraint c = constraints[id];

  uint idx1 = c.idx0;
  uint idx2 = c.idx1;
  uint idx3 = c.idx2;
  uint idx4 = c.idx3;
  Scalar expected_angle = c.theta;

  Scalar w1 = nodes[idx1]->inv_mass;
  Scalar w2 = nodes[idx2]->inv_mass;
  Scalar w3 = nodes[idx3]->inv_mass;
  Scalar w4 = nodes[idx4]->inv_mass;

  Vector3 p1 = nodes[idx1]->x;
  Vector3 p2 = nodes[idx2]->x - p1;
  Vector3 p3 = nodes[idx3]->x - p1;
  Vector3 p4 = nodes[idx4]->x - p1;
  Vector3 n1 = glm::normalize(glm::cross(p2, p3));
  Vector3 n2 = glm::normalize(glm::cross(p2, p4));

  Scalar d =
      MathFunctions::Clamp(glm::dot(n1, n2), static_cast<Scalar>(0.0), static_cast<Scalar>(1.0));
  Scalar angle = acos(d);

  // cross product for two equal vector produces NAN
  if (angle < EPSILON || isnan(d))
    return;

  Vector3 q3 =
      (glm::cross(p2, n2) + glm::cross(n1, p2) * d) / (glm::length(glm::cross(p2, p3)) + EPSILON);
  Vector3 q4 =
      (glm::cross(p2, n1) + glm::cross(n2, p2) * d) / (glm::length(glm::cross(p2, p4)) + EPSILON);
  Vector3 q2 =
      -(glm::cross(p3, n2) + glm::cross(n1, p3) * d) / (glm::length(glm::cross(p2, p3)) + EPSILON) -
      (glm::cross(p4, n1) + glm::cross(n2, p4) * d) / (glm::length(glm::cross(p2, p4)) + EPSILON);
  Vector3 q1 = -q2 - q3 - q4;

  Scalar xpbd_bend = c.compliance / dt / dt;
  Scalar denom = xpbd_bend + (w1 * glm::dot(q1, q1) + w2 * glm::dot(q2, q2) +
                              w3 * glm::dot(q3, q3) + w4 * glm::dot(q4, q4));
  if (denom < EPSILON)
    return;
  Scalar lambda = sqrt(static_cast<Scalar>(1.0) - d * d) * (angle - expected_angle) / denom;
  Scalar scaling = static_cast<Scalar>(1.0);

  Vector3 corr1 = scaling * w1 * lambda * q1;
  Vector3 corr2 = scaling * w2 * lambda * q2;
  Vector3 corr3 = scaling * w3 * lambda * q3;
  Vector3 corr4 = scaling * w4 * lambda * q4;

  for (int j = 0; j < 3; j++) {
    atomicAdd(&(nodes[idx1]->x[j]), corr1[j]);
    atomicAdd(&(nodes[idx2]->x[j]), corr2[j]);
    atomicAdd(&(nodes[idx3]->x[j]), corr3[j]);
    atomicAdd(&(nodes[idx4]->x[j]), corr4[j]);
  }
}

void SolveBend(BendConstraint* constraints, Node** nodes, uint color, CONST(uint*) colors,
               const Scalar dt, int iter, const uint n_constraints) {
  CUDA_CALL(SolveBend_Kernel, n_constraints)
  (constraints, nodes, color, colors, dt, iter, n_constraints);
}

__global__ void SolveStretch_Kernel(StretchConstraint* constraints, Node** nodes, Vector3* prev,
                                    uint color, CONST(uint*) colors, const Scalar dt, int iter,
                                    Scalar sor, const uint num_constraints) {
  GET_CUDA_ID(id, num_constraints);

  if (colors[id] != color)
    return;

  if (iter == 0)
    constraints[id].lambda = static_cast<Scalar>(0);

  StretchConstraint c = constraints[id];

  int idx1 = c.idx0;
  int idx2 = c.idx1;
  Scalar expected_distance = c.rest_length;

  Vector3 diff = nodes[idx1]->x - nodes[idx2]->x;
  Scalar distance = glm::length(diff);
  Scalar w1 = nodes[idx1]->inv_mass;
  Scalar w2 = nodes[idx2]->inv_mass;

  if (distance != expected_distance && w1 + w2 > static_cast<Scalar>(0.0)) {
    Vector3 gradient = diff / (distance + EPSILON);

    Scalar alpha_tilde = c.compliance / (dt * dt);
    Scalar constraint = distance - expected_distance;

    // eq. 18
    // gradient of C_{j} is normal so that grad_C_{j}*inv_mass*grad_{j}^{transpose} is inv_mass
    Scalar delta_lambda =
        (-constraint - alpha_tilde * constraints[id].lambda) / (w1 + w2 + alpha_tilde);

    Vector3 corr = delta_lambda * gradient;

    atomicAdd(&constraints[id].lambda, delta_lambda);

    Vector3 corr1 = corr * w1;
    Vector3 corr2 = -corr * w2;

    //glm::vec3 prev1 = prev[idx1];
    //glm::vec3 prev2 = prev[idx2];
    //glm::vec3 pred1 = nodes[idx1]->x + corr1;
    //glm::vec3 pred2 = nodes[idx2]->x + corr2;

    //corr1 = sor * (pred1 - prev1) + prev1 - nodes[idx1]->x;
    //corr2 = sor * (pred2 - prev2) + prev2 - nodes[idx2]->x;

    for (int j = 0; j < 3; j++) {
      atomicAdd(&nodes[idx1]->x[j], corr1[j]);
      atomicAdd(&nodes[idx2]->x[j], corr2[j]);
    }
  }
}

void SolveStretch(StretchConstraint* constraints, Node** nodes, Vector3* prev, uint color,
                  uint* colors, const Scalar dt, int iter, Scalar sor, int stretch_size) {
  CUDA_CALL(SolveStretch_Kernel, stretch_size)
  (constraints, nodes, prev, color, colors, dt, iter, sor, stretch_size);
}

}  // namespace BasicConstraint
}  // namespace XRTailor