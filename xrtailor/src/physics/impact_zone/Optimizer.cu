#include <xrtailor/physics/impact_zone/Optimizer.cuh>

#include <xrtailor/utils/FileSystemUtils.hpp>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>

namespace XRTailor {

//#define IZO_SEQ_WRITE_LOG

__host__ __device__ Scalar ClampViolation(Scalar x, int sign) {
  return sign < 0 ? MathFunctions::max(x, static_cast<Scalar>(0.0))
                  : (sign > 0 ? MathFunctions::min(x, static_cast<Scalar>(0.0)) : x);
}

__global__ void Initialize_Kernel(int n_nodes, const Node* const* nodes, Vector3* x) {
  GET_CUDA_ID(idx, n_nodes);

  x[idx] = nodes[idx]->x;
}

__global__ void Finalize_Kernel(int n_nodes, const Vector3* x, Node** nodes) {
  GET_CUDA_ID(idx, n_nodes);

  nodes[idx]->x = x[idx];
}

__global__ void ComputeCoefficient_Kernel(int n_constraints, const Scalar* lambda, Scalar mu,
                                          const int* signs, Scalar* c) {
  GET_CUDA_ID(idx, n_constraints);

  c[idx] = ClampViolation(c[idx] + lambda[idx] / mu, signs[idx]);
}

__global__ void ComputeSquare_Kernel(int n_constraints, Scalar* c) {
  GET_CUDA_ID(idx, n_constraints);

  c[idx] = MathFunctions::sqr(c[idx]);
}

__global__ void ComputeLambda2_Kernel(int n_constraints, Scalar* lambda, Scalar* lambda2) {
  GET_CUDA_ID(idx, n_constraints);

  lambda2[idx] = MathFunctions::sqr(lambda[idx]);
}

__global__ void UpdateMultiplier_Kernel(int n_constraints, const Scalar* c, const int* signs,
                                        Scalar mu, Scalar* lambda) {
  GET_CUDA_ID(idx, n_constraints);

  lambda[idx] = ClampViolation(lambda[idx] + mu * c[idx], signs[idx]);
}

__global__ void ComputeNorm2_Kernel(int n_nodes, const Vector3* x, Scalar* x2) {
  GET_CUDA_ID(idx, n_nodes);

  x2[idx] = glm::dot(x[idx], x[idx]);
}

/**
 * @brief Evaluate next position using the step size
 * @param n_nodes Number of nodes
 * @param x Current position
 * @param gradient The descent direction
 * @param s The step size
 * @param xt The next position
 * @return 
*/
__global__ void ComputeNextX_Kernel(int n_nodes, const Vector3* x, const Vector3* gradient,
                                    Scalar s, Vector3* xt) {
  GET_CUDA_ID(idx, n_nodes);

  xt[idx] = x[idx] - s * gradient[idx];
}

__global__ void ChebyshevAccelerate_Kernel(int n_nodes, Scalar omega, Vector3* next_X,
                                           Vector3* previous_X) {
  GET_CUDA_ID(idx, n_nodes);

  next_X[idx] = omega * (next_X[idx] - previous_X[idx]) + previous_X[idx];
}

Optimizer::Optimizer() {}

Optimizer::~Optimizer() {}

void Optimizer::Initialize(thrust::device_vector<Vector3>& x) const {
  CUDA_CALL(Initialize_Kernel, n_nodes_)
  (n_nodes_, pointer(nodes_), pointer(x));
}

void Optimizer::Finalize(const thrust::device_vector<Vector3>& x) {
  CUDA_CALL(Finalize_Kernel, n_nodes_)
  (n_nodes_, pointer(x), pointer(nodes_));
}

Scalar Optimizer::Value(const thrust::device_vector<Vector3>& x, Scalar& O2, Scalar& C2,
                        Scalar& L2) {
  thrust::device_vector<Scalar> c(n_constraints_);
  thrust::device_vector<int> signs(n_constraints_);
  Constraint(x, c, signs);

  CUDA_CALL(ComputeCoefficient_Kernel, n_constraints_)
  (n_constraints_, pointer(lambda_), mu_, pointer(signs), pointer(c));
  CUDA_CHECK_LAST();

  CUDA_CALL(ComputeSquare_Kernel, n_constraints_)
  (n_constraints_, pointer(c));
  CUDA_CHECK_LAST();

  C2 = thrust::reduce(c.begin(), c.end());
  thrust::device_vector<Scalar> lambda2(n_constraints_);

  CUDA_CALL(ComputeLambda2_Kernel, n_constraints_)
  (n_constraints_, pointer(lambda_), pointer(lambda2));
  CUDA_CHECK_LAST();

  L2 = thrust::reduce(lambda2.begin(), lambda2.end());

  return Objective(x, O2) + 0.5f * mu_ * C2 - 0.5f * L2 / mu_;
}

void Optimizer::ValueAndGradient(const thrust::device_vector<Vector3>& x, Scalar& value,
                                 thrust::device_vector<Vector3>& gradient, Scalar& O2, Scalar& C2,
                                 Scalar& L2) {
  thrust::device_vector<Scalar> c(n_constraints_);
  thrust::device_vector<int> signs(n_constraints_);
  Constraint(x, c, signs);

  ObjectiveGradient(x, gradient);

  CUDA_CALL(ComputeCoefficient_Kernel, n_constraints_)
  (n_constraints_, pointer(lambda_), mu_, pointer(signs), pointer(c));
  CUDA_CHECK_LAST();

  ConstraintGradient(x, c, mu_, gradient);

  CUDA_CALL(ComputeSquare_Kernel, n_constraints_)
  (n_constraints_, pointer(c));
  CUDA_CHECK_LAST();

  C2 = thrust::reduce(c.begin(), c.end());

  thrust::device_vector<Scalar> lambda2(n_constraints_);
  CUDA_CALL(ComputeLambda2_Kernel, n_constraints_)
  (n_constraints_, pointer(lambda_), pointer(lambda2));
  CUDA_CHECK_LAST();

  L2 = thrust::reduce(lambda2.begin(), lambda2.end());

  value = Objective(x, O2) + 0.5f * mu_ * C2 - 0.5f * L2 / mu_;
}

void Optimizer::UpdateMultiplier(const thrust::device_vector<Vector3>& x) {
  thrust::device_vector<Scalar> c(n_constraints_);
  thrust::device_vector<int> signs(n_constraints_);
  Constraint(x, c, signs);

  CUDA_CALL(UpdateMultiplier_Kernel, n_constraints_)
  (n_constraints_, pointer(c), pointer(signs), mu_, pointer(lambda_));
  CUDA_CHECK_LAST();
}

void Optimizer::Solve(int frame_index, int global_iter) {
  mu_ = 1e3f;
  // backtracking line search
  Scalar f;                     // objective function
  Scalar ft;                    // advanced objective function
  Scalar s = 1e-3f * n_nodes_;  // step length
  Scalar alpha = 0.5f;          // control paramter in the Wolfe condition
  Scalar omega = 1.0f;          // Chebyshev control parameter

  bool success = false;
  thrust::device_vector<Vector3> next_X(n_nodes_), current_X(n_nodes_), previous_X(n_nodes_),
      gradient(n_nodes_);
  thrust::device_vector<Scalar> gradient2(n_nodes_);
  Vector3* nextXPointer = pointer(next_X);
  Vector3* current_X_pointer = pointer(current_X);
  Vector3* previous_X_pointer = pointer(previous_X);
  Vector3* gradient_pointer = pointer(gradient);
  Scalar* gradient2_pointer = pointer(gradient2);
  lambda_.assign(n_constraints_, 0);

  Initialize(current_X);
  previous_X = current_X;

  Scalar O2, C2, L2;

  for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
#ifdef IZO_SEQ_WRITE_LOG
    ofs << "[iter " + std::to_string(iter) + "]\n";
#endif
    // Step1: compute gradient descent direction
    ValueAndGradient(current_X, f, gradient, O2, C2, L2);

    CUDA_CALL(ComputeNorm2_Kernel, n_nodes_)
    (n_nodes_, gradient_pointer, gradient2_pointer);
    CUDA_CHECK_LAST();

    Scalar norm2 = thrust::reduce(gradient2.begin(), gradient2.end());

    int t = 0;
    // Step2: find advancing timestep
    s /= 0.7f;

#ifdef IZO_SEQ_WRITE_LOG
    checkCudaErrors(cudaDeviceSynchronize());
    ofs << std::fixed << std::setprecision(14) << "  - z"
        << " | "
        << "s: " << s << ",f: " << f << ",ft: " << ft << ",C2: " << C2 << ",O: " << O2
        << ",G2: " << norm2 << ",L2: " << L2 << "\n";
#endif
#ifdef IZO_SEQ_WRITE_LOG
    ofs << "  - Line search\n";
#endif

    do {
      s *= 0.7f;
      CUDA_CALL(ComputeNextX_Kernel, n_nodes_)
      (n_nodes_, current_X_pointer, gradient_pointer, s, nextXPointer);
      CUDA_CHECK_LAST();

      ft = Value(next_X, O2, C2, L2);

#ifdef IZO_SEQ_WRITE_LOG
      checkCudaErrors(cudaDeviceSynchronize());
      ofs << std::fixed << std::setprecision(14) << "    t" << t << " | "
          << "s: " << s << ",f: " << f << ",ft: " << ft << ",C2: " << C2 << ",O: " << O2 << "\n";
#endif
      t++;
    } while (ft >= f - alpha * s * norm2  // Wolfe condition
             && s >= EPSILON_S            // step length threshold
             && MathFunctions::abs(f - ft) >= EPSILON_F);

#ifdef IZO_SEQ_WRITE_LOG
    checkCudaErrors(cudaDeviceSynchronize());
    ofs << "  - line search finished in " << t << " steps\n";
    ofs << std::fixed << std::setprecision(14) << "  - z"
        << " | "
        << "s: " << s << ",f: " << f << ",ft: " << ft << ",C2: " << C2 << ",L2: " << L2
        << ",O: " << O2 << ",G2: " << norm2 << ",Wolfe: 1, reason: ";
    if (ft < f - alpha * s * norm2) {
      ofs << "(Wolfe)\n";
    } else if (s < EPSILON_S) {
      ofs << "(EPSILON_S)\n";
    } else if (MathFunctions::abs(f - ft) < EPSILON_F) {
      ofs << "(OBJECTIVE not descending)\n";
    }
#endif

    if (s < EPSILON_S || MathFunctions::abs(f - ft) < EPSILON_F) {
#ifdef IZO_SEQ_WRITE_LOG
      ofs << "  - update convergency: ";
      if (s < EPSILON_S) {
        ofs << "s < EPSILON_S" << std::endl;
      } else if (MathFunctions::abs(f - ft) < EPSILON_F) {
        ofs << "OBJ not descending" << std::endl;
      }
      ofs << "Optimizer converged in " + std::to_string(iter) + " iters\n";
#endif
      success = true;
      break;
    }

    // Step3: update vertex position
    //if (iter == 10)
    //	omega = 2.0f / (2.0f - RHO2);
    //else if (iter > 10)
    //	omega = 4.0 / (4.0 - RHO2 * omega);

    //CUDA_CALL(ChebyshevAccelerate_Kernel, n_nodes)
    //	(n_nodes, omega, nextXPointer, previous_X_pointer);
    //CUDA_CHECK_LAST();

    previous_X = current_X;
    current_X = next_X;

    UpdateMultiplier(current_X);
  }
  Finalize(current_X);

  if (success) {
#ifdef IZO_SEQ_WRITE_LOG
    ofs << "Finalize success\n";
#endif  // IZO_SEQ_WRITE_LOG

  } else {
#ifdef IZO_SEQ_WRITE_LOG
    ofs << "Local: Impact zone solver failed to converge in " << std::to_string(MAX_ITERATIONS)
        << " iterations";
#endif  // IZO_SEQ_WRITE_LOG
  }
}

}  // namespace XRTailor
