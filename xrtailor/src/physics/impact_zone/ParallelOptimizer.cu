#include <xrtailor/physics/impact_zone/ParallelOptimizer.cuh>

#include <xrtailor/utils/FileSystemUtils.hpp>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <cuda/std/type_traits>

namespace XRTailor {
namespace ImpactZoneOptimization {

//#define IZO_WRITE_LOG

#ifdef IZO_WRITE_LOG
void PrintVector3(const thrust::device_vector<Vector3>& data) {
  checkCudaErrors(cudaDeviceSynchronize());

  thrust::host_vector<Vector3> h_data = data;

  int cnt = 0;
  for (int i = 0; i < h_data.size(); i++) {
    std::cout << h_data[i].x << "," << h_data[i].y << "," << h_data[i].z << " ";
  }
  std::cout << std::endl;
}

void PrintScalar(const thrust::device_vector<Scalar>& data) {
  checkCudaErrors(cudaDeviceSynchronize());

  thrust::host_vector<Scalar> h_data = data;

  int cnt = 0;
  for (int i = 0; i < h_data.size(); i++) {
    std::cout << h_data[i] << " ";
  }
  std::cout << std::endl;
}
#endif

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

__global__ void UpdateMultiplier_Kernel(int n_constraints, const Scalar* c, const int* signs,
                                        Scalar mu, Scalar* lambda) {
  GET_CUDA_ID(idx, n_constraints);

  lambda[idx] = ClampViolation(lambda[idx] + mu * c[idx], signs[idx]);
}

__global__ void ComputeNextX_Kernel(int n_nodes, const Vector3* x, const Vector3* gradient,
                                    Scalar s, Vector3* xt) {
  GET_CUDA_ID(idx, n_nodes);

  xt[idx] = x[idx] - s * gradient[idx];
}

__global__ void NextX_Kernel(CONST(int*) z_global_indices, CONST(int*) z_local_indices,
                             const Vector3* x, const Vector3* gradient, Vector3* xt,
                             ZoneAttribute* zone_attributes) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int zid = z_global_indices[bid];
  int z_local_id = z_local_indices[bid];

  const ZoneAttribute& z = zone_attributes[zid];

  if (z_local_id * blockDim.x + tid >= z.n_nodes || z.wolfe_condition_satisfied || z.converged)
    return;

  const int& node_idx = z.node_offset + z_local_id * blockDim.x + tid;

  xt[node_idx] = x[node_idx] - z.s * gradient[node_idx];
}

__global__ void ChebyshevAccelerate_Kernel(CONST(int*) z_global_indices,
                                           CONST(int*) z_local_indices, Scalar omega,
                                           Vector3* next_X, Vector3* previous_X,
                                           ZoneAttribute* zone_attributes) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int zid = z_global_indices[bid];
  int z_local_id = z_local_indices[bid];

  const ZoneAttribute& z = zone_attributes[zid];

  if (z_local_id * blockDim.x + tid >= z.n_nodes || z.converged)
    return;

  const int& node_idx = z.node_offset + z_local_id * blockDim.x + tid;

  next_X[node_idx] = omega * (next_X[node_idx] - previous_X[node_idx]) + previous_X[node_idx];
}

ParallelOptimizer::ParallelOptimizer() {}

ParallelOptimizer::~ParallelOptimizer() {}

void ParallelOptimizer::Initialize(thrust::device_vector<Vector3>& x) const {
  CUDA_CALL(Initialize_Kernel, n_nodes_)
  (n_nodes_, pointer(nodes_), pointer(x));
}

void ParallelOptimizer::Finalize(const thrust::device_vector<Vector3>& x) {
  CUDA_CALL(Finalize_Kernel, n_nodes_)
  (n_nodes_, pointer(x), pointer(nodes_));
}

void ParallelOptimizer::TotalGradient(const thrust::device_vector<Vector3>& x,
                                      thrust::device_vector<Vector3>& gradients,
                                      thrust::device_vector<Scalar>& constraints,
                                      thrust::device_vector<int>& signs) {
  ObjectiveGradient(x, gradients);

  CUDA_CALL(ComputeCoefficient_Kernel, n_constraints_)
  (n_constraints_, pointer(lambda_), mu_, pointer(signs), pointer(constraints));
  CUDA_CHECK_LAST();

  ConstraintGradient(x, constraints, mu_, gradients);
}

void ParallelOptimizer::UpdateMultiplier(const thrust::device_vector<Vector3>& x) {
  thrust::device_vector<Scalar> c(n_constraints_);
  thrust::device_vector<int> signs(n_constraints_);
  Constraint(x, c, signs);

  CUDA_CALL(UpdateMultiplier_Kernel, n_constraints_)
  (n_constraints_, pointer(c), pointer(signs), mu_, pointer(lambda_));
  CUDA_CHECK_LAST();
}

__global__ void ResetWolfeCondition_Kernel(ZoneAttribute* zone_attributes, int n_zones) {
  GET_CUDA_ID(zid, n_zones);

  ZoneAttribute& z = zone_attributes[zid];
  z.wolfe_condition_satisfied = z.converged ? true : false;
}

__global__ void InitStepLength_Kernel(ZoneAttribute* zone_attributes, int n_zones) {
  GET_CUDA_ID(zid, n_zones);

  ZoneAttribute& z = zone_attributes[zid];
  z.s = STEP_LENGTH_COEFF_SCALAR * z.n_nodes;
}

__global__ void PreconditionStepLength_Kernel(ZoneAttribute* zone_attributes, int n_zones) {
  GET_CUDA_ID(zid, n_zones);
  ZoneAttribute& z = zone_attributes[zid];
  if (z.converged)
    return;
  z.s /= 0.7f;
}

__global__ void AdjustStepLength_Kernel(ZoneAttribute* zone_attributes, int n_zones) {
  GET_CUDA_ID(zid, n_zones);
  ZoneAttribute& z = zone_attributes[zid];
  if (z.converged || z.wolfe_condition_satisfied)
    return;
  z.s *= 0.7f;
}

int GetNumBlocks(int n) {
  int num_blocks = (n + kBlockSize - 1) / kBlockSize;
  return num_blocks;
}

template <unsigned int blockSize>
__device__ void WarpReduce(volatile Scalar* cache, int tid) {
  if (blockSize >= 64)
    cache[tid] += cache[tid + 32];
  if (blockSize >= 32)
    cache[tid] += cache[tid + 16];
  if (blockSize >= 16)
    cache[tid] += cache[tid + 8];
  if (blockSize >= 8)
    cache[tid] += cache[tid + 4];
  if (blockSize >= 4)
    cache[tid] += cache[tid + 2];
  if (blockSize >= 2)
    cache[tid] += cache[tid + 1];
}

template <typename InType, int reduce_type, bool norm2, unsigned int blockSize>
__global__ void WarpReduceLocal_Kernel(CONST(InType*) in, Scalar* out, int* z_global_indices,
                                       int* z_local_indices, CONST(ZoneAttribute*) zs) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int zid = z_global_indices[bid];
  int z_local_id = z_local_indices[bid];

  const ZoneAttribute& z = zs[zid];

  if (z.converged)
    return;

  int offset, N;
  if constexpr (reduce_type == REDUCE_IMPACT) {
    offset = z.impact_offset + IZO_BLOCK_SIZE * z_local_id + tid;
    N = z.n_impacts;
  } else if constexpr (reduce_type == REDUCE_NODE) {
    offset = z.node_offset + IZO_BLOCK_SIZE * z_local_id + tid;
    N = z.n_nodes;
  }

  Scalar val = 0;

  if ((IZO_BLOCK_SIZE * z_local_id + tid) < N) {
    if (norm2) {
      if constexpr (::cuda::std::is_same_v<InType, Vector3>)
        val = glm::dot(in[offset], in[offset]);
      else if constexpr (::cuda::std::is_same_v<InType, Scalar>)
        val = in[offset] * in[offset];
    } else {
      if constexpr (::cuda::std::is_same_v<InType, Scalar>)
        val = in[offset];
    }
  }

  __shared__ Scalar sdata[IZO_BLOCK_SIZE];

  sdata[tid] = val;

  __syncthreads();

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

  if (tid == 0) {
    out[bid] = sdata[0];
  }
}

template <int reduce_object, unsigned int blockSize>
__global__ void WarpReduceGlobal_Kernel(CONST(Scalar*) in, int* z_block_offsets,
                                        ZoneAttribute* zs) {
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

  if (tid == 0) {
    if (reduce_object == REDUCE_OBJECTIVE)
      z.O = sdata[0];
    else if (reduce_object == REDUCE_CONSTRAINT)
      z.C2 = sdata[0];
    else if (reduce_object == REDUCE_CONSTRAINT_FT)
      z.C2t = sdata[0];
    else if (reduce_object == REDUCE_GRADIENT)
      z.G2 = sdata[0];
    else if (reduce_object == REDUCE_LAMBDA)
      z.lambda2 = sdata[0];
  }
}

template <typename InType, int reduce_type, bool norm2>
__global__ void ReduceBaselineLocal_Kernel(CONST(InType*) in, Scalar* out, int* z_global_indices,
                                           int* z_local_indices, CONST(ZoneAttribute*) zs) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int zid = z_global_indices[bid];
  int z_local_id = z_local_indices[bid];

  const ZoneAttribute& z = zs[zid];

  if (z.converged)
    return;

  int offset, N;
  if constexpr (reduce_type == REDUCE_IMPACT) {
    offset = z.impact_offset + IZO_BLOCK_SIZE * z_local_id + tid;
    N = z.n_impacts;
  } else if constexpr (reduce_type == REDUCE_NODE) {
    offset = z.node_offset + IZO_BLOCK_SIZE * z_local_id + tid;
    N = z.n_nodes;
  }

  Scalar val = 0;

  if ((IZO_BLOCK_SIZE * z_local_id + tid) < N) {
    if (norm2) {
      if constexpr (::cuda::std::is_same_v<InType, Vector3>)
        val = glm::dot(in[offset], in[offset]);
      else if constexpr (::cuda::std::is_same_v<InType, Scalar>)
        val = in[offset] * in[offset];
    } else {
      if constexpr (::cuda::std::is_same_v<InType, Scalar>)
        val = in[offset];
    }
  }

  __shared__ Scalar s_local_sum[IZO_BLOCK_SIZE];

  s_local_sum[tid] = val;

  __syncthreads();

  for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
    if (tid % (2 * stride) == 0) {
      s_local_sum[tid] += s_local_sum[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    out[bid] = s_local_sum[0];
  }
}

template <int reduce_object>
__global__ void ReduceBaselineGlobal_Kernel(CONST(Scalar*) in, int* z_block_offsets,
                                            ZoneAttribute* zs) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  ZoneAttribute& z = zs[bid];

  if (z.converged)
    return;

  const int& n_z_blocks = z_block_offsets[bid + 1] - z_block_offsets[bid];
  const int& offset = z_block_offsets[bid];

  __shared__ Scalar s_local_sum[IZO_BLOCK_SIZE];

  s_local_sum[tid] = tid < n_z_blocks ? in[offset + tid] : 0;

  __syncthreads();

  for (unsigned int stride = 1; stride < IZO_BLOCK_SIZE; stride *= 2) {
    if (tid % (2 * stride) == 0) {
      s_local_sum[tid] += s_local_sum[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    if (reduce_object == REDUCE_OBJECTIVE)
      z.O = s_local_sum[0];
    else if (reduce_object == REDUCE_CONSTRAINT)
      z.C2 = s_local_sum[0];
    else if (reduce_object == REDUCE_CONSTRAINT_FT)
      z.C2t = s_local_sum[0];
    else if (reduce_object == REDUCE_GRADIENT) {
      z.G2 = s_local_sum[0];
    } else if (reduce_object == REDUCE_LAMBDA)
      z.lambda2 = s_local_sum[0];
  }
}

template <typename InType, int reduce_type, bool norm2>
void ParallelOptimizer::ReduceLocal(const thrust::device_vector<InType>& in,
                                    thrust::device_vector<Scalar>& out) {

  if constexpr (reduce_type == REDUCE_IMPACT) {
#if USE_BASELINE_REDUCE
    ReduceBaselineLocal_Kernel<InType, reduce_type, norm2>
        <<<n_total_impact_blocks_, IZO_BLOCK_SIZE>>>(
            pointer(in), pointer(out), pointer(impact_block_indices_),
            pointer(impact_block_local_indices_), pointer(zone_attributes_));
#else
    WarpReduceLocal_Kernel<InType, reduce_type, norm2, IZO_BLOCK_SIZE>
        <<<n_total_impact_blocks_, IZO_BLOCK_SIZE>>>(
            pointer(in), pointer(out), pointer(impact_block_indices_),
            pointer(impact_block_local_indices_), pointer(zone_attributes_));
#endif  // USE_BASELINE_REDUCE
    CUDA_CHECK_LAST();
  } else if constexpr (reduce_type == REDUCE_NODE) {
#if USE_BASELINE_REDUCE
    ReduceBaselineLocal_Kernel<InType, reduce_type, norm2>
        <<<n_total_node_blocks_, IZO_BLOCK_SIZE>>>(
            pointer(in), pointer(out), pointer(node_block_indices_),
            pointer(node_block_local_indices_), pointer(zone_attributes_));
#else
    WarpReduceLocal_Kernel<InType, reduce_type, norm2, IZO_BLOCK_SIZE>
        <<<n_total_node_blocks_, IZO_BLOCK_SIZE>>>(
            pointer(in), pointer(out), pointer(node_block_indices_),
            pointer(node_block_local_indices_), pointer(zone_attributes_));
#endif  // USE_BASELINE_REDUCE
    CUDA_CHECK_LAST();
  }
}

__global__ void ComputeNorm2_Kernel(int n_nodes, const Vector3* x, Scalar* x2) {
  GET_CUDA_ID(idx, n_nodes);

  x2[idx] = glm::dot(x[idx], x[idx]);
}

#ifdef IZO_WRITE_LOG
__global__ void ComputeNorm2Debug_Kernel(int n_nodes, int node_offset, const Vector3* x,
                                        Scalar* x2) {
  GET_CUDA_ID(idx, n_nodes);

  x2[node_offset + idx] = glm::dot(x[node_offset + idx], x[node_offset + idx]);
}
#endif

template <typename InType, int reduce_type, int reduce_object, bool norm2>
void ParallelOptimizer::Reduce(const thrust::device_vector<InType>& in) {
  int reduce_buffer_size;
  int* block_offsets_ptr;
  if (reduce_type == REDUCE_IMPACT) {
    block_offsets_ptr = pointer(impact_block_offsets_);
    reduce_buffer_size = n_total_impact_blocks_;
  } else if (reduce_type == REDUCE_NODE) {
    block_offsets_ptr = pointer(node_block_offsets_);
    reduce_buffer_size = n_total_node_blocks_;
  }

  thrust::device_vector<Scalar> reduce1_out(reduce_buffer_size);

  ReduceLocal<InType, reduce_type, norm2>(in, reduce1_out);

#if USE_BASELINE_REDUCE
  ReduceBaselineGlobal_Kernel<reduce_object><<<zones_.n_colors, IZO_BLOCK_SIZE>>>(
      pointer(reduce1_out), block_offsets_ptr, pointer(zone_attributes_));
#else
  WarpReduceGlobal_Kernel<reduce_object, IZO_BLOCK_SIZE><<<zones_.n_colors, IZO_BLOCK_SIZE>>>(
      pointer(reduce1_out), block_offsets_ptr, pointer(zone_attributes_));
#endif
  CUDA_CHECK_LAST();
}

struct WolfeConditionSatisfiedComp {
  __host__ __device__ bool operator()(const ZoneAttribute& z) const {
    return z.wolfe_condition_satisfied == false;
  }
};

struct ZoneConvergeComp {
  __host__ __device__ bool operator()(const ZoneAttribute& z) const { return z.converged == false; }
};

void ParallelOptimizer::Solve(int frame_index, int global_iter) {
#ifdef IZO_WRITE_LOG
  filesystem::path base_dir = ".\\Debug\\ImpactZone\\local";
  filesystem::path file_name =
      "frame" + std::to_string(frame_index) + "-global_iter" + std::to_string(global_iter) + ".txt";
  filesystem::path full_path = base_dir / file_name;
  std::ofstream ofs(full_path, std::ios::app);

  thrust::host_vector<ZoneAttribute> h_zone_attributes = zone_attributes_;

  ofs << "zone overview:" << std::endl;
  for (int i = 0; i < zones_.n_colors; i++) {
    auto& z = h_zone_attributes[i];
    ofs << "  - z" << i << " | "
        << "n_nodes: " << z.n_nodes << ",n_impacts: " << z.n_impacts
        << ",node_offset: " << z.node_offset << ",impact_offset: " << z.impact_offset << "\n";
  }
  checkCudaErrors(cudaDeviceSynchronize());
#endif  // IZO_WRITE_LOG

  mu_ = 1e3f;
  Scalar omega = 1.0f;
  int n_zones = zones_.n_colors;

  thrust::device_vector<Vector3> next_X(n_nodes_), current_X(n_nodes_), previous_X(n_nodes_),
      gradient(n_nodes_);
  thrust::device_vector<Scalar> objectives(n_nodes_), constraints(n_constraints_);
  thrust::device_vector<int> signs(n_constraints_);

  Vector3* next_X_pointer = pointer(next_X);
  Vector3* current_X_pointer = pointer(current_X);
  Vector3* previous_X_pointer = pointer(previous_X);
  Vector3* gradient_pointer = pointer(gradient);
  Scalar* lambda_pointer = pointer(lambda_);
  lambda_.assign(n_constraints_, 0);

  // block partition
  impact_block_offsets_.resize(1, 0);
  node_block_offsets_.resize(1, 0);
  n_total_impact_blocks_ = n_total_node_blocks_ = 0;

  thrust::host_vector<ZoneAttribute> h_zs = zone_attributes_;
  for (int i = 0; i < zones_.n_colors; i++) {
    const ZoneAttribute& z = h_zs[i];
    int n_impact_blocks = GetNumBlocks(z.n_impacts);
    impact_block_indices_.insert(impact_block_indices_.end(), n_impact_blocks, i);
    n_total_impact_blocks_ += n_impact_blocks;
    impact_block_offsets_.push_back(n_total_impact_blocks_);

    int n_node_blocks = GetNumBlocks(z.n_nodes);
    node_block_indices_.insert(node_block_indices_.end(), n_node_blocks, i);
    n_total_node_blocks_ += n_node_blocks;
    node_block_offsets_.push_back(n_total_node_blocks_);
#ifdef IZO_WRITE_LOG
    ofs << "zone" << std::to_string(i) << " | node_block_size: " << std::to_string(n_node_blocks)
        << ", impact_block_size: " << std::to_string(n_impact_blocks) << std::endl;
#endif
  }

  impact_block_local_indices_.resize(n_total_impact_blocks_, 1);
  thrust::inclusive_scan_by_key(impact_block_indices_.begin(), impact_block_indices_.end(),
                                impact_block_local_indices_.begin(),
                                impact_block_local_indices_.begin());
  thrust::transform(impact_block_local_indices_.begin(), impact_block_local_indices_.end(),
                    impact_block_local_indices_.begin(), thrust::placeholders::_1 - 1);

  node_block_local_indices_.resize(n_total_node_blocks_, 1);
  thrust::inclusive_scan_by_key(node_block_indices_.begin(), node_block_indices_.end(),
                                node_block_local_indices_.begin(),
                                node_block_local_indices_.begin());
  thrust::transform(node_block_local_indices_.begin(), node_block_local_indices_.end(),
                    node_block_local_indices_.begin(), thrust::placeholders::_1 - 1);

#ifdef IZO_WRITE_LOG
  {
    checkCudaErrors(cudaDeviceSynchronize());
    ofs << "Total node blocks: " << std::to_string(n_total_node_blocks_)
        << ", impact blocks: " << std::to_string(n_total_impact_blocks_) << std::endl;

    thrust::host_vector<int> h_node_block_global_indices = node_block_indices_;
    ofs << "h_node_block_global_indices: ";
    for (int i = 0; i < h_node_block_global_indices.size(); i++) {
      ofs << h_node_block_global_indices[i] << " ";
    }
    ofs << std::endl;

    thrust::host_vector<int> h_node_block_local_indices = node_block_local_indices_;
    ofs << "h_node_block_local_indices: ";
    for (int i = 0; i < h_node_block_local_indices.size(); i++) {
      ofs << h_node_block_local_indices[i] << " ";
    }
    ofs << std::endl;

    thrust::host_vector<int> h_node_block_offsets = node_block_offsets_;
    ofs << "h_node_block_offsets: ";
    for (int i = 0; i < h_node_block_offsets.size(); i++) {
      ofs << h_node_block_offsets[i] << " ";
    }
    ofs << std::endl;

    thrust::host_vector<int> h_impact_block_global_indices = impact_block_indices_;
    ofs << "h_impact_block_global_indices: ";
    for (int i = 0; i < h_impact_block_global_indices.size(); i++) {
      ofs << h_impact_block_global_indices[i] << " ";
    }
    ofs << std::endl;

    thrust::host_vector<int> h_impact_block_local_indices = impact_block_local_indices_;
    ofs << "h_impact_block_local_indices: ";
    for (int i = 0; i < h_impact_block_local_indices.size(); i++) {
      ofs << h_impact_block_local_indices[i] << " ";
    }
    ofs << std::endl;

    thrust::host_vector<int> h_impact_block_offsets = impact_block_offsets_;
    ofs << "h_impact_block_offsets: ";
    for (int i = 0; i < h_impact_block_offsets.size(); i++) {
      ofs << h_impact_block_offsets[i] << " ";
    }
    ofs << std::endl;
  }
#endif

  Initialize(current_X);
  previous_X = current_X;

  bool success = false;
#ifdef IZO_DEBUG
  printf("### frame %d, global iter: %d, solve step\n", frame_index, global_iter);
#endif
#ifdef IZO_WRITE_LOG
  ofs << "### frame " + std::to_string(frame_index) +
             ", global iter: " + std::to_string(global_iter) + ", solve step\n";
#endif

  CUDA_CALL(InitStepLength_Kernel, n_zones)
  (pointer(zone_attributes_), n_zones);
  CUDA_CHECK_LAST();

  for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
#ifdef IZO_DEBUG
    printf("[iter %d]\n", iter);
    checkCudaErrors(cudaDeviceSynchronize());
#endif
#ifdef IZO_WRITE_LOG
    ofs << "[iter " + std::to_string(iter) + "]\n";
#endif

    CUDA_CALL(ResetWolfeCondition_Kernel, n_zones)
    (pointer(zone_attributes_), n_zones);
    CUDA_CHECK_LAST();

    // Step1: compute objective function values and gradient descent direction
    Constraint(current_X, constraints, signs);
    TotalGradient(current_X, gradient, constraints, signs);

#ifdef IZO_WRITE_LOG
    {
      std::cout << "grad: ";
      PrintVector3(gradient);
      thrust::host_vector<ZoneAttribute> h_zone_attributes = zone_attributes;
      Scalar sum = 0;
      for (int i = 0; i < h_zone_attributes.size(); i++) {
        const ZoneAttribute& z = h_zone_attributes[i];
        int n_nodes = z.n_nodes;
        int node_offset = z.node_offset;
        thrust::device_vector<Scalar> grad2(n_nodes);
        CUDA_CALL(ComputeNorm2Debug_Kernel, n_nodes)
        (n_nodes, node_offset, pointer(gradient), pointer(grad2));
        CUDA_CHECK_LAST();
        Scalar local_sum =
            thrust::reduce(grad2.begin() + node_offset, grad2.begin() + node_offset + n_nodes);
        std::cout << std::fixed << std::setprecision(14) << "z" << i << " G2 " << local_sum << "  ";
        sum += local_sum;
      }

      std::cout << std::fixed << std::setprecision(14) << "thrust sum: " << sum << std::endl;
    }
#endif
    Reduce<Vector3, REDUCE_NODE, REDUCE_GRADIENT, true>(gradient);

    // objectives
    Objective(current_X, objectives);

    Reduce<Scalar, REDUCE_IMPACT, REDUCE_CONSTRAINT, true>(constraints);
    Reduce<Scalar, REDUCE_IMPACT, REDUCE_LAMBDA, true>(lambda_);
    Reduce<Scalar, REDUCE_NODE, REDUCE_OBJECTIVE, false>(objectives);

#ifdef IZO_DEBUG
    printf("  - objective function values && gradient descent direction\n");
    checkCudaErrors(cudaDeviceSynchronize());
#endif  // IZO_DEBUG
#ifdef IZO_WRITE_LOG
    ofs << "  - objective function values && gradient descent direction\n";
#endif
    int t = 0;
    CUDA_CALL(PreconditionStepLength_Kernel, n_zones)
    (pointer(zone_attributes_), n_zones);
    CUDA_CHECK_LAST();

#ifdef IZO_WRITE_LOG
    {
      checkCudaErrors(cudaDeviceSynchronize());
      thrust::host_vector<ZoneAttribute> h_zone_attributes = zone_attributes_;
      for (int i = 0; i < n_zones; i++) {
        auto& z = h_zone_attributes[i];
        ofs << std::fixed << std::setprecision(14) << "  - z" << i << " | "
            << "s: " << z.s << ",f: " << z.f << ",ft: " << z.ft << ",C2: " << z.C2 << ",O: " << z.O
            << ",G2: " << z.G2 << ",L2: " << z.lambda2 << ",wolfe: " << z.wolfe_condition_satisfied
            << ",converged: " << z.converged << "\n";
      }
    }
#endif
#ifdef IZO_WRITE_LOG
    ofs << "  - Line search\n";
#endif
    do {
      CUDA_CALL(AdjustStepLength_Kernel, n_zones)
      (pointer(zone_attributes_), n_zones);
      CUDA_CHECK_LAST();

      NextX_Kernel<<<n_total_node_blocks_, IZO_BLOCK_SIZE>>>(
          pointer(node_block_indices_), pointer(node_block_local_indices_), current_X_pointer,
          gradient_pointer, next_X_pointer, pointer(zone_attributes_));
      CUDA_CHECK_LAST();

      // objectives
      Objective(next_X, objectives);
      Constraint(next_X, constraints, signs);

      CUDA_CALL(ComputeCoefficient_Kernel, n_constraints_)
      (n_constraints_, pointer(lambda_), mu_, pointer(signs), pointer(constraints));
      CUDA_CHECK_LAST();
      Reduce<Scalar, REDUCE_IMPACT, REDUCE_CONSTRAINT_FT, true>(constraints);

      LineSearchStep(objectives);
#ifdef IZO_WRITE_LOG
      checkCudaErrors(cudaDeviceSynchronize());
      thrust::host_vector<ZoneAttribute> h_zone_attributes = zone_attributes_;
      for (int i = 0; i < h_zone_attributes.size(); i++) {
        auto z = h_zone_attributes[i];
        ofs << std::fixed << std::setprecision(14) << "    t" << t << " z" << i << "("
            << z.converged << ")"
            << " | "
            << "s: " << z.s << ",f: " << z.f << ",ft: " << z.ft << ",C2: " << z.C2t
            << ",O: " << z.Ot << std::endl;
      }
#endif

      t++;
    } while (thrust::find_if(zone_attributes_.begin(), zone_attributes_.end(),
                             WolfeConditionSatisfiedComp()) != zone_attributes_.end());
#ifdef IZO_DEBUG
    {
      printf("  - line search finished in %d steps: ", t);
      checkCudaErrors(cudaDeviceSynchronize());
      thrust::host_vector<ZoneAttribute> h_zone_attributes = zone_attributes_;
      for (int i = 0; i < h_zone_attributes.size(); i++) {
        printf("%d(s:%.14f/f:%.14f/ft:%.14f) ", h_zone_attributes[i].wolfe_condition_satisfied,
               h_zone_attributes[i].s, h_zone_attributes[i].f, h_zone_attributes[i].ft);
      }
      printf("\n");
    }
#endif  // IZO_DEBUG

#ifdef IZO_WRITE_LOG
    ofs << "  - line search finished in " << t << " steps\n";
    {
      checkCudaErrors(cudaDeviceSynchronize());
      thrust::host_vector<ZoneAttribute> h_zone_attributes = zone_attributes_;
      for (int i = 0; i < h_zone_attributes.size(); i++) {
        auto z = h_zone_attributes[i];
        ofs << std::fixed << std::setprecision(14) << "    " << i << "(" << z.converged << ")"
            << " | "
            << "s: " << z.s << ",f: " << z.f << ",ft: " << z.ft << ",C2: " << z.C2 << ",O: " << z.O
            << ",wolfe: " << z.wolfe_condition_satisfied;
        if (z.line_search_reason == 1) {
          ofs << "(EPSILON_S)\n";
        } else if (z.line_search_reason == 2) {
          ofs << "(Wolfe)\n";
        } else if (z.line_search_reason == 3) {
          ofs << "(OBJECTIVE not descending)\n";
        }
      }
    }

#endif

    UpdateConvergency();

#ifdef IZO_DEBUG
    {
      checkCudaErrors(cudaDeviceSynchronize());
      thrust::host_vector<ZoneAttribute> h_zone_attributes = zone_attributes_;
      printf("  - update convergency: ");
      for (int i = 0; i < h_zone_attributes.size(); i++) {
        printf("%d ", h_zone_attributes[i].converged);
      }
      printf("\n");
    }
#endif  // IZO_DEBUG

#ifdef IZO_WRITE_LOG
    {
      checkCudaErrors(cudaDeviceSynchronize());
      thrust::host_vector<ZoneAttribute> h_zone_attributes = zone_attributes_;
      ofs << "  - update convergency:" << std::endl;
      ;
      for (int i = 0; i < h_zone_attributes.size(); i++) {
        auto z = h_zone_attributes[i];
        if (z.converge_reason == 0) {
          ofs << "    z" << i << ": " << z.converged << std::endl;
        }
        if (z.converge_reason == 1) {
          ofs << "    z" << i << ": " << z.converged << ", reason: EPSILON_S" << std::endl;
        } else if (z.converge_reason == 2) {
          ofs << "    z" << i << ": " << z.converged << ", reason: OBJ not descending" << std::endl;
        }
      }
    }
#endif

    if (thrust::find_if(zone_attributes_.begin(), zone_attributes_.end(), ZoneConvergeComp()) ==
        zone_attributes_.end()) {
#ifdef IZO_DEBUG
      //printf("  - optimizer converged in %d iters\n", iter);
      printf("Local: Optimizer converged in %d iters\n", iter);
#endif  // IZO_DEBUG

#ifdef IZO_WRITE_LOG
      ofs << "Optimizer converged in " + std::to_string(iter) + " iters\n";
#endif  // IZO_WRITE_DEBUG

      success = true;
      break;
    }
#ifdef IZO_DEBUG
    checkCudaErrors(cudaDeviceSynchronize());
    printf("  - chebyshev accelerate && update Lagrange multiplier\n");
#endif  // IZO_DEBUG

#ifdef IZO_WRITE_LOG
    ofs << "  - chebyshev accelerate && update Lagrange multiplier\n";
#endif  // IZO_WRITE_DEBUG

    //if (iter == 10)
    //	omega = 2.0f / (2.0f - RHO2);
    //else if (iter > 10)
    //	omega = 4.0 / (4.0 - RHO2 * omega);

    //ChebyshevAccelerate_Kernel <<<n_total_node_blocks_, IZO_BLOCK_SIZE>>>
    //	(pointer(node_block_indices_), pointer(node_block_local_indices_), omega, next_X_pointer, previous_X_pointer, pointer(zone_attributes));

    previous_X = current_X;
    current_X = next_X;

    UpdateMultiplier(current_X);
  }

  Finalize(current_X);

  if (success) {
#ifdef IZO_WRITE_LOG
    ofs << "Finalize success\n";
#endif  // IZO_WRITE_LOG

  } else {
#ifdef IZO_WRITE_LOG
    ofs << "Local: Impact zone solver failed to converge in " << std::to_string(MAX_ITERATIONS)
        << " iterations";
#endif  // IZO_WRITE_LOG

    //exit(TAILOR_EXIT::SUCCESS);
  }
#ifdef IZO_WRITE_LOG
  ofs.close();
#endif  // IZO_WRITE_LOG
}

template void ParallelOptimizer::ReduceLocal<Scalar, REDUCE_NODE, false>(
    const thrust::device_vector<Scalar>&, thrust::device_vector<Scalar>&);

template void ParallelOptimizer::ReduceLocal<Scalar, REDUCE_IMPACT, false>(
    const thrust::device_vector<Scalar>&, thrust::device_vector<Scalar>&);

}  // namespace ImpactZoneOptimization
}  // namespace XRTailor
