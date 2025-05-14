#pragma once

#include <tuple>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "helper_cuda.h"
#include "helper_math.h"

#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/sort.h>

#include <xrtailor/core/Scalar.hpp>

#define CONST(type) const type const
#define GET_CUDA_ID(id, maxID)                     \
  uint id = blockIdx.x * blockDim.x + threadIdx.x; \
  if (id >= maxID)                                 \
  return
#define GET_CUDA_ID_NO_RETURN(id, maxID) uint id = blockIdx.x * blockDim.x + threadIdx.x
#define LBVH_MAX_BUFFER_SIZE 256

// __CUDACC__ defines whether nvcc is steering compilation or not
#ifdef __CUDACC__
#define CUDA_CALL(func, total_threads)                                             \
  uint func##_num_blocks, func##_num_threads;                                      \
  XRTailor::ComputeGridSize(total_threads, func##_num_blocks, func##_num_threads); \
  func<<<func##_num_blocks, func##_num_threads>>>
#define CUDA_CALL_S(func, total_threads, stream)                                   \
  if (total_threads == 0)                                                          \
    return;                                                                        \
  uint func##_num_blocks, func##_num_threads;                                      \
  XRTailor::ComputeGridSize(total_threads, func##_num_blocks, func##_num_threads); \
  func<<<func##_num_blocks, func##_num_threads, stream>>>
#define CUDA_CALL_V(func, ...) func<<<__VA_ARGS__>>>
#else
#define CUDA_CALL(func, total_threads)
#define CUDA_CALL_S(func, total_threads, stream)
#define CUDA_CALL_V(func, ...)
#endif

namespace XRTailor {

const uint GRID_SIZE = 64;
const uint BLOCK_SIZE = 256;

inline void ComputeGridSize(const uint& n, uint& num_blocks, uint& num_threads) {
  if (n == 0) {
    num_blocks = GRID_SIZE;
    num_threads = BLOCK_SIZE;
    return;
  }
  num_threads = min(n, BLOCK_SIZE);
  num_blocks = (n % num_threads != 0) ? (n / num_threads + 1) : (n / num_threads);
}

}  // namespace XRTailor