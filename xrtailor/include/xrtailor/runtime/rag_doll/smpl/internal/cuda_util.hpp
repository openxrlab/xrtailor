#pragma once
#ifndef SMPLX_CUDA_UTIL_A2261F55_FC88_4379_AADF_81E399304BB7
#define SMPLX_CUDA_UTIL_A2261F55_FC88_4379_AADF_81E399304BB7

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace smplx {

namespace cuda_util {

template <class ScalarType, class MatType>
__host__ void to_host_eigen_matrix(ScalarType* d_data, int rows, int cols, MatType& out) {
  out.resize(rows, cols);
  cudaMemcpy(out.data(), d_data, out.size() * sizeof(ScalarType), cudaMemcpyDeviceToHost);
}

template <class ScalarType, class EigenType>
__host__ void from_host_eigen_matrix(ScalarType*& d_data, const EigenType& src) {
  const size_t dsize = src.size() * sizeof(src.data()[0]);
  cudaMalloc((void**)&d_data, dsize);
  cudaMemcpy(d_data, src.data(), dsize, cudaMemcpyHostToDevice);
}

template <int Option>
__host__ void from_host_eigen_sparse_matrix(internal::GPUSparseMatrix& d_data,
                                            const Eigen::SparseMatrix<float, Option>& src) {
  const size_t nnz = src.nonZeros();
  d_data.nnz = (int)nnz;
  d_data.cols = (int)src.cols();
  d_data.rows = (int)src.rows();
  cudaMalloc((void**)&d_data.values, nnz * sizeof(float));
  cudaMemcpy(d_data.values, src.valuePtr(), nnz * sizeof(float), cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_data.inner, nnz * sizeof(int));
  cudaMemcpy(d_data.inner, src.innerIndexPtr(), nnz * sizeof(int), cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_data.outer, (src.outerSize() + 1) * sizeof(int));
  cudaMemcpy(d_data.outer, src.outerIndexPtr(), (src.outerSize() + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
}

}  // namespace cuda_util
}  // namespace smplx
#endif  // ifndef SMPLX_CUDA_UTIL_A2261F55_FC88_4379_AADF_81E399304BB7
