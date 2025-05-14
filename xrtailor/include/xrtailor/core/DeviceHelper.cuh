#pragma once

#include <vector_types.h>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include <xrtailor/physics/impact_zone/Impact.cuh>
#include <xrtailor/physics/predictive_contact/Constraint.cuh>
#include <xrtailor/physics/repulsion/Proximity.cuh>

namespace XRTailor {

#define CUDA_CHECK_LAST() cudaCheckLast(__FILE__, __LINE__)

struct IsNull {
  template <typename T>
  __device__ bool operator()(const T* p) const {
    return p == nullptr;
  }

  __device__ bool operator()(int index) const { return index < 0; }

  __device__ bool operator()(const Impact& impact) const {
    return impact.t < static_cast<Scalar>(0.0);
  }

  __device__ bool operator()(const PredictiveContact::EEContact& contact) const {
    return contact.e0p0 == -1;
  }

  __device__ bool operator()(const VFProximity& proximity) const {
    return proximity.node == nullptr;
  }
};

__device__ void AtomicAddX(Node** address, int index, Vector3 val, int reorder);

__device__ void AtomicAdd(Vector3* address, int index, Vector3 val, int reorder);

__device__ void AtomicAdd(Vector3* address, int index, Vector3 val);

void cudaCheckLast(const char* file, int line);

template <typename T>
static T* pointer(thrust::host_vector<T>& v, int offset = 0) {
  return thrust::raw_pointer_cast(v.data() + offset);
}

template <typename T>
static const T* pointer(const thrust::host_vector<T>& v, int offset = 0) {
  return thrust::raw_pointer_cast(v.data() + offset);
}

template <typename T>
static T* pointer(thrust::device_vector<T>& v, int offset = 0) {
  return thrust::raw_pointer_cast(v.data() + offset);
}

template <typename T>
static const T* pointer(const thrust::device_vector<T>& v, int offset = 0) {
  return thrust::raw_pointer_cast(v.data() + offset);
}

template <typename T>
void MemcpyHostToHost(const thrust::host_vector<T>& src_data, T* tgt_data, const uint count) {
  checkCudaErrors(cudaMemcpy(tgt_data, pointer(src_data), count * sizeof(T), cudaMemcpyHostToHost));
}

}  // namespace XRTailor