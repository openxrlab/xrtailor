#pragma once

#include <vector_types.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <xrtailor/physics/representative_triangle/RTriangle.cuh>
#include <xrtailor/utils/Timer.hpp>
#include <xrtailor/core/Common.hpp>
#include <xrtailor/core/Common.cuh>
#include <xrtailor/core/DeviceHelper.cuh>
#include <xrtailor/physics/predictive_contact/Constraint.cuh>

namespace XRTailor {

class BVH;

namespace PredictiveContact {

void SetSelfCollisionParams(SimParams* host_params);

class EEConstraint {
 public:
  EEConstraint();

  void GenerateStackless(std::shared_ptr<BVH> bvh, std::shared_ptr<RTriangle> r_tri, uint* overlaps,
                          uint* n_overlaps);

  void Solve(const Vector3* positions, Vector3* predicted, uint* indices, const Scalar* inv_masses,
             Vector3* deltas, int* delta_counts, Scalar radius);

  ~EEConstraint();

 public:
  thrust::device_vector<EEContact> contacts;
  thrust::device_vector<uint> n_contacts;
  thrust::device_vector<uint> insertion_idx;
};

class VFConstraint {
 public:
  VFConstraint();

  void GenerateStackless(std::shared_ptr<BVH> bvh, std::shared_ptr<RTriangle> r_tri, uint* overlaps,
                          uint* n_overlaps);

  void Solve(const Vector3* positions, Vector3* predicted, uint* indices, const Scalar* inv_masses,
             Vector3* deltas, int* delta_counts, Scalar radius);

  ~VFConstraint();

 public:
  thrust::device_vector<VFContact> contacts;
  thrust::device_vector<uint> n_contacts;
  thrust::device_vector<uint> insertion_idx;
};

class VVConstraint {
 public:
  VVConstraint();

  void GenerateStackless(std::shared_ptr<BVH> bvh, std::shared_ptr<XRTailor::RTriangle> r_tri,
                          uint* overlaps, uint* n_overlaps);

  void Solve(const Vector3* positions, Vector3* predicted, const Scalar* inv_masses, Vector3* deltas,
             int* delta_counts, Scalar radius);

  ~VVConstraint();

 public:
  thrust::device_vector<VVContact> contacts;
  thrust::device_vector<uint> n_contacts;
  thrust::device_vector<uint> insertion_idx;
};

}  // namespace PredictiveContact
}  // namespace XRTailor
