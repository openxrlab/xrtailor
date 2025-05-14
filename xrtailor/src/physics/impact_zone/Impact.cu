#include <xrtailor/physics/impact_zone/Impact.cuh>

namespace XRTailor {

__host__ __device__ Impact::Impact() : t(static_cast<Scalar>(-1)), n(Vector3(0)) {}

__host__ __device__ bool Impact::operator<(const Impact& impact) const {
  return t < impact.t;
}

__host__ __device__ ImpactDebug::ImpactDebug() : t(static_cast<Scalar>(-1)), n(Vector3(0)) {}

}  // namespace XRTailor