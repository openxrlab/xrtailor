#pragma once

#include <xrtailor/core/DeviceHelper.cuh>

namespace XRTailor {

struct RTriParam {
  // a 6-bit mask, the first three bits indicates the vertices the
  // R-Triangle owns, and the last three, the edges
  uint* _info;
  uint _size;
};

__host__ __device__ __forceinline__ void SetRTriVertex(uint& info, uint n);

__host__ __device__ __forceinline__ void SetRTriEdge(uint& info, uint n);

__host__ __device__ bool RTriVertex(const uint info, uint n);

__host__ __device__ bool RTriEdge(const uint info, uint n);

/*
	* \brief Implementation of 'representative triangles' introduced in 'Curtis S,
	* Tamstorf R, Manocha D. Fast collision detection for deformable models using
	* representative-triangles[C]//Proceedings of the 2008 symposium on Interactive
	* 3D graphics and games. 2008: 61-69.'
	*/
class RTriangle {
 public:
  RTriangle();
  ~RTriangle();

  inline RTriParam param();

  /*
		*  \brief Initialize the R-triangles
		*
		*  \param indices: face indices
		*  \param num_faces: number of faces
		*  \param nb_vf: vertex neighbor faces
		*  \param nb_vf_prefix: prefix of nb_vf
		*/
  void Init(thrust::device_vector<uint> indices, const thrust::device_vector<uint>& nb_vf,
            const thrust::device_vector<uint>& nb_vf_prefix);

  thrust::device_vector<uint> r_tris;
};

}  // namespace XRTailor