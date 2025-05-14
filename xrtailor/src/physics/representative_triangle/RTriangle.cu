#include <xrtailor/physics/representative_triangle/RTriangle.cuh>

#include <xrtailor/core/DeviceHelper.cuh>

namespace XRTailor {
__host__ __device__ __forceinline__ void SetRTriVertex(uint& info, uint n) {
  info |= (1u << n);
}
__host__ __device__ __forceinline__ void SetRTriEdge(uint& info, uint n) {
  info |= (1u << (n + 3u));
}
__host__ __device__ bool RTriVertex(const uint info, uint n) {
  return (info >> n) & 1u;
}
__host__ __device__ bool RTriEdge(const uint info, uint n) {
  return (info >> (n + 3u)) & 1u;
}

__global__ void RTriBuild_Kernel(uint* indices, CONST(uint*) nb_vf, CONST(uint*) nb_vf_prefix,
                                 RTriParam r_tri) {
  GET_CUDA_ID(fid, r_tri._size);

  uint f_verts_indices[3];
  uint f_idx_offset = fid * 3u;

  f_verts_indices[0] = indices[f_idx_offset + 0u];
  f_verts_indices[1] = indices[f_idx_offset + 1u];
  f_verts_indices[2] = indices[f_idx_offset + 2u];

  uint i_end, j_start, j_end;
  uint i_v, j_v, i_f, j_f;
  uint info = 0u;  // this is where records the representative assignment schemas
  uint jno;

  j_start = nb_vf_prefix[f_verts_indices[0]];
  j_end = nb_vf_prefix[f_verts_indices[0] + 1u];
  for (i_v = 0u; i_v < 3u; i_v++) {
    f_idx_offset = j_start;      // index offset start of neighbor faces
    i_end = j_end;                // index offset end	of neighbor faces
    i_f = nb_vf[f_idx_offset++];  // face index
    if (i_f == fid) {
      SetRTriVertex(info, i_v);
    }

    j_v = (i_v + 1u) % 3u;  // second vertex on the edge
    j_start = nb_vf_prefix[f_verts_indices[j_v]];
    j_end = nb_vf_prefix[f_verts_indices[j_v] + 1u];

    uint of = 0xffffffff;
    for (jno = j_start; jno < j_end; jno++) {
      j_f = nb_vf[jno];
      if (j_f == i_f && fid != i_f) {
        of = i_f;
        break;
      }
    }

    for (; f_idx_offset < i_end && of == 0xffffffff; f_idx_offset++) {
      i_f = nb_vf[f_idx_offset];
      for (jno = j_start; jno < j_end; jno++) {
        j_f = nb_vf[jno];
        if (j_f == i_f && fid != i_f) {
          of = i_f;
          break;
        }
      }
    }
    if (fid < of) {
      SetRTriEdge(info, i_v);
    }
  }

  r_tri._info[fid] = info;
}

inline RTriParam RTriangle::param() {
  RTriParam p;
  p._info = r_tris.data().get();
  p._size = r_tris.size();

  return p;
}

RTriangle::RTriangle() {}
RTriangle::~RTriangle() {}

void RTriangle::Init(thrust::device_vector<uint> indices, const thrust::device_vector<uint>& nb_vf,
                     const thrust::device_vector<uint>& nb_vf_prefix) {
  checkCudaErrors(cudaDeviceSynchronize());

  uint num_faces = indices.size() / 3u;
  this->r_tris.resize(num_faces);

  CUDA_CALL(RTriBuild_Kernel, num_faces)
  (pointer(indices), pointer(nb_vf), pointer(nb_vf_prefix), param());

  checkCudaErrors(cudaPeekAtLastError());

  checkCudaErrors(cudaDeviceSynchronize());
}

}  // namespace XRTailor