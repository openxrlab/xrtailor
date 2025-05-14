#include <xrtailor/core/DeviceHelper.cuh>

namespace XRTailor {

__device__ void AtomicAddX(Node** address, int index, Vector3 val, int reorder) {
  int r1 = reorder % 3;
  int r2 = (reorder + 1) % 3;
  int r3 = (reorder + 2) % 3;
  atomicAdd(&(address[index]->x.x) + r1, val[r1]);
  atomicAdd(&(address[index]->x.x) + r2, val[r2]);
  atomicAdd(&(address[index]->x.x) + r3, val[r3]);
}

__device__ void AtomicAdd(Vector3* address, int index, Vector3 val, int reorder) {
  int r1 = reorder % 3;
  int r2 = (reorder + 1) % 3;
  int r3 = (reorder + 2) % 3;
  atomicAdd(&(address[index].x) + r1, val[r1]);
  atomicAdd(&(address[index].x) + r2, val[r2]);
  atomicAdd(&(address[index].x) + r3, val[r3]);
}

__device__ void AtomicAdd(Vector3* address, int index, Vector3 val) {
  atomicAdd(&(address[index].x), val.x);
  atomicAdd(&(address[index].y), val.y);
  atomicAdd(&(address[index].z), val.z);
}

void cudaCheckLast(const char* file, int line) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << std::endl;
  }
}

}  // namespace XRTailor