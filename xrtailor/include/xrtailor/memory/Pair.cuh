#pragma once

#include <cuda_runtime.h>

#include <xrtailor/memory/Face.cuh>

namespace XRTailor {

template <typename S, typename T>
class Pair {
 public:
  S first;
  T second;

  Pair() = default;

  Pair(const Pair<S, T>& p) = default;

  __host__ __device__ Pair(S first, T second) : first(first), second(second) {};

  ~Pair() = default;

  __host__ __device__ bool operator==(const Pair<S, T>& p) const {
    return first == p.first && second == p.second;
  };

  __host__ __device__ bool operator!=(const Pair<S, T>& p) const {
    return first != p.first || second != p.second;
  };

  __host__ __device__ bool operator<(const Pair<S, T>& p) const {
    return first < p.first || first == p.first && second < p.second;
  };

  __host__ __device__ Pair<S, T> operator+(const Pair<S, T>& p) const {
    Pair<S, T> ans(first + p.first, second + p.second);
    return ans;
  };
};

typedef Pair<int, int> Pairii;
typedef Pair<Node*, Face*> PairVF;
typedef Pair<Edge*, Face*> PairEF;
typedef Pair<Face*, Face*> PairFF;
typedef Pair<Node*, int> PairVi;

}  // namespace XRTailor