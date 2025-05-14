#pragma once

#include <iostream>
#include <random>

#include <xrtailor/core/Scalar.hpp>

namespace XRTailor {

class SeedGenerator {
 public:
  SeedGenerator();

  uint Get();

 private:
  std::random_device rd_;
  std::mt19937 rand_engine_;
  std::uniform_int_distribution<uint> dis_;
};

}  // namespace XRTailor