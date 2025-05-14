#include <climits>
#include <xrtailor/utils/SeedGenerator.hpp>

namespace XRTailor {
SeedGenerator::SeedGenerator() {
  rand_engine_ = std::mt19937(rd_());
  dis_ = std::uniform_int_distribution<unsigned int>(0, UINT_MAX);
}

unsigned int SeedGenerator::Get() {
  unsigned int val = dis_(rand_engine_);

  return val;
}

}  // namespace XRTailor