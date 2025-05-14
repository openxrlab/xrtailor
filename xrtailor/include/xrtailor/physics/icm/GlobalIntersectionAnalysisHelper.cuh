#pragma once

#define GIA_DEFAULT_COLOR -1

namespace XRTailor {
namespace Untangling {

struct EdgeState {
  bool active{false};  // the edge is involved into an intersection
  int color{GIA_DEFAULT_COLOR};
};

struct FaceState {
  bool active{false};
  int color{GIA_DEFAULT_COLOR};
};

struct IntersectionState {
  bool visited{false};  // the intersection has been traversed
  int color{-1};
};

}  // namespace Untangling
}  // namespace XRTailor