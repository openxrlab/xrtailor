#pragma once

#include <xrtailor/physics/representative_triangle/RTriangle.cuh>
#include <xrtailor/utils/Timer.hpp>
#include <xrtailor/memory/Edge.cuh>
#include <xrtailor/physics/pbd_collision/Contact.cuh>

namespace XRTailor {

class BVH;

namespace PBDCollision {
namespace DCD {

class VFConstraint {
 public:
  VFConstraint(){};

  VFConstraint(uint n_verts);

  ~VFConstraint();

  void Generate(Node** nodes, std::shared_ptr<BVH> bvh);

  void Solve(Node** nodes, std::shared_ptr<BVH> bvh);

 public:
  uint n_verts;
  thrust::device_vector<VFContact> contacts;
};

class EEConstraint {
 public:
  EEConstraint(){};

  EEConstraint(uint n_edges, std::shared_ptr<XRTailor::RTriangle> obstacle_r_tri);

  ~EEConstraint();

  void Generate(Node** nodes, Edge** edges, std::shared_ptr<BVH> bvh);

  void Solve(Edge** edges, Vector3* deltas, int* delta_counts, std::shared_ptr<BVH> bvh);

 public:
  uint n_edges;
  thrust::device_vector<EEContact> contacts;
  std::shared_ptr<RTriangle> obstacle_r_tri;
};

}  // namespace DCD

namespace CCD {

class RTConstraint {
 public:
  RTConstraint(){};

  RTConstraint(uint n_verts);

  ~RTConstraint();

  void Generate(Node** nodes, std::shared_ptr<BVH> bvh);

  void Solve(Node** nodes, std::shared_ptr<BVH> bvh);

 public:
  uint n_verts;
  thrust::device_vector<RTContact> contacts;
};

}  // namespace CCD
}  // namespace PBDCollision
}  // namespace XRTailor