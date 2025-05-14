#pragma once

#define PBSTR "||||||||||||||||||||"
#define PBWIDTH 20
#include <set>
#include <vector>
#include <iomanip>

#include <xrtailor/runtime/input/MouseGrabber.hpp>

namespace XRTailor {

class Mesh;
class AABB;
class LineRenderer;
class AABBRenderer;
class PointRenderer;
class ArrowRenderer;
class SpatialHashGPU;
class BVH;
class RTriangle;
class Collider;
class PhysicsMesh;
class MemoryPool;
class SeedGenerator;
class ImminentRepulsion;
class PBDRepulsionSolver;

namespace PBDCollision {
namespace DCD {
class VFConstraint;
class EEConstraint;
}  // namespace DCD
namespace CCD {
class RTConstraint;
}
}  // namespace PBDCollision

namespace BasicConstraint {
class StretchSolver;
class BendSolver;
}  // namespace BasicConstraint

namespace FEM {
class StrainSolver;
class IsometricBendingSolver;
}  // namespace FEM

namespace Untangling {
class ICM;
}

class BindingConstraint;

class ClothSolverBase : public Component {
 public:
  ClothSolverBase();

  void Start() override;

  void Update() override;

  void FixedUpdate() override;

  void OnDestroy() override;

  void SimulateQuality();

  void SimulateSwift();

  virtual bool UpdateObstacleAnimationFrame() { return false; };

 public:
  virtual void SaveCache() {};

  void ClearAnimationCache();

  void PrintPreSimulationProgress();

  void PrintSimulationProgress();

  void SetupBinding();

  void SetupInternalDynamics();

  void SetupRTriangle();

  void AddEuclidean(int particle_index, int slot_index, Scalar distance);

  void AddGeodesic(uint src_index, uint tgt_index, Scalar rest_length);

  void UpdateColliders(std::vector<Collider*>& colliders);

  void SolveLongRangeAttachments(bool geodesic_LRA);

  std::string GetIdentifier();

  void SetIdentifier(const std::string& identifier);

 public:
  uint prev_num_attached_slots;

  Vector3* h_obstacle_positions_ptr;
  Vector3* h_obstacle_normals_ptr;
  Vector3* h_cloth_positions_ptr;
  Vector3* h_cloth_normals_ptr;
  std::shared_ptr<std::vector<Vector3*>> h_cloth_positions_cache;
  std::shared_ptr<std::vector<Vector3*>> h_cloth_normals_cache;
  std::shared_ptr<std::vector<Vector3*>> h_obstacle_positions_cache;
  std::shared_ptr<std::vector<Vector3*>> h_obstacle_normals_cache;
  uint num_cached_cloth_frames;
  uint num_cached_obstacle_frames;
  bool simulation_finished = false;
  uint frame_step_size;

  std::string selected_sequence_label;

  double solver_time;
  double total_time = 0.0;

#ifdef DEBUG_MODE
  std::shared_ptr<AABBRenderer> obstacle_aabb_renderer = std::make_shared<AABBRenderer>();
  std::shared_ptr<AABBRenderer> internal_aabb_renderer = std::make_shared<AABBRenderer>();
  std::shared_ptr<AABBRenderer> external_aabb_renderer = std::make_shared<AABBRenderer>();

  std::shared_ptr<AABBRenderer> cloth_aabb_renderer = std::make_shared<AABBRenderer>();
  std::shared_ptr<AABBRenderer> proximity_aabb_renderer = std::make_shared<AABBRenderer>();

  std::shared_ptr<PointRenderer> quadrature_renderer = std::make_shared<PointRenderer>();

  std::shared_ptr<LineRenderer> line_renderer = std::make_shared<LineRenderer>();
  std::shared_ptr<PointRenderer> point_renderer = std::make_shared<PointRenderer>();
  std::shared_ptr<ArrowRenderer> arrow_renderer = std::make_shared<ArrowRenderer>();

  std::shared_ptr<LineRenderer> line_renderer_A = std::make_shared<LineRenderer>();
  std::shared_ptr<LineRenderer> line_renderer_B = std::make_shared<LineRenderer>();
  std::shared_ptr<PointRenderer> point_renderer_A = std::make_shared<PointRenderer>();
  std::shared_ptr<PointRenderer> point_renderer_B = std::make_shared<PointRenderer>();
  std::shared_ptr<PointRenderer> point_renderer_C = std::make_shared<PointRenderer>();
  std::shared_ptr<PointRenderer> point_renderer_D = std::make_shared<PointRenderer>();
  std::shared_ptr<PointRenderer> point_renderer_E = std::make_shared<PointRenderer>();
  std::shared_ptr<AABBRenderer> aabb_renderer_A = std::make_shared<AABBRenderer>();
  std::shared_ptr<AABBRenderer> aabb_renderer_B = std::make_shared<AABBRenderer>();
  std::shared_ptr<AABBRenderer> aabb_renderer_C = std::make_shared<AABBRenderer>();
#endif

 protected:
  std::shared_ptr<PhysicsMesh> cloth_;
  std::shared_ptr<PhysicsMesh> obstacle_;
  std::shared_ptr<MemoryPool> memory_pool_;
  Mat4 obstacle_model_matrix_;

  std::shared_ptr<SpatialHashGPU> spatial_hash_;
  std::shared_ptr<BVH> obstacle_bvh_;
  std::shared_ptr<BVH> cloth_bvh_;

  std::shared_ptr<RTriangle> obstacle_r_tri_;
  std::shared_ptr<RTriangle> cloth_r_tri_;

  std::vector<Collider*> sdf_colliders_;
  MouseGrabber mouse_grabber_;

  std::string solver_identifier_;

  std::vector<std::set<uint>> h_cloth_nb_ef_;
  std::vector<std::set<uint>> h_obstacle_nb_ef_;

  std::shared_ptr<SeedGenerator> seed_generator_;

  std::shared_ptr<PBDCollision::DCD::VFConstraint> pbd_vf_dcd_;
  std::shared_ptr<PBDCollision::CCD::RTConstraint> pbd_vf_ccd_;
  std::shared_ptr<PBDCollision::DCD::EEConstraint> pbd_ee_dcd_;

  std::shared_ptr<BindingConstraint> binding_;

  std::shared_ptr<BasicConstraint::StretchSolver> basic_stretch_solver_;
  std::shared_ptr<BasicConstraint::BendSolver> basic_bend_solver_;
  std::shared_ptr<FEM::StrainSolver> fem_strain_solver_;
  std::shared_ptr<FEM::IsometricBendingSolver> fem_isometric_bending_solver_;
  std::shared_ptr<ImminentRepulsion> imminent_repulsion_solver_;
  std::shared_ptr<PBDRepulsionSolver> pbd_repulsion_solver_;
#ifdef DEBUG_MODE
  void DrawObstacleAABBs();
  void DrawInternalNodes(BVH* bvh);
  void DrawExternalNodes(BVH* bvh);
  void DrawPairFFs(const thrust::host_vector<int>& h_types,
                   const thrust::host_vector<int>& h_indices);
#endif  // DEBUG_MODE
};

void GIADetangleStep(std::shared_ptr<Untangling::ICM> solver, std::shared_ptr<PhysicsMesh> cloth,
                     std::shared_ptr<BVH> cloth_bvh, std::shared_ptr<PhysicsMesh> obstacle,
                     std::shared_ptr<BVH> obstacle_bvh);

}  // namespace XRTailor