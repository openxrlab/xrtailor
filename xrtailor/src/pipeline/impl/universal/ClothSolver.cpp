#include <xrtailor/pipeline/impl/universal/ClothSolver.hpp>
#include <xrtailor/pipeline/impl/gltf/ClothSolver.hpp>
#include <xrtailor/pipeline/base/ClothSolverBase.hpp>

#include <xrtailor/physics/ClothSolverHelper.cuh>
#include <xrtailor/physics/ClothSolverHelper.hpp>
#include <xrtailor/physics/DebugDrawingHelper.hpp>
#include <xrtailor/physics/PhysicsMesh.cuh>
#include <xrtailor/physics/PhysicsMeshHelper.cuh>

#include <xrtailor/physics/broad_phase/lbvh/BVH.cuh>
#include <xrtailor/physics/broad_phase/spatial_hash/SpatialHashGPU.cuh>

#include <xrtailor/physics/pbd_collision/Constraint.cuh>
#include <xrtailor/physics/impact_zone/Entry.cuh>
#include <xrtailor/physics/impact_zone/ImpactZoneOptimizer.cuh>
#include <xrtailor/physics/predictive_contact/Solver.cuh>

#include <xrtailor/physics/dynamics/BindingConstraint.cuh>
#include <xrtailor/physics/dynamics/BasicConstraint.cuh>
#include <xrtailor/physics/dynamics/FEMSolver.cuh>
#include <xrtailor/physics/dynamics/BasicSolver.cuh>
#include <xrtailor/physics/dynamics/LongRangeAttachments.cuh>

#include <xrtailor/physics/icm/IntersectionContourMinimization.cuh>
#include <xrtailor/physics/icm/GlobalIntersectionAnalysis.cuh>
#include <xrtailor/physics/sdf/Collider.hpp>
#include <xrtailor/physics/repulsion/ImminentRepulsion.cuh>
#include <xrtailor/physics/repulsion/PBDRepulsion.cuh>

#include <xrtailor/math/BasicPrimitiveTests.cuh>
#include <xrtailor/utils/ObjUtils.hpp>
#include <xrtailor/utils/SeedGenerator.hpp>

#include <xrtailor/runtime/mesh/Mesh.hpp>
#include <xrtailor/runtime/rendering/LineRenderer.hpp>
#include <xrtailor/runtime/rendering/AABBRenderer.hpp>
#include <xrtailor/runtime/rendering/PointRenderer.hpp>
#include <xrtailor/runtime/rendering/ArrowRenderer.hpp>

namespace XRTailor {
ClothSolverUniversal::ClothSolverUniversal() {
  name = "ClothSolverUniversal";
}

void ClothSolverUniversal::Start() {
  ClothSolverBase::Start();

  int n_obstacle_nodes = obstacle_->NumNodes();
  obstacle_->UpdateNormals();
  LOG_TRACE("Update BVH data");
  obstacle_bvh_->UpdateData(pointer(obstacle_->faces), Global::sim_params.bvh_tolerance, false,
                            obstacle_->NumFaces());
  LOG_DEBUG("#nodes: {}, #faces: {}", obstacle_->nodes.size(), obstacle_->faces.size());
  LOG_TRACE("Save initial positions");
  UpdateVertices(pointer(memory_pool_->initial_positions), cloth_);
  checkCudaErrors(cudaDeviceSynchronize());
  cloth_->Sync();
  obstacle_->Sync();

  obstacle_bvh_->UpdateData(pointer(obstacle_->faces), 0.0f, false, obstacle_->NumFaces());
  obstacle_bvh_->Construct();

  SetupRTriangle();

  int n_garment_nodes = cloth_->NumNodes();
  int n_garment_edges = cloth_->NumEdges();

  pbd_vf_dcd_ = std::make_shared<PBDCollision::DCD::VFConstraint>(n_garment_nodes);
  pbd_vf_ccd_ = std::make_shared<PBDCollision::CCD::RTConstraint>(n_garment_nodes);
  pbd_ee_dcd_ = std::make_shared<PBDCollision::DCD::EEConstraint>(n_garment_edges, obstacle_r_tri_);

  BuildDeviceEFAdjacency(h_cloth_nb_ef_, memory_pool_->cloth_nb_ef,
                         memory_pool_->cloth_nb_ef_prefix);
  BuildDeviceEFAdjacency(h_obstacle_nb_ef_, memory_pool_->obstacle_nb_ef,
                         memory_pool_->obstacle_nb_ef_prefix);

  cloth_->RegisterEFAdjacency(memory_pool_->cloth_nb_ef, memory_pool_->cloth_nb_ef_prefix);
  obstacle_->RegisterEFAdjacency(memory_pool_->obstacle_nb_ef, memory_pool_->obstacle_nb_ef_prefix);

  checkCudaErrors(cudaDeviceSynchronize());

  LOG_TRACE("Setup spatial hash");
  spatial_hash_ = std::make_shared<SpatialHashGPU>(
      Global::sim_params.particle_diameter, Global::sim_params.num_particles,
      Global::sim_config.swift.collision.self_contact.hash_cell_size,
      Global::sim_params.max_num_neighbors);
  spatial_hash_->SetInitialPositions(pointer(cloth_->nodes), cloth_->NumNodes());

  SetupInternalDynamics();

  checkCudaErrors(cudaDeviceSynchronize());
}

void ClothSolverUniversal::Update() {
  ClothSolverBase::Update();
}

void ClothSolverUniversal::FixedUpdate() {
  ClothSolverBase::FixedUpdate();
}

void ClothSolverUniversal::OnDestroy() {
  ClothSolverBase::OnDestroy();
}

void ClothSolverUniversal::SaveCache() {
  ScopedTimerGPU timer("Solver_SaveCache");
  if (Global::sim_params.frame_index > 0 &&
      Global::sim_params.frame_index <=
          Global::sim_params.num_frames  // stop recording when animation is finished
  ) {
    if (Global::sim_params.record_obstacle) {
      int n_obstacle_nodes = obstacle_->NumNodes();
      h_obstacle_positions_ptr = static_cast<Vector3*>(malloc(sizeof(Vector3) * n_obstacle_nodes));
      h_obstacle_normals_ptr = static_cast<Vector3*>(malloc(sizeof(Vector3) * n_obstacle_nodes));

      thrust::host_vector<Vector3> h_obstacle_positions_vec = obstacle_->HostPositions();
      thrust::host_vector<Vector3> h_obstacle_normals_vec = obstacle_->HostNormals();
      MemcpyHostToHost<Vector3>(h_obstacle_positions_vec, h_obstacle_positions_ptr,
                                n_obstacle_nodes);
      MemcpyHostToHost<Vector3>(h_obstacle_normals_vec, h_obstacle_normals_ptr, n_obstacle_nodes);

      h_obstacle_positions_cache->push_back(h_obstacle_positions_ptr);
      h_obstacle_normals_cache->push_back(h_obstacle_normals_ptr);
      num_cached_obstacle_frames++;
    }

    if (Global::sim_params.record_cloth) {
      int n_garment_nodes = cloth_->NumNodes();
      h_cloth_positions_ptr = static_cast<Vector3*>(malloc(sizeof(Vector3) * n_garment_nodes));
      h_cloth_normals_ptr = static_cast<Vector3*>(malloc(sizeof(Vector3) * n_garment_nodes));

      thrust::host_vector<Vector3> h_cloth_positions_vec = cloth_->HostPositions();
      thrust::host_vector<Vector3> h_cloth_normals_vec = cloth_->HostNormals();
      MemcpyHostToHost<Vector3>(h_cloth_positions_vec, h_cloth_positions_ptr, n_garment_nodes);
      MemcpyHostToHost<Vector3>(h_cloth_normals_vec, h_cloth_normals_ptr, n_garment_nodes);

      h_cloth_positions_cache->push_back(h_cloth_positions_ptr);
      h_cloth_normals_cache->push_back(h_cloth_normals_ptr);
      num_cached_cloth_frames++;
    }
  }
}

bool ClothSolverUniversal::UpdateObstacleAnimationFrame() {
  if (!Global::sim_params.update_obstacle_animation) {
    return false;
  }

  int n_obstacle_nodes = obstacle_->NumNodes();

  if (Global::sim_params.frame_index < Global::sim_params.num_frames) {
    PrintSimulationProgress();
    {
      ScopedTimerGPU timer("Solver_UpdateObstacle");
      uint target_frame_index =
          (Global::sim_params.frame_index > Global::sim_params.num_lerped_frames)
              ? (static_cast<size_t>(Global::sim_params.frame_index) -
                 Global::sim_params.num_lerped_frames) *
                        frame_step_size +
                    Global::sim_params.num_lerped_frames
              : Global::sim_params.frame_index;
      // TODO: update obstacle
    }
    {
      ScopedTimerGPU timer("Solver_UpdateBVH");
      obstacle_bvh_->UpdateData(pointer(obstacle_->faces), 0.0f, false, obstacle_->NumFaces());
      obstacle_bvh_->Construct();
    }
    Global::sim_params.frame_index++;

    if (Global::sim_params.frame_index >= Global::sim_params.num_frames) {
      simulation_finished = true;
    }
#ifdef DEBUG_MODE
    // early exit
    if (Global::sim_params.frame_index >= 199) {
      simulation_finished = true;
    }
#endif  // DEBUG_MODE

    return true;
  }

  return false;
}

int ClothSolverUniversal::AddObstacle(std::shared_ptr<Mesh> mesh, Mat4 model_matrix,
                                      unsigned int obstacle_id) {
  LOG_TRACE("AddObstacle.");

  Timer::StartTimer("INIT_BODY_GPU");

  int prev_num_obstacle_vertices = Global::sim_params.num_obstacle_vertices;
  int prev_num_obstacle_edges = Global::sim_params.num_obstacle_edges;
  int prev_num_obstacle_faces = Global::sim_params.num_obstacle_faces;

  auto new_num_obstacle_vertices = static_cast<int>(mesh->Positions().size());
  auto new_num_obstacle_edges = static_cast<int>(mesh->TMesh().edge.size());
  auto new_num_obstacle_faces = static_cast<int>(mesh->TMesh().face.size());
  int prev_num_vertices = Global::sim_params.num_overall_particles;

  Global::sim_params.num_overall_particles += new_num_obstacle_vertices;
  Global::sim_params.num_obstacle_vertices += new_num_obstacle_vertices;
  Global::sim_params.num_obstacle_edges += new_num_obstacle_edges;
  Global::sim_params.num_obstacle_faces += new_num_obstacle_faces;

  LOG_TRACE("Allocate managed buffers.");

  BuildEFAdjacency(mesh, h_obstacle_nb_ef_, prev_num_obstacle_edges, prev_num_obstacle_faces);
  checkCudaErrors(cudaDeviceSynchronize());
  obstacle_->SetPtrOffset(cloth_->NumNodes(), cloth_->NumEdges(), cloth_->NumFaces());

  thrust::host_vector<uint> h_edges;
  for (int i = 0; i < new_num_obstacle_edges; i++) {
    auto e = &mesh->TMesh().edge[i];
    h_edges.push_back(Mesh::GetIndex(e->V(0), mesh->TMesh()) + prev_num_obstacle_vertices);
    h_edges.push_back(Mesh::GetIndex(e->V(1), mesh->TMesh()) + prev_num_obstacle_vertices);
  }

  thrust::host_vector<uint> h_fe_indices = mesh->FaceEdgeIndices();

  LOG_DEBUG("Register new buffer");
  obstacle_->RegisterNewBuffer(mesh->VBO());
  LOG_DEBUG("Register new mesh, VBO {}, #verts {}, #faces {}, #uvs {}", mesh->VBO(),
            mesh->Positions().size(), mesh->Indices().size() / 3, mesh->UVs().size());
  obstacle_->RegisterNewMesh(mesh->Positions(), mesh->Indices(), h_edges, h_fe_indices,
                             prev_num_obstacle_vertices, false);

  UpdateX(pointer(obstacle_->nodes), model_matrix, obstacle_->NumNodes());
  obstacle_->PredictedToPositions();

  LOG_DEBUG("Register masked index buffer");
  int n_obstacle_faces = obstacle_->NumFaces();
  thrust::host_vector<uint> obstacle_indices = obstacle_->HostIndices();
  memory_pool_->SetupObstacleMaskedIndex(obstacle_indices);

  obstacle_->UpdateNodeGeometries();

  double time = Timer::EndTimer("INIT_BODY_GPU") * 1000;
  LOG_TRACE("AddObstacle done. Took time {:.2f} ms", time);

  return prev_num_obstacle_vertices;
}

int ClothSolverUniversal::AddCloth(std::shared_ptr<Mesh> mesh, Mat4 model_matrix,
                                   Scalar particle_diameter) {
  LOG_TRACE("AddCloth");

  Timer::StartTimer("INIT_SOLVER_GPU");

  int prev_num_particles = Global::sim_params.num_particles;
  int prev_num_edges = Global::sim_params.num_edges;
  int prev_num_faces = Global::sim_params.num_faces;
  LOG_DEBUG("Prev num edges: {}", prev_num_edges);
  BuildEFAdjacency(mesh, h_cloth_nb_ef_, prev_num_edges, prev_num_faces);

  auto new_num_particles = static_cast<int>(mesh->Positions().size());
  auto new_num_edges = static_cast<int>(mesh->TMesh().edge.size());
  auto new_num_faces = static_cast<int>(mesh->TMesh().face.size());

  prev_num_attached_slots = Global::sim_params.num_attached_slots;
  auto new_attached_slots = static_cast<int>(mesh->AttachedIndices().size());

  // Set global parameters
  Global::sim_params.num_overall_particles += new_num_particles;
  Global::sim_params.num_particles += new_num_particles;
  Global::sim_params.num_attached_slots += new_attached_slots;
  Global::sim_params.num_edges += new_num_edges;
  Global::sim_params.num_faces += new_num_faces;
  Global::sim_params.particle_diameter = particle_diameter;
  Global::sim_params.delta_time = Timer::FixedDeltaTime();

  memory_pool_->SetupInitialPositions(mesh->Positions());

  LOG_TRACE("Allocate managed buffers.");

  thrust::host_vector<uint> h_edges;
  for (int i = 0; i < new_num_edges; i++) {
    auto e = &mesh->TMesh().edge[i];
    h_edges.push_back(Mesh::GetIndex(e->V(0), mesh->TMesh()) + prev_num_particles);
    h_edges.push_back(Mesh::GetIndex(e->V(1), mesh->TMesh()) + prev_num_particles);
  }

  thrust::host_vector<uint> h_fe_indices = mesh->FaceEdgeIndices();

  LOG_INFO("VBO: {}", mesh->VBO());
  cloth_->RegisterNewBuffer(mesh->VBO());
  cloth_->RegisterNewMesh(mesh->Positions(), mesh->Indices(), h_edges, h_fe_indices,
                          prev_num_particles);

  memory_pool_->SetupTmpPositions(new_num_particles);
  memory_pool_->SetupDeltas(new_num_particles);

  LOG_DEBUG("Add binding");
  binding_->Add(mesh->BindedIndices(), mesh->BindStiffnesses(), mesh->BindDistances(),
                prev_num_particles);

  LOG_DEBUG("Initialize positions");
  InitializePositions(cloth_, prev_num_particles, new_num_particles, model_matrix);

  LOG_TRACE("Setup fixed nodes");
  for (int i = 0; i < mesh->FixedIndices().size(); i++) {
    cloth_->SetInvMass(0.0f, mesh->FixedIndices()[i] + prev_num_particles);
    cloth_->SetFree(false, mesh->FixedIndices()[i] + prev_num_particles);
  }

  checkCudaErrors(cudaDeviceSynchronize());
  cloth_->Sync();
  checkCudaErrors(cudaDeviceSynchronize());

  std::vector<uint> bend_indices;
  std::vector<Scalar> bend_angles;
  GenerateBendingElements(mesh, prev_num_particles, bend_indices, bend_angles);

  if (Global::sim_params.solver_mode == SOLVER_MODE::SWIFT) {
    Timer::StartTimer("BASIC_STRETCH_ADD_CLOTH");
    const auto& fabric_settings = Global::sim_config.swift.fabric;

    std::vector<int> ev0_indices, ev1_indices;
    std::vector<Scalar> e_lengths;
    PrepareBasicStretchData(mesh, ev0_indices, ev1_indices, e_lengths, prev_num_particles);
    basic_stretch_solver_->AddCloth(ev0_indices, ev1_indices, e_lengths,
                                    fabric_settings.stretch_compliance);

    double t_add_cloth = Timer::EndTimer("BASIC_STRETCH_ADD_CLOTH") * 1000;
    LOG_TRACE("[BasicStretchSolver] Generate stretch constraints done, took time {:.2f} ms",
              t_add_cloth);

    if (Global::sim_params.solve_bending) {
      Timer::StartTimer("BASIC_BEND_ADD_CLOTH");
      basic_bend_solver_->AddCloth(bend_indices, bend_angles, fabric_settings.bend_compliance);
      double t_add_cloth = Timer::EndTimer("BASIC_BEND_ADD_CLOTH") * 1000;
      LOG_TRACE(
          "[BasicBendSolver] Generate bend constraints using compliance {:.2f} done, took time "
          "{:.2f} ms",
          fabric_settings.bend_compliance, t_add_cloth);
    }
  } else if (Global::sim_params.solver_mode == SOLVER_MODE::QUALITY) {
    LOG_DEBUG("FEMIsometricBendingSolver add cloth");
    fem_isometric_bending_solver_->AddCloth(bend_indices, bend_angles);
  }

  double time = Timer::EndTimer("INIT_SOLVER_GPU") * 1000;
  LOG_TRACE("AddCloth done. Took time {:.2f} ms", time);
  LOG_TRACE("Use recommond max vel = {}", Global::sim_params.max_speed);

  return prev_num_particles;
}
}  // namespace XRTailor
