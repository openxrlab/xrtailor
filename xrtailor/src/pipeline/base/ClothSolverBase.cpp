#include <xrtailor/core/Precompiled.h>
#include <xrtailor/pipeline/base/ClothSolverBase.hpp>

#include <xrtailor/runtime/engine/GameInstance.hpp>
#include <xrtailor/runtime/mesh/Mesh.hpp>
#include <xrtailor/utils/ObjUtils.hpp>

#include <xrtailor/physics/broad_phase/spatial_hash/SpatialHashGPU.cuh>
#include <xrtailor/physics/pbd_collision/Constraint.cuh>
#include <xrtailor/physics/dynamics/BindingConstraint.cuh>
#include <xrtailor/physics/dynamics/LongRangeAttachments.cuh>
#include <xrtailor/physics/icm/IntersectionContourMinimization.cuh>
#include <xrtailor/physics/icm/GlobalIntersectionAnalysis.cuh>
#include <xrtailor/physics/dynamics/FEMSolver.cuh>
#include <xrtailor/physics/dynamics/BasicSolver.cuh>
#include <xrtailor/math/BasicPrimitiveTests.cuh>
#include <xrtailor/physics/impact_zone/ImpactZoneOptimizer.cuh>
#include <xrtailor/physics/predictive_contact/Solver.cuh>
#include <xrtailor/utils/SeedGenerator.hpp>
#include <xrtailor/physics/sdf/Collider.hpp>
#include <xrtailor/physics/PhysicsMesh.cuh>
#include <xrtailor/physics/PhysicsMeshHelper.cuh>
#include <xrtailor/physics/repulsion/ImminentRepulsion.cuh>
#include <xrtailor/physics/repulsion/PBDRepulsion.cuh>

#include <xrtailor/physics/ClothSolverHelper.cuh>
#include <xrtailor/physics/impact_zone/Entry.cuh>
#include <xrtailor/physics/broad_phase/lbvh/BVH.cuh>
#include <xrtailor/physics/ClothSolverHelper.hpp>
#include <xrtailor/physics/DebugDrawingHelper.hpp>


namespace XRTailor {
ClothSolverBase::ClothSolverBase() {
  name = "ClothSolverBase";
  Global::sim_params.num_particles = 0;
  Global::sim_params.num_skinned_slots = 0;
  Global::sim_params.num_obstacle_vertices = 0;
  Global::sim_params.pre_simulation_frame_index = 0;
  Global::sim_params.frame_index = 0;
  //Global::sim_params.num_frames = 0;

  memory_pool_ = std::make_shared<MemoryPool>();

  cloth_ = std::make_shared<PhysicsMesh>(memory_pool_);
  obstacle_ = std::make_shared<PhysicsMesh>(memory_pool_);

  // ------------------- begin initialzie animation buffer ----------------------
  num_cached_obstacle_frames = 0;
  num_cached_cloth_frames = 0;
  h_cloth_positions_cache = std::make_shared<std::vector<Vector3*>>();
  h_cloth_normals_cache = std::make_shared<std::vector<Vector3*>>();
  h_obstacle_positions_cache = std::make_shared<std::vector<Vector3*>>();
  h_obstacle_normals_cache = std::make_shared<std::vector<Vector3*>>();
  // ------------------- end initialzie animation buffer ------------------------

  obstacle_bvh_ = std::make_shared<BVH>();
  cloth_bvh_ = std::make_shared<BVH>();

  seed_generator_ = std::make_shared<SeedGenerator>();

  binding_ = std::make_shared<BindingConstraint>();
  basic_stretch_solver_ = std::make_shared<BasicConstraint::StretchSolver>();
  basic_bend_solver_ = std::make_shared<BasicConstraint::BendSolver>();
  fem_isometric_bending_solver_ = std::make_shared<FEM::IsometricBendingSolver>();

#ifdef DEBUG_MODE
  auto debugger = Global::game->FindActor("Debugger");

  debugger->AddComponent(obstacle_aabb_renderer);
  obstacle_aabb_renderer->SetLineWidth(2.0f);

  debugger->AddComponent(internal_aabb_renderer);
  internal_aabb_renderer->SetLineWidth(2.0f);

  debugger->AddComponent(external_aabb_renderer);
  external_aabb_renderer->SetLineWidth(2.0f);
  external_aabb_renderer->SetColor(Vector3(0.0f, 1.0f, 0.2f));

  debugger->AddComponent(cloth_aabb_renderer);
  cloth_aabb_renderer->SetLineWidth(2.0f);
  cloth_aabb_renderer->SetColor(Vector3(0.0f, 1.0f, 0.2f));

  debugger->AddComponent(proximity_aabb_renderer);
  proximity_aabb_renderer->SetLineWidth(2.0f);
  proximity_aabb_renderer->SetColor(Vector3(0.0f, 0.0f, 1.0f));

  debugger->AddComponent(point_renderer);
  point_renderer->SetPointSize(15.0f);
  point_renderer->SetColor(Vector3(1.0f, 0.0f, 0.0f));

  debugger->AddComponent(line_renderer);
  line_renderer->SetLineWidth(10.0f);
  line_renderer->SetColor(Vector3(1.0f, 0.0f, 1.0f));

  debugger->AddComponent(arrow_renderer);
  arrow_renderer->SetLineWidth(10.0f);
  arrow_renderer->SetColor(Vector3(0.0f, 0.5f, 0.5f));

  debugger->AddComponent(point_renderer_A);
  point_renderer_A->SetPointSize(15.0f);
  point_renderer_A->SetColor(Vector3(1.0f, 0.0f, 0.0f));

  debugger->AddComponent(point_renderer_B);
  point_renderer_B->SetPointSize(15.0f);
  point_renderer_B->SetColor(Vector3(0.0f, 1.0f, 0.0f));

  debugger->AddComponent(point_renderer_C);
  point_renderer_C->SetPointSize(15.0f);
  point_renderer_C->SetColor(Vector3(0.0f, 0.0f, 1.0f));

  debugger->AddComponent(point_renderer_D);
  point_renderer_D->SetPointSize(15.0f);
  point_renderer_D->SetColor(Vector3(1.0f, 0.0f, 1.0f));

  debugger->AddComponent(point_renderer_E);
  point_renderer_E->SetPointSize(15.0f);
  point_renderer_E->SetColor(Vector3(0.0f, 1.0f, 1.0f));

  debugger->AddComponent(line_renderer_A);
  line_renderer_A->SetLineWidth(10.0f);
  line_renderer_A->SetColor(Vector3(1.0f, 0.0f, 1.0f));

  debugger->AddComponent(line_renderer_B);
  line_renderer_B->SetLineWidth(10.0f);
  line_renderer_B->SetColor(Vector3(0.0f, 1.0f, 0.2f));
#endif  // DEBUG_MODE
}

void ClothSolverBase::Start() {
  LOG_TRACE(" Starting {0}'s component: {1}", actor->name, name);
  sdf_colliders_ = Global::game->FindComponents<Collider>();
  mouse_grabber_.Initialize(cloth_);

  imminent_repulsion_solver_ = std::make_shared<ImminentRepulsion>();
  imminent_repulsion_solver_->InitializeDeltas(cloth_->NumNodes());

  pbd_repulsion_solver_ = std::make_shared<PBDRepulsionSolver>(cloth_.get());
}

void ClothSolverBase::Update() {
  if (!simulation_finished) {
    mouse_grabber_.HandleMouseInteraction();
  }
}

void ClothSolverBase::FixedUpdate() {
  mouse_grabber_.UpdateGrappedVertex();
  UpdateColliders(sdf_colliders_);

  Timer::StartTimer("GPU_TIME");

  if (!simulation_finished) {
    if (Global::sim_params.solver_mode == SOLVER_MODE::QUALITY) {
      SimulateQuality();
    } else {
      SimulateSwift();
    }
  }
  Timer::EndTimer("GPU_TIME");
  solver_time = Timer::GetTimerGPU("Solver_Total");
}

void ClothSolverBase::OnDestroy() {
  cloth_->Destroy();
  obstacle_->Destroy();

  ClearAnimationCache();
}

void ClothSolverBase::SimulateSwift() {
  Timer::StartTimerGPU("Solver_Total");
  //==========================
  // Prepare
  //==========================
  Scalar frame_time = Timer::FixedDeltaTime();
  Scalar substep_time = Timer::FixedDeltaTime() / Global::sim_params.num_substeps;
  //==========================
  // Launch kernel
  //==========================
  SetSimulationParams(&Global::sim_params);
  auto seed = seed_generator_->Get();

  bool b_obstacle_updated = UpdateObstacleAnimationFrame();
  uint current_step = 0u;

  uint frequency = (Global::sim_params.num_substeps * Global::sim_params.num_iterations) /
                   Global::sim_params.num_collision_passes;
  uint cnt = 0;
  for (int substep = 0; substep < Global::sim_params.num_substeps; substep++) {
    PredictPositions(pointer(cloth_->nodes), substep_time, cloth_->NumNodes());

    if (Global::sim_params.enable_self_collision) {
      if (substep % Global::sim_config.swift.collision.self_contact.inter_leaved_hash == 0) {
        spatial_hash_->Hash(pointer(cloth_->nodes), cloth_->NumNodes());
      }
    }
    for (int iteration = 0; iteration < Global::sim_params.num_iterations; iteration++) {
      if (Global::sim_params.enable_self_collision) {
        CollideParticles(pointer(memory_pool_->deltas), pointer(memory_pool_->delta_counts),
                         pointer(cloth_->nodes), pointer(spatial_hash_->neighbors));
      }
      current_step = substep * Global::sim_params.num_iterations + iteration;
      if (current_step % frequency == 0) {
        if (b_obstacle_updated) {
          obstacle_->Interpolate(cnt, Global::sim_params.num_collision_passes);
          obstacle_->UpdateMidstepNormals();
          obstacle_bvh_->RefitMidstep(pointer(obstacle_->faces), Global::sim_params.bvh_tolerance);
        }
        pbd_vf_dcd_->Generate(pointer(cloth_->nodes), obstacle_bvh_);
        pbd_vf_ccd_->Generate(pointer(cloth_->nodes), obstacle_bvh_);
        pbd_ee_dcd_->Generate(pointer(cloth_->nodes), pointer(cloth_->edges), obstacle_bvh_);
        cnt++;
      }
      for (int c = 0; c < basic_stretch_solver_->Coloring()->n_colors; c++) {
        basic_stretch_solver_->Solve(pointer(cloth_->nodes), pointer(memory_pool_->tmp_positions),
                                     c, substep_time, iteration);
      }

      if (Global::sim_params.solve_bending) {
        for (int c = 0; c < basic_bend_solver_->Coloring()->n_colors; c++) {
          basic_bend_solver_->Solve(pointer(cloth_->nodes), c, substep_time, iteration);
        }
      }

      SolveLongRangeAttachments(Global::sim_params.geodesic_LRA);

      pbd_vf_ccd_->Solve(pointer(cloth_->nodes), obstacle_bvh_);

      pbd_vf_dcd_->Solve(pointer(cloth_->nodes), obstacle_bvh_);

      pbd_ee_dcd_->Solve(pointer(cloth_->edges), pointer(memory_pool_->deltas),
                         pointer(memory_pool_->delta_counts), obstacle_bvh_);
      ApplyDeltas(pointer(cloth_->nodes), pointer(memory_pool_->deltas),
                  pointer(memory_pool_->delta_counts));

      binding_->Solve(pointer(cloth_->nodes), pointer(obstacle_->nodes));
    }
    Finalize(pointer(cloth_->nodes), substep_time);
  }

  obstacle_->PredictedToPositions();
  cloth_->UpdateNormals();
  obstacle_->UpdateNormals();
  //==========================
  // Sync
  //==========================
  Timer::EndTimerGPU("Solver_Total");
  checkCudaErrors(cudaDeviceSynchronize());

  cloth_->Sync();
  obstacle_->Sync();

  checkCudaErrors(cudaDeviceSynchronize());
#ifdef DEBUG_MODE
  //DrawObstacleAABBs();
  //DrawInternalNodes(obstacle_bvh_.get());
  //DrawExternalNodes(obstacle_bvh_.get());
#endif  //  DEBUG_MODE
  SaveCache();
}

void ClothSolverBase::SimulateQuality() {
  Timer::StartTimerGPU("Solver_Total");
  //==========================
  // Prepare
  //==========================
  Scalar frame_time = Timer::FixedDeltaTime();
  Scalar substep_time = Timer::FixedDeltaTime() / Global::sim_params.num_substeps;
  //==========================
  // Launch kernel
  //==========================
  SetSimulationParams(&Global::sim_params);

  bool b_obstacle_updated = UpdateObstacleAnimationFrame();

  cloth_bvh_->UpdateData(pointer(cloth_->faces), 0.0f, false, cloth_->NumFaces());
  cloth_bvh_->Construct();

  if (Global::sim_params.imminent_repulsion) {
    imminent_repulsion_solver_->UpdateProximity(cloth_.get(), obstacle_.get(), cloth_bvh_.get(),
                                                obstacle_bvh_.get(), 1e-6f);
    imminent_repulsion_solver_->Generate();
  }

  std::shared_ptr<Untangling::ICM> icm_solver =
      std::make_shared<Untangling::ICM>(cloth_->NumNodes(), cloth_->NumEdges());
#ifdef DEBUG_MODE
  //thrust::host_vector<Untangling::IntersectionWithGradient> h_intersections;
#endif  // DEBUG_MODE
  obstacle_bvh_->Update(pointer(obstacle_->faces), true);

  for (int substep = 0; substep < Global::sim_params.num_substeps; substep++) {
    PredictPositions(pointer(cloth_->nodes), substep_time, cloth_->NumNodes());

    for (int iteration = 0; iteration < Global::sim_params.num_iterations; iteration++) {
      if (Global::sim_params.imminent_repulsion) {
        imminent_repulsion_solver_->Solve(
            cloth_.get(), Global::sim_config.quality.repulsion.imminent_thickness, true,
            Global::sim_config.quality.repulsion.relaxation_rate, substep_time,
            Global::sim_params.frame_index, iteration);
      }

      if (Global::sim_params.pbd_repulsion) {
        if (iteration % 10 == 0) {
          pbd_repulsion_solver_->UpdatePairs(cloth_.get(), obstacle_.get(), cloth_bvh_.get(),
                                             obstacle_bvh_.get(), 1e-6f);
        }
        pbd_repulsion_solver_->SolveNoProximity(
            cloth_.get(), Global::sim_config.quality.repulsion.pbd_thickness,
            Global::sim_config.quality.repulsion.relaxation_rate, substep_time,
            Global::sim_params.frame_index, iteration);
      }

      for (int c = 0; c < fem_strain_solver_->Coloring()->n_colors; c++) {
        fem_strain_solver_->Solve(pointer(cloth_->nodes), c);
      }

      if (Global::sim_params.solve_bending) {
        for (int c = 0; c < fem_isometric_bending_solver_->Coloring()->n_colors; c++) {
          fem_isometric_bending_solver_->Solve(pointer(cloth_->nodes), c, iteration, substep_time);
        }
      }

      SolveLongRangeAttachments(Global::sim_params.geodesic_LRA);

      binding_->Solve(pointer(cloth_->nodes), pointer(obstacle_->nodes));
    }

    Timer::StartTimerGPU("Solver_CollisionStep");
    int total_impacts = 0;
    ImpactZoneOptimization::CollisionStep(cloth_, obstacle_, memory_pool_,
                                          Global::sim_params.frame_index, substep_time,
                                          Global::sim_config.quality.impact_zone.obstacle_mass,
                                          Global::sim_config.quality.impact_zone.thickness);
    Timer::EndTimerGPU("Solver_CollisionStep");

    if (Global::sim_params.icm_enable) {
      GIADetangleStep(icm_solver, cloth_, cloth_bvh_, obstacle_, obstacle_bvh_);
    }

    Finalize(pointer(cloth_->nodes), substep_time);
  }
  obstacle_->PredictedToPositions();

  cloth_->UpdateNormals();
  obstacle_->UpdateNormals();
  //==========================
  // Sync
  //==========================
  Timer::EndTimerGPU("Solver_Total");
  cloth_->Sync();
  obstacle_->Sync();
#ifdef DEBUG_MODE
  //auto h_quadInds = rRepSolver->HostQuadIndices();
  //Debug::DrawEEProximities(line_renderer, m_garmentPhysicsMesh, h_quadInds);
  //Debug::DrawICMIntersections(line_renderer, line_renderer_B, point_renderer, arrow_renderer, m_garmentPhysicsMesh, h_intersections);
  //DrawObstacleAABBs();
#endif  // DEBUG_MODE

  SaveCache();
}

void GIADetangleStep(std::shared_ptr<Untangling::ICM> solver, std::shared_ptr<PhysicsMesh> cloth,
                     std::shared_ptr<BVH> cloth_bvh, std::shared_ptr<PhysicsMesh> obstacle,
                     std::shared_ptr<BVH> obstacle_bvh) {
  auto* gia = new Untangling::GlobalIntersectionAnalysis();
  cloth_bvh->Update(pointer(cloth->faces), true);
  Timer::StartTimerGPU("Solver_GIA");
  int n_islands = gia->FloodFillIntersectionIslands(cloth, cloth_bvh);
  //#ifdef DEBUG_MODE
  //		Debug::DrawGIAContours(point_renderer_A, point_renderer_B, point_renderer_C, point_renderer_D, point_renderer_E,
  //			gia->HostIntersections(), gia->HostIntersectionStates(), m_garmentPhysicsMesh);
  //#endif // DEBUG_MODE
  Timer::EndTimerGPU("Solver_GIA");
  double time_gia = Timer::GetTimerGPU("Solver_GIA");

  bool has_collisions = true;
  for (int iter = 0; iter < 100; iter++) {
    if (iter % 5 == 0) {
      cloth_bvh->Update(pointer(cloth->faces), true);
      solver->UpdatePairs(cloth.get(), cloth.get(), cloth_bvh.get(), cloth_bvh.get());
    }
    // local
    //icmSolver->UpdateGradient(cloth.get(), obstacle.get());
    // global
    has_collisions = solver->UpdateGradientGIA(cloth.get(), obstacle.get(), n_islands, gia);
    if (!has_collisions) {
      break;
    }

#ifdef DEBUG_MODE
    //if (iter == 0)
    //{
    //	h_intersections = icmSolver->HostIntersections();
    //}
#endif  // DEBUG_MODE

    solver->ApplyImpulse(cloth.get(), Global::sim_params.icm_h0, Global::sim_params.icm_g0);
    //solver->ApplyGradient(m_garmentPhysicsMesh.get(), Global::sim_params.icm_h0, Global::sim_params.icm_g0);
  }
  delete gia;
}

void ClothSolverBase::ClearAnimationCache() {
  for (size_t i = 0; i < h_cloth_positions_cache->size(); i++) {
    free((*h_cloth_positions_cache)[i]);
    free((*h_cloth_normals_cache)[i]);
  }

  for (size_t i = 0; i < h_obstacle_positions_cache->size(); i++) {
    free((*h_obstacle_positions_cache)[i]);
    free((*h_obstacle_normals_cache)[i]);
  }

  h_cloth_positions_cache->clear();
  h_cloth_normals_cache->clear();
  h_obstacle_positions_cache->clear();
  h_obstacle_normals_cache->clear();

  num_cached_obstacle_frames = 0;
  num_cached_cloth_frames = 0;
}

void ClothSolverBase::PrintPreSimulationProgress() {
  Scalar percentage = static_cast<Scalar>(Global::sim_params.pre_simulation_frame_index + 1) /
                      static_cast<Scalar>(Global::sim_params.pre_simulation_frames);
  auto val = static_cast<int>(percentage * 100);
  auto lpad = static_cast<int>(percentage * PBWIDTH);
  int rpad = PBWIDTH - lpad;
  printf("\rPre-simulation %d/%d [%.*s%*s] %3d%%", Global::sim_params.pre_simulation_frame_index,
         Global::sim_params.pre_simulation_frames, lpad, PBSTR, rpad, "", val);
  fflush(stdout);
}

void ClothSolverBase::PrintSimulationProgress() {
  Scalar percentage = static_cast<Scalar>(Global::sim_params.frame_index + 1) /
                      static_cast<Scalar>(Global::sim_params.num_frames);
  auto val = static_cast<int>(percentage * 100);
  auto lpad = static_cast<int>(percentage * PBWIDTH);
  int rpad = PBWIDTH - lpad;
  printf("\rSimulating sequence %s %d/%d [%.*s%*s] %3d%% %.2f fps", selected_sequence_label.c_str(),
         Global::sim_params.frame_index, Global::sim_params.num_frames, lpad, PBSTR, rpad, "", val,
         solver_time > 0 ? (1000.0 / solver_time) : 0);
  fflush(stdout);
}

void ClothSolverBase::SetupBinding() {
  binding_->Generate(pointer(cloth_->nodes), pointer(obstacle_->nodes),
                     pointer(memory_pool_->skin_params));
}

void ClothSolverBase::SetupInternalDynamics() {
  LOG_TRACE("Setup internal cloth dynamics");
  auto seed = seed_generator_->Get();

  if (Global::sim_params.solver_mode == SOLVER_MODE::SWIFT) {
    // stretch
    LOG_INFO("Using basic stretch model");
    int n_cloth_edges = cloth_->NumEdges();
    basic_stretch_solver_->SetupGraphColoring(pointer(cloth_->edges), n_cloth_edges, seed);
    const auto& fabric_settings = Global::sim_config.swift.fabric;
    if (fabric_settings.solve_bending) {
      // bending
      LOG_INFO("Using basic bending model");
      basic_bend_solver_->SetupGraphColoring(seed);
    }
  } else if (Global::sim_params.solver_mode == SOLVER_MODE::QUALITY) {
    const auto& fabric_settings = Global::sim_config.quality.fabric;
    // strain
    LOG_INFO("Using FEM-based strain model");
    fem_strain_solver_ =
        std::make_shared<FEM::StrainSolver>(pointer(cloth_->faces), cloth_->NumFaces(), seed);
    fem_strain_solver_->Init(pointer(memory_pool_->initial_positions), pointer(cloth_->faces),
                             fabric_settings.xx_stiffness, fabric_settings.yy_stiffness,
                             fabric_settings.xy_stiffness, fabric_settings.xy_poisson_ratio,
                             fabric_settings.yx_poisson_ratio);

    // bending
    if (fabric_settings.solve_bending) {
      LOG_INFO("Using isometric bending model");
      fem_isometric_bending_solver_->SetupGraphColoring(seed);
      fem_isometric_bending_solver_->Init(pointer(memory_pool_->initial_positions),
                                          fabric_settings.bending_stiffness);
    }
  }
}

void ClothSolverBase::SetupRTriangle() {
  LOG_DEBUG("Build obstacle R-Triangle");
  {
    ScopedTimerGPU timer("Solver_buildDeviceAdj");
    BuildVFAdjacency(obstacle_, memory_pool_->obstacle_nb_vf, memory_pool_->obstacle_nb_vf_prefix);
  }
  double device_adj_build_time = Timer::GetTimerGPU("Solver_buildDeviceAdj");
  LOG_INFO("Obstacle device adjacency build done, took {:.2f}ms", device_adj_build_time);
  {
    ScopedTimerGPU timer("Solver_buildRTri");

    obstacle_r_tri_ = std::make_shared<RTriangle>();
    obstacle_r_tri_->Init(obstacle_->DeviceIndices(), memory_pool_->obstacle_nb_vf,
                          memory_pool_->obstacle_nb_vf_prefix);
  }
  double r_tri_build_time = Timer::GetTimerGPU("Solver_buildRTri");
  LOG_INFO("Obstacle R-Tri build done, took {:.2f}ms", r_tri_build_time);
  obstacle_->RegisterRTris(obstacle_r_tri_->r_tris);

  LOG_DEBUG("Build cloth R-Triangle");
  {
    ScopedTimerGPU timer("Solver_buildDeviceAdj");
    BuildVFAdjacency(cloth_, memory_pool_->cloth_nb_vf, memory_pool_->cloth_nb_vf_prefix);
  }
  device_adj_build_time = Timer::GetTimerGPU("Solver_buildDeviceAdj");
  LOG_INFO("Garment device adjacency build done, took {:.2f}ms", device_adj_build_time);
  {
    ScopedTimerGPU timer("Solver_buildRTri");

    cloth_r_tri_ = std::make_shared<RTriangle>();
    cloth_r_tri_->Init(cloth_->DeviceIndices(), memory_pool_->cloth_nb_vf,
                       memory_pool_->cloth_nb_vf_prefix);
  }
  r_tri_build_time = Timer::GetTimerGPU("Solver_buildRTri");
  LOG_INFO("Garment R-Tri build done, took {:.2f}ms", r_tri_build_time);
  cloth_->RegisterRTris(cloth_r_tri_->r_tris);
}

void ClothSolverBase::AddEuclidean(int particle_index, int slot_index, Scalar distance) {
  if (distance == 0) {
    cloth_->SetInvMass(0.0f, particle_index);
  }

  memory_pool_->AddEuclidean(particle_index, slot_index, distance);
}

void ClothSolverBase::AddGeodesic(unsigned int src_index, unsigned int tgt_index,
                                  Scalar rest_length) {
  memory_pool_->AddGeodesic(src_index, tgt_index, rest_length);
}

void ClothSolverBase::UpdateColliders(std::vector<Collider*>& colliders) {
  for (int i = 0; i < colliders.size(); i++) {
    const Collider* c = colliders[i];
    if (!c->enabled) {
      continue;
    }
    SDFCollider sc;
    sc.type = c->type;
    sc.position = c->actor->transform->position;
    sc.scale = c->actor->transform->scale;
    sc.cur_transform = c->cur_transform;
    sc.inv_cur_transform = glm::inverse(c->cur_transform);
    sc.last_transform = c->last_transform;
    sc.delta_time = Timer::FixedDeltaTime();
    memory_pool_->AddSDFCollider(sc);
  }
}

void ClothSolverBase::SolveLongRangeAttachments(bool geodesic_LRA) {
  ScopedTimerGPU timer("Solver_SolveLRA");
  if (geodesic_LRA) {
    // Geodesic LRA
    SolveGeodesicLRA(pointer(cloth_->nodes), pointer(memory_pool_->geodesic_src_indices),
                     pointer(memory_pool_->geodesic_tgt_indices),
                     pointer(memory_pool_->geodesic_rest_length), memory_pool_->NumGeodesics(),
                     Global::sim_params.long_range_stretchiness);
  } else {
    // Euclidean LRA
    //SolveEuclideanLRA
    //(
    //	predicted,
    //	deltas,
    //	delta_counts,
    //	invMasses,
    //	attach_particle_ids,
    //	attach_slot_ids,
    //	midstepAttachSlotPositions,
    //	attach_distances,
    //	attach_particle_ids.size(),
    //	Global::sim_params.long_range_stretchiness
    //);
  }
}

std::string ClothSolverBase::GetIdentifier() {
  return solver_identifier_;
}

void ClothSolverBase::SetIdentifier(const std::string& identifier) {
  solver_identifier_ = identifier;
}

#ifdef DEBUG_MODE
void ClothSolverBase::DrawObstacleAABBs() {
  if (!Global::sim_params.draw_obstacle_aabbs) {
    obstacle_aabb_renderer->Reset();
    return;
  }

  std::vector<AABB> aabbs;
  auto h_aabbs = obstacle_bvh_->AABBsHost();
  int n_internal_nodes = obstacle_bvh_->NumInternalNodes();
  checkCudaErrors(cudaDeviceSynchronize());
  for (int i = 0; i < n_internal_nodes * 2 + 1; i++) {
    Vector3 upper = h_aabbs[i].upper;
    Vector3 lower = h_aabbs[i].lower;
    aabbs.emplace_back(Vector3(lower.x, lower.y, lower.z), Vector3(upper.x, upper.y, upper.z));
  }
  obstacle_aabb_renderer->SetAABBs(aabbs);
}

void ClothSolverBase::DrawInternalNodes(BVH* bvh) {
  if (!Global::sim_params.draw_internal_nodes) {
    internal_aabb_renderer->Reset();
    return;
  }

  std::vector<AABB> aabbs;
  auto h_aabbs = bvh->AABBsHost();
  int n_internal_nodes = bvh->NumInternalNodes();

  for (int i = 0; i < n_internal_nodes; i++) {
    Vector3 upper = h_aabbs[i].upper;
    Vector3 lower = h_aabbs[i].lower;
    aabbs.emplace_back(Vector3(lower.x, lower.y, lower.z), Vector3(upper.x, upper.y, upper.z));
  }

  internal_aabb_renderer->SetAABBs(aabbs);
}

void ClothSolverBase::DrawExternalNodes(BVH* bvh) {
  if (!Global::sim_params.draw_external_nodes) {
    external_aabb_renderer->Reset();
    return;
  }

  std::vector<AABB> aabbs;
  auto h_aabbs = bvh->AABBsHost();
  int n_internal_nodes = bvh->NumInternalNodes();

  for (int i = 0; i < n_internal_nodes + 1; i++) {
    Vector3 upper = h_aabbs[n_internal_nodes + i].upper;
    Vector3 lower = h_aabbs[n_internal_nodes + i].lower;
    aabbs.emplace_back(Vector3(lower.x, lower.y, lower.z), Vector3(upper.x, upper.y, upper.z));
  }

  external_aabb_renderer->SetAABBs(aabbs);
}

void ClothSolverBase::DrawPairFFs(const thrust::host_vector<int>& h_types,
                                  const thrust::host_vector<int>& h_indices) {
  if (!Global::sim_params.draw_obstacle_aabbs) {
    cloth_aabb_renderer->Reset();
    obstacle_aabb_renderer->Reset();
    return;
  }
  std::vector<AABB> cloth_aabbs, obstacle_aabbs;
  auto h_cloth_default_aabbs = cloth_bvh_->AABBsHost();
  auto h_obstacle_default_aabbs = obstacle_bvh_->AABBsHost();
  int n_cloth_bvh_internal_nodes = cloth_bvh_->NumInternalNodes();
  int n_obstacle_bvh_internal_nodes = obstacle_bvh_->NumInternalNodes();
  checkCudaErrors(cudaDeviceSynchronize());
  for (int i = 0; i < h_types.size() / 2; i++) {
    int f1_idx = h_indices[i * 2];
    int f1_type = h_types[i * 2];
    int f2_idx = h_indices[i * 2 + 1];
    int f2_type = h_types[i * 2 + 1];

    if (f1_type == 0) {
      Vector3 upper = h_cloth_default_aabbs[n_cloth_bvh_internal_nodes + f1_idx].upper;
      Vector3 lower = h_cloth_default_aabbs[n_cloth_bvh_internal_nodes + f1_idx].lower;
      cloth_aabbs.emplace_back(Vector3(lower.x, lower.y, lower.z),
                               Vector3(upper.x, upper.y, upper.z));
    } else {
      Vector3 upper = h_obstacle_default_aabbs[n_obstacle_bvh_internal_nodes + f1_idx].upper;
      Vector3 lower = h_obstacle_default_aabbs[n_obstacle_bvh_internal_nodes + f1_idx].lower;
      obstacle_aabbs.emplace_back(Vector3(lower.x, lower.y, lower.z),
                                  Vector3(upper.x, upper.y, upper.z));
    }
    if (f2_type == 0) {
      Vector3 upper = h_cloth_default_aabbs[n_cloth_bvh_internal_nodes + f2_idx].upper;
      Vector3 lower = h_cloth_default_aabbs[n_cloth_bvh_internal_nodes + f2_idx].lower;
      cloth_aabbs.emplace_back(Vector3(lower.x, lower.y, lower.z),
                               Vector3(upper.x, upper.y, upper.z));
    } else {
      Vector3 upper = h_obstacle_default_aabbs[n_obstacle_bvh_internal_nodes + f2_idx].upper;
      Vector3 lower = h_obstacle_default_aabbs[n_obstacle_bvh_internal_nodes + f2_idx].lower;
      obstacle_aabbs.emplace_back(Vector3(lower.x, lower.y, lower.z),
                                  Vector3(upper.x, upper.y, upper.z));
    }
  }

  cloth_aabb_renderer->SetAABBs(cloth_aabbs);
  obstacle_aabb_renderer->SetAABBs(obstacle_aabbs);
}
#endif  // DEBUG_MODE
}  // namespace XRTailor
