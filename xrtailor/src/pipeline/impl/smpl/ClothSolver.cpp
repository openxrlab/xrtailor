#include <xrtailor/pipeline/impl/smpl/ClothSolver.hpp>

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

using SMPL = smplx::model_config::SMPL;
using SMPLH = smplx::model_config::SMPLH;
using SMPLX = smplx::model_config::SMPLX;
using AMASS_SMPLH_G = smplx::sequence_config::AMASS_SMPLH_G;
using AMASS_SMPLX_G = smplx::sequence_config::AMASS_SMPLX_G;

namespace XRTailor {
template <class ModelConfig, class SequenceConfig>
ClothSolverSMPL<ModelConfig, SequenceConfig>::ClothSolverSMPL() {
  name = "ClothSolverSMPL";
  amass_x_rotation_ = Global::sim_config.smpl.amass_x_rotation;
}

template <class ModelConfig, class SequenceConfig>
void ClothSolverSMPL<ModelConfig, SequenceConfig>::Start() {
  ClothSolverBase::Start();

  LOG_TRACE("Setup spatial hash");
  spatial_hash_ = std::make_shared<SpatialHashGPU>(
      Global::sim_params.particle_diameter, Global::sim_params.num_particles,
      Global::sim_config.swift.collision.self_contact.hash_cell_size,
      Global::sim_params.max_num_neighbors);
  spatial_hash_->SetInitialPositions(pointer(cloth_->nodes), cloth_->NumNodes());

  Global::sim_params.frame_index = 0;
  Global::sim_params.frame_rate = sequence_->frame_rate;

  unsigned int mocap_frame_rate = Global::sim_params.frame_rate;
  unsigned int target_frame_rate = Global::sim_config.animation.target_frame_rate;
  unsigned int num_lerped_frames = Global::sim_params.num_lerped_frames;
  LOG_DEBUG("Target framerate: {}", Global::sim_config.animation.target_frame_rate);

  if (mocap_frame_rate < target_frame_rate) {
    LOG_ERROR("Mocap framerate {0} below target framerate {1}", mocap_frame_rate,
              target_frame_rate);
    exit(TAILOR_EXIT::INVALID_TARGET_FRAMERATE);
  }
  frame_step_size = static_cast<int>(mocap_frame_rate) / static_cast<int>(target_frame_rate);
  Global::sim_params.num_frames =
      ((sequence_->n_frames - num_lerped_frames) / frame_step_size) + num_lerped_frames;

  // Build bvh to find nearest vertices from cloth to body
  // Notice that we do this before reading motion sequence
  // so that the body is in the canonical space
  body_->Update();
  int n_obstacle_nodes = obstacle_->NumNodes();
  UpdateObstaclePredicted(pointer(obstacle_->nodes), body_->verts(), n_obstacle_nodes,
                          obstacle_model_matrix_, 0.0f);
  obstacle_->PredictedToPositions();
  obstacle_->UpdateNormals();
  LOG_TRACE("Update BVH data");
  checkCudaErrors(cudaDeviceSynchronize());
  obstacle_bvh_->UpdateData(pointer(obstacle_->faces), Global::sim_params.bvh_tolerance, false,
                            obstacle_->NumFaces());
  checkCudaErrors(cudaDeviceSynchronize());
  LOG_DEBUG("#nodes: {}, #faces: {}", obstacle_->nodes.size(), obstacle_->faces.size());
  LOG_TRACE("Construct BVH.");
  obstacle_bvh_->Construct();
  checkCudaErrors(cudaDeviceSynchronize());

  LOG_TRACE("Save initial positions");
  UpdateVertices(pointer(memory_pool_->initial_positions), cloth_);
  checkCudaErrors(cudaDeviceSynchronize());
  memory_pool_->SetupSkinnedObstacle(Global::sim_params.num_particles);

  QueryNearestBarycentricClothToObstacle(cloth_, pointer(memory_pool_->skin_params), obstacle_bvh_);

  thrust::host_vector<SkinParam> h_skin_params = memory_pool_->skin_params;

  LOG_TRACE("Embed skinning weights into garment");
  int n_cloth_nodes = cloth_->NumNodes();
  int n_cloth_edges = cloth_->NumEdges();
  LOG_DEBUG("Num garment nodes: {}", n_cloth_nodes);
  body_->EmbedBarycentricSkinningWeightsIntoCloth(body_->model.weights, body_->cloth_weights,
                                                  n_cloth_nodes, pointer(h_skin_params));
  checkCudaErrors(cudaDeviceSynchronize());

  LOG_TRACE("embed_barycentric_blend_shapes_into_garment");
  body_->EmbedBarycentricBlendShapesIntoCloth(n_cloth_nodes, pointer(h_skin_params));
  checkCudaErrors(cudaDeviceSynchronize());

  body_->_cuda_set_cloth_vertices(cloth_);

  if (sequence_->n_frames) {
    sequence_->set_shape(*body_);
    sequence_->set_pose(*body_, 0);
  }

  body_->Update(false, pose_blendshape_, true, false);
  UpdateObstaclePredicted(pointer(obstacle_->nodes), body_->verts(), n_obstacle_nodes,
                          obstacle_model_matrix_, amass_x_rotation_);
  obstacle_->PredictedToPositions();
  UpdateSkinnedVertices(cloth_, body_->device_cloth_verts(), amass_x_rotation_);

  obstacle_->UpdateNodeGeometries();
  checkCudaErrors(cudaDeviceSynchronize());
  cloth_->Sync();
  obstacle_->Sync();
  checkCudaErrors(cudaDeviceSynchronize());

  obstacle_bvh_->UpdateData(pointer(obstacle_->faces), 0.0f, false, obstacle_->NumFaces());
  obstacle_bvh_->Construct();
  SetupRTriangle();

  pbd_vf_dcd_ = std::make_shared<PBDCollision::DCD::VFConstraint>(n_cloth_nodes);
  pbd_vf_ccd_ = std::make_shared<PBDCollision::CCD::RTConstraint>(n_cloth_nodes);
  pbd_ee_dcd_ = std::make_shared<PBDCollision::DCD::EEConstraint>(n_cloth_edges, obstacle_r_tri_);

  BuildDeviceEFAdjacency(h_cloth_nb_ef_, memory_pool_->cloth_nb_ef,
                         memory_pool_->cloth_nb_ef_prefix);
  BuildDeviceEFAdjacency(h_obstacle_nb_ef_, memory_pool_->obstacle_nb_ef,
                         memory_pool_->obstacle_nb_ef_prefix);

  cloth_->RegisterEFAdjacency(memory_pool_->cloth_nb_ef, memory_pool_->cloth_nb_ef_prefix);
  obstacle_->RegisterEFAdjacency(memory_pool_->obstacle_nb_ef, memory_pool_->obstacle_nb_ef_prefix);

  SetupInternalDynamics();
  SetupBinding();
}

template <class ModelConfig, class SequenceConfig>
void ClothSolverSMPL<ModelConfig, SequenceConfig>::Update() {
  ClothSolverBase::Update();
}

template <class ModelConfig, class SequenceConfig>
void ClothSolverSMPL<ModelConfig, SequenceConfig>::FixedUpdate() {
  ClothSolverBase::FixedUpdate();
}

template <class ModelConfig, class SequenceConfig>
void ClothSolverSMPL<ModelConfig, SequenceConfig>::OnDestroy() {
  ClothSolverBase::OnDestroy();
}

template <class ModelConfig, class SequenceConfig>
void ClothSolverSMPL<ModelConfig, SequenceConfig>::SaveCache() {
  ScopedTimerGPU timer("Solver_SaveCache");
  if (Global::sim_params.pre_simulation_frame_index >=
          Global::sim_params.pre_simulation_frames &&  // ignore pre-simulation frames
      Global::sim_params.frame_index > 0 &&
      Global::sim_params.frame_index <=
          sequence_->n_frames &&  // stop recording when animation is finished
      Global::sim_params.frame_index > Global::sim_params.num_lerped_frames  // ignore lerped frames
  ) {
    if (Global::sim_params.record_obstacle) {
      int n_nodes = obstacle_->NumNodes();
      h_obstacle_positions_ptr = static_cast<Vector3*>(malloc(sizeof(Vector3) * n_nodes));
      h_obstacle_normals_ptr = static_cast<Vector3*>(malloc(sizeof(Vector3) * n_nodes));

      thrust::host_vector<Vector3> h_obstacle_positions_vec = obstacle_->HostPositions();
      thrust::host_vector<Vector3> h_obstacle_normals_vec = obstacle_->HostNormals();
      MemcpyHostToHost<Vector3>(h_obstacle_positions_vec, h_obstacle_positions_ptr, n_nodes);
      MemcpyHostToHost<Vector3>(h_obstacle_normals_vec, h_obstacle_normals_ptr, n_nodes);

      h_obstacle_positions_cache->push_back(h_obstacle_positions_ptr);
      h_obstacle_normals_cache->push_back(h_obstacle_normals_ptr);

      num_cached_obstacle_frames++;
    }

    if (Global::sim_params.record_cloth) {
      int n_nodes = cloth_->NumNodes();
      h_cloth_positions_ptr = static_cast<Vector3*>(malloc(sizeof(Vector3) * n_nodes));
      h_cloth_normals_ptr = static_cast<Vector3*>(malloc(sizeof(Vector3) * n_nodes));

      thrust::host_vector<Vector3> h_garment_positions_vec = cloth_->HostPositions();
      thrust::host_vector<Vector3> h_garment_normals_vec = cloth_->HostNormals();
      MemcpyHostToHost<Vector3>(h_garment_positions_vec, h_cloth_positions_ptr, n_nodes);
      MemcpyHostToHost<Vector3>(h_garment_normals_vec, h_cloth_normals_ptr, n_nodes);

      h_cloth_positions_cache->push_back(h_cloth_positions_ptr);
      h_cloth_normals_cache->push_back(h_cloth_normals_ptr);

      num_cached_cloth_frames++;
    }
  }
}

template <class ModelConfig, class SequenceConfig>
bool ClothSolverSMPL<ModelConfig, SequenceConfig>::UpdateObstacleAnimationFrame() {
  if (Global::sim_params.pre_simulation_frame_index < Global::sim_params.pre_simulation_frames) {
    PrintPreSimulationProgress();
    Global::sim_params.pre_simulation_frame_index++;

    return false;
  }

  if (!Global::sim_params.update_obstacle_animation) {
    return false;
  }

  int n_obstacle_nodes = obstacle_->NumNodes();

  if (Global::sim_params.frame_index < Global::sim_params.num_frames) {
    PrintSimulationProgress();
    {
      ScopedTimerGPU timer("Solver_UpdateSMPL");
      uint target_frame_index =
          (Global::sim_params.frame_index > Global::sim_params.num_lerped_frames)
              ? (static_cast<size_t>(Global::sim_params.frame_index) -
                 Global::sim_params.num_lerped_frames) *
                        frame_step_size +
                    Global::sim_params.num_lerped_frames
              : Global::sim_params.frame_index;
      sequence_->set_pose(*body_, target_frame_index);
      body_->Update(false, pose_blendshape_, false, false);
      UpdateObstaclePredicted(pointer(obstacle_->nodes), body_->verts(), n_obstacle_nodes,
                              obstacle_model_matrix_, amass_x_rotation_);
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
    //#ifdef DEBUG_MODE
    //    // early exit
    //    if (Global::sim_params.frame_index >= 199) {
    //      simulation_finished = true;
    //    }
    //    return true;
    //#endif  // DEBUG_MODE
    return true;
  }

  return false;
}

template <class ModelConfig, class SequenceConfig>
void ClothSolverSMPL<ModelConfig, SequenceConfig>::InstantSkinning() {
  uint frame_index = Global::sim_params.frame_index;
  sequence_->set_pose(*body_, static_cast<size_t>(frame_index > 0 ? frame_index - 1 : 0));
  body_->Update(false, pose_blendshape_, true, false);
  UpdateSkinnedVertices(cloth_, body_->device_cloth_verts(), amass_x_rotation_);
  cloth_->UpdateNormals();
  checkCudaErrors(cudaDeviceSynchronize());
  cloth_->Sync();

  cloth_->ResetX();
  cloth_->ResetVelocities();
  checkCudaErrors(cudaDeviceSynchronize());
}

template <class ModelConfig, class SequenceConfig>
int ClothSolverSMPL<ModelConfig, SequenceConfig>::AddObstacle(
    std::shared_ptr<Mesh> mesh, Mat4 model_matrix, unsigned int obstacle_id,
    std::shared_ptr<smplx::Body<ModelConfig>> body,
    std::shared_ptr<smplx::Sequence<SequenceConfig>> sequence,
    const std::vector<unsigned int>& masked_indices) {
  LOG_TRACE("Add obstacle.");

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

  LOG_DEBUG("Register masked index buffer");
  int n_obstacle_faces = obstacle_->NumFaces();
  if (Global::sim_config.smpl.enable_collision_filter) {
    memory_pool_->SetupObstacleMaskedIndex(masked_indices);
  } else {
    thrust::host_vector<uint> obstacle_indices = obstacle_->HostIndices();
    memory_pool_->SetupObstacleMaskedIndex(obstacle_indices);
  }

  sequence_ = sequence;
  body_ = body;
  obstacle_model_matrix_ = model_matrix;

  double time = Timer::EndTimer("INIT_BODY_GPU") * 1000;
  LOG_TRACE("AddObstacle done. Took time {:.2f} ms", time);
  checkCudaErrors(cudaDeviceSynchronize());

  return prev_num_obstacle_vertices;
}

template <class ModelConfig, class SequenceConfig>
int ClothSolverSMPL<ModelConfig, SequenceConfig>::AddCloth(std::shared_ptr<Mesh> mesh,
                                                           Mat4 model_matrix,
                                                           Scalar particle_diameter) {
  LOG_TRACE("Add cloth");

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

  // Global parameters
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

  checkCudaErrors(cudaDeviceSynchronize());
  cloth_->Sync();

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

    if (fabric_settings.solve_bending) {
      basic_bend_solver_->AddCloth(bend_indices, bend_angles, fabric_settings.bend_compliance);
      Timer::StartTimer("BASIC_BEND_ADD_CLOTH");
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
  LOG_TRACE("Add cloth done. Took time {:.2f} ms", time);
  LOG_TRACE("Use recommond max vel = {}", Global::sim_params.max_speed);
  checkCudaErrors(cudaDeviceSynchronize());
  return prev_num_particles;
}

template class ClothSolverSMPL<smplx::model_config::SMPL, smplx::sequence_config::AMASS_SMPLH_G>;
template class ClothSolverSMPL<smplx::model_config::SMPLH, smplx::sequence_config::AMASS_SMPLH_G>;
template class ClothSolverSMPL<smplx::model_config::SMPLX, smplx::sequence_config::AMASS_SMPLX_G>;
}  // namespace XRTailor
