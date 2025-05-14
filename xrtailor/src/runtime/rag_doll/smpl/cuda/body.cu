#include <iostream>

#include <xrtailor/runtime/rag_doll/smpl/smplx.hpp>
#include <xrtailor/runtime/rag_doll/smpl/util.hpp>
#include <xrtailor/runtime/rag_doll/smpl/internal/cuda_util.cuh>
#include <xrtailor/core/DeviceHelper.cuh>

namespace smplx {
namespace {
using cuda_util::from_host_eigen_matrix;
using cuda_util::from_host_eigen_sparse_matrix;
using cuda_util::to_host_eigen_matrix;
using cuda_util::device::BLOCK_SIZE;

namespace device {

/** Joint regressor: multiples sparse matrix in CSR represented by
 *  (model_jr_values(nnz), ..inner(nnz), ..outer(#joints+1)) to
 *  d_verts_shaped(#verts,3) row-major
 *  -> outputs to out(#joints, 3) row-major
 *  TODO: Optimize. The matrix is very wide and this is not efficient */
__global__ void joint_regressor(float* RESTRICT d_verts_shaped, float* RESTRICT model_jr_values,
                                int* RESTRICT model_jr_inner, int* RESTRICT model_jr_outer,
                                float* RESTRICT out_joints) {
  const int joint = threadIdx.y, idx = threadIdx.x;
  out_joints[joint * 3 + idx] = 0.f;
  for (int i = model_jr_outer[joint]; i < model_jr_outer[joint + 1]; ++i) {
    out_joints[joint * 3 + idx] += model_jr_values[i] * d_verts_shaped[model_jr_inner[i] * 3 + idx];
  }
}

/** Linear blend skinning kernel.
  * d_joint_global_transform (#joints, 12) row-major;
  *   global-space homogeneous transforms (bottom row dropped)
  *   at each joint from local_to_global
  * d_points_shaped (#points, 3) row-major; vertices after blendshapes applied
  * (model_weights_values(nnz), ..inner(nnz), ..outer(#joints+1)) sparse LBS weights in CSR
  * -> out_verts(#points, 3) resulting vertices after deformation */
__global__ void lbs(float* RESTRICT d_joint_global_transform, float* RESTRICT d_verts_shaped,
                    float* RESTRICT model_weights_values, int* RESTRICT model_weights_inner,
                    int* RESTRICT model_weights_outer,
                    float* RESTRICT out_verts,  // transformed joint pos
                    const int n_joints, const int n_verts) {
  const int vert = blockDim.x * blockIdx.x + threadIdx.x;  // Vert idx
  if (vert < n_verts) {
    for (int i = 0; i < 3; ++i) {
      out_verts[vert * 3 + i] = 0.f;
      for (int joint_it = model_weights_outer[vert]; joint_it < model_weights_outer[vert + 1];
           ++joint_it) {
        const int joint_row_idx = model_weights_inner[joint_it] * 12 + i * 4;
        for (int j = 0; j < 3; ++j) {
          out_verts[vert * 3 + i] += model_weights_values[joint_it] *
                                     d_joint_global_transform[joint_row_idx + j] *
                                     d_verts_shaped[vert * 3 + j];
        }
        out_verts[vert * 3 + i] +=
            model_weights_values[joint_it] * d_joint_global_transform[joint_row_idx + 3];
      }
    }
  }
}

}  // namespace device
}  // namespace

template <class ModelConfig>
__host__ void Body<ModelConfig>::_cuda_load() {
  cudaCheck(cudaMalloc((void**)&device.verts, model.n_verts() * 3 * sizeof(float)));
  cudaCheck(cudaMalloc((void**)&device.blendshape_params, model.n_blend_shapes() * sizeof(float)));
  cudaCheck(cudaMalloc((void**)&device.joint_transforms, model.n_joints() * 12 * sizeof(float)));
  cudaCheck(cudaMalloc((void**)&device.verts_shaped, model.n_verts() * 3 * sizeof(float)));
  cudaCheck(cudaMalloc((void**)&device.joints_shaped, model.n_joints() * 3 * sizeof(float)));
}

template <class ModelConfig>
__host__ void Body<ModelConfig>::_cuda_load_cloth(int _num_garment_verts) {
  num_cloth_verts = _num_garment_verts;
  cudaCheck(cudaMalloc((void**)&device.initial_cloth_verts,
                       num_cloth_verts * 3 * sizeof(smplx::Scalar)));
  cudaCheck(
      cudaMalloc((void**)&device.cloth_verts, num_cloth_verts * 3 * sizeof(smplx::Scalar)));
  cudaCheck(cudaMalloc((void**)&device.cloth_verts_shaped,
                       num_cloth_verts * 3 * sizeof(smplx::Scalar)));
}

template <class ModelConfig>
__host__ void Body<ModelConfig>::_cuda_load_slots(int _num_slots) {
  num_slots = _num_slots;
  cudaCheck(cudaMalloc((void**)&device.initial_cloth_slots, num_slots * 3 * sizeof(float)));
  cudaCheck(cudaMalloc((void**)&device.cloth_slots, num_slots * 3 * sizeof(float)));
  cudaCheck(cudaMalloc((void**)&device.cloth_slots_shaped, num_slots * 3 * sizeof(float)));
}


__global__ void assign_cloth_vertex_kernel(XRTailor::Node** nodes, const unsigned int n_nodes,
                                             smplx::Scalar* skinning_verts) {
  GET_CUDA_ID(id, n_nodes);
  XRTailor::Vector3 pos = nodes[id]->x0;
  skinning_verts[id * 3] = static_cast<smplx::Scalar>(pos.x);
  skinning_verts[id * 3 + 1] = static_cast<smplx::Scalar>(pos.y);
  skinning_verts[id * 3 + 2] = static_cast<smplx::Scalar>(pos.z);
}

template <class ModelConfig>
__host__ void Body<ModelConfig>::_cuda_set_cloth_vertices(
    std::shared_ptr<XRTailor::PhysicsMesh> physics_mesh) {
  int n_nodes = physics_mesh->nodes.size();

  _cuda_load_cloth(n_nodes);
  CUDA_CALL(assign_cloth_vertex_kernel, n_nodes)
  (XRTailor::pointer(physics_mesh->nodes), n_nodes, device.initial_cloth_verts);
  cudaDeviceSynchronize();
}

template <class ModelConfig>
__host__ void Body<ModelConfig>::_cuda_free() {
  if (device.verts)
    cudaFree(device.verts);
  if (device.blendshape_params)
    cudaFree(device.blendshape_params);
  if (device.joint_transforms)
    cudaFree(device.joint_transforms);
  if (device.verts_shaped)
    cudaFree(device.verts_shaped);
  if (device.joints_shaped)
    cudaFree(device.joints_shaped);

  if (device.initial_cloth_verts)
    cudaFree(device.initial_cloth_verts);
  if (device.cloth_verts)
    cudaFree(device.cloth_verts);
  if (device.cloth_verts_shaped)
    cudaFree(device.cloth_verts_shaped);

  if (device.initial_cloth_slots)
    cudaFree(device.initial_cloth_slots);
  if (device.cloth_slots)
    cudaFree(device.cloth_slots);
  if (device.cloth_slots_shaped)
    cudaFree(device.cloth_slots_shaped);

  if (device.cloth_blend_shapes)
    cudaFree(device.cloth_blend_shapes);
  if (device.slot_blend_shapes)
    cudaFree(device.slot_blend_shapes);
}

template <class ModelConfig>
__host__ void Body<ModelConfig>::_cuda_maybe_retrieve_verts() const {
  if (!_verts_retrieved) {
    _verts.resize(model.n_verts(), 3);
    cudaMemcpy(_verts.data(), device.verts, _verts.size() * sizeof(float), cudaMemcpyDeviceToHost);
    _verts_retrieved = true;
  }
}

template <class ModelConfig>
__host__ void Body<ModelConfig>::_cuda_maybe_retrieve_cloth_verts() const {
  if (!_cloth_verts_retrieved) {
    _garment_verts.resize(num_cloth_verts, 3);
    cudaMemcpy(_garment_verts.data(), device.cloth_verts, _garment_verts.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    _cloth_verts_retrieved = true;
  }
}

template <class ModelConfig>
__host__ void Body<ModelConfig>::_cuda_maybe_retrieve_cloth_slots() const {
  if (!_cloth_slots_retrieved) {
    _garment_slots.resize(num_slots, 3);
    cudaMemcpy(_garment_slots.data(), device.cloth_slots, _garment_slots.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    _cloth_slots_retrieved = true;
  }
}

template <class ModelConfig>
__host__ void Body<ModelConfig>::_cuda_maybe_retrieve_verts_shaped() const {
  if (!_verts_shaped_retrieved) {
    _verts_shaped.resize(model.n_verts(), 3);
    cudaMemcpy(_verts_shaped.data(), device.verts_shaped, _verts_shaped.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    _verts_shaped_retrieved = true;
  }
}

template <class ModelConfig>
__host__ void Body<ModelConfig>::_cuda_maybe_retrieve_cloth_verts_shaped() const {
  if (!_cloth_verts_shaped_retrieved) {
    _garment_verts_shaped.resize(num_cloth_verts, 3);
    cudaMemcpy(_garment_verts_shaped.data(), device.cloth_verts_shaped,
               _garment_verts_shaped.size() * sizeof(float), cudaMemcpyDeviceToHost);
    _cloth_verts_shaped_retrieved = true;
  }
}

template <class ModelConfig>
__host__ void Body<ModelConfig>::_cuda_maybe_retrieve_cloth_slots_shaped() const {
  if (!_cloth_slots_shaped_retrieved) {
    _garment_slots_shaped.resize(num_slots, 3);
    cudaMemcpy(_garment_slots_shaped.data(), device.cloth_slots_shaped,
               _garment_slots_shaped.size() * sizeof(float), cudaMemcpyDeviceToHost);
    _cloth_slots_shaped_retrieved = true;
  }
}

template <class ModelConfig>
SMPLX_HOST void Body<ModelConfig>::_cuda_update(float* h_blendshape_params,
                                                float* h_joint_transforms,
                                                bool enable_pose_blendshapes,
                                                bool enable_garment_skinning,
                                                bool enable_slot_skinning) {
  // Verts will be updated
  _verts_retrieved = false;
  _verts_shaped_retrieved = false;

  _cloth_verts_retrieved = false;
  _cloth_verts_shaped_retrieved = false;

  _cloth_slots_retrieved = false;
  _cloth_slots_shaped_retrieved = false;

  // Copy parameters to GPU
  cudaCheck(cudaMemcpyAsync(device.blendshape_params, h_blendshape_params,
                            ModelConfig::n_blend_shapes() * sizeof(float), cudaMemcpyHostToDevice));

  // Shape blendshapes
  cudaCheck(cudaMemcpyAsync(device.verts_shaped, model.device.verts,
                            model.n_verts() * 3 * sizeof(float), cudaMemcpyDeviceToDevice));
  cuda_util::mmv_block<float, true>(model.device.blend_shapes, device.blendshape_params,
                                    device.verts_shaped, ModelConfig::n_verts() * 3,
                                    ModelConfig::n_shape_blends());

  // Joint regressor
  // TODO: optimize sparse matrix multiplication, maybe use ELL format
  dim3 jr_blocks(3, model.n_joints());
  device::joint_regressor<<<1, jr_blocks>>>(device.verts_shaped, model.device.joint_reg.values,
                                            model.device.joint_reg.inner,
                                            model.device.joint_reg.outer, device.joints_shaped);

  if (enable_pose_blendshapes) {
    // Pose blendshapes.
    // Note: this is the most expensive operation.
    cuda_util::mmv_block<float, true>(
        model.device.blend_shapes + ModelConfig::n_shape_blends() * 3 * ModelConfig::n_verts(),
        device.blendshape_params + ModelConfig::n_shape_blends(), device.verts_shaped,
        ModelConfig::n_verts() * 3, ModelConfig::n_pose_blends());
  }

  // Compute global joint transforms, this part can't be parallized and
  // is horribly slow on GPU; we do it on CPU instead
  // Actually, this is pretty bad too, TODO try implementing on GPU again
  cudaCheck(cudaMemcpyAsync(_joints_shaped.data(), device.joints_shaped,
                            model.n_joints() * 3 * sizeof(float), cudaMemcpyDeviceToHost));
  _local_to_global();
  cudaCheck(cudaMemcpyAsync(device.joint_transforms, _joint_transforms.data(),
                            _joint_transforms.size() * sizeof(float), cudaMemcpyHostToDevice));

  // perform LBS
  // weights: (#verts, #joints)
  device::lbs<<<(model.verts.size() - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(
      device.joint_transforms, device.verts_shaped, model.device.weights.values,
      model.device.weights.inner, model.device.weights.outer, device.verts, model.n_joints(),
      model.n_verts());

  if (enable_garment_skinning) {
    // Shape blendshapes for garment
    cudaCheck(cudaMemcpyAsync(device.cloth_verts_shaped, device.initial_cloth_verts,
                              num_cloth_verts * 3 * sizeof(float), cudaMemcpyDeviceToDevice));
    cuda_util::mmv_block<float, true>(device.cloth_blend_shapes,
                                      //model.device.blend_shapes,
                                      device.blendshape_params, device.cloth_verts_shaped,
                                      num_cloth_verts * 3, ModelConfig::n_shape_blends());

    if (enable_pose_blendshapes) {
      // Pose blendshapes.
      // Note: this is the most expensive operation.
      cuda_util::mmv_block<float, true>(
          device.cloth_blend_shapes + ModelConfig::n_shape_blends() * 3 * num_cloth_verts,
          //model.device.blend_shapes + ModelConfig::n_shape_blends() * 3 * num_cloth_verts,
          device.blendshape_params + ModelConfig::n_shape_blends(), device.cloth_verts_shaped,
          num_cloth_verts * 3, ModelConfig::n_pose_blends());
    }

    // LBS for cloth
    device::lbs<<<(num_cloth_verts - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(
        device.joint_transforms, device.cloth_verts_shaped, device.cloth_weights.values,
        device.cloth_weights.inner, device.cloth_weights.outer, device.cloth_verts,
        model.n_joints(), num_cloth_verts);
  }

  if (enable_slot_skinning) {
    // Shape blendshapes for slots
    cudaCheck(cudaMemcpyAsync(device.cloth_slots_shaped, device.initial_cloth_slots,
                              num_slots * 3 * sizeof(float), cudaMemcpyDeviceToDevice));
    cuda_util::mmv_block<float, true>(device.slot_blend_shapes, device.blendshape_params,
                                      device.cloth_slots_shaped, num_slots * 3,
                                      ModelConfig::n_shape_blends());

    if (enable_pose_blendshapes) {
      // Pose blendshapes.
      // Note: this is the most expensive operation.
      cuda_util::mmv_block<float, true>(
          device.slot_blend_shapes + ModelConfig::n_shape_blends() * 3 * num_slots,
          device.blendshape_params + ModelConfig::n_shape_blends(), device.cloth_slots_shaped,
          num_slots * 3, ModelConfig::n_pose_blends());
    }

    //printf("LBS for slots...\n");
    // LBS for garment
    device::lbs<<<(num_slots - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(
        device.joint_transforms, device.cloth_slots_shaped, device.cloth_slot_weights.values,
        device.cloth_slot_weights.inner, device.cloth_slot_weights.outer, device.cloth_slots,
        model.n_joints(), num_slots);
  }
}

// Instantiation
template class Body<model_config::SMPL>;
template class Body<model_config::SMPL_v1>;
template class Body<model_config::SMPLH>;
template class Body<model_config::SMPLX>;
template class Body<model_config::SMPLXpca>;
template class Body<model_config::SMPLX_v1>;
template class Body<model_config::SMPLXpca_v1>;

}  // namespace smplx
