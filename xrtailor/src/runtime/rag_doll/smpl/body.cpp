#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>  // <-- Added

#include <xrtailor/runtime/rag_doll/smpl/smplx.hpp>
#include <xrtailor/runtime/rag_doll/smpl/util.hpp>
#include <xrtailor/runtime/rag_doll/smpl/internal/cuda_util.hpp>

namespace smplx {

template <class ModelConfig>
Body<ModelConfig>::Body(const Model<ModelConfig>& model, bool set_zero)
    : model(model), params(model.n_params()) {
  if (set_zero) {
    this->set_zero();
  }
  // Point cloud after applying shape keys but before lbs (num points, 3)
  _verts_shaped.resize(model.n_verts(), 3);

  // Joints after applying shape keys but before lbs (num joints, 3)
  _joints_shaped.resize(model.n_joints(), 3);

  // Final deformed point cloud
  _verts.resize(model.n_verts(), 3);

  // Affine joint transformation, as 3x4 matrices stacked horizontally (bottom
  // row omitted) NOTE: col major
  _joint_transforms.resize(model.n_joints(), 12);
#ifdef SMPLX_CUDA_ENABLED
  _cuda_load();
#endif
}

template <class ModelConfig>
Body<ModelConfig>::~Body() {
#ifdef SMPLX_CUDA_ENABLED
  _cuda_free();
#endif
}

template <class ModelConfig>
const Points& Body<ModelConfig>::verts() const {
#ifdef SMPLX_CUDA_ENABLED
  if (_last_update_used_gpu) {
    _cuda_maybe_retrieve_verts();
  }
#endif
  return _verts;
}

template <class ModelConfig>
const Points& Body<ModelConfig>::host_cloth_verts() const {
#ifdef SMPLX_CUDA_ENABLED
  if (_last_update_used_gpu) {
    _cuda_maybe_retrieve_cloth_verts();
  }
#endif
  return _garment_verts;
}

template <class ModelConfig>
const float* Body<ModelConfig>::device_cloth_verts() const {
  return device.cloth_verts;
}

template <class ModelConfig>
const Points& Body<ModelConfig>::host_cloth_slots() const {
#ifdef SMPLX_CUDA_ENABLED
  if (_last_update_used_gpu) {
    _cuda_maybe_retrieve_cloth_slots();
  }
#endif
  return _garment_slots;
}

template <class ModelConfig>
const float* Body<ModelConfig>::device_cloth_slots() const {
  return device.cloth_slots;
}

template <class ModelConfig>
const Points& Body<ModelConfig>::verts_shaped() const {
#ifdef SMPLX_CUDA_ENABLED
  if (_last_update_used_gpu) {
    _cuda_maybe_retrieve_verts_shaped();
  }
#endif
  return _verts_shaped;
}

template <class ModelConfig>
const Points& Body<ModelConfig>::cloth_verts_shaped() const {
#ifdef SMPLX_CUDA_ENABLED
  if (_last_update_used_gpu) {
    _cuda_maybe_retrieve_cloth_verts_shaped();
  }
#endif
  return _garment_verts_shaped;
}

template <class ModelConfig>
const Points& Body<ModelConfig>::joints() const {
  return _joints;
}

template <class ModelConfig>
const Eigen::Matrix<Scalar, Eigen::Dynamic, 12, Eigen::RowMajor>&
Body<ModelConfig>::joint_transforms() const {
  return _joint_transforms;
}

template <class ModelConfig>
const Eigen::Matrix<Scalar, Eigen::Dynamic, 12, Eigen::RowMajor>&
Body<ModelConfig>::vert_transforms() const {
  if (_vert_transforms.rows() == 0) {
    _vert_transforms.noalias() = model.weights * _joint_transforms;
  }
  return _vert_transforms;
}

using Eigen::EigenBase;
using std::ostringstream;

template <typename Derived>
std::string GetShape(const EigenBase<Derived>& x) {
  std::ostringstream oss;
  oss << " size: " << x.size() << ", (" << x.rows() << ", " << x.cols() << ")";
  return oss.str();
}

template <class ModelConfig>
void Body<ModelConfig>::EmbedBlendShapesIntoCloth(unsigned int num_garment_vertices,
                                                    const unsigned int* nearest_body_indices) {
  Eigen::MatrixXf body_blend_shapes = model.blend_shapes;
  Eigen::MatrixXf cloth_blend_shapes(num_garment_vertices * 3, body_blend_shapes.cols());

  for (unsigned int i = 0; i < num_garment_vertices; i++) {
    cloth_blend_shapes.row(i * 3) = body_blend_shapes.row(nearest_body_indices[i] * 3);
    cloth_blend_shapes.row(i * 3 + 1) = body_blend_shapes.row(nearest_body_indices[i] * 3 + 1);
    cloth_blend_shapes.row(i * 3 + 2) = body_blend_shapes.row(nearest_body_indices[i] * 3 + 2);
  }
  cuda_util::from_host_eigen_matrix(device.cloth_blend_shapes, cloth_blend_shapes);
}

template <class ModelConfig>
void Body<ModelConfig>::EmbedBarycentricBlendShapesIntoCloth(unsigned int num_garment_vertices,
                                                             XRTailor::SkinParam* skin_params) {
  Eigen::MatrixXf body_blend_shapes = model.blend_shapes;
  Eigen::MatrixXf cloth_blend_shapes(num_garment_vertices * 3, body_blend_shapes.cols());
  for (unsigned int i = 0; i < num_garment_vertices; i++) {
    cloth_blend_shapes.row(i * 3) =
        skin_params[i].u * body_blend_shapes.row(skin_params[i].idx0 * 3) +
        skin_params[i].v * body_blend_shapes.row(skin_params[i].idx1 * 3) +
        skin_params[i].w * body_blend_shapes.row(skin_params[i].idx2 * 3);

    cloth_blend_shapes.row(i * 3 + 1) =
        skin_params[i].u * body_blend_shapes.row(skin_params[i].idx0 * 3 + 1) +
        skin_params[i].v * body_blend_shapes.row(skin_params[i].idx1 * 3 + 1) +
        skin_params[i].w * body_blend_shapes.row(skin_params[i].idx2 * 3 + 1);

    cloth_blend_shapes.row(i * 3 + 2) =
        skin_params[i].u * body_blend_shapes.row(skin_params[i].idx0 * 3 + 2) +
        skin_params[i].v * body_blend_shapes.row(skin_params[i].idx1 * 3 + 2) +
        skin_params[i].w * body_blend_shapes.row(skin_params[i].idx2 * 3 + 2);
  }

  cuda_util::from_host_eigen_matrix(device.cloth_blend_shapes, cloth_blend_shapes);
}

template <class ModelConfig>
void Body<ModelConfig>::EmbedBlendShapesIntoSlots(unsigned int num_slots,
                                                  const unsigned int* slot_indices,
                                                  const unsigned int* nearest_body_indices) {
  Eigen::MatrixXf body_blend_shapes = model.blend_shapes;
  Eigen::MatrixXf slot_blend_shapes(num_slots * 3, body_blend_shapes.cols());

  for (unsigned int i = 0; i < num_slots; i++) {
    unsigned int slot_idx = slot_indices[i];
    unsigned int nearest_body_idx = nearest_body_indices[slot_idx];
    slot_blend_shapes.row(i * 3) = body_blend_shapes.row(nearest_body_idx * 3);
    slot_blend_shapes.row(i * 3 + 1) = body_blend_shapes.row(nearest_body_idx * 3 + 1);
    slot_blend_shapes.row(i * 3 + 2) = body_blend_shapes.row(nearest_body_idx * 3 + 2);
  }
  cuda_util::from_host_eigen_matrix(device.slot_blend_shapes, slot_blend_shapes);
}


template <class ModelConfig>
void Body<ModelConfig>::EmbedSkinningWeightsIntoSlots(const SparseMatrixColMajor& body_weights,
                                                      unsigned int num_slots,
                                                      const unsigned int* slot_indices,
                                                      unsigned int* nearest_body_indices) {
  //TODO: embed weights more efficiently
  Eigen::MatrixXf body_weights_dense = body_weights.toDense();
  Eigen::MatrixXf slot_weights_dense(num_slots, body_weights.cols());
  for (unsigned int i = 0; i < num_slots; i++) {
    int slot_idx = slot_indices[i];
    slot_weights_dense.row(i) = body_weights_dense.row(nearest_body_indices[slot_idx]);
  }

  {
    SparseMatrix tmp_weights = slot_weights_dense.sparseView();  // Change to CSR
    cuda_util::from_host_eigen_sparse_matrix(device.cloth_slot_weights, tmp_weights);
    cudaDeviceSynchronize();
  }
}

template <class ModelConfig>
void Body<ModelConfig>::EmbedBarycentricSkinningWeightsIntoSlots(
    const SparseMatrixColMajor& body_weights, unsigned int num_slots,
    const unsigned int* slot_indices, XRTailor::SkinParam* skin_params) {
  //TODO: embed weights more efficiently
  Eigen::MatrixXf body_weights_dense = body_weights.toDense();
  Eigen::MatrixXf slot_weights_dense(num_slots, body_weights.cols());
  for (unsigned int i = 0; i < num_slots; i++) {
    int slot_idx = slot_indices[i];

    slot_weights_dense.row(i) =
        skin_params[slot_idx].u * body_weights_dense.row(skin_params[slot_idx].idx0) +
        skin_params[slot_idx].v * body_weights_dense.row(skin_params[slot_idx].idx1) +
        skin_params[slot_idx].w * body_weights_dense.row(skin_params[slot_idx].idx2);
  }

  {
    SparseMatrix tmp_weights = slot_weights_dense.sparseView();  // Change to CSR
    cuda_util::from_host_eigen_sparse_matrix(device.cloth_slot_weights, tmp_weights);
    cudaDeviceSynchronize();
  }
}

template <class ModelConfig>
void Body<ModelConfig>::EmbedSkinningWeightsIntoCloth(const SparseMatrixColMajor& body_weights,
                                                        SparseMatrixColMajor& cloth_weights,
                                                        unsigned int num_garment_vertices,
                                                        unsigned int* nearest_body_indices) {
  //TODO: embed weights more efficiently
  Eigen::MatrixXf body_weights_dense = body_weights.toDense();
  Eigen::MatrixXf cloth_weights_dense(num_garment_vertices, body_weights.cols());
  for (unsigned int i = 0; i < num_garment_vertices; i++) {
    cloth_weights_dense.row(i) = body_weights_dense.row(nearest_body_indices[i]);

  }
  {
    SparseMatrix tmp_weights = cloth_weights_dense.sparseView();  // Change to CSR
    cuda_util::from_host_eigen_sparse_matrix(device.cloth_weights, tmp_weights);
    cudaDeviceSynchronize();
  }
}

template <class ModelConfig>
void Body<ModelConfig>::EmbedBarycentricSkinningWeightsIntoCloth(
    const SparseMatrixColMajor& body_weights, SparseMatrixColMajor& cloth_weights,
    unsigned int num_cloth_vertices, XRTailor::SkinParam* skin_params) {
  //TODO: embed weights more efficiently
  Eigen::MatrixXf body_weights_dense = body_weights.toDense();
  Eigen::MatrixXf cloth_weights_dense(num_cloth_vertices, body_weights.cols());

  for (unsigned int i = 0; i < num_cloth_vertices; i++) {
    cloth_weights_dense.row(i) = skin_params[i].u * body_weights_dense.row(skin_params[i].idx0) +
                                   skin_params[i].v * body_weights_dense.row(skin_params[i].idx1) +
                                   skin_params[i].w * body_weights_dense.row(skin_params[i].idx2);
  }
  {
    SparseMatrix tmp_weights = cloth_weights_dense.sparseView();  // Change to CSR
    cuda_util::from_host_eigen_sparse_matrix(device.cloth_weights, tmp_weights);
  }
}

// Main LBS routine
template <class ModelConfig>
void Body<ModelConfig>::Update(bool force_cpu, bool enable_pose_blendshapes,
                               bool enable_garment_skinning, bool enable_slot_skinning) {
  // _SMPLX_BEGIN_PROFILE;
  // Will store full pose params (angle-axis), including hand
  Vector full_pose(3 * model.n_joints());

  // Shape params +/ linear joint transformations as flattened 3x3 rotation
  // matrices rowmajor, only for blend shapes
  // SMPLH: 475 = 16 shape blends + (52 - 1)*9 pose blends
  Vector blendshape_params(model.n_blend_shapes());

  using TransformMap = Eigen::Map<Eigen::Matrix<Scalar, 3, 4, Eigen::RowMajor>>;
  using TransformTransposedMap = Eigen::Map<Eigen::Matrix<Scalar, 4, 3>>;
  using RotationMap = Eigen::Map<Eigen::Matrix<Scalar, 3, 3, Eigen::RowMajor>>;

  // Copy body pose onto full pose
  full_pose.head(3 * model.n_explicit_joints()).noalias() = pose();

  // if the hand do not use PCA weights, skip
  // uses PCA weights: SMPLXpca_v1.0, SMPLXpca_v1.1
  // do not use PCA weights: SMPLX_v1.0, SMPLX_v1.1, SMPLH
  if (model.n_hand_pca_joints() > 0) {
    // Use hand PCA weights to fill in hand pose within full pose
    full_pose.segment(3 * model.n_explicit_joints(), 3 * model.n_hand_pca_joints()).noalias() =
        model.hand_mean_l + model.hand_comps_l * hand_pca_l();

    full_pose.tail(3 * model.n_hand_pca_joints()).noalias() =
        model.hand_mean_r + model.hand_comps_r * hand_pca_r();
  }

  // Copy shape params to blendshape params
  // n_shape_blends:
  //  - SMPLH: 16
  //  - SMPLX_v1: 20
  //  - SMPLX_v1.1: 300
  blendshape_params.head<ModelConfig::n_shape_blends()>() = shape();

  // Convert angle-axis to rotation matrix using Rodrigues
  // root orient
  TransformMap(_joint_transforms.topRows<1>().data())
      .template leftCols<
          3>()
      .noalias() = util::Rodrigues<float>(full_pose.head<3>());
  // explicit joints + hand joints(if available)
  for (size_t i = 1; i < model.n_joints(); ++i) {
    // notice that in Eigen, a map refers to memory address of an array/vector,
    // therefore, any modifications to the input array/vector will make effects.
    TransformMap joint_trans(_joint_transforms.row(i).data());
    joint_trans.template leftCols<3>().noalias() =
        util::Rodrigues<float>(full_pose.segment<3>(3 * i));

    RotationMap mp(blendshape_params.data() + 9 * i + (model.n_shape_blends() - 9));
    mp.noalias() = joint_trans.template leftCols<3>();
    mp.diagonal().array() -= 1.f;
  }

#ifdef SMPLX_CUDA_ENABLED
  _last_update_used_gpu = !force_cpu;
  if (!force_cpu) {
    _cuda_update(blendshape_params.data(), _joint_transforms.data(), enable_pose_blendshapes,
                 enable_garment_skinning, enable_slot_skinning);
    _vert_transforms.resize(0, 12);
    return;
  }
#endif
  // _SMPLX_PROFILE(preproc);
  // Apply blend shapes
  {
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> verts_shaped_flat(_verts_shaped.data(),
                                                                           model.n_verts() * 3);
    Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> verts_init_flat(model.verts.data(),
                                                                               model.n_verts() * 3);
    // Add shape blend shapes
    verts_shaped_flat.noalias() =
        verts_init_flat + model.blend_shapes.template leftCols<ModelConfig::n_shape_blends()>() *
                              blendshape_params.head<ModelConfig::n_shape_blends()>();
  }
  // _SMPLX_PROFILE(blendshape);

  // Apply joint regressor
  _joints_shaped = model.joint_reg * _verts_shaped;

  if (enable_pose_blendshapes) {
    // HORRIBLY SLOW, like 95% of the time is spent here yikes
    // Add pose blend shapes
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> verts_shaped_flat(_verts_shaped.data(),
                                                                           model.n_verts() * 3);
    verts_shaped_flat += model.blend_shapes.template rightCols<ModelConfig::n_pose_blends()>() *
                         blendshape_params.tail<ModelConfig::n_pose_blends()>();
  }

  // Inputs: trans(), _joints_shaped
  // Outputs: _joints
  // Input/output: _joint_transforms
  //   (input: left 3x3 should be local rotation mat for joint
  //    output: completed joint local space transform rel global)
  _local_to_global();
  // _SMPLX_PROFILE(localglobal);

  // * LBS *
  // Construct a transform for each vertex
  _vert_transforms.noalias() = model.weights * _joint_transforms;
  // _SMPLX_PROFILE(lbs weight computation);

  // Apply affine transform to each vertex and store to output
  // #pragma omp parallel for // Seems to only make it slower??
  for (size_t i = 0; i < model.n_verts(); ++i) {
    TransformTransposedMap transform_tr(_vert_transforms.row(i).data());
    _verts.row(i).noalias() = _verts_shaped.row(i).homogeneous() * transform_tr;
  }
  // _SMPLX_PROFILE(lbs point transform);
}

template <class ModelConfig>
void Body<ModelConfig>::_local_to_global() {
  _joints.resize(ModelConfig::n_joints(), 3);
  using TransformMap = Eigen::Map<Eigen::Matrix<Scalar, 3, 4, Eigen::RowMajor>>;
  using TransformTransposedMap = Eigen::Map<Eigen::Matrix<Scalar, 4, 3>>;
  // Handle root joint transforms
  TransformTransposedMap root_transform_tr(_joint_transforms.topRows<1>().data());
  root_transform_tr.bottomRows<1>().noalias() = _joints_shaped.topRows<1>() + trans().transpose();
  _joints.topRows<1>().noalias() = root_transform_tr.bottomRows<1>();

  // Complete the affine transforms for all other joint by adding translation
  // components and composing with parent
  for (int i = 1; i < ModelConfig::n_joints(); ++i) {
    TransformMap transform(_joint_transforms.row(i).data());
    const auto p = ModelConfig::parent[i];
    // Set relative translation
    transform.rightCols<1>().noalias() =
        (_joints_shaped.row(i) - _joints_shaped.row(p)).transpose();
    // Compose rotation with parent
    util::MulAffine<float, Eigen::RowMajor>(TransformMap(_joint_transforms.row(p).data()),
                                            transform);
    // Grab the joint position in case the user wants it
    _joints.row(i).noalias() = transform.rightCols<1>().transpose();
  }

  for (int i = 0; i < ModelConfig::n_joints(); ++i) {
    TransformTransposedMap transform_tr(_joint_transforms.row(i).data());
    // Translate to center at global origin
    transform_tr.bottomRows<1>().noalias() -= _joints_shaped.row(i) * transform_tr.topRows<3>();
  }
}

template <class ModelConfig>
void Body<ModelConfig>::SaveObj(const std::string& path) const {
  const auto& cur_verts = verts();
  if (cur_verts.rows() == 0) {
    return;
  }
  std::ofstream ofs(path);
  ofs << "# Generated by SMPL-X"
      << "\n";
  ofs << std::fixed << std::setprecision(6) << "o smplx\n";
  for (int i = 0; i < model.n_verts(); ++i) {
    ofs << "v " << cur_verts(i, 0) << " " << cur_verts(i, 1) << " " << cur_verts(i, 2) << "\n";
  }
  ofs << "s 1\n";
  for (int i = 0; i < model.n_faces(); ++i) {
    ofs << "f " << model.faces(i, 0) + 1 << " " << model.faces(i, 1) + 1 << " "
        << model.faces(i, 2) + 1 << "\n";
  }
  ofs.close();
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
