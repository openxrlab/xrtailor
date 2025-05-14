#include <xrtailor/runtime/rag_doll/smpl/smplx.hpp>

#include <algorithm>
#include <cstring>
#include <fstream>
#include "cnpy.h"


#include <xrtailor/runtime/rag_doll/smpl/util.hpp>
#include <xrtailor/runtime/rag_doll/smpl/util_cnpy.hpp>

#include <xrtailor/runtime/rag_doll/smpl/version.hpp>

#include <xrtailor/utils/FileSystemUtils.hpp>
#include <xrtailor/core/ErrorDefs.hpp>

namespace smplx {
namespace {
using util::AssertShape;
}  // namespace

template <class ModelConfig>
Model<ModelConfig>::Model(Gender gender) {
  Load(gender);
}

template <class ModelConfig>
Model<ModelConfig>::Model(const std::string& path, const std::string& uv_path, Gender gender) {
  Load(path, uv_path, gender);
}

template <class ModelConfig>
Model<ModelConfig>::~Model() {
#ifdef SMPLX_CUDA_ENABLED
  _cuda_free();
#endif
}

template <class ModelConfig>
void Model<ModelConfig>::Load(Gender gender) {
  filesystem::path model_base_path = XRTailor::GetBodyModelDirectory();
  filesystem::path full_model_path = model_base_path;
  full_model_path.append(std::string(ModelConfig::default_path_prefix) + util::GenderToStr(gender) +
                         ".npz");
  filesystem::path full_uv_path = model_base_path;
  full_uv_path.append(ModelConfig::default_uv_path);
  Load(full_model_path.string(), full_uv_path.string(), gender);
}

template <class ModelConfig>
void Model<ModelConfig>::Load(const std::string& path, const std::string& uv_path,
                              Gender new_gender) {
  if (!std::ifstream(path)) {
    std::cerr << "ERROR: Model '" << path
              << "' does not exist, "
                 "did you download the model following instructions in "
                 "https://github.com/sxyu/smplxpp/tree/master/data/models/"
                 "README.md?\n";
    return;
  }
  gender = new_gender;
  cnpy::npz_t npz = cnpy::npz_load(path);

  // Load kintree
  children.resize(n_joints());
  for (size_t i = 1; i < n_joints(); ++i) {
    children[ModelConfig::parent[i]].push_back(i);
  }

  // Load base template
  const auto& verts_raw = npz.at("v_template");
  AssertShape(verts_raw, {n_verts(), 3});
  verts.noalias() = util::LoadFloatMatrix(verts_raw, n_verts(), 3);
  verts_load.noalias() = verts;

  // Load triangle mesh
  const auto& faces_raw = npz.at("f");
  AssertShape(faces_raw, {n_faces(), 3});
  faces = util::LoadUintMatrix(faces_raw, n_faces(), 3);

  // Load joint regressor
  const auto& jreg_raw = npz.at("J_regressor");
  AssertShape(jreg_raw, {n_joints(), n_verts()});
  joint_reg.resize(n_joints(), n_verts());
  joint_reg = util::LoadFloatMatrix(jreg_raw, n_joints(), n_verts()).sparseView();
  joints = joint_reg * verts;
  joint_reg.makeCompressed();

  // Load LBS weights
  const auto& wt_raw = npz.at("weights");
  AssertShape(wt_raw, {n_verts(), n_joints()});
  weights.resize(n_verts(), n_joints());
  weights = util::LoadFloatMatrix(wt_raw, n_verts(), n_joints()).sparseView();
  weights.makeCompressed();

  blend_shapes.resize(3 * n_verts(), n_blend_shapes());
  // Load shape-dep blend shapes
  const auto& sb_raw = npz.at("shapedirs");
  AssertShape(sb_raw, {n_verts(), 3, n_shape_blends()});
  blend_shapes.template leftCols<n_shape_blends()>().noalias() =
      util::LoadFloatMatrix(sb_raw, 3 * n_verts(), n_shape_blends());

  // Load pose-dep blend shapes
  const auto& pb_raw = npz.at("posedirs");
  AssertShape(pb_raw, {n_verts(), 3, n_pose_blends()});
  blend_shapes.template rightCols<n_pose_blends()>().noalias() =
      util::LoadFloatMatrix(pb_raw, 3 * n_verts(), n_pose_blends());

  if ((n_hand_pca() != 0u) && (npz.count("hands_meanl") != 0u) &&
      (npz.count("hands_meanr") != 0u)) {
    // Model has hand PCA (e.g. SMPLXpca), load hand PCA
    const auto& hml_raw = npz.at("hands_meanl");
    const auto& hmr_raw = npz.at("hands_meanr");
    const auto& hcl_raw = npz.at("hands_componentsl");
    const auto& hcr_raw = npz.at("hands_componentsr");

    AssertShape(hml_raw, {util::ANY_SHAPE});
    AssertShape(hmr_raw, {hml_raw.shape[0]});

    size_t n_hand_params = hml_raw.shape[0];
    _SMPLX_ASSERT_EQ(n_hand_params, n_hand_pca_joints() * 3);

    AssertShape(hcl_raw, {n_hand_params, n_hand_params});
    AssertShape(hcr_raw, {n_hand_params, n_hand_params});

    hand_mean_l = util::LoadFloatMatrix(hml_raw, n_hand_params, 1);
    hand_mean_r = util::LoadFloatMatrix(hmr_raw, n_hand_params, 1);

    hand_comps_l = util::LoadFloatMatrix(hcl_raw, n_hand_params, n_hand_params)
                       .topRows(n_hand_pca())
                       .transpose();
    hand_comps_r = util::LoadFloatMatrix(hcr_raw, n_hand_params, n_hand_params)
                       .topRows(n_hand_pca())
                       .transpose();
  }

  // Maybe load UV (UV mapping WIP)
  if (static_cast<unsigned int>(!uv_path.empty()) != 0u) {
    std::ifstream ifs(uv_path);
    ifs >> _n_uv_verts;
    if (_n_uv_verts) {
      if (ifs) {
        // Load the uv data
        uv.resize(_n_uv_verts, 2);
        for (size_t i = 0; i < _n_uv_verts; ++i) {
          ifs >> uv(i, 0) >> uv(i, 1);
        }
        _SMPLX_ASSERT(ifs);
        uv_faces.resize(n_faces(), 3);
        for (size_t i = 0; i < n_faces(); ++i) {
          _SMPLX_ASSERT(ifs);
          for (size_t j = 0; j < 3; ++j) {
            ifs >> uv_faces(i, j);
            // Make indices 0-based
            --uv_faces(i, j);
            _SMPLX_ASSERT_LT(uv_faces(i, j), _n_uv_verts);
          }
        }
      }
    }
  }
#ifdef SMPLX_CUDA_ENABLED
  _cuda_load();
#endif
}

template <class ModelConfig>
void Model<ModelConfig>::SetDeformations(const Eigen::Ref<const Points>& d) {
  verts.noalias() = verts_load + d;
#ifdef SMPLX_CUDA_ENABLED
  _cuda_copy_template();
#endif
}

template <class ModelConfig>
void Model<ModelConfig>::SetTemplate(const Eigen::Ref<const Points>& t) {
  verts.noalias() = t;
#ifdef SMPLX_CUDA_ENABLED
  _cuda_copy_template();
#endif
}

// Instantiations
template class Model<model_config::SMPL>;
template class Body<model_config::SMPL_v1>;
template class Model<model_config::SMPLH>;
template class Model<model_config::SMPLX>;
template class Model<model_config::SMPLXpca>;
template class Model<model_config::SMPLX_v1>;
template class Model<model_config::SMPLXpca_v1>;

}  // namespace smplx
