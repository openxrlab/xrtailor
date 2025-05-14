#pragma once

#include <thrust/device_vector.h>

#include <xrtailor/memory/Node.cuh>
#include <xrtailor/memory/Edge.cuh>
#include <xrtailor/memory/Face.cuh>
#include <xrtailor/physics/sdf/SDFCollider.cuh>
#include <xrtailor/physics/sdf/Collider.hpp>
#include <xrtailor/runtime/rag_doll/smpl/defs.hpp>

const int NODE_POOL_SIZE = 1000000;
const int EDGE_POOL_SIZE = 1000000;
const int FACE_POOL_SIZE = 1000000;

namespace XRTailor {

class MemoryPool {
 private:
  int node_ptr_, vertex_ptr_, edge_ptr_, face_ptr_;
  thrust::device_vector<Node> node_pool_;
  thrust::device_vector<Edge> edge_pool_;
  thrust::device_vector<Face> face_pool_;

 public:
  MemoryPool();
  
  ~MemoryPool();

  Node* CreateNodes(int n_nodes);
    
  Edge* CreateEdges(int n_edges);
  
  Face* CreateFaces(int n_faces);
  
  int NodeOffset();
    
  int EdgeOffset();
  
  int FaceOffset();

  void SetupInitialPositions(const std::vector<Vector3>& positions);
  
  void SetupTmpPositions(int new_size);
  
  void SetupDeltas(int new_size);
  
  void AddEuclidean(int particle_index, int slot_index, Scalar distance);
  
  void AddGeodesic(unsigned int src_index, unsigned int tgt_index, Scalar rest_length);
  
  void AddSDFCollider(const SDFCollider& sc);
  
  void UpdateColliders(std::vector<Collider*>& colliders, Scalar dt);
  
  int NumGeodesics();
    
  void SetupObstacleMaskedIndex(const std::vector<uint>& indices);
  
  void SetupObstacleMaskedIndex(const thrust::host_vector<uint>& indices);
    
  void SetupSkinnedObstacle(int size);
  
  thrust::device_vector<Vector3> tmp_positions;

  /*
	* The Jacobi solver uses an additional buffer which
	* stores corrections for each particle, namely "deltas"
	*/
  thrust::device_vector<Vector3> deltas;
  thrust::device_vector<int> delta_counts;

  thrust::device_vector<uint> obstacle_nb_vf;        // vertex 1-ring neighbor faces
  thrust::device_vector<uint> obstacle_nb_vf_prefix;  // each VF neighbor index offset
  thrust::device_vector<uint> obstacle_nb_ef;
  thrust::device_vector<uint> obstacle_nb_ef_prefix;

  thrust::device_vector<uint> cloth_nb_vf;
  thrust::device_vector<uint> cloth_nb_vf_prefix;
  thrust::device_vector<uint> cloth_nb_ef;
  thrust::device_vector<uint> cloth_nb_ef_prefix;

  thrust::device_vector<uint> bend_indices;
  thrust::device_vector<Scalar> bend_angles;

  thrust::device_vector<uint> attach_slots;  // indices of attached vertices, |attach_slots|
  thrust::device_vector<int> attach_particle_ids;  // indices of target attached vertices, |V| * |attach_slots|
  thrust::device_vector<int> attach_slot_ids;  // indices of source slots, |V| * |attach_slots|
  thrust::device_vector<Scalar> attach_distances;  // euclidean distance between the source slot and target vertex, |V| * |attach_slots|

  thrust::device_vector<uint> geodesic_src_indices;  // index of source attachment, |V| * |attach_slots|
  thrust::device_vector<uint> geodesic_tgt_indices;  // index of target attached vertex, |V| * |attach_slots|
  thrust::device_vector<Scalar> geodesic_rest_length;  // geodesic distance between the source and target vertices

  thrust::device_vector<SkinParam> skin_params;

  thrust::device_vector<SDFCollider> sdf_colliders;

  thrust::device_vector<uint> obstacle_masked_indices;

  thrust::device_vector<Vector3> initial_positions;
};

}  // namespace XRTailor