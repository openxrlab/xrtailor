#include <xrtailor/utils/Geodesic.hpp>

#include <queue>

namespace XRTailor {
void Geodesic::ComputeGeodesicDistance(const unsigned int& source_index) {
  // Check index of source
  if (source_index >= mesh_->Positions().size()) {
    throw(std::ios_base::failure(std::string("[Geodesic] Invalid source index")));
  }

  // Initialize distance field with infinite values
  distances.emplace_back(mesh_->Positions().size(), SCALAR_MAX_HOST);

  //unordered_map<unsigned int, unsigned int> previous;
  std::vector<unsigned int> previous(mesh_->Positions().size(), INT_MAX);
  previous[sources[source_index]] = sources[source_index];


  // Compute shortest path for source vertex
  // Initialize priority queue
  // Pair composed of distance, vertex
  typedef std::pair<Scalar, VertexIter> FieldPair;

  // Priority queue for using Dijkstra's algorithm efficiently
  std::priority_queue<FieldPair, std::vector<FieldPair>, std::greater<>> pq;

  // Get source to be processed
  unsigned int source_id = sources[source_index];

  auto source = mesh_->TMesh().vert.begin() + source_id;

  // Insert source to priority queue and initialize its distance to 0
  pq.push(std::make_pair(0, source));
  distances[source_index][source_id] = 0;

  // Process the priority queue
  while (!pq.empty()) {
    // Extract top element and its id
    VertexIter current = pq.top().second;
    pq.pop();

    int current_id = Mesh::GetIndex(&(*current), mesh_->TMesh());
    // Go through the neighbors of the top element
    std::vector<TailorVertex*> neighbors;
    Mesh::GetNeighbors(&(*current), neighbors);
    for (auto v_iter = neighbors.begin(); v_iter != neighbors.end(); v_iter++) {
      unsigned int vertex_id = Mesh::GetIndex((*v_iter), mesh_->TMesh());

      // Compute distance between current and its neighbor
      Scalar weight = glm::distance(mesh_->Positions()[Mesh::GetIndex(&(*current), mesh_->TMesh())],
                                    mesh_->Positions()[vertex_id]);

      //  Check if there is a shorter path to n through current
      if (distances[source_index][vertex_id] > distances[source_index][current_id] + weight) {
        // Update distance of n
        distances[source_index][vertex_id] = distances[source_index][current_id] + weight;
        //printf("insert %d into (%d, %d)\n", vertex_id, source_id, current_id);
        previous[vertex_id] = current_id;
        pq.push(std::make_pair(distances[source_index][vertex_id],
                               mesh_->TMesh().vert.begin() + vertex_id));
      }
    }
  }

  vec_previous.push_back(previous);

  MapType map;
  for (size_t i = 0; i < mesh_->Positions().size(); i++) {
    PathType path;
    unsigned int cur = previous[i];
    while (cur != source_id) {
      path.emplace_back(cur);
      cur = previous[cur];
    }
    map.emplace_back(path);
  }
  maps.push_back(map);
}
}  // namespace XRTailor