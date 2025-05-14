#pragma once

#include <iostream>
#include <vector>

#include <xrtailor/runtime/mesh/Mesh.hpp>
#include <xrtailor/utils/Logger.hpp>
#include <xrtailor/core/Common.cuh>

namespace XRTailor {

void GenerateBendingElements(std::shared_ptr<Mesh> mesh, const uint& index_offset,
                             std::vector<uint>& bend_indices, std::vector<Scalar>& bend_angles);

void PrepareBasicStretchData(std::shared_ptr<Mesh> mesh, std::vector<int>& ev0_indices,
                             std::vector<int>& ev1_indices, std::vector<Scalar>& e_lengths,
                             int prev_num_particles);

void BuildEFAdjacency(std::shared_ptr<Mesh> mesh, std::vector<std::set<uint>>& nb_ef,
                      int& prev_num_edges, int& prev_num_faces);

}  // namespace XRTailor