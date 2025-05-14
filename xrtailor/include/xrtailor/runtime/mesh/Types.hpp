#pragma once

#include <vcg/complex/complex.h>

namespace XRTailor {
class TailorVertex;
class TailorEdge;
class TailorFace;

struct TailorUsedTypes
    : public vcg::UsedTypes<vcg::Use<TailorVertex>::AsVertexType, vcg::Use<TailorEdge>::AsEdgeType,
                            vcg::Use<TailorFace>::AsFaceType> {};

class TailorVertex
    : public vcg::Vertex<TailorUsedTypes, vcg::vertex::VFAdj, vcg::vertex::VEAdj,
                         vcg::vertex::Coord3f, vcg::vertex::Normal3f, vcg::vertex::Color4b,
                         vcg::vertex::Mark, vcg::vertex::BitFlags> {};

class TailorEdge
    : public vcg::Edge<TailorUsedTypes, vcg::edge::VertexRef, vcg::edge::VEAdj, vcg::edge::EEAdj,
                       vcg::edge::EFAdj, vcg::edge::Mark, vcg::edge::BitFlags> {};

class TailorFace
    : public vcg::Face<TailorUsedTypes, vcg::face::VertexRef, vcg::face::WedgeRealNormal3f,
                       vcg::face::WedgeTexCoord2f, vcg::face::FFAdj, vcg::face::VFAdj,
                       vcg::face::EFAdj, vcg::face::FEAdj, vcg::face::Color4b, vcg::face::Mark,
                       vcg::face::BitFlags> {};

class TailorMesh : public vcg::tri::TriMesh<std::vector<TailorVertex>, std::vector<TailorEdge>,
                                            std::vector<TailorFace>> {};

typedef std::vector<TailorVertex>::const_iterator CVertexIter;
typedef std::vector<TailorVertex>::iterator VertexIter;
typedef std::vector<TailorFace>::const_iterator CFaceIter;
typedef std::vector<TailorFace>::iterator FaceIter;
typedef std::vector<TailorEdge>::const_iterator CEdgeIter;
typedef std::vector<TailorEdge>::iterator EdgeIter;
}  // namespace XRTailor