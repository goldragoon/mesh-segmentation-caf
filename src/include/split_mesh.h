#ifndef SPLIT_MESH_H
#define SPLIT_MESH_H

#include <boost/graph/adjacency_list.hpp>

struct FaceVertexProperties {
    int id;
    double area;

    FaceVertexProperties();
};

struct FaceEdgeProperties {
    FaceEdgeProperties();
};

typedef boost::adjacency_list<boost::setS, boost::vecS, boost::undirectedS, FaceVertexProperties, FaceEdgeProperties> FaceGraph;

std::vector<double> split_mesh(Eigen::MatrixXi &F, Eigen::MatrixXi &E, Eigen::MatrixXi &IF, Eigen::MatrixXd &dblA,
                               std::vector<int> &contour_faces, FaceGraph &g,
                               std::set<boost::graph_traits<FaceGraph>::vertex_descriptor> vertex_blacklist,
                               std::set<boost::graph_traits<FaceGraph>::edge_descriptor> edge_blacklist);

void valid_cuts(Eigen::MatrixXd &V, Eigen::MatrixXd &N, Eigen::MatrixXi &F, Eigen::MatrixXd &dblA, Eigen::MatrixXi &IF,
                Eigen::MatrixXi &E,
                std::vector<std::vector<int>> &candidate_face_ids, std::vector<double> &candidate_score,
                std::vector<int> &applied_isolines);

void compute_candidate_svs_by_sdf(std::vector<double> &candidate_length,
                                  std::vector<std::vector<Eigen::MatrixXd>> &candidate_vertices,
                                  std::vector<double> &candidate_svs,
                                  std::vector<std::vector<int>> &candidate_faces,
                                  Eigen::MatrixXi &F, Eigen::MatrixXd &V);

void compute_candidate_svs_new(std::vector<double> &candidate_length,
                               std::vector<double> &candidate_svs,
                               std::vector<std::vector<int>> &candidate_faces,
                               Eigen::MatrixXi &F, Eigen::MatrixXd &V);

#endif