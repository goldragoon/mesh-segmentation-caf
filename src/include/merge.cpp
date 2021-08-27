#include "merge.hpp"
#include "emd.h"
#include "emd.c"
#include "sdf.hpp"
#include <igl/histc.h>

struct MyVertexProperties {
    int id;
    int region;
    pcl::PointNormal point;

    MyVertexProperties() : id(0), point(pcl::PointNormal()), region(-1) {}
};

struct MyEdgeProperties {
    int id;
    float normal_difference;
    float weight;
    bool is_internal;
    bool is_concave;
    bool cut;
    bool valid;
    int region;

    MyEdgeProperties() : id(0), is_internal(false), is_concave(false), normal_difference(0), weight(0.0), region(-1),
                         valid(true), cut(false) {}
};

typedef boost::adjacency_list<boost::setS, boost::vecS, boost::undirectedS, MyVertexProperties, MyEdgeProperties> Graph;
typedef Graph::edge_descriptor edge_id_t;
typedef Graph::vertex_descriptor vertex_id_t;

class edge_predicate_border {
public:
    edge_predicate_border() : graph_m(0) {}

    edge_predicate_border(Graph &graph) : graph_m(&graph) {}

    bool operator()(const edge_id_t &edge_id) const {
        //return !((*graph_m)[edge_id].cut);
        return !((*graph_m)[boost::source(edge_id, *graph_m)].region !=
                 (*graph_m)[boost::target(edge_id, *graph_m)].region);
    }

private:
    Graph *graph_m;
};

float dist(float *F1, float *F2) {
    return abs(std::max(*F1, *F2) - std::min(*F1, *F2));
}

/**
 *
 * @param V
 * @param F
 * @param label
 * @param min_part
 */
void mergeSegments(Eigen::MatrixXd &V, Eigen::MatrixXi &F, Eigen::MatrixXd &label, double min_part) {
    Graph g(V.rows());
    /// Add edges
    Eigen::MatrixXi E;
    igl::edges(F, E);
    for (int i = 0; i < E.rows(); i++) {
        boost::add_edge(E(i, 0), E(i, 1), g);
    }
    for (int i = 0; i < label.rows(); i++) {
        g[i].region = label(i, 0);
    }

    typedef boost::graph_traits<Graph>::edge_iterator edge_iter;
    typedef boost::graph_traits<Graph>::vertex_iterator vertex_iterator;
    boost::graph_traits<Graph>::vertex_descriptor v1, v2;
    boost::filtered_graph<Graph, edge_predicate_border> lg(g, edge_predicate_border(g));

    /// Obtain SDF Values for all vertices
    Eigen::MatrixXd face_sdf;
    Eigen::MatrixXd vertex_sdf;
    Eigen::MatrixXd skeleton_vertices;
    sdf(V, F, face_sdf, vertex_sdf, skeleton_vertices);

    /// While something is still done in last iteration: find smallest region and merge it with closest neighbor
    bool merged_region = true;
    while (merged_region) {
        merged_region = false;

        /// Compute connected components
        std::vector<int> componentCuts(boost::num_vertices(lg));
        size_t regions = boost::connected_components(lg, &componentCuts[0]);
        for (int j = 0; j != componentCuts.size(); ++j) {
            g[j].region = componentCuts[j];
        }

        /// Find smallest component
        /// Iterate over all clusters:
        int smallest_region_index = 0;
        int smallest_region_size = num_vertices(g);
        std::vector<int> region_size(regions);
        for (int i = 0; i < regions; i++) {
            region_size[i] = 0;
        }

        for (int i = 0; i < componentCuts.size(); i++)
            region_size[componentCuts[i]]++;

        smallest_region_size = *min_element(region_size.begin(), region_size.end());
        smallest_region_index = distance(region_size.begin(), min_element(region_size.begin(), region_size.end()));

        /// check if it is smaller than the smallest one to merge
        if (smallest_region_size < (componentCuts.size() * min_part)) {

            int ind = smallest_region_index;

            std::vector<std::vector<float>> region_vertex_sdf_values(regions);

            for (int i = 0; i < componentCuts.size(); i++) {
                region_vertex_sdf_values[componentCuts[i]].push_back((float) vertex_sdf(i, 0));
            }

            std::vector<Eigen::MatrixXf> region_sdf_hist(regions);
            for (int i = 0; i < regions; i++) {
                region_sdf_hist[i] = Eigen::MatrixXf::Zero(region_vertex_sdf_values[i].size(), 1);
                for (int j = 0; j < region_vertex_sdf_values[i].size(); j++) {
                    region_sdf_hist[i](j, 0) = region_vertex_sdf_values[i][j];
                }
            }

            /// Find all neighbors
            std::vector<int> connections(regions);
            for (int i = 0; i < regions; i++) {
                connections.at(i) = 0;
            }

            auto es = boost::edges(g);
            for (auto eit = es.first; eit != es.second; ++eit) {
                if (g[boost::source(*eit, g)].region != g[boost::target(*eit, g)].region) {
                    auto v1 = boost::source(*eit, g);
                    auto v2 = boost::target(*eit, g);
                    if ((g[v1].region == ind || g[v2].region == ind) && g[v1].region != g[v2].region) {
                        if (g[v1].region == ind) {
                            connections.at(g[v2].region) = connections.at(g[v2].region) + 1;
                        }
                        if (g[v2].region == ind) {
                            connections.at(g[v1].region) = connections.at(g[v1].region) + 1;
                        }
                    }
                }
            }

            /// From set of neighboring clusters, find the closest one
            /// Create first histogram
            Eigen::VectorXf N, B;
            Eigen::VectorXf E = Eigen::VectorXf::LinSpaced(20, 0.0, 1.0);
            Eigen::VectorXf feature_vec = region_sdf_hist[ind];
            feature_vec.resize(feature_vec.cols() * feature_vec.rows(), 1);
            igl::histc(feature_vec, E, N, B);
            N = N.array() / feature_vec.rows();
            
            feature_t *f1 = new feature_t[N.rows()];
            feature_t *f2 = new feature_t[N.rows()];
            for (int i = 0; i < E.rows(); i++) {
                f1[i] = E(i);
                f2[i] = E(i);
            }

            feature_t *w1 = new feature_t[N.rows()];
            for (int i = 0; i < N.rows(); i++)
                w1[i] = N(i);

            float min_dist = 100.0;
            int closest_neighbor = ind;

            /// Get distances to all neighbors
            for (int i = 0; i < regions; i++) {
                if (connections.at(i) > 0) {
                    Eigen::VectorXf feature_vec = region_sdf_hist[i];
                    feature_vec.resize(feature_vec.cols() * feature_vec.rows(), 1);
                    igl::histc(feature_vec, E, N, B);
                    N = N.array() / feature_vec.rows();

                    feature_t *w2 = new feature_t[N.rows()];
                    for (int j = 0; j < N.rows(); j++)
                        w2[j] = N(j);

                    signature_t s1 = {(int) N.rows(), f1, w1},
                            s2 = {(int) N.rows(), f2, w2};
                    float e;

                    e = emd(&s1, &s2, dist, 0, 0);
                    if (e < min_dist) {
                        min_dist = e;
                        closest_neighbor = i;
                    }
                }
            }

            std::cout << "Merging Cluster " << ind << " with " << closest_neighbor << ", distance = " << min_dist
                      << std::endl;

            /// Now merge!
            if (closest_neighbor != ind) {
                std::pair<vertex_iterator, vertex_iterator> vp;
                for (vp = boost::vertices(g); vp.first != vp.second; ++vp.first) {
                    if (g[*vp.first].region == ind) {
                        g[*vp.first].region = closest_neighbor;
                        //std::cout << "Changed vertex\n";
                    }

                }

                merged_region = true;
            }
        }
    }

    for (int i = 0; i < V.rows(); i++) {
        label(i, 0) = g[i].region;
    }
}

/**
 *
 * @param V
 * @param F
 * @param label
 * @param min_part
 */
void mergeSegments_old(Eigen::MatrixXd &V, Eigen::MatrixXi &F, Eigen::MatrixXd &label, double min_part) {
    Graph g(V.rows());
    /// Add edges
    Eigen::MatrixXi E;
    igl::edges(F, E);
    for (int i = 0; i < E.rows(); i++) {
        boost::add_edge(E(i, 0), E(i, 1), g);
    }
    for (int i = 0; i < label.rows(); i++) {
        g[i].region = label(i, 0);
    }

    typedef boost::graph_traits<Graph>::edge_iterator edge_iter;
    typedef boost::graph_traits<Graph>::vertex_iterator vertex_iterator;
    boost::graph_traits<Graph>::vertex_descriptor v1, v2;
    boost::filtered_graph<Graph, edge_predicate_border> lg(g, edge_predicate_border(g));

    std::vector<int> componentCuts(boost::num_vertices(lg));
    size_t regions = boost::connected_components(lg, &componentCuts[0]);

    for (int j = 0; j != componentCuts.size(); ++j) {
        g[j].region = componentCuts[j];
    }

    std::vector<int> connections(regions);
    for (int i = 0; i < regions; i++) {
        connections.at(i) = 0;
    }

    ///Merge small parts with closest part
    std::cout << "Merging Regions...\n";
    double min_dist = std::numeric_limits<double>::max();
    int min_dist_seg = 0;
    int vert_count = 0;

    for (int ind = 0; ind < regions; ind++) {

        /// Count number of vertices of this segment
        int vert_count = 0;
        for (int i = 0; i < componentCuts.size(); i++) {
            if (componentCuts[i] == ind) {
                vert_count++;
            }
        }


        /// Check if Segment is small
        if (vert_count < (componentCuts.size() * min_part)) {
            for (int i = 0; i < regions; i++) {
                connections.at(i) = 0;
            }

            /// Obtain connected Segments and the frequency of their connections with the current segment
            auto es = boost::edges(g);
            for (auto eit = es.first; eit != es.second; ++eit) {
                if (g[boost::source(*eit, g)].region != g[boost::target(*eit, g)].region) {
                    auto v1 = boost::source(*eit, g);
                    auto v2 = boost::target(*eit, g);
                    if ((g[v1].region == ind || g[v2].region == ind) && g[v1].region != g[v2].region) {
                        if (g[v1].region == ind) {
                            connections.at(g[v2].region) = connections.at(g[v2].region) + 1;
                        }
                        if (g[v2].region == ind) {
                            connections.at(g[v1].region) = connections.at(g[v1].region) + 1;
                        }
                    }
                }
            }

            /// Find most strongly connected segment
            min_dist_seg = std::distance(connections.begin(), std::max_element(connections.begin(), connections.end()));

            std::pair<vertex_iterator, vertex_iterator> vp;
            for (vp = boost::vertices(g); vp.first != vp.second; ++vp.first) {
                if (g[*vp.first].region == ind) {
                    g[*vp.first].region = min_dist_seg;
                    componentCuts[g[*vp.first].id] == min_dist_seg;
                }
            }
        }
    }

    for (int i = 0; i < V.rows(); i++) {
        label(i, 0) = g[i].region;
    }
}
