#include "mesh_utils.hpp"
#include <pcl/PolygonMesh.h>
#include <boost/graph/filtered_graph.hpp>
#include <boost/graph/connected_components.hpp>
#include <random>
#include <igl/sort.h>

/**
 *
 * @param source_centroid
 * @param target_centroid
 * @param source_normal
 * @param target_normal
 * @param normal_angle
 * @return
 */
bool
connIsConvex(const Eigen::Vector3f &source_centroid, const Eigen::Vector3f &target_centroid,
             const Eigen::Vector3f &source_normal, const Eigen::Vector3f &target_normal, float &normal_angle) {
    bool is_convex = true;
    bool is_smooth = true;

    normal_angle = pcl::getAngle3D(source_normal, target_normal, true);
    //  Geometric comparisons
    Eigen::Vector3f vec_t_to_s, vec_s_to_t;

    vec_t_to_s = source_centroid - target_centroid;
    vec_s_to_t = -vec_t_to_s;

    Eigen::Vector3f ncross;
    ncross = source_normal.cross(target_normal);

    // vec_t_to_s is the reference direction for angle measurements
    // Convexity Criterion: Check if connection of patches is convex. If this is the case the two supervoxels should be merged.
    if ((pcl::getAngle3D(vec_t_to_s, source_normal) - pcl::getAngle3D(vec_t_to_s, target_normal)) <= 0) {
        is_convex &= true;  // connection convex
    } else {
        is_convex &= (normal_angle < 10.0);  // concave connections will be accepted  if difference of normals is small
    }
    return (is_convex && is_smooth);
}

/**
 *
 * @param cloud
 * @param meshData
 * @param g
 * @param vertex_list
 * @param edge_list
 */
void meshToGraph(pcl::PointCloud<pcl::PointNormal>::Ptr cloud, pcl::PolygonMeshPtr meshData, Graph &g,
                 std::vector<boost::graph_traits<Graph>::vertex_descriptor> &vertex_list,
                 std::vector<boost::graph_traits<Graph>::edge_descriptor> &edge_list) {
    boost::graph_traits<Graph>::vertex_descriptor vertex_id;

    vertex_list.clear();
    edge_list.clear();

    /// Add Vertices
    for (int i = 0; i < cloud->size(); i++) {
        MyVertexProperties vertex = MyVertexProperties();
        vertex.id = (uint32_t) i;
        vertex.point = cloud->points[i];
        vertex.weight = 0.0;
        vertex_id = boost::add_vertex(vertex, g);
        vertex_list.push_back(vertex_id);
    }

    /// Add Edges
    std::pair<Graph::edge_descriptor, bool> check_edge;

    int run_index = 0;
    for (auto polyItr : meshData->polygons) {
        check_edge = boost::add_edge(vertex_list[polyItr.vertices[0]], vertex_list[polyItr.vertices[1]], g);
        if (check_edge.second) {
            g[check_edge.first].id = (uint32_t) run_index;
            edge_list.push_back(check_edge.first);
            run_index++;
        }
        check_edge = boost::add_edge(vertex_list[polyItr.vertices[1]], vertex_list[polyItr.vertices[2]], g);
        if (check_edge.second) {
            g[check_edge.first].id = (uint32_t) run_index;
            edge_list.push_back(check_edge.first);
            run_index++;
        }
        check_edge = boost::add_edge(vertex_list[polyItr.vertices[2]], vertex_list[polyItr.vertices[0]], g);
        if (check_edge.second) {
            g[check_edge.first].id = (uint32_t) run_index;
            edge_list.push_back(check_edge.first);
            run_index++;
        }
    }
}

/**
 *
 * @param g
 */
void labelEdgesByVertexRegions(Graph &g) {
    boost::graph_traits<Graph>::vertex_descriptor v1, v2;
    typedef boost::graph_traits<Graph>::edge_iterator edge_iter;
    std::pair<edge_iter, edge_iter> ep;

    for (ep = edges(g); ep.first != ep.second; ++ep.first) {
        v1 = source(*ep.first, g);
        v2 = target(*ep.first, g);
        if (g[v1].region == g[v2].region) {
            g[*ep.first].region = g[v1].region;
        }
    }
}

void getConcaveEdgesFromGraph(Graph &g, std::vector<boost::graph_traits<Graph>::vertex_descriptor> &vertex_list,
                              std::set<boost::graph_traits<Graph>::edge_descriptor> &concave_edges, bool thinning) {

    boost::filtered_graph<Graph, edge_predicate_cn, vertex_predicate_c> fg(g, edge_predicate_cn(g),
                                                                           vertex_predicate_c(g));

    std::vector<int> component(num_vertices(fg));
    int num = boost::connected_components(fg, &component[0]);

    //cout << "Total number of components: " << num << endl;
    std::vector<int>::size_type j;
    for (j = 0; j != component.size(); ++j) {
        g[vertex_list.at(j)].region = component[j];
    }

    labelEdgesByVertexRegions(g);

    /// Add all vertices of the graph
    if (!thinning) {
        for (int i = 0; i <= num; i++) {
            ///Find all vertices/edges of this concave region:
            std::set<boost::graph_traits<Graph>::vertex_descriptor> component_vertices;
            for (j = 0; j < component.size(); j++) {
                if (component[j] == i)
                    component_vertices.insert(vertex_list.at(j));
            }
            std::set<boost::graph_traits<Graph>::vertex_descriptor> vertex_blacklist;
            vertex_blacklist.clear();
            std::set<boost::graph_traits<Graph>::edge_descriptor> edge_blacklist;
            edge_blacklist.clear();

            /// Filtered, temporary graph that holds the current region
            boost::filtered_graph<Graph, edge_predicate_region_bl, vertex_predicate_region_bl>
                    tg(g, edge_predicate_region_bl(g, i, edge_blacklist),
                       vertex_predicate_region_bl(g, i, vertex_blacklist));


            /// Get all edges of this concave region based on the set of vertices
            auto es = boost::edges(tg);
            ///  Add Concave Edges to Set of Concave Edges
            for (auto eit = es.first; eit != es.second; ++eit) {
                auto vu1 = source(*eit, tg);
                auto vu2 = target(*eit, tg);
                if (component_vertices.find(vu1) != component_vertices.end() &&
                    component_vertices.find(vu2) != component_vertices.end()) {
                    /// Add edge
                    concave_edges.insert(*eit);
                    g[*eit].concave_region = i;
                }
            }
        }
    } else {
        std::set<boost::graph_traits<Graph>::edge_descriptor> globalEdgeList;
        typedef boost::graph_traits<Graph>::vertex_iterator VertexIterator;
        for (int i = 0; i <= num; i++) {
            //Find all vertices/edges of this concave region:
            std::set<boost::graph_traits<Graph>::vertex_descriptor> ComponentVertices;
            for (j = 0; j < component.size(); j++) {
                if (component[j] == i)
                    ComponentVertices.insert(vertex_list.at(j));
            }
            std::set<boost::graph_traits<Graph>::vertex_descriptor> vertex_blacklist;
            vertex_blacklist.clear();
            std::set<boost::graph_traits<Graph>::edge_descriptor> edge_blacklist;
            edge_blacklist.clear();

            /// Filtered, temporary graph that holds the current region
            boost::filtered_graph<Graph, edge_predicate_region_bl, vertex_predicate_region_bl>
                    tg(g, edge_predicate_region_bl(g, i, edge_blacklist),
                       vertex_predicate_region_bl(g, i, vertex_blacklist));

            /// Sort the boundary edges depending on their weight
            /// Get all edges of this concave region based on the set of vertices
            std::multimap<float, boost::graph_traits<Graph>::edge_descriptor> WeightEdgeMap;
            std::map<boost::graph_traits<Graph>::edge_descriptor, boost::graph_traits<Graph>::edge_descriptor> EdgeToEdgeMap;
            auto es = boost::edges(tg);
            ///  Get multimap a la weight -> edge_index
            for (auto eit = es.first; eit != es.second; ++eit) {
                auto vu1 = source(*eit, tg);
                auto vu2 = target(*eit, tg);
                if (ComponentVertices.find(vu1) != ComponentVertices.end() &&
                    ComponentVertices.find(vu2) != ComponentVertices.end()) {
                    /// Add weight of edge and edge itself to map
                    WeightEdgeMap.insert(
                            std::pair<float, boost::graph_traits<Graph>::edge_descriptor>(tg[*eit].weight, *eit));
                }
            }

            auto next_it = WeightEdgeMap.begin();
            int l = 0;
            int comp = 0;
            int comp_old = 0;

            //std::cout << "Region " << i << ", before thinning: " << WeightEdgeMap.size() << " Edges" << std::endl;

            ///For all edges inside the WeightEdgeMap
            for (auto it = WeightEdgeMap.begin(); it != WeightEdgeMap.end(); it = next_it) {
                l++;
                //std::ofstream dotfile ("test" + std::to_string(l) + ".dot");
                //write_graphviz (dotfile, tg);
                //std::cout << "[" << l << "/" << WeightEdgeMap.size() << "] ";
                //std::cout << "s: " << source(it->second, tg) << ", t: " << target(it->second, tg) << std::endl;
                std::set<boost::graph_traits<Graph>::vertex_descriptor> vvu1_neighbors;
                std::set<boost::graph_traits<Graph>::vertex_descriptor> vvu2_neighbors;
                next_it = std::next(it, 1);
                std::vector<int> component_temp(num_vertices(tg));
                comp_old = boost::connected_components(tg, &component_temp[0]);
                /// Remove Edge from temp graph
                auto pair_ai = boost::adjacent_vertices(source(it->second, tg), tg);
                for (; pair_ai.first != pair_ai.second; pair_ai.first++) {
                    vvu1_neighbors.insert(*pair_ai.first);
                }
                ///Check if all these vertices are really inside
                vvu1_neighbors.erase(target(it->second, tg));
                vvu1_neighbors.erase(source(it->second, tg));
                //std::cout << "s: " << source(it->second, tg) << ", t: " << target(it->second, tg) << std::endl;
                pair_ai = boost::adjacent_vertices(target(it->second, tg), tg);
                for (; pair_ai.first != pair_ai.second; pair_ai.first++) {
                    vvu2_neighbors.insert(*pair_ai.first);
                }
                vvu2_neighbors.erase(source(it->second, tg));
                vvu2_neighbors.erase(target(it->second, tg));

                if (vvu1_neighbors.empty()) {
                    vertex_blacklist.insert(source(it->second, tg));
                }
                if (vvu2_neighbors.empty()) {
                    vertex_blacklist.insert(target(it->second, tg));
                }

                edge_blacklist.insert(it->second);

                /// Calculate number of components of the temporary graph
                std::vector<int> component_temp2(num_vertices(tg));
                comp = boost::connected_components(tg, &component_temp2[0]);
                if (comp_old == comp) {
                    WeightEdgeMap.erase(it);
                } else {
                    // Graph connectivity will break if this edge is deleted, so add it back and go to next edge
                    edge_blacklist.erase(it->second);

                    // Also add back the vertices that were deleted, if done
                    if (vvu1_neighbors.empty())
                        vertex_blacklist.erase(source(it->second, tg));
                    if (vvu2_neighbors.empty())
                        vertex_blacklist.erase(target(it->second, tg));
                    //std::cout << "Edge not deleted since this would break the region.\n";
                }
            }
            //Iterate over the rest of the temporary graph to copy it's values to the permanent graph cg
            for (auto it = WeightEdgeMap.begin(); it != WeightEdgeMap.end(); it++) {
                globalEdgeList.insert(it->second);
            }
        }

        /// Remove all cuts
        auto es = boost::edges(g);
        ///  Add Concave Edges to Set of Concave Edges
        for (auto eit = es.first; eit != es.second; ++eit) {
            g[*eit].is_concave = false;
        }
        for (auto curr_edge : globalEdgeList) {
            g[curr_edge].cut = true;
        }
    }
}

/**
 *
 * @param g
 */
void setConcaveEdges(Graph &g) {
    boost::graph_traits<Graph>::vertex_descriptor v1, v2;
    typedef boost::graph_traits<Graph>::edge_iterator edge_iter;
    std::pair<edge_iter, edge_iter> ep;
    edge_iter ei, ei_end;

    typedef boost::graph_traits<Graph>::adjacency_iterator AdjacencyIterator1;

    for (ep = edges(g); ep.first != ep.second; ++ep.first) {
        v1 = source(*ep.first, g);
        v2 = target(*ep.first, g);

        float normal_angle(0.0);
        auto x = g[v1].point.getVector3fMap();
        auto y = g[v2].point.getVector3fMap();
        auto xn = g[v1].point.getNormalVector3fMap();
        auto yn = g[v2].point.getNormalVector3fMap();

        /*if (g[v1].weight < g[v2].weight){
          g[*ep.first].principle_direction = g[v1].principle_direction;
        }else{
          g[*ep.first].principle_direction = g[v2].principle_direction;
        }

        g[*ep.first].weight = (g[v1].weight + g[v2].weight) / 2;
        */
        //if ((g[v1].weight < -0.8 || g[v2].weight < -0.8) && !connIsConvex(x, y,xn, yn, normal_angle)){
        /*if ((g[v1].weight < -0.8 && g[v2].weight < -0.8)){
          g[*ep.first].is_concave = true;
          g[v1].is_concave = true;
          g[v2].is_concave = true;
        }*/
        if (!connIsConvex(x, y, xn, yn, normal_angle)) {
            g[*ep.first].is_concave = true;
            g[v1].is_concave = true;
            g[v2].is_concave = true;
            g[v1].weight = -1.0;
            g[v2].weight = -1.0;
            g[*ep.first].weight = -1.0;
        }

    }

    /// Concave Edge Growing Mechanism:
    for (ep = edges(g); ep.first != ep.second; ++ep.first) {
        v1 = source(*ep.first, g);
        v2 = target(*ep.first, g);
        if (g[v1].is_concave && g[v2].is_concave) {
            g[*ep.first].is_concave = true;
        }
    }

    /*for (ep = edges(g); ep.first != ep.second; ++ep.first) {
      v1 = source(*ep.first, g);
      v2 = target(*ep.first, g);

      if (g[*ep.first].is_concave) {
        /// Refining
        typedef boost::graph_traits<Graph>::adjacency_iterator AdjacencyIterator1;
        AdjacencyIterator1 ai, a_end;

        boost::tie(ai, a_end) = boost::adjacent_vertices(v1, g);
        std::set<boost::graph_traits<Graph>::vertex_descriptor> v1_neighbors;
        std::set<boost::graph_traits<Graph>::vertex_descriptor> v2_neighbors;
        std::set<boost::graph_traits<Graph>::vertex_descriptor> v1_neighbors_n;
        std::set<boost::graph_traits<Graph>::vertex_descriptor> v2_neighbors_n;

        for (; ai != a_end; ai++) {
          v1_neighbors.insert(*ai);
        }
        v1_neighbors_n = v1_neighbors;
        v1_neighbors.erase(v2);

        boost::tie(ai, a_end) = boost::adjacent_vertices(v2, g);
        for (; ai != a_end; ai++) {
          v2_neighbors.insert(*ai);
        }
        v2_neighbors_n = v2_neighbors;
        v2_neighbors.erase(v1);

        std::vector<boost::graph_traits<Graph>::vertex_descriptor> common;
        std::set_intersection(v1_neighbors.begin(), v1_neighbors.end(), v2_neighbors.begin(), v2_neighbors.end(),
                              std::back_inserter(common));

        auto voppo1 = common[0];
        auto voppo2 = common[1];

        if (g[*ep.first].is_concave && g[voppo1].weight > g[*ep.first].weight && g[voppo2].weight > g[*ep.first].weight) {
          g[*ep.first].is_concave = true;
        }else{
          g[*ep.first].is_concave = false;
        }
      }
    }*/
}

/**
 *
 * @param g
 * @param min_part
 * @param vertex_list
 */
void mergeSegments(Graph &g, double min_part,
                   std::vector<boost::graph_traits<Graph>::vertex_descriptor> &vertex_list) {
    typedef boost::graph_traits<Graph>::edge_iterator edge_iter;
    typedef boost::graph_traits<Graph>::vertex_iterator vertex_iterator;
    boost::graph_traits<Graph>::vertex_descriptor v1, v2;
    boost::filtered_graph<Graph, edge_predicate_cnCut> lg(g, edge_predicate_cnCut(g));

    std::vector<int> componentCuts(num_vertices(lg));
    size_t regions = boost::connected_components(lg, &componentCuts[0]);

    //cout << "Total number of Parts before Merging Step: " << regions << endl;
    for (int j = 0; j != componentCuts.size(); ++j) {
        g[vertex_list.at(j)].region = componentCuts[j];
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
                if (g[*eit].cut) {
                    auto v1 = source(*eit, g);
                    auto v2 = target(*eit, g);
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
            //TODO: Instead merge the smallest region with the next smallest neighbor it has connections with

            std::pair<vertex_iterator, vertex_iterator> vp;
            for (vp = vertices(g); vp.first != vp.second; ++vp.first) {
                if (g[*vp.first].region == ind) {
                    g[*vp.first].region = min_dist_seg;
                    componentCuts[g[*vp.first].id] == min_dist_seg;
                }
            }
            std::cout << "Merged Region " << ind << " with Region " << min_dist_seg << std::endl;
        }
    }
}

/**
 *
 * @param plane_normal
 * @param plane_point
 * @param pb1
 * @param pb2
 * @param w
 * @param D
 * @param N
 * @return
 */
Eigen::VectorXi
planeLineIntersectionBatch(Eigen::Vector3f plane_normal, Eigen::Vector3f plane_point, Eigen::MatrixXf &pb1,
                           Eigen::MatrixXf &pb2,
                           Eigen::MatrixXf &w, Eigen::VectorXf &D, Eigen::VectorXf &N) {
    Eigen::MatrixXf Id(1, pb1.cols());
    Eigen::MatrixXf u = pb2 - pb1;
    //Eigen::MatrixXf w = pb1 - plane_point * Eigen::RowVectorXf::Ones(pb1.cols());
    //Eigen::VectorXf D = ((plane_normal * Eigen::RowVectorXf::Ones(pb1.cols())).cwiseProduct(u)).colwise().sum();
    //Eigen::VectorXf N = - ((plane_normal * Eigen::RowVectorXf::Ones(pb1.cols())).cwiseProduct(w)).colwise().sum();
    w = pb1 - plane_point * Eigen::RowVectorXf::Ones(pb1.cols());
    D = ((plane_normal * Eigen::RowVectorXf::Ones(pb1.cols())).cwiseProduct(u)).colwise().sum();
    N = -((plane_normal * Eigen::RowVectorXf::Ones(pb1.cols())).cwiseProduct(w)).colwise().sum();
    Eigen::VectorXi check = Eigen::VectorXi::Zero(pb1.cols());
    for (int i = 0; i < D.rows(); i++) {
        if (abs(D(i)) < 0.0000001) {
            if (N(i) == 0) {
                check(i) = 1;
                //check(i) = 2;
            } else {
                check(i) = 0;
            }
        }
    }

    Eigen::VectorXf sI = N.cwiseQuotient(D);
    for (int i = 0; i < D.rows(); i++) {
        if (sI(i) < 0.0 || sI(i) > 1.0) {
            check(i) = 0;
            //check(i) = 3;
        } else {
            check(i) = 1;
        }
    }
    return check;
}

/**
 *
 * @param g
 * @param curr_cluster
 * @param cluster_edge_list
 * @param cluster_vertex_list
 * @param cluster_concave_vertices
 * @param cluster_concave_edges
 * @param globalEdgeIndices
 */
void getSegmentEdgesAndVertices(Graph &g, int curr_cluster,
                                std::vector<boost::graph_traits<Graph>::edge_descriptor> &cluster_edge_list,
                                std::vector<boost::graph_traits<Graph>::vertex_descriptor> &cluster_vertex_list,
                                std::vector<Graph::vertex_descriptor> &cluster_concave_vertices,
                                std::vector<boost::graph_traits<Graph>::edge_descriptor> &cluster_concave_edges,
                                Eigen::VectorXi &globalEdgeIndices) {
    cluster_edge_list.clear();
    cluster_vertex_list.clear();
    cluster_concave_vertices.clear();
    cluster_concave_edges.clear();

    /// Get all (concave) vertices of this cluster
    typedef boost::graph_traits<Graph>::vertex_iterator vertex_iterator;
    std::pair<vertex_iterator, vertex_iterator> vp;
    for (vp = vertices(g); vp.first != vp.second; ++vp.first) {
        if (g[*vp.first].region == curr_cluster && g[*vp.first].is_concave) {
            cluster_concave_vertices.push_back(*vp.first);
        }
    }

    std::vector<int> concaveEdgeIndices;
    /// Get all (concave) edges of this cluster
    int edge_index = 0;
    for (auto ep = edges(g); ep.first != ep.second; ++ep.first) {
        auto v1 = source(*ep.first, g);
        auto v2 = target(*ep.first, g);
        if (g[v1].region == curr_cluster && g[v2].region == curr_cluster) {
            cluster_edge_list.push_back(*ep.first);
            if (g[*ep.first].is_concave && !(g[*ep.first].cut) && g[*ep.first].valid) {
                cluster_concave_edges.push_back(*ep.first);
                concaveEdgeIndices.push_back(edge_index);
            }
            edge_index++;
        } //else{
        //std::cout << " ";
        //}
    }
    globalEdgeIndices = Eigen::VectorXi::Zero(cluster_edge_list.size());
    for (auto val : concaveEdgeIndices) {
        globalEdgeIndices(val) = 1;
    }
}

/**
 *
 * @param g
 * @param ransac_iterations
 * @param vertex_list
 * @param edge_list
 * @param concave_edges
 * @param min_cut_score
 */
void
MCPCSegment(Graph &g, int ransac_iterations, std::vector<boost::graph_traits<Graph>::vertex_descriptor> &vertex_list,
            std::vector<boost::graph_traits<Graph>::edge_descriptor> &edge_list,
            std::set<boost::graph_traits<Graph>::edge_descriptor> &concave_edges, double min_cut_score) {

    /// Initialization
    //std::cout << num_vertices(g) << " vertices, " << num_edges(g) << " edges\n";
    std::random_device rd;
    //std::mt19937 rng(rd());
    std::mt19937 rng(0); // TODO: Switch to rd-seed
    std::vector<boost::graph_traits<Graph>::edge_descriptor> edges_to_be_cut;
    edges_to_be_cut.clear();
    int num_cut = 0;
    int index = 0;
    float cut_score, max_score, last_iter_score;
    max_score = 0;
    Eigen::Vector3f plane_normal, best_plane_normal;
    Eigen::Vector3f plane_point, best_plane_point;
    std::vector<Graph::edge_descriptor> concave_vertices;
    for (auto it : concave_edges) {
        concave_vertices.push_back(it);
    }
    std::vector<boost::graph_traits<Graph>::edge_descriptor> cluster_edge_list;
    std::vector<boost::graph_traits<Graph>::vertex_descriptor> cluster_vertex_list;
    std::vector<Graph::vertex_descriptor> cluster_concave_vertices;
    std::vector<boost::graph_traits<Graph>::edge_descriptor> cluster_concave_edges;

    /// Pick first three points and get plane out of them
    int pa, pb, pc;
    Eigen::Vector3f point_a;
    Eigen::Vector3f point_b;
    Eigen::Vector3f point_c;
    //pcl::ModelCoefficients::Ptr plane(new pcl::ModelCoefficients);
    /// Inital Plane Model Coefficients
    Eigen::Hyperplane<float, 3> eigen_plane;

    last_iter_score = 0;
    cut_score = -0.1;
    std::set<int> cut_regions;
    cut_regions.clear();
    int best_cut_segment = 0;
    int curr_cluster = 0;
    bool found_any_cut = false;

    /// Main RANSAC LOOP
    do {
        num_cut++;
        std::cout << "Searching Cut # " << num_cut << std::endl;
        found_any_cut = false;

        /// Get current clustered graph
        boost::filtered_graph<Graph, edge_predicate_cnCut> tlg(g, edge_predicate_cnCut(g));
        std::vector<int> clusters(num_vertices(tlg));
        size_t num_clusters = boost::connected_components(tlg, &clusters[0]);

        /// Refresh Vertex labels of the main graph
        for (int j = 0; j != clusters.size(); ++j) {
            g[vertex_list.at(j)].region = clusters[j];
        }

        /// Refresh Edge labels of the main graph
        boost::graph_traits<Graph>::vertex_descriptor v1, v2;
        typedef boost::graph_traits<Graph>::edge_iterator edge_iter;
        std::pair<edge_iter, edge_iter> ep;

        for (ep = edges(g); ep.first != ep.second; ++ep.first) {
            v1 = source(*ep.first, g);
            v2 = target(*ep.first, g);
            if (g[v1].region == g[v2].region) {
                g[*ep.first].region = g[v1].region;
            } else {
                g[*ep.first].region = -1; // Undefined region
            }
        }

        last_iter_score = 0;
        cut_score = -0.1;

        max_score = 0;
        /// Iterate over all clusters of the graph
        for (curr_cluster = 0; curr_cluster < num_clusters; curr_cluster++) {

            /// Extract the cluster's lists of (concave) edges and vertices
            Eigen::VectorXi globalEdgeIndices;
            getSegmentEdgesAndVertices(g, curr_cluster, cluster_edge_list, cluster_vertex_list,
                                       cluster_concave_vertices,
                                       cluster_concave_edges, globalEdgeIndices);

            if (cluster_concave_edges.size() > 3) {

                std::uniform_int_distribution<int> uni(0, cluster_concave_edges.size() - 1);
                /// Source-Target Vertex Table 2xN to Eigen::MatrixXf
                Eigen::MatrixXf source_vertex_of_edge_table(3, cluster_edge_list.size());
                Eigen::MatrixXf target_vertex_of_edge_table(3, cluster_edge_list.size());
                int rowCount(0);
                for (auto it = cluster_edge_list.begin(); it != cluster_edge_list.end(); it++) {
                    auto p1 = g[source(*it, g)].point.getVector3fMap();
                    auto p2 = g[target(*it, g)].point.getVector3fMap();
                    source_vertex_of_edge_table.col(rowCount) = p1;
                    target_vertex_of_edge_table.col(rowCount) = p2;
                    rowCount++;
                }
                std::vector<Graph::edge_descriptor> cut_edges;

                /// Main Loop for Iterations
                Eigen::MatrixXf w;
                Eigen::VectorXf D;
                Eigen::VectorXf N;
                for (index = 0; index < ransac_iterations; index++) {
                    cut_regions.clear();
                    cut_edges.clear();
                    int count_weights(0);
                    int counter(0);

                    ///Select new inliers
                    if (last_iter_score >= cut_score) {
                        //TODO: Eliminate this loop, probably going back to shuffling vectors
                        //do {
                        Eigen::MatrixXi randindices;
                        igl::randperm(cluster_concave_edges.size(), randindices);
                        //  pa = uni(rng);
                        //  pb = uni(rng);
                        //  pc = uni(rng);
                        //}while (pa == pb || pa == pc || pb == pc);
                        pa = randindices(0, 0);
                        pb = randindices(1, 0);
                        pc = randindices(2, 0);

                        point_a = Eigen::Vector3f(
                                (g[source(cluster_concave_edges[pa], g)].point.x +
                                 g[target(cluster_concave_edges[pa], g)].point.x) / 2.0,
                                (g[source(cluster_concave_edges[pa], g)].point.y +
                                 g[target(cluster_concave_edges[pa], g)].point.y) / 2.0,
                                (g[source(cluster_concave_edges[pa], g)].point.z +
                                 g[target(cluster_concave_edges[pa], g)].point.z) / 2.0
                        );

                        point_b = Eigen::Vector3f(
                                (g[source(cluster_concave_edges[pb], g)].point.x +
                                 g[target(cluster_concave_edges[pb], g)].point.x) / 2.0,
                                (g[source(cluster_concave_edges[pb], g)].point.y +
                                 g[target(cluster_concave_edges[pb], g)].point.y) / 2.0,
                                (g[source(cluster_concave_edges[pb], g)].point.z +
                                 g[target(cluster_concave_edges[pb], g)].point.z) / 2.0
                        );

                        point_c = Eigen::Vector3f(
                                (g[source(cluster_concave_edges[pc], g)].point.x +
                                 g[target(cluster_concave_edges[pc], g)].point.x) / 2.0,
                                (g[source(cluster_concave_edges[pc], g)].point.y +
                                 g[target(cluster_concave_edges[pc], g)].point.y) / 2.0,
                                (g[source(cluster_concave_edges[pc], g)].point.z +
                                 g[target(cluster_concave_edges[pc], g)].point.z) / 2.0
                        );

                        plane_normal = (point_a - point_c).cross(point_b - point_c);
                        plane_point = point_c;
                    }

                    /// Check if plane intersects line segments
                    //auto start = std::chrono::system_clock::now();
                    /// Repeating this function 10000 times takes a long time, try to call it only once?
                    auto res1 = planeLineIntersectionBatch(plane_normal, plane_point, source_vertex_of_edge_table,
                                                           target_vertex_of_edge_table,
                                                           w, D, N);
                    //auto end = std::chrono::system_clock::now();
                    //std::chrono::duration<double> elapsed_seconds = end-start;
                    //std::time_t end_time = std::chrono::system_clock::to_time_t(end);
                    //std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

                    /// Create list of edges that were actually cut by given plane
                    for (auto it = cluster_edge_list.begin(); it != cluster_edge_list.end(); it++) {
                        if (res1(counter) == 1)
                            cut_edges.push_back(*it);
                        counter++;
                    }

                    /// Calculate score & update highest score, if necessary
                    count_weights = (globalEdgeIndices.cwiseProduct(res1)).sum();
                    last_iter_score = cut_score;
                    cut_score = (float) count_weights / (float) cut_edges.size();
                    if (cut_score > max_score) {
                        max_score = cut_score;
                        best_plane_normal = plane_normal;
                        best_plane_point = plane_point;
                        edges_to_be_cut = cut_edges;
                        best_cut_segment = curr_cluster;
                        found_any_cut = true;
                    }
                }
            }
        }
        std::cout << "Best Score found: " << max_score << std::endl;
        if (max_score >= min_cut_score) {
            plane_normal = best_plane_normal;
            plane_point = best_plane_point;
            curr_cluster = best_cut_segment;
            Eigen::VectorXi globalEdgeIndices;
            getSegmentEdgesAndVertices(g, curr_cluster, cluster_edge_list, cluster_vertex_list,
                                       cluster_concave_vertices,
                                       cluster_concave_edges, globalEdgeIndices);

            Eigen::MatrixXf source_vertex_of_edge_table(3, cluster_edge_list.size());
            Eigen::MatrixXf target_vertex_of_edge_table(3, cluster_edge_list.size());
            int rowCount(0);
            for (auto it = cluster_edge_list.begin(); it != cluster_edge_list.end(); it++) {
                auto p1 = g[source(*it, g)].point.getVector3fMap();
                auto p2 = g[target(*it, g)].point.getVector3fMap();
                source_vertex_of_edge_table.col(rowCount) = p1;
                target_vertex_of_edge_table.col(rowCount) = p2;
                rowCount++;
            }

            Eigen::MatrixXf w;
            Eigen::VectorXf D;
            Eigen::VectorXf N;
            auto res1 = planeLineIntersectionBatch(plane_normal, plane_point, source_vertex_of_edge_table,
                                                   target_vertex_of_edge_table, w, D, N);
            int counter = 0;
            /// Remove the cut edges from global Concave Edge List and mark them as cut in graph
            for (auto it = edges_to_be_cut.begin(); it != edges_to_be_cut.end(); it++) {
                g[*it].cut = true;
            }

            /// Now get all cut regions and remove all member edges of these regions from the Concave Edge List
            for (auto itr4 = edges_to_be_cut.begin(); itr4 != edges_to_be_cut.end(); itr4++) {
                cut_regions.insert(g[*itr4].concave_region);
            }

        }
    } while (max_score >= min_cut_score);

    std::cout << "No more cuts found.\n";
}

/**
 *
 * @param mesh
 * @param V
 * @param F
 * @param cloud
 */
void mesh2EigenAndCloud(pcl::PolygonMeshPtr &mesh, Eigen::MatrixXd &V, Eigen::MatrixXd &F,
                        pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
    pcl::fromPCLPointCloud2(mesh->cloud, *cloud);
    V.resize(mesh->cloud.height * mesh->cloud.width, 3);
    pcl::fromPCLPointCloud2(mesh->cloud, *cloud);

    /// Get Vertices
    for (int i = 0; i < cloud->size(); i++) {
        V(i, 0) = cloud->points[i].x;
        V(i, 1) = cloud->points[i].y;
        V(i, 2) = cloud->points[i].z;
    }

    /// Get Faces
    F.resize(mesh->polygons.size(), 3);
    int i = 0;
    for (auto itr = mesh->polygons.begin(); itr != mesh->polygons.end(); ++itr) {
        F(i, 0) = itr->vertices[0];
        F(i, 1) = itr->vertices[1];
        F(i, 2) = itr->vertices[2];
        i++;
    }
}

/**
 *
 * @param mesh
 * @param V
 * @param F
 * @param cloud
 */
void mesh2EigenAndCloud(pcl::PolygonMeshPtr &mesh, Eigen::MatrixXf &V, Eigen::MatrixXf &F,
                        pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
    pcl::fromPCLPointCloud2(mesh->cloud, *cloud);
    V.resize(mesh->cloud.height * mesh->cloud.width, 3);
    pcl::fromPCLPointCloud2(mesh->cloud, *cloud);

    /// Get Vertices
    for (int i = 0; i < cloud->size(); i++) {
        V(i, 0) = cloud->points[i].x;
        V(i, 1) = cloud->points[i].y;
        V(i, 2) = cloud->points[i].z;
    }

    /// Get Faces
    F.resize(mesh->polygons.size(), 3);
    int i = 0;
    for (auto itr = mesh->polygons.begin(); itr != mesh->polygons.end(); ++itr) {
        F(i, 0) = itr->vertices[0];
        F(i, 1) = itr->vertices[1];
        F(i, 2) = itr->vertices[2];
        i++;
    }
}

/**
 *
 * @param PV1
 * @param k
 */
void smoothOutliers(Eigen::MatrixXf &PV1, int k) {
    Eigen::MatrixXf PP(PV1.size(), 1);
    Eigen::MatrixXf PPJ(PV1.size(), 1);
    igl::sort(PV1, 1, true, PP, PPJ);

    for (int i = 0; i < k; i++) {
        PV1(PPJ(i)) = PP(k);
        PV1(PPJ(PV1.size() - i - 1)) = PP(PV1.size() - k - 1);
    }
}

/**
 *
 * @param PV1
 */
void normalizeCurvature(Eigen::MatrixXf &PV1) {
    Eigen::MatrixXf a = PV1 - PV1.mean() * Eigen::MatrixXf::Ones(PV1.size(), 1);
    Eigen::MatrixXf b = a.cwiseProduct(a);
    double c = b.sum();
    double stddev = sqrt(c / (PV1.size() - 1));
    Eigen::MatrixXf PV2 = a.cwiseQuotient(stddev * Eigen::MatrixXf::Ones(PV1.size(), 1));
    PV1 = PV2;
}

/**
 *
 * @param PV1
 * @return
 */
Eigen::MatrixXf normalizeCurvatureAndCopy(Eigen::MatrixXf &PV1) {
    Eigen::MatrixXf a = PV1 - PV1.mean() * Eigen::MatrixXf::Ones(PV1.size(), 1);
    Eigen::MatrixXf b = a.cwiseProduct(a);
    double c = b.sum();
    double stddev = sqrt(c / (PV1.size() - 1));
    Eigen::MatrixXf PV2 = a.cwiseQuotient(stddev * Eigen::MatrixXf::Ones(PV1.size(), 1));
    return PV2;
}

/**
 *
 * @param meshData
 * @param pc
 * @param inversion
 */
void computeVertexNormals(pcl::PolygonMeshPtr &meshData, pcl::PointCloud<pcl::PointNormal>::Ptr &pc, bool inversion) {
    pcl::fromPCLPointCloud2(meshData->cloud, *pc);

    int vertex_size = meshData->cloud.height * meshData->cloud.width;
    std::vector<Eigen::Vector3f> *normal_buffer = new std::vector<Eigen::Vector3f>[vertex_size];

    int i = 0;
    for (auto polyItr = meshData->polygons.begin(); polyItr < meshData->polygons.end(); polyItr++) {
        std::vector<Eigen::Vector3f> p(3);
        i = 0;
        for (auto vertItr = polyItr->vertices.begin(); vertItr < polyItr->vertices.end(); vertItr++) {
            p[i] << pc->points[*vertItr].x, pc->points[*vertItr].y, pc->points[*vertItr].z;
            i++;
        }
        Eigen::Vector3f v1 = p[1] - p[0];
        Eigen::Vector3f v2 = p[2] - p[0];
        Eigen::Vector3f normal = v1.cross(v2);

        normal.normalize();
        for (auto vertItr = polyItr->vertices.begin(); vertItr < polyItr->vertices.end(); vertItr++) {
            normal_buffer[*vertItr].push_back(normal);
        }
    }

    // Now loop through each vertex vector, and avarage out all the normals stored.
    for (int i = 0; i < vertex_size; ++i) {
        for (int j = 0; j < normal_buffer[i].size(); ++j) {
            pc->points[i].normal_x += normal_buffer[i][j](0);
            pc->points[i].normal_y += normal_buffer[i][j](1);
            pc->points[i].normal_z += normal_buffer[i][j](2);
        }
        pc->points[i].normal_x /= normal_buffer[i].size();
        pc->points[i].normal_y /= normal_buffer[i].size();
        pc->points[i].normal_z /= normal_buffer[i].size();

        if (inversion) {
            pc->points[i].normal_x = -pc->points[i].normal_x;
            pc->points[i].normal_y = -pc->points[i].normal_y;
            pc->points[i].normal_z = -pc->points[i].normal_z;
        }
    }
}