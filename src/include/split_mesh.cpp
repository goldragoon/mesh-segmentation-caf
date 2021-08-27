#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/filtered_graph.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/undirected_dfs.hpp>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/mesh_segmentation.h>
#include <CGAL/property_map.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <igl/triangle_triangle_adjacency.h>
#include <igl/vertex_triangle_adjacency.h>
#include <igl/grad.h>
#include <igl/copyleft/cgal/mesh_to_polyhedron.h>

#include <iostream>
#include <fstream>

#include "split_mesh.h"
#define _USE_MATH_DEFINES
#include <math.h>

#include <pcl/common/common.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef CGAL::Polyhedron_3<Kernel> Polyhedron;

FaceVertexProperties::FaceVertexProperties() : area(0.0), id(0) {};

FaceEdgeProperties::FaceEdgeProperties() {};

typedef FaceGraph::edge_descriptor edge_id_t;
typedef FaceGraph::vertex_descriptor vertex_id_t;

class vertex_in_whitelist {
public:
    vertex_in_whitelist();

    vertex_in_whitelist(FaceGraph &graph, std::set<boost::graph_traits<FaceGraph>::vertex_descriptor> &whitelist);

    bool operator()(const vertex_id_t &vertex_id) const;

private:
    FaceGraph *graph_m;
    std::set<boost::graph_traits<FaceGraph>::vertex_descriptor> *whitelist_m;
};

class edge_in_whitelist {
public:
    edge_in_whitelist();

    edge_in_whitelist(FaceGraph &graph, std::set<boost::graph_traits<FaceGraph>::vertex_descriptor> &vertex_whitelist,
                      std::set<boost::graph_traits<FaceGraph>::edge_descriptor> &edge_whitelist);

    bool operator()(const edge_id_t &edge_id) const;

private:
    FaceGraph *graph_m;
    std::set<boost::graph_traits<FaceGraph>::vertex_descriptor> *vertex_whitelist_m;
    std::set<boost::graph_traits<FaceGraph>::edge_descriptor> *edge_whitelist_m;
};

typedef typename boost::filtered_graph<FaceGraph, edge_in_whitelist, vertex_in_whitelist> WhitelistFilteredFaceGraph;

vertex_in_whitelist::vertex_in_whitelist() : graph_m(0), whitelist_m() {};

vertex_in_whitelist::vertex_in_whitelist(FaceGraph &graph,
                                         std::set<boost::graph_traits<FaceGraph>::vertex_descriptor> &whitelist) :
        graph_m(&graph), whitelist_m(&whitelist) {};

bool vertex_in_whitelist::operator()(const vertex_id_t &vertex_id) const {
    return (*whitelist_m).find(vertex_id) != (*whitelist_m).end();
};

edge_in_whitelist::edge_in_whitelist() : graph_m(0), vertex_whitelist_m(), edge_whitelist_m() {};

edge_in_whitelist::edge_in_whitelist(FaceGraph &graph,
                                     std::set<boost::graph_traits<FaceGraph>::vertex_descriptor> &vertex_whitelist,
                                     std::set<boost::graph_traits<FaceGraph>::edge_descriptor> &edge_whitelist) :
        graph_m(&graph), vertex_whitelist_m(&vertex_whitelist), edge_whitelist_m(&edge_whitelist) {};

bool edge_in_whitelist::operator()(const edge_id_t &edge_id) const {
    return ((*vertex_whitelist_m).find(source(edge_id, *graph_m)) != (*vertex_whitelist_m).end() &&
            (*vertex_whitelist_m).find(target(edge_id, *graph_m)) != (*vertex_whitelist_m).end() &&
            (*edge_whitelist_m).find(edge_id) != (*edge_whitelist_m).end());
};

class vertex_in_blacklist {
public:
    vertex_in_blacklist();

    vertex_in_blacklist(FaceGraph &graph, std::set<boost::graph_traits<FaceGraph>::vertex_descriptor> &blacklist);

    bool operator()(const vertex_id_t &vertex_id) const;

private:
    FaceGraph *graph_m;
    std::set<boost::graph_traits<FaceGraph>::vertex_descriptor> *blacklist_m;
};

class edge_in_blacklist {
public:
    edge_in_blacklist();

    edge_in_blacklist(FaceGraph &graph, std::set<boost::graph_traits<FaceGraph>::edge_descriptor> &blacklist);

    bool operator()(const edge_id_t &edge_id) const;

private:
    FaceGraph *graph_m;
    std::set<boost::graph_traits<FaceGraph>::edge_descriptor> *blacklist_m;
};

typedef typename boost::filtered_graph<FaceGraph, edge_in_blacklist, vertex_in_blacklist> FilteredFaceGraph;

vertex_in_blacklist::vertex_in_blacklist() : graph_m(0), blacklist_m() {};

vertex_in_blacklist::vertex_in_blacklist(FaceGraph &graph,
                                         std::set<boost::graph_traits<FaceGraph>::vertex_descriptor> &blacklist) :
        graph_m(&graph), blacklist_m(&blacklist) {};

bool vertex_in_blacklist::operator()(const vertex_id_t &vertex_id) const {
    return (*blacklist_m).find(vertex_id) == (*blacklist_m).end();
};

edge_in_blacklist::edge_in_blacklist() : graph_m(0), blacklist_m() {};

edge_in_blacklist::edge_in_blacklist(FaceGraph &graph,
                                     std::set<boost::graph_traits<FaceGraph>::edge_descriptor> &blacklist) :
        graph_m(&graph), blacklist_m(&blacklist) {};

bool edge_in_blacklist::operator()(const edge_id_t &edge_id) const {
    return (*blacklist_m).find(edge_id) == (*blacklist_m).end();
};

/**
 *
 * @param F
 * @param E
 * @param IF
 * @param dblA
 * @param contour_faces
 * @param g
 * @param vertex_blacklist
 * @param edge_blacklist
 * @return
 */
std::vector<double> split_mesh(Eigen::MatrixXi &F, Eigen::MatrixXi &E, Eigen::MatrixXi &IF, Eigen::MatrixXd &dblA,
                               std::vector<int> &contour_faces, FaceGraph &g,
                               std::set<boost::graph_traits<FaceGraph>::vertex_descriptor> vertex_blacklist,
                               std::set<boost::graph_traits<FaceGraph>::edge_descriptor> edge_blacklist) {
    using namespace boost;
    /// Initialization
    std::vector<double> comp_area;

    for (int i = 0; i < IF.rows(); i++) {
        if ((std::find(contour_faces.begin(), contour_faces.end(), IF(i, 0)) != contour_faces.end() ||
             std::find(contour_faces.begin(), contour_faces.end(), IF(i, 1)) != contour_faces.end())) {
            auto tmp_edge = boost::edge(IF(i, 0), IF(i, 1), g).first;
            edge_blacklist.insert(tmp_edge);
        }
    }

    for (int i = 0; i < contour_faces.size(); i++) {
        vertex_blacklist.insert(contour_faces[i]);
    }

    FilteredFaceGraph fg(g, edge_in_blacklist(g, edge_blacklist), vertex_in_blacklist(g, vertex_blacklist));

    /// Compute connected components
    std::vector<int> component(num_vertices(fg));
    int num = connected_components(fg, &component[0]);

    /// Create vector of all components' areas
    std::vector<int>::size_type i;
    comp_area.resize(num);

    auto vs = vertices(fg);
    for (auto vit = vs.first; vit != vs.second; ++vit) {
        comp_area[component[*vit]] += fg[*vit].area;
    }
    /// Normalize area-vector
    double total_area = 0.0;
    for (int i = 0; i < comp_area.size(); i++) {
        total_area += comp_area[i];
    }

    for (int i = 0; i < comp_area.size(); i++) {
        comp_area[i] = comp_area[i] / total_area;
    }

    /// Return area-vector
    return comp_area;
}


class DFSVisitorFG : public boost::default_dfs_visitor {
public:
    DFSVisitorFG() : traversal_edges(
            new std::vector<typename boost::graph_traits<WhitelistFilteredFaceGraph>::edge_descriptor>),
                     traversal_vertices(
                             new std::vector<typename boost::graph_traits<WhitelistFilteredFaceGraph>::vertex_descriptor>) {}

    void discover_vertex(typename boost::graph_traits<WhitelistFilteredFaceGraph>::vertex_descriptor v,
                         const WhitelistFilteredFaceGraph &g) const {
        traversal_vertices->push_back(v);
        return;
    }

    void examine_edge(typename boost::graph_traits<WhitelistFilteredFaceGraph>::edge_descriptor e,
                      const WhitelistFilteredFaceGraph &g) const {
        traversal_edges->push_back(e);
        return;
    }

    typename std::vector<typename boost::graph_traits<WhitelistFilteredFaceGraph>::edge_descriptor> &
    GetEdgeTraversal() const { return *traversal_edges; }

    typename std::vector<typename boost::graph_traits<WhitelistFilteredFaceGraph>::vertex_descriptor> &
    GetVertexTraversal() const { return *traversal_vertices; }

private:
    boost::shared_ptr<std::vector<typename boost::graph_traits<WhitelistFilteredFaceGraph>::edge_descriptor>> traversal_edges;
    boost::shared_ptr<std::vector<typename boost::graph_traits<WhitelistFilteredFaceGraph>::vertex_descriptor>> traversal_vertices;
};


template<typename typeT>
class CyclesVisitor : public boost::default_dfs_visitor {
public:
    CyclesVisitor() : traversal1(new std::vector<typename boost::graph_traits<typeT>::edge_descriptor>),
                      vv(new std::vector<typename boost::graph_traits<typeT>::vertex_descriptor>()),
                      start_end_vertex(new std::vector<typename boost::graph_traits<typeT>::vertex_descriptor>()),
                      has_cycle(new bool(false)),
                      parent(new std::map<typename boost::graph_traits<typeT>::vertex_descriptor,
                              typename boost::graph_traits<typeT>::vertex_descriptor>) {}

    void discover_vertex(typename boost::graph_traits<typeT>::vertex_descriptor v, const typeT &g) const {
        vv->push_back(v);
        return;
    }

    void tree_edge(typename boost::graph_traits<typeT>::edge_descriptor e, const typeT &g) const {
        (*traversal1).push_back(e);
        (*parent)[boost::target(e, g)] = boost::source(e, g);
        return;
    }

    void back_edge(typename boost::graph_traits<typeT>::edge_descriptor e, const typeT &g) const {
        start_end_vertex->push_back(boost::target(e, g));
        (*traversal1).push_back(e);
        *has_cycle = true;
        return;
    }

    std::vector<typename boost::graph_traits<typeT>::vertex_descriptor> &GetVector() const { return *vv; }

    typename std::vector<typename boost::graph_traits<typeT>::vertex_descriptor> &
    GetStart_end_vertex() const { return *start_end_vertex; }

    typename std::vector<typename boost::graph_traits<typeT>::edge_descriptor> &
    GetTraversal() const { return *traversal1; }

    bool &GetHasCycle() { return *has_cycle; }

    std::map<typename boost::graph_traits<typeT>::vertex_descriptor,
            typename boost::graph_traits<typeT>::vertex_descriptor> &GetPredecessors() const { return *parent; };

private:
    boost::shared_ptr<std::vector<typename boost::graph_traits<typeT>::edge_descriptor>> traversal1;
    boost::shared_ptr<std::vector<typename boost::graph_traits<typeT>::vertex_descriptor> > vv;
    boost::shared_ptr<std::vector<typename boost::graph_traits<typeT>::vertex_descriptor> > start_end_vertex;
    boost::shared_ptr<bool> has_cycle;
    boost::shared_ptr<std::map<typename boost::graph_traits<typeT>::vertex_descriptor,
            typename boost::graph_traits<typeT>::vertex_descriptor>> parent;
};

/**
 *
 * @param fg
 * @return
 */
std::vector<std::vector<int>> find_cycles(WhitelistFilteredFaceGraph &fg) {
    std::vector<std::vector<int>> all_cycles;
    typedef typename boost::graph_traits<WhitelistFilteredFaceGraph>::edge_descriptor edge_id;
    typedef typename boost::graph_traits<WhitelistFilteredFaceGraph>::vertex_descriptor vertex_id;

    vertex_id src;
    std::set<WhitelistFilteredFaceGraph::edge_descriptor> whitelist_edges;
    std::set<WhitelistFilteredFaceGraph::vertex_descriptor> whitelist_vertices;

    auto es = boost::edges(fg);
    for (auto eit = es.first; eit != es.second; ++eit) {
        whitelist_edges.insert(*eit);
    }

    auto vs = vertices(fg);
    for (auto vit = vs.first; vit != vs.second; ++vit) {
        whitelist_vertices.insert(*vit);
    }

    src = (*whitelist_vertices.begin());
    //std::cout << src << std::endl;
    using namespace boost;
    std::vector<default_color_type> vertex_color(num_vertices(fg));
    std::map<edge_id, default_color_type> edge_color;

    auto idmap = get(vertex_index, fg);
    auto vcmap = make_iterator_property_map(vertex_color.begin(), idmap);
    auto ecmap = make_assoc_property_map(edge_color);

    CyclesVisitor<WhitelistFilteredFaceGraph> vis;

    std::vector<vertex_id> predecessors(num_vertices(fg));

    undirected_dfs(fg, root_vertex(src).visitor(vis).vertex_color_map(vcmap).edge_color_map(ecmap));

    std::vector<edge_id> cycle_indicating_edges;
    std::vector<std::set<edge_id>> cycles;

    /// Build spanning tree
    FaceGraph spanning_tree(0);
    std::map<vertex_id, vertex_id> g2st;
    //auto vs = vertices(fg);
    for (auto vit = vs.first; vit != vs.second; ++vit) {
        g2st[*vit] = add_vertex(spanning_tree);
    }
    /// Add edges to spanning_tree according to parent-variable
    auto testVar = vis.GetTraversal();
    std::map<vertex_id, vertex_id> pred = vis.GetPredecessors();
    for (auto map_it : pred) {
        add_edge(g2st[map_it.first], g2st[map_it.second], spanning_tree);
        //std::cout << "Added edge " << map_it.first << " - " << map_it.second << std::endl;
    }

    /// Check for each edge if it is still in the spanning tree
    for (auto eit : testVar) {
        if (edge(g2st[boost::source(eit, fg)], g2st[boost::target(eit, fg)], spanning_tree).second) {
            //std::cout << "Found edge\n";
        } else {
            /// Found cycle! Backtrack!
            cycle_indicating_edges.push_back(eit);
        }
    }

    for (auto edge1 : cycle_indicating_edges) {
        std::vector<edge_id> tmp_path;
        std::set<vertex_id> tmp_vertices;
        tmp_path.push_back(edge1);
        vertex_id v = source(edge1, fg);
        vertex_id stop_v = target(edge1, fg);
        tmp_path.push_back(edge1);
        for (auto u = pred[v]; v != stop_v && pred[v] != 0; v = u, u = pred[v]) {
            std::pair<edge_id, bool> edge_pair = boost::edge(v, u, fg);
            tmp_path.push_back(edge_pair.first);
        }
        /// Connect last vertex with first one!
        tmp_path.back() = edge(target(edge1, fg), source(tmp_path.back(), fg), fg).first;
        std::set<edge_id> tmp_edges = std::set<edge_id>(tmp_path.begin(), tmp_path.end());
        for (edge_id edge_it : tmp_edges) {
            tmp_vertices.insert(source(edge_it, fg));
            tmp_vertices.insert(target(edge_it, fg));
        }
        std::vector<int> cycle(tmp_vertices.begin(), tmp_vertices.end());
        all_cycles.push_back(cycle);
    }
    return all_cycles;
}

/**
 *
 * @param g
 * @param candidate_faces
 * @param F
 * @param VF
 * @param k
 * @param strip1
 * @param strip2
 * @param TT
 */
void get_strip(FaceGraph &g, std::vector<int> &candidate_faces, Eigen::MatrixXi &F, std::vector<std::vector<int>> &VF,
               int k, std::vector<int> &strip1, std::vector<int> &strip2, Eigen::MatrixXi &TT) {
    std::set<FaceGraph::vertex_descriptor> face_basis(candidate_faces.begin(), candidate_faces.end());
    std::set<FaceGraph::vertex_descriptor> next_face_basis;
    std::set<FaceGraph::vertex_descriptor> last_strips_faces(candidate_faces.begin(), candidate_faces.end());

    std::set<FaceGraph::edge_descriptor> edge_basis;
    for (int i = 0; i < candidate_faces.size(); i++) {
        if (i > 0)
            edge_basis.insert(boost::edge(candidate_faces[i], candidate_faces[i - 1], g).first);
        else
            edge_basis.insert(boost::edge(candidate_faces[i], candidate_faces.back(), g).first);
    }

    WhitelistFilteredFaceGraph fg(g, edge_in_whitelist(g, face_basis, edge_basis), vertex_in_whitelist(g, face_basis));

    for (int index_k = 1; index_k <= k; index_k++) {
        /// Compute First Left and Right Extended Face Strip:
        /// Add all faces of the original strip and all adjacent faces to a new set

        ///For all three corners/vertices of any given face
        for (int l = 0; l <= 2; l++) {
            /// For all vertices of the current face
            for (auto ele : face_basis) {
                auto vertex_faces = VF[F(ele, l)];
                std::set<int> vertex_face_set(vertex_faces.begin(), vertex_faces.end());
                std::set<int> face_adj_set;
                std::vector<int> intersect;
                for (auto face: vertex_faces) {
                    next_face_basis.insert(face);
                    face_adj_set.clear();
                    face_adj_set.insert(TT(face, 0));
                    face_adj_set.insert(TT(face, 1));
                    face_adj_set.insert(TT(face, 2));
                    intersect.clear();
                    set_intersection(face_adj_set.begin(), face_adj_set.end(), vertex_face_set.begin(),
                                     vertex_face_set.end(),
                                     std::inserter(intersect, intersect.begin()));
                    for (auto neighbor : intersect) {
                        if (std::find(face_basis.begin(), face_basis.end(), neighbor) == face_basis.end() &&
                            std::find(face_basis.begin(), face_basis.end(), face) == face_basis.end())
                            edge_basis.insert(boost::edge(neighbor, face, g).first);
                    }
                }
            }
        }

        face_basis = next_face_basis;

        /// Remove the faces of the original isoline from this set
        for (auto ele : last_strips_faces) {
            face_basis.erase(ele);
        }

        /// Remove redundant faces
        auto vs = vertices(fg);
        bool is_clean = true;
        do {
            vs = vertices(fg);
            is_clean = true;
            for (auto vit = vs.first; vit != vs.second; ++vit) {
                if (boost::degree(*vit, fg) == 1) {
                    face_basis.erase(*vit);
                    is_clean = false;
                }
            }
        } while (is_clean == false);

        /// Remove redundant faces
        vs = vertices(fg);
        bool fault_thrown = false;
        for (auto vit = vs.first; vit != vs.second; ++vit) {
            if (boost::degree(*vit, fg) != 2 && !fault_thrown) {
                fault_thrown = true;

                //std::ofstream myfile;
                //myfile.open ("debug.dot");
                //boost::write_graphviz(myfile, fg);

                /// Print original
                //std::cout << "Original: ";
                //for (auto vertex : face_basis)
                //  std::cout << vertex << " ";
                //std::cout << std::endl;

                //TODO: Get largest cycle for each of the two connected components!
                int num_cycles = 1;
                auto all_cycles = find_cycles(fg);
                for (auto cycle : all_cycles) {
                    //std::cout << "Cycle # " << num_cycles << ": ";
                    num_cycles++;
                    //for (auto vertex : cycle)
                    //std::cout << vertex << " ";
                    //std::cout << std::endl;
                }
                //std::cerr << ">> Warning: k-step strip contains non-cycle element!\n";
            }
        }

        /// Remove the faces of the original isoline from this set
        last_strips_faces.insert(next_face_basis.begin(), next_face_basis.end());
    }

    std::vector<std::vector<int>> contours_unsorted;

    DFSVisitorFG vis;
    boost::depth_first_search(fg, visitor(vis));
    auto vertex_traversal = vis.GetVertexTraversal();

    /// Create Contours
    int last_vertex = 0;
    contours_unsorted.clear();
    std::vector<int> contour;
    for (auto vertex_it : vertex_traversal) {
        if (!boost::edge(vertex_it, last_vertex, fg).second) {
            contours_unsorted.push_back(contour);
            contour.clear();
        }
        contour.push_back(vertex_it);
        last_vertex = vertex_it;
    }
    contours_unsorted.push_back(contour);
    contours_unsorted.erase(contours_unsorted.begin());

    strip1 = contours_unsorted[0];
    strip2 = contours_unsorted[1];
}

/**
 *
 * @param strip
 * @param V
 * @param F
 * @return
 */
double get_strip_length(std::vector<int> &strip, Eigen::MatrixXd &V, Eigen::MatrixXi &F) {
    double curr_length = 0.0;
    if (strip.size() > 0) {
        int last_face = strip.back();
        int next_face = strip.front();
        int curr_face;
        for (int k = 0; k < strip.size(); k++) {
            if (k != 0)
                last_face = strip[k - 1];
            if (k != strip.size() - 1)
                next_face = strip[k + 1];
            else
                next_face = strip.front();
            curr_face = strip[k];

            std::set<int> face_vertex_set1;
            std::set<int> face_vertex_set2;
            std::set<int> face_vertex_set3;
            face_vertex_set1.insert(F(curr_face, 0));
            face_vertex_set1.insert(F(curr_face, 1));
            face_vertex_set1.insert(F(curr_face, 2));
            face_vertex_set2.insert(F(last_face, 0));
            face_vertex_set2.insert(F(last_face, 1));
            face_vertex_set2.insert(F(last_face, 2));
            face_vertex_set3.insert(F(next_face, 0));
            face_vertex_set3.insert(F(next_face, 1));
            face_vertex_set3.insert(F(next_face, 2));
            std::set<int> intersect1;
            set_intersection(face_vertex_set1.begin(), face_vertex_set1.end(), face_vertex_set2.begin(),
                             face_vertex_set2.end(),
                             std::inserter(intersect1, intersect1.begin()));
            std::set<int> intersect2;
            set_intersection(face_vertex_set1.begin(), face_vertex_set1.end(), face_vertex_set3.begin(),
                             face_vertex_set3.end(),
                             std::inserter(intersect2, intersect2.begin()));
            std::vector<int> edge1(intersect1.begin(), intersect1.end());
            std::vector<int> edge2(intersect2.begin(), intersect2.end());
            if (edge1.size() == 2 && edge2.size() == 2) {
                Eigen::MatrixXd edge_center1 = (V.row(edge1[0]) + V.row(edge1[1])) / 2.0;
                Eigen::MatrixXd edge_center2 = (V.row(edge2[0]) + V.row(edge2[1])) / 2.0;
                curr_length += (edge_center1 - edge_center2).norm();
            } else {
                //std::cout << "Strip: ";
                //for (int i = 0; i < strip.size(); i++){
                //  std::cout << strip[i] << " ";
                //}
                //std::cout << std::endl;
                //std::cerr << "Last Face: " << last_face << "\n";
                //std::cerr << "Current Face: " << curr_face << "\n";
                //std::cerr << "Next Face: " << next_face << "\n";
                //std::cerr << ">> Warning: k-strip not connected!\n";
            }
        }
    }

    return curr_length;
}

/**
 *
 * @param candidate_length
 * @param candidate_vertices
 * @param candidate_svs
 * @param candidate_faces
 * @param F
 * @param V
 */
void compute_candidate_svs_by_geodesics(std::vector<double> &candidate_length,
                                        std::vector<std::vector<Eigen::MatrixXd>> &candidate_vertices,
                                        std::vector<double> &candidate_svs,
                                        std::vector<std::vector<int>> &candidate_faces,
                                        Eigen::MatrixXi &F, Eigen::MatrixXd &V) {

    /// For each Candidate:
    /*for (int i = 0; i < candidate_length.size(); i++){
      double sdf = 0.0;
      Eigen::MatrixXd last_vertex;
      for (int j = 0; j < candidate_faces[i].size(); j++){
        /// If first element: create edge to very last vertex
        if (j == 0)
          last_vertex = candidate_vertices[i].back();
        /// Multiply edgelength by faces magnitude
        auto p1 = candidate_vertices[i][j];
        auto p2 = last_vertex;
        sdf += ((p1 - p2).norm() * GU_mat(candidate_faces[i][j], 0));
        last_vertex = candidate_vertices[i][j];
      }
      sdf = sdf / candidate_length[i];
      candidate_svs[i] = sdf;
    }*/

}

/**
 *
 * @param candidate_length
 * @param candidate_vertices
 * @param candidate_svs
 * @param candidate_faces
 * @param F
 * @param V
 */
void compute_candidate_svs_by_sdf(std::vector<double> &candidate_length,
                                  std::vector<std::vector<Eigen::MatrixXd>> &candidate_vertices,
                                  std::vector<double> &candidate_svs,
                                  std::vector<std::vector<int>> &candidate_faces,
                                  Eigen::MatrixXi &F, Eigen::MatrixXd &V) {

    Polyhedron mesh;

    igl::copyleft::cgal::mesh_to_polyhedron(V, F, mesh);
    // create a property-map
    typedef std::map<Polyhedron::Facet_const_handle, double> Facet_double_map;
    Facet_double_map internal_map;
    boost::associative_property_map<Facet_double_map> sdf_property_map(internal_map);
    // compute SDF values
    //std::pair<double, double> min_max_sdf = CGAL::sdf_values(mesh, sdf_property_map);
    // It is possible to compute the raw SDF values and post-process them using
    // the following lines:
    const std::size_t number_of_rays = 25;  // cast 25 rays per facet
    const double cone_angle = 2.0 / 3.0 * CGAL_PI; // set cone opening-angle
    CGAL::sdf_values(mesh, sdf_property_map, cone_angle, number_of_rays, false);
    std::pair<double, double> min_max_sdf = CGAL::sdf_values_postprocessing(mesh, sdf_property_map);

    Eigen::MatrixXd L = Eigen::MatrixXd::Zero(F.rows(), 1);

    int index = 0;
    for (Polyhedron::Facet_const_iterator facet_it = mesh.facets_begin();
         facet_it != mesh.facets_end(); ++facet_it) {
        L(index, 0) = sdf_property_map[facet_it];
        index++;
    }

    /// Convert Face SDF-attribute back to vertex SDF-attribute
    std::vector<std::vector<int>> VF, VI;
    igl::vertex_triangle_adjacency(F.maxCoeff() + 1, F, VF, VI);
    Eigen::MatrixXd vertex_sdf = Eigen::MatrixXd::Zero(V.rows(), 1);
    for (int i = 0; i < V.rows(); i++) {
        double sdf = 0.0;
        int neighbors = 0;
        for (int j = 0; j < VF[i].size(); j++) {
            sdf += L(VF[i][j], 0);
            neighbors++;
        }
        sdf = sdf / (double) neighbors;
        vertex_sdf(i, 0) = sdf;
    }

    Eigen::SparseMatrix<double> gradient;
    igl::grad(V, F, gradient, false);

    Eigen::MatrixXd GU = Eigen::Map<const Eigen::MatrixXd>((gradient * vertex_sdf).eval().data(), F.rows(), 3);
    const Eigen::VectorXd GU_mag = GU.rowwise().norm();
    Eigen::MatrixXd GU_mat = Eigen::MatrixXd(GU_mag.size(), 1);
    GU_mat << GU_mag;

    candidate_svs.resize(candidate_length.size());

    /*
     * for (int i = 0; i < candidate_vertices.size(); i++){
      double tmp_score = 0.0;
      Eigen::MatrixXd last_vertex;
      for (int j = 0; j < candidate_vertices[i].size(); j++){
        /// If first element: create edge to very last vertex
        if (j == 0)
          last_vertex = candidate_vertices[i].back();
        /// Multiply edgelength by faces magnitude
        auto p1 = candidate_vertices[i][j];
        auto p2 = last_vertex;
        tmp_score = tmp_score + ((p1 - p2).norm() * g_hat(candidate_face_ids[i][j],0));
        last_vertex = candidate_vertices[i][j];
      }
      candidate_gs[i] = tmp_score / candidate_length[i];
    }
     */

    /// For each Candidate:
    for (int i = 0; i < candidate_length.size(); i++) {
        double sdf = 0.0;
        Eigen::MatrixXd last_vertex;
        for (int j = 0; j < candidate_faces[i].size(); j++) {
            /// If first element: create edge to very last vertex
            if (j == 0)
                last_vertex = candidate_vertices[i].back();
            /// Multiply edgelength by faces magnitude
            auto p1 = candidate_vertices[i][j];
            auto p2 = last_vertex;
            sdf += ((p1 - p2).norm() * GU_mat(candidate_faces[i][j], 0));
            last_vertex = candidate_vertices[i][j];
        }
        sdf = sdf / candidate_length[i];
        candidate_svs[i] = sdf;
    }

    /// For each Candidate:
    /*for (int i = 0; i < candidate_length.size(); i++){
      double sdf = 0.0;
      for (int j = 0; j < candidate_faces[i].size(); j++){
        sdf += GU_mat(candidate_faces[i][j], 0);
      }
      sdf = sdf / candidate_length[i];
      candidate_svs[i] = sdf;
    }*/
}

/**
 *
 * @param candidate_length
 * @param candidate_svs
 * @param candidate_faces
 * @param F
 * @param V
 */
void compute_candidate_svs_new(std::vector<double> &candidate_length,
                               std::vector<double> &candidate_svs,
                               std::vector<std::vector<int>> &candidate_faces,
                               Eigen::MatrixXi &F, Eigen::MatrixXd &V) {
    /// Create Graph
    FaceGraph g(F.rows());

    Eigen::MatrixXi TT;
    std::vector<std::vector<int>> VF;
    std::vector<std::vector<int>> VI;
    igl::triangle_triangle_adjacency(F, TT);
    igl::vertex_triangle_adjacency(F.maxCoeff() + 1, F, VF, VI);

    /// Add edges between faces
    for (int i = 0; i < F.rows(); i++) {
        try {
            if (TT(i, 0) >= 0 && TT(i, 0) < F.rows()) {
                boost::add_edge(i, TT(i, 0), g);
            }
            if (TT(i, 1) >= 0 && TT(i, 1) < F.rows()) {
                boost::add_edge(i, TT(i, 1), g);
            }
            if (TT(i, 2) >= 0 && TT(i, 2) < F.rows()) {
                boost::add_edge(i, TT(i, 2), g);
            }
        } catch (...) {

        }
    }

    /// Compute longest Candidate
    double longest_candidate = 0.0;
    for (int i = 0; i < candidate_length.size(); i++) {
        longest_candidate = std::max(longest_candidate, candidate_length[i]);
    }
    candidate_svs.resize(candidate_length.size());

    /// For each Candidate:
    for (int i = 0; i < candidate_length.size(); i++) {
        std::vector<double> delta_ij;
        for (int j = 1; j <= 5; j++) {
            std::vector<int> strip1, strip2;
            get_strip(g, candidate_faces[i], F, VF, j, strip1, strip2, TT);
            double strip1_length = get_strip_length(strip1, V, F);
            double strip2_length = get_strip_length(strip2, V, F);
            if (strip1_length == 0.0)
                strip1_length = longest_candidate;
            if (strip2_length == 0.0)
                strip2_length = longest_candidate;
            delta_ij.push_back(((strip1_length + strip2_length - (candidate_length[i] * 2)) / longest_candidate));
            candidate_svs[i] += exp(-((j * j) / 2 * (2 * 2))) * delta_ij.back();
        }
    }
}

/**
 *
 * @param v1
 * @param v2
 * @param in_degree
 * @return
 */

double _getAngle3D(const Eigen::Vector3d &v1, const Eigen::Vector3d &v2, const bool in_degree) {
    // Compute the actual angle
    double rad = v1.normalized().dot(v2.normalized());
    if (rad < -1.0)
        rad = -1.0;
    else if (rad > 1.0)
        rad = 1.0;
    return (in_degree ? acos(rad) * 180.0 / M_PI : acos(rad));
}

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
_connIsConvex(const Eigen::Vector3d &source_centroid, const Eigen::Vector3d &target_centroid,
             const Eigen::Vector3d &source_normal, const Eigen::Vector3d &target_normal, double &normal_angle) {
    bool is_convex = true;
    bool is_smooth = true;

    normal_angle = _getAngle3D(source_normal, target_normal, true);
    //  Geometric comparisonsa
    Eigen::Vector3d vec_t_to_s, vec_s_to_t;

    vec_t_to_s = source_centroid - target_centroid;
    vec_s_to_t = -vec_t_to_s;

    Eigen::Vector3d ncross;
    ncross = source_normal.cross(target_normal);

    // vec_t_to_s is the reference direction for angle measurements
    // Convexity Criterion: Check if connection of patches is convex. If this is the case the two supervoxels should be merged.
    if ((_getAngle3D(vec_t_to_s, source_normal, true) - _getAngle3D(vec_t_to_s, target_normal, true)) <= 0) {
        normal_angle = -normal_angle;
        is_convex &= true;  // connection convex
    } else {
        is_convex &= (normal_angle < 0.0);  // concave connections will be accepted  if difference of normals is small
    }
    return (is_convex && is_smooth);
}

/**
 *
 * @param V
 * @param N
 * @param F
 * @param dblA
 * @param IF
 * @param E
 * @param candidate_face_ids
 * @param candidate_score
 * @param applied_isolines
 */
void valid_cuts(Eigen::MatrixXd &V, Eigen::MatrixXd &N, Eigen::MatrixXi &F, Eigen::MatrixXd &dblA, Eigen::MatrixXi &IF,
                Eigen::MatrixXi &E,
                std::vector<std::vector<int>> &candidate_face_ids, std::vector<double> &candidate_score,
                std::vector<int> &applied_isolines) {
    /// Sort Isoline Candidates
    std::vector<std::pair<double, int>> sorted_scores;
    for (int i = 0; i < candidate_score.size(); i++) {
        std::pair<double, int> P = std::make_pair(candidate_score[i], i);
        sorted_scores.push_back(P);
    }

    std::sort(sorted_scores.rbegin(), sorted_scores.rend());

    /// First create FaceGraph
    std::set<boost::graph_traits<FaceGraph>::vertex_descriptor> vertex_blacklist;
    std::set<boost::graph_traits<FaceGraph>::edge_descriptor> edge_blacklist;
    FaceGraph g(F.rows());

    /// Add faces and their properties
    for (int i = 0; i < F.rows(); i++)
        g[i].area = dblA(i, 0);

    /// Add edges between faces
    for (int i = 0; i < IF.rows(); i++)
        boost::add_edge(IF(i, 0), IF(i, 1), g);

    int parts_before_cut = 0;

    std::set<int> faces_adjacent_to_cut;
    std::set<int> faces_adjacent_to_cut_tmp;
    std::vector<std::vector<int>> VF;
    std::vector<std::vector<int>> VI;
    igl::vertex_triangle_adjacency(F.maxCoeff() + 1, F, VF, VI);

    double last_score = sorted_scores[0].first;

    /// Now iterate over the isoline candidates
    for (int i = 0; i < sorted_scores.size() && !(sorted_scores[i].first < 0.1 * last_score); i++) {
        //std::cout << "Current Candidate score: " << sorted_scores[i].first << std::endl;
        auto parts = split_mesh(F, E, IF, dblA, candidate_face_ids[sorted_scores[i].second], g, vertex_blacklist,
                                edge_blacklist);
        bool valid_cut = true;
        for (int j = 0; j < parts.size(); j++) {
            if (parts[j] <= 0.01)
                valid_cut = false;
            //std::cout << "Part " << j << ": " << parts[j] << std::endl;
        }

        int num_concave_edges_cut = 0;
        int num_total_edges_cut = 0;
        /// Sanity Check
        if (valid_cut) {
            auto faces = candidate_face_ids[sorted_scores[i].second];
            for (int j = 0; j < faces.size(); j++) {
                num_total_edges_cut++;
                /// Handle extreme cases (first and last)
                int curr_face = faces[j];
                int next_face;
                if (j == faces.size() - 1) {
                    next_face = faces[0];
                } else {
                    next_face = faces[j + 1];
                }
                /// The two faces share an adge/a vertex pair, check it's concavity
                std::vector<int> v1;
                v1.push_back(F(curr_face, 0));
                v1.push_back(F(curr_face, 1));
                v1.push_back(F(curr_face, 2));
                std::vector<int> v2;
                v2.push_back(F(next_face, 0));
                v2.push_back(F(next_face, 1));
                v2.push_back(F(next_face, 2));
                std::sort(v1.begin(), v1.end());
                std::sort(v2.begin(), v2.end());
                std::vector<int> v_intersection;
                std::set_intersection(v1.begin(), v1.end(),
                                      v2.begin(), v2.end(),
                                      std::back_inserter(v_intersection));
                if (v_intersection.size() == 2) {
                    double normal_angle;
                    _connIsConvex(V.row(v_intersection[0]), V.row(v_intersection[1]), N.row(v_intersection[0]),
                                 N.row(v_intersection[1]), normal_angle);
                    //std::cout << normal_angle << std::endl;
                    if (normal_angle > 8.0) {
                        num_concave_edges_cut++;
                    }
                }
            }

            if ((float) num_concave_edges_cut / (float) num_total_edges_cut < 0.16) {
                valid_cut = false;
                //std::cout << ">> Invalidated Contour!\n";
            }
        }

        faces_adjacent_to_cut_tmp = faces_adjacent_to_cut;
        for (int j = 0; j < candidate_face_ids[sorted_scores[i].second].size(); j++) {
            faces_adjacent_to_cut_tmp.insert(candidate_face_ids[sorted_scores[i].second][j]);
        }
        if (valid_cut && faces_adjacent_to_cut_tmp.size() > faces_adjacent_to_cut.size()) {
            //std::cout << "# Concave Edges: " << (float) num_concave_edges_cut  << std::endl;
            //std::cout << "# Total Edges: " << (float) num_total_edges_cut << std::endl;
            //std::cout << "Contour Score: " << (float) num_concave_edges_cut / (float) num_total_edges_cut << std::endl;
            //last_score = sorted_scores[i].first;
            //std::cout << "Valid!\n";
            for (int j = 0; j < candidate_face_ids[sorted_scores[i].second].size(); j++) {
                faces_adjacent_to_cut.insert(candidate_face_ids[sorted_scores[i].second][j]);
                auto vertex_faces = VF[F(candidate_face_ids[sorted_scores[i].second][j], 0)];
                for (auto face: vertex_faces) {
                    faces_adjacent_to_cut.insert(face);
                }
                vertex_faces = VF[F(candidate_face_ids[sorted_scores[i].second][j], 1)];
                for (auto face: vertex_faces) {
                    faces_adjacent_to_cut.insert(face);
                }
                vertex_faces = VF[F(candidate_face_ids[sorted_scores[i].second][j], 2)];
                for (auto face: vertex_faces) {
                    faces_adjacent_to_cut.insert(face);
                }
            }
            /// Add elements to vertex blacklist
            for (int j = 0; j < candidate_face_ids[sorted_scores[i].second].size(); j++) {
                vertex_blacklist.insert(candidate_face_ids[sorted_scores[i].second][j]);
            }
            /// Add elements to edge_blacklist
            for (int j = 0; j < IF.rows(); j++) {
                if ((std::find(candidate_face_ids[sorted_scores[i].second].begin(),
                               candidate_face_ids[sorted_scores[i].second].end(), IF(j, 0)) !=
                     candidate_face_ids[sorted_scores[i].second].end() ||
                     std::find(candidate_face_ids[sorted_scores[i].second].begin(),
                               candidate_face_ids[sorted_scores[i].second].end(), IF(j, 1)) !=
                     candidate_face_ids[sorted_scores[i].second].end())) {
                    auto tmp_edge = boost::edge(IF(j, 0), IF(j, 1), g).first;
                    edge_blacklist.insert(tmp_edge);
                }
            }
            applied_isolines.push_back(sorted_scores[i].second);
            parts_before_cut = parts.size();
        }
    }
}