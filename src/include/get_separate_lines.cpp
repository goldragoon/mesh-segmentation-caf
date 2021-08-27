#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/format.hpp>
#include "get_separate_lines.h"

struct VertexProperties {
    VertexProperties() {};
};

struct EdgeProperties {
    int face;
    int isoline_id;

    EdgeProperties() : face(-1), isoline_id(-1) {};
};

typedef boost::adjacency_list<boost::setS, boost::vecS, boost::undirectedS, VertexProperties, EdgeProperties> Graph;

class DFSVisitor : public boost::default_dfs_visitor {
public:
    DFSVisitor() : traversal_edges(new std::vector<typename boost::graph_traits<Graph>::edge_descriptor>),
                   traversal_vertices(new std::vector<typename boost::graph_traits<Graph>::vertex_descriptor>) {}

    void discover_vertex(typename boost::graph_traits<Graph>::vertex_descriptor v, const Graph &g) const {
        traversal_vertices->push_back(v);
        return;
    }

    void examine_edge(typename boost::graph_traits<Graph>::edge_descriptor e, const Graph &g) const {
        traversal_edges->push_back(e);
        return;
    }

    typename std::vector<typename boost::graph_traits<Graph>::edge_descriptor> &
    GetEdgeTraversal() const { return *traversal_edges; }

    typename std::vector<typename boost::graph_traits<Graph>::vertex_descriptor> &
    GetVertexTraversal() const { return *traversal_vertices; }

private:
    boost::shared_ptr<std::vector<typename boost::graph_traits<Graph>::edge_descriptor>> traversal_edges;
    boost::shared_ptr<std::vector<typename boost::graph_traits<Graph>::vertex_descriptor>> traversal_vertices;
};

/**
 *
 * @param E
 * @param F
 * @param segmentation_lines
 * @param vertex_labels
 */
void color_mesh_by_isolines(Eigen::MatrixXi &E, Eigen::MatrixXi &F, std::vector<std::vector<int>> &segmentation_lines,
                            Eigen::MatrixXd &vertex_labels) {
    /// 1. Create Mesh graph (vertices are actual vertices)
    Graph g(F.maxCoeff() + 1);
    for (int i = 0; i < E.rows(); i++) {
        /// Add edge
        boost::add_edge(E(i, 0), E(i, 1), g);
    }
    /// 2. For each segmentation line:
    for (int i = 0; i < segmentation_lines.size(); i++) {
        ///   2.1 For each triplet of adjacent faces:
        int last_face, curr_face, next_face;
        for (int j = 0; j < segmentation_lines[i].size(); j++) {
            ///     2.1.1 Remove the two edges in the graph given by the common vertex of all three faces
            if (j > 0)
                last_face = segmentation_lines[i][j - 1];
            else
                last_face = segmentation_lines[i].back();

            curr_face = segmentation_lines[i][j];

            if (j < segmentation_lines[i].size() - 1)
                next_face = segmentation_lines[i][j + 1];
            else
                next_face = segmentation_lines[i].front();

            /// Get common vertex
            std::vector<int> common_vertex;
            std::set<int> last_face_vertices = {F(last_face, 0), F(last_face, 1), F(last_face, 2)};
            std::set<int> curr_face_vertices = {F(curr_face, 0), F(curr_face, 1), F(curr_face, 2)};
            std::set<int> next_face_vertices = {F(next_face, 0), F(next_face, 1), F(next_face, 2)};

            std::set_intersection(last_face_vertices.begin(), last_face_vertices.end(), next_face_vertices.begin(),
                                  next_face_vertices.end(), std::back_inserter(common_vertex));

            /// Pick the two other vertices of curr_face
            if (common_vertex.size() > 0) {
                curr_face_vertices.erase(common_vertex[0]);

                for (auto vertex : curr_face_vertices) {
                    boost::remove_edge(vertex, common_vertex[0], g);
                }
            }
        }
    }

    /// Now run connected components
    vertex_labels = Eigen::MatrixXd::Zero(F.maxCoeff() + 1, 1);
    std::vector<int> component(num_vertices(g));
    int num = boost::connected_components(g, &component[0]);

    std::vector<int>::size_type i;
    //cout << "Total number of components: " << num << endl;
    for (i = 0; i != component.size(); ++i)
        vertex_labels(i, 0) = component[i];
    //cout << "Vertex " << i <<" is in component " << component[i] << endl;
    //cout << endl;

}

/*double getAngle3D (const Eigen::Vector3d &v1, const Eigen::Vector3d &v2, const bool in_degree)
{
  // Compute the actual angle
  double rad = v1.normalized ().dot (v2.normalized ());
  if (rad < -1.0)
    rad = -1.0;
  else if (rad >  1.0)
    rad = 1.0;
  return (in_degree ? acos (rad) * 180.0 / M_PI : acos (rad));
}*/

/**
 *
 * @param V
 * @param contour1
 * @param contour2
 * @return
 */
double dist_contour_centers(Eigen::MatrixXd &V, std::vector<int> &contour1, std::vector<int> &contour2) {
    /// Calculate center1
    Eigen::MatrixXd center_contour1 = Eigen::MatrixXd::Zero(1, 3);
    for (int vertex_index : contour1) {
        center_contour1 = center_contour1 + V.row(vertex_index);
    }
    center_contour1 = center_contour1 / contour1.size();

    /// Calculate center2
    Eigen::MatrixXd center_contour2 = Eigen::MatrixXd::Zero(1, 3);
    for (int vertex_index : contour2) {
        center_contour2 = center_contour2 + V.row(vertex_index);
    }
    center_contour2 = center_contour2 / contour2.size();

    return (center_contour1 - center_contour2).norm();
}

/**
 *
 * @param isoE
 * @param isoF
 * @param contours
 * @param contour_faces
 * @param isoV
 * @param FN
 * @param isoI
 * @param contour_id
 */
void get_separate_lines(Eigen::MatrixXi &isoE, std::vector<int> &isoF, std::vector<std::vector<int>> &contours,
                        std::vector<std::vector<int>> &contour_faces, Eigen::MatrixXd &isoV, Eigen::MatrixXd &FN,
                        std::vector<int> &isoI, std::vector<std::vector<int>> &contour_id) {

    /// Create Graph
    Graph g(isoE.maxCoeff() + 1);
    for (int i = 0; i < isoE.rows(); i++) {
        auto curr_edge = boost::add_edge(isoE(i, 0), isoE(i, 1), g);
        g[curr_edge.first].face = isoF[i];
        g[curr_edge.first].isoline_id = isoI[i];
    }

    /// Count connected components
    std::vector<int> component(num_vertices(g));
    int num = connected_components(g, &component[0]);

    std::vector<int>::size_type i;
    //std::cout << "Total number of components: " << num << std::endl;

    /// Check if there are dead ends
    auto vs = vertices(g);
    for (auto vit = vs.first; vit != vs.second; ++vit) {
        auto neighbors = boost::adjacent_vertices(*vit, g);
        int num_neighbors = 0;
        for (auto nit = neighbors.first; nit != neighbors.second; ++nit) {
            num_neighbors++;
        }

        if ((num_neighbors == 1)) {
            for (auto nit = neighbors.first; nit != neighbors.second; ++nit) {
                if (*vit != *nit) {
                    //std::ofstream myfile;
                    //myfile.open ("debug_dead_end.dot");
                    //boost::write_graphviz(myfile, g);
                    std::cout << "Error: Found dead end! Please tune tolerance of -remove_duplicate_vertices-!" << "\n";
                }
            }
        }
    }

    std::vector<std::vector<int>> contours_unsorted;
    std::vector<std::vector<int>> contour_faces_unsorted;

    DFSVisitor vis;
    boost::depth_first_search(g, visitor(vis));
    auto vertex_traversal = vis.GetVertexTraversal();

    /// Create Contours
    int last_vertex = 0;
    contours_unsorted.clear();
    std::vector<int> contour;
    for (auto vertex_it : vertex_traversal) {
        if (!boost::edge(vertex_it, last_vertex, g).second) {
            contours_unsorted.push_back(contour);
            contour.clear();
        }
        contour.push_back(vertex_it);
        last_vertex = vertex_it;
    }
    contours_unsorted.push_back(contour);
    contours_unsorted.erase(contours_unsorted.begin());

    std::vector<std::vector<int>> contours2;
    for (int i = 0; i < contours_unsorted.size(); i++) {
        if (contours_unsorted[i].size() > 1)
            contours2.push_back(contours_unsorted[i]);
    }
    contours_unsorted = contours2;

    //std::cout << "Number of unsorted contours: " << contours_unsorted.size() << std::endl;

    /// Create Contour Faces
    contour_faces_unsorted.clear();
    contour_id.clear();
    std::vector<int> contour_face;
    std::vector<int> contour_ids;
    for (auto cont : contours_unsorted) {
        contour_face.clear();
        contour_ids.clear();
        auto last_vertex = -1;
        for (auto vertex_it : cont) {
            /// If this is the first vertex, chose the end vertex as previous one to create an edge
            if (last_vertex == -1)
                last_vertex = cont.back();
            contour_face.push_back(g[boost::edge(last_vertex, vertex_it, g).first].face);
            contour_ids.push_back(g[boost::edge(last_vertex, vertex_it, g).first].isoline_id);
            last_vertex = vertex_it;
        }
        contour_faces_unsorted.push_back(contour_face);
        contour_id.push_back(contour_ids);
    }

    contour_faces.resize(contours_unsorted.size());
    contours.resize(contours_unsorted.size());

    //std::cout << "Label B: Number of unsorted contours: " << contours_unsorted.size() << std::endl;

    /// First, get the maximal amount of isolines with the same isoID, such that ana entire array of parallel isoline-paths can be constructed
    int max_id = 0;
    int max_same_id = 0;

    for (int i = 0; i < contours_unsorted.size(); i++) {
        contour_id[i][0] = contour_id[i][0] - 1;
        //std::cout << "Isoline " << i << " has ID " << contour_id[i][0] << std::endl;
    }

    /// Find maximum isoID
    for (int i = 0; i < contours_unsorted.size(); i++) {
        max_id = std::max(max_id, contour_id[i][0]);
    }

    /// Find maximum amount of isolines with the same isoID
    std::vector<std::vector<int>> id_to_isolines(max_id + 1);
    for (int i = 0; i < max_id; i++) {
        int count = 0;
        for (int j = 0; j < contours_unsorted.size(); j++) {
            if (contour_id[j][0] == i) {
                count++;
                id_to_isolines[i].push_back(j);
            }
        }
        //std::cout << "ID " << i << ": " << count << " occurences.\n";
        max_same_id = std::max(max_same_id, count);
    }

    /// Now embedd all isolines into a 2D array: A column can be interpreted as one path
    /// Check if one of the two previous isoline ids had multiple isolines as well and then cross-check the
    /// distances between all of them to append the current isoline to the correct column
    Eigen::MatrixXi isoline_path = Eigen::MatrixXi::Zero(max_id + 1, max_same_id);
    for (int i = 0; i < isoline_path.rows(); i++) {
        for (int j = 0; j < isoline_path.cols(); j++) {
            isoline_path(i, j) = -1;
        }
    }

    //std::cout << "isoline_path: " << isoline_path.rows() << " x " << isoline_path.cols() << std::endl;

    for (int i = 0; i < max_id; i++) {
        if (id_to_isolines[i].size() > 1) {
            /// Multiple isolines for this id
            if (i > 0) {
                /// Not the very first id
                if (id_to_isolines[i - 1].size() > 1) {
                    /// Multiple isolines for previous id
                    std::vector<int> prev_isoline_candidates = id_to_isolines[i - 1];
                    std::vector<int> curr_isoline_candidates = id_to_isolines[i];
                    int curr_index = 0;
                    for (auto prev_isoline: prev_isoline_candidates) {
                        std::vector<double> score;
                        for (auto curr_isoline: curr_isoline_candidates) {
                            score.push_back(dist_contour_centers(isoV, contours_unsorted[prev_isoline],
                                                                 contours_unsorted[curr_isoline]));
                        }
                        /// Find index with smallest index and append current isoline to this col
                        std::vector<double>::iterator result = std::min_element(score.begin(), score.end());
                        int smallest_score_index = std::distance(std::begin(score), result);
                        isoline_path(i, curr_index) = curr_isoline_candidates[smallest_score_index];
                        curr_index++;
                    }
                } else {
                    /// Single isolines for previous id
                    std::vector<double> score;
                    for (auto curr_isoline: id_to_isolines[i]) {
                        score.push_back(dist_contour_centers(isoV, contours_unsorted[curr_isoline],
                                                             contours_unsorted[id_to_isolines[i - 1][0]]));
                    }
                    /// Find index with smallest index and append current isoline to this col
                    std::vector<double>::iterator result = std::min_element(std::begin(score), std::end(score));
                    int smallest_score_index = std::distance(std::begin(score), result);
                    isoline_path(i, 0) = id_to_isolines[i][smallest_score_index];
                    int curr_index = 0;
                    for (auto curr_isoline: id_to_isolines[i]) {
                        if (curr_isoline != id_to_isolines[i][smallest_score_index]) {
                            curr_index++;
                            isoline_path(i, curr_index) = curr_isoline;
                        }
                    }
                }
            } else {
                /// Very first id
                /// Simply embedd all isolines of id 0 however wanted
                for (int j = 0; j < id_to_isolines[i].size(); j++)
                    isoline_path(i, j) = id_to_isolines[i][j];
            }
        } else {
            /// Single isolines for this id
            if (i > 0) {
                /// Not the very first id
                if (id_to_isolines[i - 1].size() > 1) {
                    /// Multiple isolines for previous id
                    std::vector<double> score;
                    for (auto prev_isoline: id_to_isolines[i - 1]) {
                        score.push_back(dist_contour_centers(isoV, contours_unsorted[prev_isoline],
                                                             contours_unsorted[id_to_isolines[i][0]]));
                    }
                    /// Find index with smallest index and append current isoline to this col
                    std::vector<double>::iterator result = std::min_element(std::begin(score), std::end(score));
                    int smallest_score_index = std::distance(std::begin(score), result);
                    isoline_path(i, smallest_score_index) = id_to_isolines[i][0];
                } else {
                    /// Single isoline for previous id
                    isoline_path(i, 0) = id_to_isolines[i][0];
                }
            } else {
                /// Very first id
                /// Simply embed isolines of id 0
                isoline_path(i, 0) = id_to_isolines[i][0];
            }
        }
    }

    //std::cout << isoline_path << std::endl;

    /// The following part has to be rewritten since it currently overwrites isolines
    //for (int i = 0; i < contours_unsorted.size(); i++){
    //  int id = contour_id[i][0] - 1;
    //  contour_faces[id] = contour_faces_unsorted[i];
    //  contours[id] = contours_unsorted[i];
    //}

    int curr_index = 0;
    contour_faces.resize(0);
    contours.resize(0);
    for (int i = 0; i < isoline_path.cols(); i++) {
        for (int j = 0; j < isoline_path.rows(); j++) {
            if (isoline_path(j, i) != -1) {
                contour_faces.push_back(contour_faces_unsorted[isoline_path(j, i)]);
                contours.push_back(contours_unsorted[isoline_path(j, i)]);
                curr_index++;
            }
        }
    }

    //std::cout << "curr_index: " << curr_index << std::endl;

    contours_unsorted.clear();
    contour_faces_unsorted.clear();
    for (int i = 0; i < contours.size(); i++) {
        if (contours[i].size() > 1) {
            contours_unsorted.push_back(contours[i]);
            contour_faces_unsorted.push_back(contour_faces[i]);
        }
    }

    contours = contours_unsorted;
    contour_faces = contour_faces_unsorted;

    //std::cout << "Number of exported contours: " << contours.size() << std::endl;

}
