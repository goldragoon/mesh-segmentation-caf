#include "sdf.hpp"

#define CGAL_EIGEN3_ENABLED

#include <CGAL/Polyhedron_3.h>
#include <CGAL/Polyhedron_items_with_id_3.h>
#include <CGAL/boost/graph/graph_traits_Polyhedron_3.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/extract_mean_curvature_flow_skeleton.h>
#include <CGAL/mesh_segmentation.h>
#include <igl/copyleft/cgal/mesh_to_polyhedron.h>
#include <igl/vertex_triangle_adjacency.h>
#include <igl/point_mesh_squared_distance.h>

typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_3 Point;
typedef CGAL::Polyhedron_3<Kernel, CGAL::Polyhedron_items_with_id_3> Polyhedron;
typedef boost::graph_traits<Polyhedron>::vertex_descriptor vertex_descriptor;
typedef boost::graph_traits<Polyhedron>::halfedge_descriptor halfedge_descriptor;
typedef boost::graph_traits<Polyhedron>::face_descriptor face_descriptor;
typedef CGAL::Mean_curvature_flow_skeletonization<Polyhedron> Skeletonization;
typedef Skeletonization::Skeleton Skeleton;
typedef Skeleton::vertex_descriptor Skeleton_vertex;
typedef Skeleton::edge_descriptor Skeleton_edge;

template<class ValueType>
struct Facet_with_id_pmap
        : public boost::put_get_helper<ValueType &,
                Facet_with_id_pmap<ValueType> > {
    typedef face_descriptor key_type;
    typedef ValueType value_type;
    typedef value_type &reference;
    typedef boost::lvalue_property_map_tag category;

    Facet_with_id_pmap(
            std::vector<ValueType> &internal_vector
    ) : internal_vector(internal_vector) {}

    reference operator[](key_type key) const { return internal_vector[key->id()]; }

private:
    std::vector<ValueType> &internal_vector;
};

/**
 *
 * @param V
 * @param F
 * @param face_sdf
 * @param vertex_sdf
 * @param skeleton_vertices
 * @param SE1
 * @param SE2
 * @param skeleton_diam
 */
void new_sdf(Eigen::MatrixXd &V, Eigen::MatrixXi &F, Eigen::MatrixXd &face_sdf, Eigen::MatrixXd &vertex_sdf,
             Eigen::MatrixXd &skeleton_vertices,
             Eigen::MatrixXd &SE1, Eigen::MatrixXd &SE2, Eigen::MatrixXd &skeleton_diam) {
    Polyhedron tmesh;
    igl::copyleft::cgal::mesh_to_polyhedron(V, F, tmesh);
    // extract the skeleton
    Skeleton skeleton;
    //CGAL::Mean_curvature_flow_skeletonization(tmesh, skeleton);
    CGAL::extract_mean_curvature_flow_skeleton(tmesh, skeleton);
    // init the polyhedron simplex indices
    CGAL::set_halfedgeds_items_id(tmesh);
    //for each input vertex compute its distance to the skeleton
    std::vector<double> distances(num_vertices(tmesh));
    int num_skel_vertices = 0;
    BOOST_FOREACH(Skeleton_vertex v, vertices(skeleton)) {
                    num_skel_vertices++;
                    const Point &skel_pt = skeleton[v].point;
                    BOOST_FOREACH(vertex_descriptor mesh_v, skeleton[v].vertices) {
                                    const Point &mesh_pt = mesh_v->point();
                                    distances[mesh_v->id()] = std::sqrt(CGAL::squared_distance(skel_pt, mesh_pt));
                                }
                }
    skeleton_vertices = Eigen::MatrixXd::Zero(num_skel_vertices, 3);
    skeleton_diam = Eigen::MatrixXd::Zero(num_skel_vertices, 1);
    int i = 0;
    BOOST_FOREACH(Skeleton_vertex v, vertices(skeleton)) {
                    const Point &skel_pt = skeleton[v].point;
                    skeleton_vertices(i, 0) = skel_pt[0];
                    skeleton_vertices(i, 1) = skel_pt[1];
                    skeleton_vertices(i, 2) = skel_pt[2];
                    i++;
                }

    Eigen::VectorXd sqrD;
    Eigen::VectorXi I;
    Eigen::MatrixXd C;
    igl::point_mesh_squared_distance(skeleton_vertices, V, F, sqrD, I, C);

    // create a property-map for sdf values
    std::vector<double> sdf_values(num_faces(tmesh));
    Facet_with_id_pmap<double> sdf_property_map(sdf_values);

    face_sdf = Eigen::MatrixXd::Zero(F.rows(), 1);

    // compute sdf values with skeleton
    for (int i = 0; i < V.rows(); i++) {
        distances[i] = 0.0;
    }

    std::vector<bool> untouched(V.rows());
    for (int i = 0; i < V.rows(); i++) {
        untouched[i] = true;
    }
    i = 0;

    BOOST_FOREACH(Skeleton_edge e, edges(skeleton)) {
                    i++;
                }
    SE1 = Eigen::MatrixXd::Zero(i, 3);
    SE2 = Eigen::MatrixXd::Zero(i, 3);
    i = 0;

    BOOST_FOREACH(Skeleton_edge e, edges(skeleton)) {
                    const Point &s = skeleton[source(e, skeleton)].point;
                    const Point &t = skeleton[target(e, skeleton)].point;
                    SE1(i, 0) = s[0];
                    SE1(i, 1) = s[1];
                    SE1(i, 2) = s[2];
                    SE2(i, 0) = t[0];
                    SE2(i, 1) = t[1];
                    SE2(i, 2) = t[2];
                    i++;
                    //std::cout << s << " " << t << "\n";
                }

    i = 0;
    BOOST_FOREACH(Skeleton_vertex v, vertices(skeleton)) {
                    const Point &skel_pt = skeleton[v].point;
                    double min_dist = 0.0;
                    BOOST_FOREACH(vertex_descriptor mesh_v, skeleton[v].vertices) {
                                    untouched[mesh_v->id()] = false;
                                    const Point &mesh_pt = mesh_v->point();
                                    if (distances[mesh_v->id()] == 0.0)
                                        distances[mesh_v->id()] = sqrD(i);
                                    else
                                        distances[mesh_v->id()] = std::min(sqrD(i), distances[mesh_v->id()]);
                                    if (min_dist == 0)
                                        min_dist = distances[mesh_v->id()];
                                    else
                                        min_dist = std::min(distances[mesh_v->id()], min_dist);
                                }
                    skeleton_diam(i, 0) = min_dist;
                    i++;
                }


    for (int i = 0; i < V.rows(); i++) {
        if (untouched[i])
            std::cout << "Vertex " << i << " was not touched.\n";
    }

    vertex_sdf = Eigen::MatrixXd::Zero(V.rows(), 1);
    for (int i = 0; i < V.rows(); i++) {
        vertex_sdf(i, 0) = distances[i];
    }


}

/**
 *
 * @param V
 * @param F
 * @param face_sdf
 * @param vertex_sdf
 * @param skeleton_vertices
 */
void sdf(Eigen::MatrixXd &V, Eigen::MatrixXi &F, Eigen::MatrixXd &face_sdf, Eigen::MatrixXd &vertex_sdf,
         Eigen::MatrixXd &skeleton_vertices) {
    Polyhedron tmesh;
    igl::copyleft::cgal::mesh_to_polyhedron(V, F, tmesh);
    // extract the skeleton
    Skeleton skeleton;
    //CGAL::Mean_curvature_flow_skeletonization(tmesh, skeleton);
    CGAL::extract_mean_curvature_flow_skeleton(tmesh, skeleton);
    // init the polyhedron simplex indices
    CGAL::set_halfedgeds_items_id(tmesh);
    //for each input vertex compute its distance to the skeleton
    std::vector<double> distances(num_vertices(tmesh));
    int num_skel_vertices = 0;
    BOOST_FOREACH(Skeleton_vertex v, vertices(skeleton)) {
                    num_skel_vertices++;
                    const Point &skel_pt = skeleton[v].point;
                    BOOST_FOREACH(vertex_descriptor mesh_v, skeleton[v].vertices) {
                                    const Point &mesh_pt = mesh_v->point();
                                    distances[mesh_v->id()] = std::sqrt(CGAL::squared_distance(skel_pt, mesh_pt));
                                }
                }
    skeleton_vertices = Eigen::MatrixXd::Zero(num_skel_vertices, 3);
    int i = 0;
    BOOST_FOREACH(Skeleton_vertex v, vertices(skeleton)) {
                    const Point &skel_pt = skeleton[v].point;
                    skeleton_vertices(i, 0) = skel_pt[0];
                    skeleton_vertices(i, 1) = skel_pt[1];
                    skeleton_vertices(i, 2) = skel_pt[2];
                    i++;
                }
    // create a property-map for sdf values
    std::vector<double> sdf_values(num_faces(tmesh));
    Facet_with_id_pmap<double> sdf_property_map(sdf_values);

    face_sdf = Eigen::MatrixXd::Zero(F.rows(), 1);

    // compute sdf values with skeleton
    i = 0;
    BOOST_FOREACH(face_descriptor f, faces(tmesh)) {
                    double dist = 0;
                    BOOST_FOREACH(halfedge_descriptor hd,
                                  halfedges_around_face(halfedge(f, tmesh), tmesh))dist += distances[target(hd,
                                                                                                            tmesh)->id()];
                    sdf_property_map[f] = dist / 3.;
                    i++;
                }


    std::pair<double, double> min_max_sdf = CGAL::sdf_values(tmesh, sdf_property_map);
    min_max_sdf = CGAL::sdf_values_postprocessing(tmesh, sdf_property_map);

    i = 0;
    BOOST_FOREACH(face_descriptor f, faces(tmesh)) {
                    face_sdf(i, 0) = sdf_property_map[f];
                    i++;
                }

    /// Convert Face SDF-attribute back to vertex SDF-attribute
    std::vector<std::vector<int>> VF, VI;
    igl::vertex_triangle_adjacency(F.maxCoeff() + 1, F, VF, VI);
    vertex_sdf = Eigen::MatrixXd::Zero(V.rows(), 1);
    for (int i = 0; i < V.rows(); i++) {
        double sdf = 0.0;
        int neighbors = 0;
        for (int j = 0; j < VF[i].size(); j++) {
            sdf += face_sdf(VF[i][j], 0);
            neighbors++;
        }
        sdf = sdf / (double) neighbors;
        vertex_sdf(i, 0) = sdf;
    }
}
