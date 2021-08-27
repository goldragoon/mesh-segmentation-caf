#include <iostream>
#include <random>
#include <chrono>
#include <boost/bind.hpp>
#include "basic_mesh_functions.h"
#include <igl/principal_curvature.h>
#include <igl/gaussian_curvature.h>
#include <igl/edges.h>
#include <igl/grad.h>
#include <igl/adjacency_list.h>
#include "custom_isoline.h"
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <Eigen/SparseQR>
#include <Eigen/SPQRSupport>
//#include <Eigen/SparseCholesky>
#include <chrono>
#include <cmath>  /* for std::abs(double) */

inline bool isEqual(double x, double y)
{
    const double epsilon = 1e-6/* some small number such as 1e-5 */;
    return std::abs(x - y) <= epsilon * std::abs(x);
    // see Knuth section 4.2.2 pages 217-218
}
/**
 *
 * @param V
 * @param F
 * @param N
 */
void compute_vertex_normals(Eigen::MatrixXd &V, Eigen::MatrixXi &F, Eigen::MatrixXd &N) {
    N.resize(V.rows(), 3);
    int vertex_size = V.rows();
    std::vector<Eigen::Vector3d> *normal_buffer = new std::vector<Eigen::Vector3d>[vertex_size];

    /// Calculate Face Normals
    for (int i = 0; i < F.rows(); i++) {
        std::vector<Eigen::Vector3d> p(3);
        p[0] = V.row(F(i, 0));
        p[1] = V.row(F(i, 1));
        p[2] = V.row(F(i, 2));
        Eigen::Vector3d v1 = p[1] - p[0];
        Eigen::Vector3d v2 = p[2] - p[0];
        Eigen::Vector3d normal = v1.cross(v2);

        normal.normalize();
        normal_buffer[F(i, 0)].push_back(normal);
        normal_buffer[F(i, 1)].push_back(normal);
        normal_buffer[F(i, 2)].push_back(normal);
    }

    /// Now loop through each vertex vector, and avarage out all the normals stored.
    for (int i = 0; i < V.rows(); ++i) {
        for (int j = 0; j < normal_buffer[i].size(); ++j) {
            N(i, 0) += normal_buffer[i][j](0);
            N(i, 1) += normal_buffer[i][j](1);
            N(i, 2) += normal_buffer[i][j](2);
        }
        N(i, 0) /= normal_buffer[i].size();
        N(i, 1) /= normal_buffer[i].size();
        N(i, 2) /= normal_buffer[i].size();
    }
}

/**
 *
 * @param E
 * @param F
 * @param OV
 */
void compute_opposite_vertices(Eigen::MatrixXi &E, Eigen::MatrixXi &F, Eigen::MatrixXi &OV) {
    std::map<std::set<int>, std::vector<int>> edge_to_opposite_vertices_map;

    int not_assigned = F.rows();
    OV = Eigen::MatrixXi::Ones(E.rows(), 2);
    for (int i = 0; i < F.rows(); i++) {
        /// Register this face in all three edges if not already done
        edge_to_opposite_vertices_map[{F(i, 0), F(i, 1)}].push_back(F(i, 2));
        edge_to_opposite_vertices_map[{F(i, 1), F(i, 2)}].push_back(F(i, 0));
        edge_to_opposite_vertices_map[{F(i, 2), F(i, 0)}].push_back(F(i, 1));
    }

    for (int i = 0; i < E.rows(); i++) {
        OV(i, 0) = edge_to_opposite_vertices_map[{E(i, 0), E(i, 1)}][0];
        OV(i, 1) = edge_to_opposite_vertices_map[{E(i, 0), E(i, 1)}][1];
    }
}

/**
 *
 * @param E
 * @param F
 * @param IF
 */
void compute_incident_faces(Eigen::MatrixXi &E, Eigen::MatrixXi &F, Eigen::MatrixXi &IF) {
    std::map<std::set<int>, std::vector<int>> edge_to_faces_map;

    int not_assigned = F.rows();
    IF = Eigen::MatrixXi::Ones(E.rows(), 2);
    for (int i = 0; i < F.rows(); i++) {
        /// Register this face in all three edges if not already done
        edge_to_faces_map[{F(i, 0), F(i, 1)}].push_back(i);
        edge_to_faces_map[{F(i, 1), F(i, 2)}].push_back(i);
        edge_to_faces_map[{F(i, 2), F(i, 0)}].push_back(i);
    }

    for (int i = 0; i < E.rows(); i++) {
        IF(i, 0) = edge_to_faces_map[{E(i, 0), E(i, 1)}][0];
        IF(i, 1) = edge_to_faces_map[{E(i, 0), E(i, 1)}][1];
    }
}

/**
 *
 * @param V
 * @param F
 * @param FC
 * @param FN
 */
void compute_face_normals(Eigen::MatrixXd &V, Eigen::MatrixXi &F, Eigen::MatrixXd &FC, Eigen::MatrixXd &FN) {
    FC.resize(F.rows(), 3);
    FN.resize(F.rows(), 3);

    /// Calculate Face Normals
    for (int i = 0; i < F.rows(); i++) {
        std::vector<Eigen::Vector3d> p(3);
        p[0] = V.row(F(i, 0));
        p[1] = V.row(F(i, 1));
        p[2] = V.row(F(i, 2));
        Eigen::Vector3d v1 = p[1] - p[0];
        Eigen::Vector3d v2 = p[2] - p[0];
        Eigen::Vector3d normal = v1.cross(v2);

        normal.normalize();
        Eigen::Vector3d centroid = (p[0] + p[1] + p[2]) / 3;
        FC.row(i) = centroid;
        FN.row(i) = normal;
    }
}

/**
 *
 * @param V
 * @param F
 * @param E
 * @param IF
 * @param OV
 * @param FN
 * @param DA
 */
void compute_dihedral_angle(Eigen::MatrixXd &V, Eigen::MatrixXi &F, Eigen::MatrixXi &E, Eigen::MatrixXi &IF,
                            Eigen::MatrixXi &OV, Eigen::MatrixXd &FN, Eigen::MatrixXd &DA) {
    DA.resize(E.rows(), 1);
    std::map<std::set<int>, int> vertices_to_face_index_map;
    for (int i = 0; i < F.rows(); i++) {
        vertices_to_face_index_map[{F(i, 0), F(i, 1), F(i, 2)}] = i;
    }

    for (int i = 0; i < E.rows(); i++) {
        Eigen::Vector3d p0 = V.row(E(i, 0));
        Eigen::Vector3d p1 = V.row(E(i, 1));
        Eigen::Vector3d p2 = V.row(OV(i, 0));
        Eigen::Vector3d p3 = V.row(OV(i, 1));
        int f1 = vertices_to_face_index_map[{E(i, 0), E(i, 1), OV(i, 0)}];
        int f2 = vertices_to_face_index_map[{E(i, 0), E(i, 1), OV(i, 1)}];
        Eigen::Vector3d n1 = FN.row(f1);
        Eigen::Vector3d n2 = FN.row(f2);
        double normal_angle = atan2((n1.cross(n2)).norm(), n1.dot(n2));

        if ((p3 - p2).dot(n1 - n2) < 0)
            normal_angle = -normal_angle;

        DA(i) = normal_angle;
    }
}

/**
 *
 * @param V
 * @param E
 * @param D
 */
void compute_distance(Eigen::MatrixXd &V, Eigen::MatrixXi &E, Eigen::MatrixXd &D) {
    D.resize(E.rows(), 1);
    for (int i = 0; i < E.rows(); i++) {
        D(i) = sqrt((V(E(i, 0), 0) - V(E(i, 1), 0)) * (V(E(i, 0), 0) - V(E(i, 1), 0)) +
                    (V(E(i, 0), 1) - V(E(i, 1), 1)) * (V(E(i, 0), 1) - V(E(i, 1), 1)) +
                    (V(E(i, 0), 2) - V(E(i, 1), 2)) * (V(E(i, 0), 2) - V(E(i, 1), 2)));
    }
}

/**
 *
 * @param F
 * @param V
 * @param L
 * @param index
 * @param A
 * @return
 */
int solve_poisson_equation_least_squares(Eigen::MatrixXi &F, Eigen::MatrixXd &V, Eigen::MatrixXd &L, int index,
                       std::vector<std::vector<int>> &A) {

    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(A[index].size() + 1, V.rows());
    for (int i = 0; i < A[index].size(); i++) {
        C(i, A[index][i]) = 1.0;
    }
    C(A[index].size(), index) = 1.0;

    Eigen::VectorXd B = Eigen::VectorXd::Ones(A[index].size() + 1);
    B(A[index].size(), 0) = 0.0;

    Eigen::VectorXd nulls = Eigen::VectorXd::Zero(V.rows());

    Eigen::MatrixXd MA(L.rows() + C.rows(), L.cols());
    MA << L,
            C;

    Eigen::VectorXd Mb(L.rows() + C.rows());
    Mb << nulls,
            B;

    Eigen::MatrixXd x = MA.householderQr().solve(Mb);
    Eigen::MatrixXd::Index maxRow, maxCol;
    x.maxCoeff(&maxRow, &maxCol);

    return maxRow;
}

/**
 *  Compute a field that propagates from p to the rest of the shape by constraining the value at p to be 0.
 * @param V         Vertices
 * @param index     index of selected point p from vertices V
 * @param A         Adjacency matrix
 * @param E         Edge list
 * @param basic_tripletList     triplet list
 * @return          index of vertex that is the resulting starting point
 */
void solve_poisson_equation_least_squares_fast(Eigen::MatrixXi &F, Eigen::MatrixXd &V, int index,
                            std::vector<std::vector<int>> &A, Eigen::MatrixXi &E,
                            std::vector<Eigen::Triplet<double>> &L_triplets,
                            Eigen::MatrixXd& x) {

    // Construct matrix A of Eq (1) in Paragraph 3.1. Concave-Aware Segmentation Field
    std::vector<T> A_triplets = L_triplets; // matrix for boundary conditions
    int C_rows = A[index].size() + 1;
    int L_rows = V.rows(); // or laplacian matrix's size, L(laplacian) is square matrix by definition.

    // Add triplets according to matrix L in A. (Ref. Eq. (1))
    for (int i = 0; i < A[index].size(); i++) {
        A_triplets.push_back(T(V.rows() + i, A[index][i], 1.0));
    }
    A_triplets.push_back(T(V.rows() + A[index].size(), index, 1.0));

   
    Eigen::SparseMatrix<double> LC_sparse(L_rows + C_rows, L_rows);
    LC_sparse.setFromTriplets(A_triplets.begin(), A_triplets.end());

    Eigen::VectorXd B = Eigen::VectorXd::Ones(A[index].size() + 1); // Set values at one-ring neighbors of p to be 1.
    B(A[index].size(), 0) = 0.0;// Set value at p to be 0.

    Eigen::VectorXd nulls = Eigen::VectorXd::Zero(V.rows());

    Eigen::VectorXd Mb(L_rows + C_rows);
    Mb << nulls,
            B;

    Eigen::SPQR<Eigen::SparseMatrix<double>> solver;
    solver.compute(LC_sparse);
    if (solver.info() != Eigen::Success) {
        printf("solve_poisson_equation_least_squares_fast : decomposition failed\n");
    }
    x = solver.solve(Mb);
    if (solver.info() != Eigen::Success) {
        printf("solve_poisson_equation_least_squares_fast : solving failed\n");
    }
}

/**
 *
 * @param extreme_point_set
 * @param V
 * @return
 */
double max_geodesic_dist(std::set<int> &extreme_point_set, Eigen::MatrixXd &V) {
    double dist_max = 0.0;
    for (auto ele1 : extreme_point_set) {
        for (auto ele2 : extreme_point_set) {
            if (ele1 < ele2)
                dist_max = std::max(dist_max, (V.row(ele1) - V.row(ele2)).norm());
        }
    }
    return dist_max;
}

/**
 *
 * @param p1
 * @param extreme_point_set
 * @param V
 * @param dist_prox
 * @return
 */
bool in_proximity_to(Eigen::MatrixXd p1, std::set<int> &extreme_point_set, Eigen::MatrixXd &V, double dist_prox) {
    bool is_prox = false;
    for (auto ele1 : extreme_point_set) {
        if ((V.row(ele1) - p1).norm() < dist_prox)
            is_prox = true;
    }
    return is_prox;
}

/**
 * * Implement scheme described in Section '3.2 Multiple Segmentation Fields' in the paper.
 * Obtain single point to start the search for extreme points from.
 * 
 * @param F
 * @param V
 * @param L
 * @param index_given
 * @param E
 * @return
 */
std::vector<int> get_extreme_points(Eigen::MatrixXi &F, Eigen::MatrixXd &V,  
    std::vector<T> &L_triplets, int index_given, Eigen::MatrixXi &E) {
    std::set<int> extreme_point_set;

    int prev_previous_num_points = 0;
    int previous_num_points = 0;
    int current_num_points = 0;
    int max_iters = 2;
    int iters = 0;

    // Special Random Number Generator Definition
    std::ranlux48 gen;
    std::uniform_int_distribution<int> uniform_0_255(0, V.rows() - 1);

    std::vector<std::vector<int>> vtx_adj_list;
    igl::adjacency_list(F, vtx_adj_list);
    
    do {
        iters++;

        // Getting starting point fast by randomly selected surface point p.
        int p_idx = uniform_0_255(gen); // idx of p in V.
        //std::vector<int> orn_p = vtx_adj_list[p_idx]; // one ring neighbors of point p 
        std::cout << ">> Starting with random point " << p_idx << "\n";

        // Compute a field from 'p' to the rest of the shape and get the point 'q' 
        // with maximum field value as the most prominent point of the shape.
        Eigen::MatrixXd x;
        Eigen::MatrixXd::Index maxRow, maxCol;
        Eigen::MatrixXd::Index minRow, minCol;

        solve_poisson_equation_least_squares_fast(F, V, p_idx, vtx_adj_list, E, L_triplets, x);// idx of q in V.

        x.maxCoeff(&maxRow, &maxCol);
        int q_idx = maxRow;

        solve_poisson_equation_least_squares_fast(F, V, q_idx, vtx_adj_list, E, L_triplets, x);
        
        x.maxCoeff(&maxRow, &maxCol);
        x.minCoeff(&minRow, &minCol);
      
        std::cout << ">> Field varies from " << x(minRow,0) << " to " << x(maxRow,0) << std::endl;

        std::vector<int> candidate_extreme_points;

        std::vector<std::pair<double, int>> extreme_point_queue;

        // find all the points with local max/min field value
        // (i.e. those with the largest or smallest field value among its one-ring neighbors)
        for (int i = 0; i < V.rows(); i++) {

            // Compute max/min val of one-ring of V[i]
            double max_val = std::numeric_limits<double>::lowest();
            double min_val = std::numeric_limits<double>::max();
            for (int adj_vert : vtx_adj_list[i]) { 
                max_val = std::max(x(adj_vert, 0), max_val);
                min_val = std::min(x(adj_vert, 0), min_val);
            }

            // Check that V[i] is largest or smallest among one-ring
            if (x(i) >= max_val || x(i) <= min_val) {

                double max_dist = max_geodesic_dist(extreme_point_set, V);

                // Ignore the newly found extreme point if it is within coeff * max_dist(L in paper) of any existing extreme point.
                if (!in_proximity_to(V.row(i), extreme_point_set, V, max_dist * 0.3)) {
                    /*
                    double neighbors = 0.0;
                    for (int j = 0; j < vtx_adj_list[i].size(); j++) {
                        neighbors = neighbors + x(vtx_adj_list[i][j]);
                    }
                    neighbors = abs(x(i) - (neighbors / vtx_adj_list[i].size()));
                    */
                    //std::cout << "Neighbor_values: " << neighbors << "\n";
                    //extreme_point_queue.push_back(std::pair<double, int>(neighbors, i));
                    extreme_point_set.insert(i);
                }
                /*
                if (extreme_point_set.size() < 2) {
                    candidate_extreme_points.push_back(i);
                }
                else {
                }
                */
            }
        }
        /*
        if (extreme_point_queue.size() > 0 && extreme_point_set.size() >= 2) {
            std::sort(extreme_point_queue.begin(), extreme_point_queue.end(),
                      boost::bind(&std::pair<double, int>::first, _1) <
                      boost::bind(&std::pair<double, int>::first, _2));

            for (int i = extreme_point_queue.size() - 1; i >= 0; i--) {
                double max_dist = max_geodesic_dist(extreme_point_set, V);
                if (!in_proximity_to(V.row(extreme_point_queue[i].second), extreme_point_set, V, max_dist * 0.15)) {
                    extreme_point_set.insert(extreme_point_queue[i].second);
                    //std::cout << ">> vtx_adj_listdded extreme point " << extreme_point_queue[i].second << std::endl;
                } else {
                    //std::cout << ">> Rejected extreme point " << extreme_point_queue[i].second << std::endl;
                }

            }
        }
       
        if (candidate_extreme_points.size() > 0) {
            std::vector<double> pair_dist;
            std::vector<int> candidate1;
            std::vector<int> candidate2;
            int selected_1 = 0;
            int selected_2 = 0;
            /// vtx_adj_listdd pair, that is the farthest apart first
            for (int i = 0; i < candidate_extreme_points.size(); i++) {
                for (int j = 0; j < i; j++) {
                    pair_dist.push_back(
                            (V.row(candidate_extreme_points[i]) - V.row(candidate_extreme_points[j])).norm());
                    candidate1.push_back(candidate_extreme_points[i]);
                    candidate2.push_back(candidate_extreme_points[j]);
                }
            }
            double max_dist = 0.0;
            for (int i = 0; i < pair_dist.size(); i++) {
                if (max_dist < pair_dist[i]) {
                    max_dist = pair_dist[i];
                    selected_1 = candidate1[i];
                    selected_2 = candidate2[i];
                }
            }
            extreme_point_set.insert(selected_1);
            extreme_point_set.insert(selected_2);
        } */

    } while (iters < max_iters);

    return std::vector<int>(extreme_point_set.begin(), extreme_point_set.end());
}

/**
 *
 * @param L
 * @param index1 idx of some point u
 * @param index2 idx of some point v
 * @param z
 * @param basic_tripletList
 */
void get_segmentation_field(Eigen::MatrixXd &L,
                            int index1, int index2, Eigen::MatrixXd &z, std::vector<T> &basic_tripletList) {

    int C_rows = 2; // boundary condition, one is u and other is v.
    int L_rows = L.rows(); // or laplacian matrix's size, L(laplacian) is square matrix by definition.

    // Construct Matrix A = [L  C] with sparse matrix.
    std::vector<T> tripletList = basic_tripletList;
    tripletList.push_back(T(L.rows(), index1, 1.0));
    tripletList.push_back(T(L.rows() + 1, index2, 1.0));

    Eigen::SparseMatrix<double> LC_sparse(L.rows() + 2, L.rows());
    LC_sparse.setFromTriplets(tripletList.begin(), tripletList.end());

    Eigen::VectorXd B = Eigen::VectorXd::Ones(2);
    B(0, 0) = 0.0;
    Eigen::VectorXd nulls = Eigen::VectorXd::Zero(L.rows());

    Eigen::VectorXd Mb(L_rows + C_rows);
    Mb << nulls,
            B;

    Eigen::SPQR<Eigen::SparseMatrix<double>> solver;
    solver.compute(LC_sparse);
    if (solver.info() != Eigen::Success) {
        printf("get_segmentation_field : decomposition failed\n");
    }
    z = solver.solve(Mb);
    if (solver.info() != Eigen::Success) {
        printf("get_segmentation_field : solving failed\n");
    }

}

/**
 *
 * @param isoV
 * @param isoE
 * @param contours
 * @param contour_faces
 * @param z
 * @param score
 */
void get_isoline_gradient_scores(Eigen::MatrixXd &isoV, Eigen::MatrixXi &isoE, std::vector<std::vector<int>> &contours,
                                 std::vector<std::vector<int>> &contour_faces, Eigen::MatrixXd &z,
                                 std::vector<double> &score) {
    score.clear();
    for (int i = 0; i < contours.size(); i++) {
        double tmp_score = 0.0;
        double tmp_length = 0.0;
        int last_vertex = -1;
        for (int j = 0; j < contours[i].size(); j++) {
            /// Calculate Distance of current edge
            /// If first element: create edge to very last vertex
            if (j == 0)
                last_vertex = contours[i].back();
            /// Multiply edgelength by faces magnitude
            auto p1 = isoV.row(contours[i][j]);
            auto p2 = isoV.row(last_vertex);

            tmp_score = tmp_score + ((p1 - p2).norm() * z(contour_faces[i][j], 0));
            tmp_length = tmp_length + (p1 - p2).norm();

            last_vertex = contours[i][j];
        }
        score.push_back(tmp_score / tmp_length);
    }
}

/**
 *
 * @param isoV
 * @param isoE
 * @param contours
 * @param contour_faces
 * @param z
 * @param score
 */
void get_isoline_shape_scores(Eigen::MatrixXd &isoV, Eigen::MatrixXi &isoE, std::vector<std::vector<int>> &contours,
                              std::vector<std::vector<int>> &contour_faces, Eigen::MatrixXd &z,
                              std::vector<double> &score) {
    score.clear();
    for (int i = 0; i < contours.size(); i++) {
        double tmp_score = 0.0;
        double tmp_length = 0.0;
        int last_vertex = -1;
        for (int j = 0; j < contours[i].size(); j++) {
            /// Calculate Distance of current edge
            /// If first element: create edge to very last vertex
            if (j == 0)
                last_vertex = contours[i].back();
            /// Multiply edgelength by faces magnitude
            auto p1 = isoV.row(contours[i][j]);
            auto p2 = isoV.row(last_vertex);

            tmp_score = tmp_score + ((p1 - p2).norm() * z(contour_faces[i][j], 0));
            tmp_length = tmp_length + (p1 - p2).norm();

            last_vertex = contours[i][j];
        }
        score.push_back(tmp_score / tmp_length);
    }
}

/**
 *
 * @param isoV
 * @param isoE
 * @param contours
 * @param contour_faces
 * @param length
 */
void get_isoline_length(Eigen::MatrixXd &isoV, Eigen::MatrixXi &isoE, std::vector<std::vector<int>> &contours,
                        std::vector<std::vector<int>> &contour_faces, std::vector<double> &length) {
    length.clear();
    for (int i = 0; i < contours.size(); i++) {
        double tmp_length = 0.0;
        int last_vertex = -1;
        for (int j = 0; j < contours[i].size(); j++) {
            if (j == 0)
                last_vertex = contours[i].back();
            auto p1 = isoV.row(contours[i][j]);
            auto p2 = isoV.row(last_vertex);
            tmp_length = tmp_length + (p1 - p2).norm();
            last_vertex = contours[i][j];
        }
        length.push_back(tmp_length);
    }
}

/**
 *
 * @param V1
 * @param F1
 * @param V2
 * @param F2
 */
void add_mesh(Eigen::MatrixXd &V1, Eigen::MatrixXi &F1, Eigen::MatrixXd &V2, Eigen::MatrixXi &F2) {
    Eigen::MatrixXd V(V1.rows() + V2.rows(), V1.cols());
    V << V1, V2;
    Eigen::MatrixXi F(F1.rows() + F2.rows(), F1.cols());
    F << F1, (F2.array() + V1.rows());
    V1 = V;
    F1 = F;
}