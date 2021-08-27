#include <igl/doublearea.h>
#include <igl/vertex_triangle_adjacency.h>
#include <igl/per_vertex_normals.h>
#include <igl/edges.h>
#include <igl/grad.h>
#include <igl/gaussian_curvature.h>
#include <igl/adjacency_list.h>
#include <igl/principal_curvature.h>
#include <fstream>

#include "custom_isoline.h"
#include <igl/isolines.h> // to replace custom_isoline.h

//#include <igl/all_edges.h>
#include <igl/edges.h>
#include <igl/sort.h>
#include <igl/per_face_normals.h>
#include <igl/triangle_triangle_adjacency.h>
#include "create_laplacian.h"
#include "basic_mesh_functions.h"
#include "get_separate_lines.h"

#define _USE_MATH_DEFINES
#include <math.h>
/**
 *
 * @param V
 * @param F
 * @param E
 * @param N
 * @param VV
 * @param IF
 * @param OV
 * @param FC
 * @param FN
 * @param DA
 * @param D
 * @param L
 * @param G
 * @param dblA
 */
void compute_all_features(Eigen::MatrixXd &V, Eigen::MatrixXi &F, Eigen::MatrixXi &E, Eigen::MatrixXd &N,
                          std::vector<std::vector<int>> &VV, 
                           Eigen::MatrixXi &IF, Eigen::MatrixXi &OV,
                          //Eigen::MatrixXd &FC, // remove this
                          Eigen::MatrixXd &FN, Eigen::MatrixXd &DA, Eigen::MatrixXd &D,
                          Eigen::MatrixXd &G, Eigen::MatrixXd &dblA) {

    igl::edges(F, E);
    igl::adjacency_list(F, VV);

    igl::per_face_normals(V, F, FN);
    igl::per_vertex_normals(V, F, igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_DEFAULT, FN , N);

    compute_incident_faces(E, F, IF); //temporarily disabled
    compute_opposite_vertices(E, F, OV);

    //compute_dihedral_angle(V, F, E, IF, OV, FN, DA);
    //compute_distance(V, E, D);
    igl::doublearea(V, F, dblA);
    igl::gaussian_curvature(V, F, G);


    ///Normalize G
    //G = G.array() / G.maxCoeff();

    /// Smooth Normals
    /*Eigen::MatrixXd FN_new = FN;
    for (int iters = 0; iters < 5; iters++){
      for (int i = 0; i < F.rows(); i++){
        //std::cout << i << std::endl;
        Eigen::MatrixXd x = FN.row(i);
        Eigen::MatrixXd N_smooth = FN.row(i);
        std::set<int> faces;
        for (int j = 0; j < VF[F(i,0)].size(); j++)
          faces.insert(VF[F(i,0)][j]);
        for (int j = 0; j < VF[F(i,1)].size(); j++)
          faces.insert(VF[F(i,1)][j]);
        for (int j = 0; j < VF[F(i,2)].size(); j++)
          faces.insert(VF[F(i,2)][j]);
        faces.erase(i);
        for (int face : faces){
          Eigen::MatrixXd x_hat = FN.row(face);
          //double Kxx = exp(- ((x - x_hat).norm()) / (2 * 0.1 * 0.1));
          double Kxx = exp(- ((x - x_hat).squaredNorm()) / (2 * 0.1 * 0.1));
          N_smooth += (x_hat * Kxx);
        }
        FN_new.row(i) = N_smooth.normalized();
      }
      FN = FN_new;
    }*/


    /*Eigen::MatrixXd N_new = N;
    for (int iters = 0; iters < 5; iters++) {
      for (int i = 0; i < V.rows(); i++) {
        //std::cout << i << std::endl;
        Eigen::MatrixXd x = N.row(i);
        Eigen::MatrixXd N_smooth = N.row(i);
        for (int vertex : VF[i]) {
          Eigen::MatrixXd x_hat = N.row(vertex);
          double Kxx = exp(- ((x - x_hat).norm()) / (2 * 0.1 * 0.1));
          //double Kxx = exp(-((x - x_hat).squaredNorm()) / (2 * 0.1 * 0.1));
          N_smooth += (x_hat * Kxx);
        }
        N_new.row(i) = N_smooth.normalized();
      }
      N = N_new;
    }*/

}

/**
 *
 * @param v1
 * @param v2
 * @param in_degree
 * @return
 */

double getAngle3D(const Eigen::Vector3d &v1, const Eigen::Vector3d &v2, const bool in_degree) {
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
 * @param source_centroidB
 * @param target_centroid
 * @param source_normal
 * @param target_normal
 * @param normal_angle
 * @return
 */
bool
connIsConvex(const Eigen::Vector3d &source_centroid, const Eigen::Vector3d &target_centroid,
             const Eigen::Vector3d &source_normal, const Eigen::Vector3d &target_normal, double &normal_angle) {
    bool is_convex = true;
    bool is_smooth = true;

    normal_angle = getAngle3D(source_normal, target_normal, true);
    //  Geometric comparisons
    Eigen::Vector3d vec_t_to_s, vec_s_to_t;

    vec_t_to_s = source_centroid - target_centroid;
    vec_s_to_t = -vec_t_to_s;

    Eigen::Vector3d ncross;
    ncross = source_normal.cross(target_normal);

    // vec_t_to_s is the reference direction for angle measurements
    // Convexity Criterion: Check if connection of patches is convex. If this is the case the two supervoxels should be merged.
    if ((getAngle3D(vec_t_to_s, source_normal, true) - getAngle3D(vec_t_to_s, target_normal, true)) <= 0) {
        normal_angle = -normal_angle;
        is_convex &= true;  // connection convex
    } else {
        is_convex &= (normal_angle < 0.0);  // concave connections will be accepted  if difference of normals is small
    }
    return (is_convex && is_smooth);
}

/**
 *
 * @param M
 * @param filename
 */
void var_to_file(Eigen::MatrixXd &M, std::string filename) {
    std::ofstream myfile;
    myfile.open(filename);
    myfile << M;
    myfile.close();
}

/**
 *
 * @param M
 * @param filename
 */
void var_to_file(Eigen::MatrixXi &M, std::string filename) {
    std::ofstream myfile;
    myfile.open(filename);
    myfile << M;
    myfile.close();
}

/**
 *
 * @param mat
 * @param cov
 */
void covariance_matrix(Eigen::MatrixXd &mat, Eigen::MatrixXd &cov) {
    Eigen::MatrixXd centered = mat.rowwise() - mat.colwise().mean();
    cov = Eigen::MatrixXd::Zero(mat.cols(), mat.cols());
    cov = (centered.adjoint() * centered) / double(mat.rows() - 1);
}

/**
 *
 * @param mat
 * @param ev
 */
void eigenvalues(Eigen::MatrixXd &mat, Eigen::MatrixXd &ev) {
    Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(mat.cols(), mat.cols());
    covariance_matrix(mat, cov);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(cov);
    ev = eigensolver.eigenvalues();
}

/**
 *
 * @param id
 * @param A
 * @return
 */
std::set<int> two_ring_neighbors(int id, std::vector<std::vector<int>> &A) {
    std::set<int> neighbors;
    for (int j = 0; j < A[id].size(); j++) {
        int neighbor = A[id][j];
        for (int k = 0; k < A[neighbor].size(); k++) {
            neighbors.insert(A[neighbor][k]);
        }
    }
    return neighbors;
}

/**
 *
 * @param V
 * @param N
 * @param A
 * @param i
 * @return
 */
bool is_concave(Eigen::MatrixXd &V, Eigen::MatrixXd &N, std::vector<std::vector<int>> &A, int i, double zeta) {
    bool concave = false;
    for (int j = 0; j < A[i].size(); j++) {
        if ((((V.row(i) - V.row(A[i][j])) / ((V.row(i) - V.row(A[i][j])).norm())).dot(N.row(A[i][j]) - N.row(i))) >
            zeta) {// [Further Improvement] 0.01 could be replace with zeta
            concave = true;
            break;
        }
    }
    return concave;
}

/**
 *
 * @param V
 * @param N
 * @param i
 * @param j
 * @return
 */
bool is_connection_concave(Eigen::MatrixXd &V, Eigen::MatrixXd &N, int i, int j) {
    bool concave = false;
    if ((((V.row(i) - V.row(j)) / ((V.row(i) - V.row(j)).norm())).dot(N.row(j) - N.row(i))) > 0.01) {
        concave = true;
    }
    return concave;
}

/**
 *
 * @param V
 * @param N
 * @param i
 * @param A
 * @param ratio
 * @return
 */
bool filter_1b(Eigen::MatrixXd &V, Eigen::MatrixXd &N, int i, std::vector<std::vector<int>> &A, double ratio) {
    std::set<int> neighbors = two_ring_neighbors(i, A);
    neighbors.erase(i);
    double concave_neighbors = 0.0;
    for (int vertex_id : neighbors) {
        if (is_connection_concave(V, N, vertex_id, i)) {
            concave_neighbors += 1.0;
        }
    }
    return (concave_neighbors / neighbors.size() > ratio);
}


#define MAXBUFSIZE  ((int) 1e6)
/**
 *
 * @param filename
 * @return
 */
Eigen::MatrixXd readMatrix(std::string filename) {
    int cols = 0, rows = 0;
    double buff[MAXBUFSIZE];

    // Read numbers from file into buffer.
    std::ifstream infile;
    infile.open(filename);
    while (!infile.eof()) {
        std::string line;
        getline(infile, line);

        int temp_cols = 0;
        std::stringstream stream(line);
        while (!stream.eof())
            stream >> buff[cols * rows + temp_cols++];

        if (temp_cols == 0)
            continue;

        if (cols == 0)
            cols = temp_cols;

        rows++;
    }

    infile.close();
    //rows--;

    // Populate matrix with numbers.
    Eigen::MatrixXd result(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result(i, j) = buff[cols * i + j];

    return result;
}

/**
 *
 * @param V
 * @param i
 * @param A
 * @return
 */
bool filter_2(Eigen::MatrixXd &V, int i, std::vector<std::vector<int>> &A) {
    //Eigen::MatrixXd one_ring = Eigen::MatrixXd::Zero(A[i].size(), 3);
    Eigen::MatrixXd one_ring = Eigen::MatrixXd::Zero(A[i].size() + 1, 3);
    for (int l = 0; l < A[i].size(); l++) {
        int j = A[i][l];
        one_ring.row(l) = V.row(j);
    }
    one_ring.row(A[i].size()) = V.row(i);
    Eigen::MatrixXd ev;
    eigenvalues(one_ring, ev);
    return (ev(0, 0) / ev.sum() > 0.001);
}

/**
 *
 * @param V
 * @param F
 * @param E
 * @param L
 * @param N
 * @param beta
 * @param zeta
 */
void compute_laplacian_harmonic(Eigen::MatrixXd &V, Eigen::MatrixXi &F, Eigen::MatrixXi &E, Eigen::MatrixXd &L,
                                Eigen::MatrixXd &N,
                                double beta, double zeta) {
    //Eigen::MatrixXi E;
    //igl::edges(F, E);
    Eigen::MatrixXd E2 = Eigen::MatrixXd::Zero(E.rows(), 1);
    for (int i = 0; i < E.rows(); i++) {
        E2(i, 0) = (V.row(E(i, 0)) - V.row(E(i, 1))).norm();
    }
    double mean_dist = E2.mean();

    std::vector<std::vector<int>> A;
    igl::adjacency_list(F, A);
    Eigen::MatrixXd w = Eigen::MatrixXd::Zero(V.rows(), V.rows());

    Eigen::MatrixXd V_is_concave = Eigen::MatrixXd::Zero(V.rows(), 1);

    for (int i = 0; i < V.rows(); i++) {
        bool is_concave = false;
        for (int j = 0; j < A[i].size(); j++) {
            if ((((V.row(i) - V.row(A[i][j])) / ((V.row(i) - V.row(A[i][j])).norm())).dot(N.row(A[i][j]) - N.row(i))) >
                zeta) {
                V_is_concave(i, 0) = 1.0;
            }
        }
    }

    for (int i = 0; i < V.rows(); i++) {
        double sum_w_ik = 0.0;
        for (int l = 0; l < A[i].size(); l++) {
            int j = A[i][l];
            ///Convex
            if (V_is_concave(i, 0) == 1.0 || V_is_concave(j, 0) == 1.0)
                w(i, j) = (((V.row(i) - V.row(j)).norm()) / mean_dist * beta);
            else
                w(i, j) = 1.0;

            sum_w_ik += w(i, j);
        }
    }

    L = w;

    for (int i = 0; i < L.rows(); i++) {
        L(i, i) = -1.0;
    }
}

/**
 * Compute Laplacian Matrix according to (2) and (3) of the paper.
 * @param V
 * @param F
 * @param E
 * @param G
 * @param N
 * @param L
 * @param vertex_is_concave
 * @param beta
 * @param eps
 * @param zeta

 */
void compute_laplacian(Eigen::MatrixXd &V, Eigen::MatrixXi &F, Eigen::MatrixXi &E, Eigen::MatrixXd &G,
                       Eigen::MatrixXd &N, Eigen::MatrixXd &L, Eigen::MatrixXi &vertex_is_concave, double beta,
                       double eps, double zeta) {

    std::cout << "Beta: " << beta << std::endl;
    std::cout << "Eps: " << eps << std::endl;
    std::cout << "zeta: " << zeta << std::endl;

    Eigen::MatrixXd E2 = Eigen::MatrixXd::Zero(E.rows(), 1);

    // Compute each edge magnitude or norm ( |e_ij| ).
    for (int i = 0; i < E.rows(); i++) {
        E2(i, 0) = (V.row(E(i, 0)) - V.row(E(i, 1))).norm();
    }

    //double mean_dist = E2.mean();
    /*
    Eigen::MatrixXd::Index maxRow, maxCol;
    E2.maxCoeff(&maxRow, &maxCol);
    double max_dist = E2(maxRow, 0);
    */
    //std::vector<std::vector<int>> VF;
    //std::vector<std::vector<int>> VI;
    //igl::vertex_triangle_adjacency(F.maxCoeff() + 1, F, VF, VI);
    //Eigen::MatrixXd area;
    //igl::doublearea(V, F, area);

    // Calculate G_ij term in equation .(3) in paper.
    std::vector<double> connections;
    for (int i = 0; i < E.rows(); i++) {
        connections.push_back( abs(G(E(i, 0))) + abs(G(E(i, 1))) );
    }

    /*
    double max_val = *std::max_element(connections.begin(), connections.end());
    std::cout << "G max_val " << max_val << ", "<< G.maxCoeff()<< std::endl;
    //double max_val = G.maxCoeff();
    G = G.array() / max_val;
    */

    std::vector<std::vector<int>> A;
    igl::adjacency_list(F, A);  // containing at row i the adjacent vertices of vertex i

    Eigen::MatrixXi V_is_concave = Eigen::MatrixXi::Zero(V.rows(), 1);

    for (int i = 0; i < V.rows(); i++) {
        // pass zeta to concave
        if (is_concave(V, N, A, i, zeta)){ //  && filter_1b(V, N, i, A, 0.3) && filter_2(V, i, A)) { 
            V_is_concave(i, 0) = 1;
        }
    }

    vertex_is_concave = V_is_concave;

    // Calculate laplacian matrix according to Equation .(2) in paper.
    // (Case default) otherwise.
    Eigen::MatrixXd w = Eigen::MatrixXd::Zero(V.rows(), V.rows());

    // (Case 2) (i,j) \in E
    for (int i = 0; i < V.rows(); i++) {
        double sum_w_ik = 0.0;
        for (int l = 0; l < A[i].size(); l++) {
            int j = A[i][l];
            //double normal_angle;
            //connIsConvex(V.row(i), V.row(j), N.row(i), N.row(j), normal_angle);
            ///Convex
            if (V_is_concave(i, 0) == 1 || V_is_concave(j, 0) == 1) {
                //printf("(%d, %d) is concave!", i, j);
                //w(i, j) = (1 - ((V.row(i) - V.row(j)).norm() / max_dist)) * (1 - (abs(G(i, 0)) + abs(G(j, 0)))) * beta + eps;
                w(i, j) = ((V.row(i) - V.row(j)).norm() * beta ) / (abs(G(i,0)) + abs(G(j,0)) + eps);
            } else {
                //w(i, j) = (1 - ((V.row(i) - V.row(j)).norm() / max_dist)) * (1 - (abs(G(i, 0)) + abs(G(j, 0)))) + eps;
                w(i, j) = ((V.row(i) - V.row(j)).norm()) / (abs(G(i, 0)) + abs(G(j, 0)) + eps);
            }
            sum_w_ik += w(i, j);
        }


        for (int l = 0; l < A[i].size(); l++) {
            int j = A[i][l];
            w(i, j) = w(i, j) / sum_w_ik;
        }

    }

    // (Case 2) i == j
    for (int i = 0; i < w.rows(); i++) {
        w(i, i) = -1.0;
    }

    L = w;
}

/**
 * Pairwise-segmentation field Phi_uv calculation according to 3.2. (last two paragraphs) in paper. 
 *
 * @param V
 * @param F
 * @param FN
 * @param E
 * @param L
 * @param point1
 * @param point2
 * @param field_id
 * @param gradient_magnitude
 * @param isoline_neighbor_length
 * @param isoline_vertices
 * @param isoline_face_ids
 * @param isoline_length
 * @param isoline_gs
 * @param isoline_field_id
 * @param basic_tripletList
 * @param fields
 */
bool compute_segmentation_field(Eigen::MatrixXd &V, Eigen::MatrixXi &F, Eigen::MatrixXd &FN, Eigen::MatrixXi &E,
                                Eigen::MatrixXd &L, int point1, int point2, int field_id,
                                std::vector<Eigen::MatrixXd> &gradient_magnitude,
                                std::vector<std::vector<double>> &isoline_neighbor_length,
                                std::vector<std::vector<Eigen::MatrixXd>> &isoline_vertices,
                                std::vector<std::vector<int>> &isoline_face_ids,
                                std::vector<double> &isoline_length,
                                std::vector<double> &isoline_gs,
                                std::vector<int> &isoline_field_id,
                                std::vector<T> &basic_tripletList,
                                std::vector<Eigen::MatrixXd> &fields) {

    /// Initialize temporary Variables
    Eigen::MatrixXd isoV;   // Isoline's Vertices
    Eigen::MatrixXd isoE;   // Isoline's Edge IDs
    Eigen::MatrixXd field;  // Field values
    std::vector<int> isoF;  // Isoline's traversed Faces
    Eigen::SparseMatrix<double> gradient;   // Gradient
    std::vector<int> isoI;
    std::vector<std::vector<int>> isoline_ids;

    std::vector<std::vector<int>> isoline_vertex_ids;


    /// Get Isolines (The output (isoV and isoE) is chaotic and has to be organized for further processing)
    get_segmentation_field(L, point1, point2, field, basic_tripletList);
    fields.push_back(field);

    isolines(V, F, field, 50, isoV, isoE, isoF, isoI); // [!!IMPORTANT!!] must check correctness
    //igl::isolines(V, F, field, 50, isoV, isoE);
    std::cout << "Field " <<  field.size() << "varies : [" << field.minCoeff() << ", " << field.maxCoeff() << std::endl;
    if (isoV.size() == 0 || isoE.size() == 0) return false;
    //return true;
    //std::cout << isoV << std::endl;
    //std::cout << isoE << std::endl;


    /// Compute Gradient Magnitudes
    igl::grad(V, F, gradient, false);

    Eigen::MatrixXd GU = Eigen::Map<const Eigen::MatrixXd>((gradient * field).eval().data(), F.rows(), 3);
    const Eigen::VectorXd GU_mag = GU.rowwise().norm();
    Eigen::MatrixXd GU_mat = Eigen::MatrixXd(GU_mag.size(), 1);
    GU_mat << GU_mag;
    gradient_magnitude.push_back(GU_mat);
    

    /// Sort Isoline data into actual Isolines and compute their Scores for finding good candidates
    Eigen::MatrixXi intIsoE = isoE.cast<int>();

    /// Separate Lines
    get_separate_lines(intIsoE, isoF, isoline_vertex_ids, isoline_face_ids, isoV, FN, isoI, isoline_ids);

    /// Get Isoline Vertices
    for (int i = 0; i < isoline_vertex_ids.size(); i++) {
        std::vector<Eigen::MatrixXd> tmp_path;
        for (int j = 0; j < isoline_vertex_ids[i].size(); j++) {
            tmp_path.push_back(isoV.row(isoline_vertex_ids[i][j]));
        }
        isoline_vertices.push_back(tmp_path);
    }

    /// Get Isoline Gradient Magnitude Scores
    get_isoline_gradient_scores(isoV, intIsoE, isoline_vertex_ids, isoline_face_ids, GU_mat, isoline_gs);

    /// Get Isolines
    get_isoline_length(isoV, intIsoE, isoline_vertex_ids, isoline_face_ids, isoline_length);

    for (int i = 0; i < isoline_vertex_ids.size(); i++) {
        isoline_field_id.push_back(field_id);
        std::vector<double> tmp_neighbor_length;
        int limit_a = std::max(i - 5, 0);
        int limit_b = std::min(i + 5, (int) isoline_vertex_ids.size() - 1);
        for (int j = i - 5; j < i; j++) {
            if (j > 0)
                tmp_neighbor_length.push_back(isoline_length[j]);
            else
                tmp_neighbor_length.push_back(0.0);
        }
        for (int j = i + 5; j > i; j--) {
            if (j < (int) isoline_vertex_ids.size())
                tmp_neighbor_length.push_back(isoline_length[j]);
            else
                tmp_neighbor_length.push_back(0.0);
        }
        isoline_neighbor_length.push_back(tmp_neighbor_length);
    }
    return true;
}

/**
 *
 * @param candidate_length
 * @param candidate_neighbor_length
 * @param candidate_svs
 */
void compute_candidate_svs(std::vector<double> &candidate_length,
                           std::vector<std::vector<double>> &candidate_neighbor_length,
                           std::vector<double> &candidate_svs) {
    /// Now compute the SVS
    double longest_candidate = 0.0;
    for (int i = 0; i < candidate_length.size(); i++) {
        longest_candidate = std::max(longest_candidate, candidate_length[i]);
    }
    candidate_svs.resize(candidate_length.size());
    for (int i = 0; i < candidate_length.size(); i++) {
        double svs = 0.0;
        std::vector<double> candidate_delta(6);
        for (int j = 1; j <= 5; j++) {
            candidate_delta[j] = candidate_neighbor_length[i][10 - j] + candidate_neighbor_length[i][5 - j] -
                                 (2 * candidate_length[i]);
        }
        for (int j = 1; j <= 5; j++) {
            svs += exp(-(j * j) / (2 * pow(2, 2))) * candidate_delta[j];
        }
        candidate_svs[i] = svs;
    }
}

/**
 *
 * @param m
 */
void scale_0_1(Eigen::MatrixXd &m) {
    double min_val = m.minCoeff();
    double max_val = m.maxCoeff();
    double dist = max_val - min_val;
    for (int i = 0; i < m.rows(); i++) {
        m(i, 0) = (m(i, 0) - min_val) / dist;
    }
}

/**
 *
 * @param gradient_magnitude
 * @param candidate_vertices
 * @param candidate_face_ids
 * @param candidate_length
 * @param candidate_gs
 * @param g_hat
 */
void compute_candidate_gs(std::vector<Eigen::MatrixXd> &gradient_magnitude,
                          std::vector<std::vector<Eigen::MatrixXd>> &candidate_vertices,
                          std::vector<std::vector<int>> &candidate_face_ids,
                          std::vector<double> &candidate_length,
                          std::vector<double> &candidate_gs,
                          Eigen::MatrixXd &g_hat) {

    for (int i = 0; i < gradient_magnitude.size(); i++) {
        scale_0_1(gradient_magnitude[i]);
    }

    g_hat = Eigen::MatrixXd::Zero(gradient_magnitude[0].rows(), 1);
    for (int i = 0; i < gradient_magnitude.size(); i++) {
        g_hat = g_hat + gradient_magnitude[i];
    }

    for (int i = 0; i < candidate_vertices.size(); i++) {
        double tmp_score = 0.0;
        Eigen::MatrixXd last_vertex;
        for (int j = 0; j < candidate_vertices[i].size(); j++) {
            /// If first element: create edge to very last vertex
            if (j == 0)
                last_vertex = candidate_vertices[i].back();
            /// Multiply edgelength by faces magnitude
            auto p1 = candidate_vertices[i][j];
            auto p2 = last_vertex;
            tmp_score = tmp_score + ((p1 - p2).norm() * g_hat(candidate_face_ids[i][j], 0));
            last_vertex = candidate_vertices[i][j];
        }
        candidate_gs[i] = tmp_score / candidate_length[i];
    }
}

/**
 *
 * @param V
 * @param F
 * @param DA
 */
void compute_face_dihedral_angle(Eigen::MatrixXd &V, Eigen::MatrixXi &F, Eigen::MatrixXd &DA) {

    Eigen::MatrixXd FN;
    igl::per_face_normals(V, F, FN);
    Eigen::MatrixXi TT, TTi;
    igl::triangle_triangle_adjacency(F, TT, TTi);
    DA = Eigen::MatrixXd::Zero(F.rows(), 1);

    for (int i = 0; i < F.rows(); i++) {
        double highest_ang = 0.0;
        for (int j = 0; j < 3; j++) {
            double normal_angle = -360.0;
            Eigen::Vector3d c1 = (V.row(F(i, 0)) + V.row(F(i, 1)) + V.row(F(i, 2))) / 3.0;
            Eigen::Vector3d c2 = (V.row(F(TT(i, j), 0)) + V.row(F(TT(i, j), 1)) + V.row(F(TT(i, j), 2))) / 3.0;
            Eigen::Vector3d n1 = FN.row(i);
            Eigen::Vector3d n2 = FN.row(TT(i, j));
            normal_angle = atan2((n1.cross(n2)).norm(), n1.dot(n2));

            if ((c1 - c2).dot(n1 - n2) < 0)
                normal_angle = -normal_angle;
            highest_ang = std::max(highest_ang, normal_angle);
        }

        DA(i) = highest_ang;
    }
}

/**
 *
 * @param gradient_magnitude
 * @param candidate_vertices
 * @param candidate_face_ids
 * @param candidate_length
 * @param candidate_gs
 * @param V
 * @param F
 */
void compute_candidate_gs2(std::vector<Eigen::MatrixXd> &gradient_magnitude,
                           std::vector<std::vector<Eigen::MatrixXd>> &candidate_vertices,
                           std::vector<std::vector<int>> &candidate_face_ids,
                           std::vector<double> &candidate_length,
                           std::vector<double> &candidate_gs,
                           Eigen::MatrixXd &V, Eigen::MatrixXi F) {

    Eigen::MatrixXd DA;
    compute_face_dihedral_angle(V, F, DA);

    for (int i = 0; i < candidate_face_ids.size(); i++) {
        double tmp_score = 0.0;
        //tmp_score = std::max()
        candidate_gs[i] = tmp_score;
    }
}

/**
 *
 * @param isoline_vertices
 * @param isoline_field_id
 * @param isoline_lines
 * @param num_fields
 */
void create_edges_from_isolines(std::vector<std::vector<Eigen::MatrixXd>> &isoline_vertices,
                                std::vector<int> &isoline_field_id,
                                std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> &isoline_lines,
                                int num_fields) {
    std::vector<std::vector<Eigen::MatrixXd>> field_vertices1(num_fields);
    std::vector<std::vector<Eigen::MatrixXd>> field_vertices2(num_fields);

    for (int i = 0; i < isoline_vertices.size(); i++) {
        Eigen::MatrixXd last_vertex;
        for (int j = 0; j < isoline_vertices[i].size(); j++) {
            /// If first element: create edge to very last vertex
            if (j == 0)
                last_vertex = isoline_vertices[i].back();
            auto p1 = isoline_vertices[i][j];
            auto p2 = last_vertex;
            last_vertex = isoline_vertices[i][j];
            field_vertices1[isoline_field_id[i]].push_back(p1);
            field_vertices2[isoline_field_id[i]].push_back(p2);
        }
    }

    for (int i = 0; i < num_fields; i++) {
        Eigen::MatrixXd tmp1 = Eigen::MatrixXd::Zero(field_vertices1[i].size(), 3);
        Eigen::MatrixXd tmp2 = Eigen::MatrixXd::Zero(field_vertices1[i].size(), 3);
        for (int j = 0; j < field_vertices1[i].size(); j++) {
            tmp1.row(j) = field_vertices1[i][j];
            tmp2.row(j) = field_vertices2[i][j];
        }
        std::pair<Eigen::MatrixXd, Eigen::MatrixXd> tmp_pair(tmp1, tmp2);
        isoline_lines.push_back(tmp_pair);
    }
}

/**
 *
 * @param v
 * @param m
 */
void std_vector_to_eigen_matrix(std::vector<Eigen::MatrixXd> &v, Eigen::MatrixXd &m) {
    m.resize(v.size(), v[0].cols());
    for (int i = 0; i < v.size(); i++) {
        m.row(i) = v[i];
    }
}