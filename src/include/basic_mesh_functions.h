#ifndef BASIC_MESH_FUNCTIONS_H
#define BASIC_MESH_FUNCTIONS_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <map>
#include <set>

typedef Eigen::Triplet<double> T;
inline bool isEqual(double x, double y);

/**
 *
 * @param V
 * @param F
 * @param N
 */
void compute_vertex_normals(Eigen::MatrixXd &V, Eigen::MatrixXi &F, Eigen::MatrixXd &N);

/**
 * Compute opposite vertices
 * @param E
 * @param F
 * @param OV
 */
void compute_opposite_vertices(Eigen::MatrixXi &E, Eigen::MatrixXi &F, Eigen::MatrixXi &OV);

/**
 * Compute incident faces
 * @param E
 * @param F
 * @param IF
 */
void compute_incident_faces(Eigen::MatrixXi &E, Eigen::MatrixXi &F, Eigen::MatrixXi &IF);

/**
 * Compute face normals
 * @param V
 * @param F
 * @param FC
 * @param FN
 */
void compute_face_normals(Eigen::MatrixXd &V, Eigen::MatrixXi &F, Eigen::MatrixXd &FC, Eigen::MatrixXd &FN);

/**
 * Compute dihedral angle
 * @param V
 * @param F
 * @param E
 * @param IF
 * @param OV
 * @param FN
 * @param DA
 */
void compute_dihedral_angle(Eigen::MatrixXd &V, Eigen::MatrixXi &F, Eigen::MatrixXi &E, Eigen::MatrixXi &IF,
                            Eigen::MatrixXi &OV, Eigen::MatrixXd &FN, Eigen::MatrixXd &DA);

/**
 * Compute distance
 * @param V
 * @param E
 * @param D
 */
void compute_distance(Eigen::MatrixXd &V, Eigen::MatrixXi &E, Eigen::MatrixXd &D);

/**
 * Obtain starting point
 * @param F
 * @param V
 * @param L
 * @param index
 * @param A
 * @return
 */
int solve_poisson_equation_least_squares(Eigen::MatrixXi &F, Eigen::MatrixXd &V, Eigen::MatrixXd &L, int index,
                       std::vector<std::vector<int>> &A);

void solve_poisson_equation_least_squares_fast(
                            Eigen::MatrixXi &F, Eigen::MatrixXd &V, int index,
                            std::vector<std::vector<int>> &A, Eigen::MatrixXi &E,
                            std::vector<Eigen::Triplet<double>> & L_triplets,
                            Eigen::MatrixXd &x);

/**
 * Get maximum geodesic distance
 * @param extreme_point_set
 * @param V
 * @return
 */
double max_geodesic_dist(std::set<int> &extreme_point_set, Eigen::MatrixXd &V);

bool in_proximity_to(Eigen::MatrixXd p1, std::set<int> &extreme_point_set, Eigen::MatrixXd &V, double dist_prox);

std::vector<int> get_extreme_points(Eigen::MatrixXi &F, Eigen::MatrixXd &V, std::vector<T> &L_triplets, 
                                    int index_given, Eigen::MatrixXi &E);

/**
 * Obtain segmentation field
 * @param F
 * @param V
 * @param L
 * @param E
 * @param index1
 * @param index2
 * @param isoV
 * @param isoE
 * @param z
 * @param isoF
 * @param isoI
 * @param basic_tripletList
 */
void get_segmentation_field(Eigen::MatrixXd &L,
                            int index1, int index2, Eigen::MatrixXd &z, 
                            std::vector<T> &basic_tripletList);

double getAngle3D(const Eigen::Vector3d &v1, const Eigen::Vector3d &v2, const bool in_degree);

void get_isoline_gradient_scores(Eigen::MatrixXd &isoV, Eigen::MatrixXi &isoE, std::vector<std::vector<int>> &contours,
                                 std::vector<std::vector<int>> &contour_faces, Eigen::MatrixXd &z,
                                 std::vector<double> &score);

void get_isoline_length(Eigen::MatrixXd &isoV, Eigen::MatrixXi &isoE, std::vector<std::vector<int>> &contours,
                        std::vector<std::vector<int>> &contour_faces, std::vector<double> &length);

/**
 * Add meshes
 * @param V
 * @param F
 * @param V2
 * @param F2
 */
void add_mesh(Eigen::MatrixXd &V, Eigen::MatrixXi &F, Eigen::MatrixXd &V2, Eigen::MatrixXi &F2);

#endif