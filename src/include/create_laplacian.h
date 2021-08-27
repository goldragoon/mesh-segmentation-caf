#ifndef CREATE_LAPLACIAN_H
#define CREATE_LAPLACIAN_H

void compute_all_features(Eigen::MatrixXd &V, Eigen::MatrixXi &F,
                          Eigen::MatrixXi &E, Eigen::MatrixXd &N, 
                            std::vector<std::vector<int>> &VV,
                        Eigen::MatrixXi &IF, Eigen::MatrixXi &OV,
                          Eigen::MatrixXd &FN, Eigen::MatrixXd &DA, Eigen::MatrixXd &D,
                          Eigen::MatrixXd &G, Eigen::MatrixXd &dblA);

void compute_laplacian_harmonic(Eigen::MatrixXd &V, Eigen::MatrixXi &F, Eigen::MatrixXi &E, Eigen::MatrixXd &L,
                                Eigen::MatrixXd &N,
                                double beta, double sigma);

void compute_laplacian(Eigen::MatrixXd &V, Eigen::MatrixXi &F, Eigen::MatrixXi &E, Eigen::MatrixXd &G,
                       Eigen::MatrixXd &N, Eigen::MatrixXd &L, Eigen::MatrixXi &vertex_is_concave, double beta,
                       double eps, double sigma);

bool compute_segmentation_field(Eigen::MatrixXd &V, Eigen::MatrixXi &F, Eigen::MatrixXd &FN, Eigen::MatrixXi &E,
                                Eigen::MatrixXd &L, int point1, int point2, int field_id,
                                std::vector<Eigen::MatrixXd> &gradient_magnitude,
                                std::vector<std::vector<double>> &isoline_neighbor_length,
                                std::vector<std::vector<Eigen::MatrixXd>> &isoline_vertices,
                                std::vector<std::vector<int>> &isoline_face_ids,
                                std::vector<double> &isoline_length,
                                std::vector<double> &isoline_gs,
                                std::vector<int> &isoline_field_id,
                                std::vector<Eigen::Triplet<double>> &basic_tripletList,
                                std::vector<Eigen::MatrixXd> &fields);

void compute_candidate_svs(std::vector<double> &candidate_length,
                           std::vector<std::vector<double>> &candidate_neighbor_length,
                           std::vector<double> &candidate_svs);

void compute_candidate_gs(std::vector<Eigen::MatrixXd> &gradient_magnitude,
                          std::vector<std::vector<Eigen::MatrixXd>> &candidate_vertices,
                          std::vector<std::vector<int>> &candidate_face_ids,
                          std::vector<double> &candidate_length,
                          std::vector<double> &candidate_gs,
                          Eigen::MatrixXd &g_hat);

void create_edges_from_isolines(std::vector<std::vector<Eigen::MatrixXd>> &isoline_vertices,
                                std::vector<int> &isoline_field_id,
                                std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> &isoline_lines,
                                int num_fields);

void std_vector_to_eigen_matrix(std::vector<Eigen::MatrixXd> &v, Eigen::MatrixXd &m);

#endif