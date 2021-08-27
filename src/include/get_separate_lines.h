#ifndef GET_SEPARATE_LINES_H
#define GET_SEPARATE_LINES_H

#include <Eigen/Dense>
#include <iostream>

void color_mesh_by_isolines(Eigen::MatrixXi &E, Eigen::MatrixXi &F, std::vector<std::vector<int>> &segmentation_lines,
                            Eigen::MatrixXd &vertex_labels);

void get_separate_lines(Eigen::MatrixXi &isoE, std::vector<int> &isoF, std::vector<std::vector<int>> &contours,
                        std::vector<std::vector<int>> &contour_faces, Eigen::MatrixXd &isoV, Eigen::MatrixXd &FN,
                        std::vector<int> &isoI, std::vector<std::vector<int>> &contour_id);

#endif