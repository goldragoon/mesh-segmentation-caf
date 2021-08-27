#ifndef FUNCTIONAL_OBJECT_UNDERSTANDING_SDF_HPP
#define FUNCTIONAL_OBJECT_UNDERSTANDING_SDF_HPP

#include <Eigen/Dense>

void new_sdf(Eigen::MatrixXd &V, Eigen::MatrixXi &F, Eigen::MatrixXd &face_sdf, Eigen::MatrixXd &vertex_sdf,
             Eigen::MatrixXd &skeleton_vertices, Eigen::MatrixXd &SE1, Eigen::MatrixXd &SE2,
             Eigen::MatrixXd &skeleton_diam);

void sdf(Eigen::MatrixXd &V, Eigen::MatrixXi &F, Eigen::MatrixXd &face_sdf, Eigen::MatrixXd &vertex_sdf,
         Eigen::MatrixXd &skeleton_vertices);

#endif //FUNCTIONAL_OBJECT_UNDERSTANDING_SDF_HPP
