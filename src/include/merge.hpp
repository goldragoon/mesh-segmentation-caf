#ifndef FUNCTIONAL_OBJECT_UNDERSTANDING_MERGE_HPP
#define FUNCTIONAL_OBJECT_UNDERSTANDING_MERGE_HPP

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/filtered_graph.hpp>
#include <igl/edges.h>

void mergeSegments(Eigen::MatrixXd &V, Eigen::MatrixXi &F, Eigen::MatrixXd &label, double min_part);

void mergeSegments_old(Eigen::MatrixXd &V, Eigen::MatrixXi &F, Eigen::MatrixXd &label, double min_part);

#endif //FUNCTIONAL_OBJECT_UNDERSTANDING_MERGE_HPP
