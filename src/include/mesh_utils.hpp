#ifndef FUNCTIONAL_OBJECT_UNDERSTANDING_MESH_UTILS_HPP
#define FUNCTIONAL_OBJECT_UNDERSTANDING_MESH_UTILS_HPP

#include <igl/randperm.h>
#include <chrono>
#include <ctime>
#include <pcl/common/common.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/PolygonMesh.h>
#include <boost/graph/adjacency_list.hpp>
#include <pcl/io/obj_io.h>

/// *****  Type Definitions ***** ///

typedef pcl::PointXYZRGBA PointT;  // The point type used for input

struct MyVertexProperties {
    uint32_t id;
    float weight;
    bool is_concave;
    int region;
    pcl::PointNormal point;
    pcl::PointNormal principle_direction;
    pcl::PointNormal principle_direction2;

    void get_xyz(float xyz[]);

    MyVertexProperties() : id(0), weight(0.0), principle_direction(pcl::PointNormal()),
                           principle_direction2(pcl::PointNormal()), point(pcl::PointNormal()),
                           is_concave(false), region(-1) {}
};

struct MyEdgeProperties {
    uint32_t id;
    float normal_difference;
    float weight;
    bool is_internal;
    bool is_concave;
    bool cut;
    bool valid;
    int region;
    int concave_region;
    pcl::PointNormal principle_direction;
    pcl::PointNormal principle_direction2;

    MyEdgeProperties() : id(0), is_internal(false), is_concave(false), normal_difference(0), weight(0.0), region(-1),
                         valid(true), concave_region(-1), cut(false), principle_direction(pcl::PointNormal()),
                         principle_direction2(pcl::PointNormal()) {}
};

typedef boost::adjacency_list<boost::setS, boost::vecS, boost::undirectedS, MyVertexProperties, MyEdgeProperties> Graph;
typedef Graph::edge_descriptor edge_id_t;
typedef Graph::vertex_descriptor vertex_id_t;

class edge_predicate_c {
public:
    edge_predicate_c() : graph_m(0) {}

    edge_predicate_c(Graph &graph) : graph_m(&graph) {}

    bool operator()(const edge_id_t &edge_id) const {
        return (*graph_m)[edge_id].is_concave;
    }

private:
    Graph *graph_m;
};

class edge_predicate_cnCut {
public:
    edge_predicate_cnCut() : graph_m(0) {}

    edge_predicate_cnCut(Graph &graph) : graph_m(&graph) {}

    bool operator()(const edge_id_t &edge_id) const {
        return !((*graph_m)[edge_id].cut);
    }

private:
    Graph *graph_m;
};

class edge_predicate_cn {
public:
    edge_predicate_cn() : graph_m(0) {}

    edge_predicate_cn(Graph &graph) : graph_m(&graph) {}

    bool operator()(const edge_id_t &edge_id) const {
        return (*graph_m)[edge_id].is_concave;
    }

private:
    Graph *graph_m;
};

class vertex_predicate_c {
public:
    vertex_predicate_c() : graph_m(0) {}

    vertex_predicate_c(Graph &graph) : graph_m(&graph) {}

    bool operator()(const vertex_id_t &vertex_id) const {
        return (*graph_m)[vertex_id].is_concave;
    }

private:
    Graph *graph_m;
};

class vertex_predicate_region {
public:
    vertex_predicate_region() : graph_m(0) {}

    vertex_predicate_region(Graph &graph, int i) : graph_m(&graph), i_m(i) {}

    bool operator()(const vertex_id_t &vertex_id) const {
        return (*graph_m)[vertex_id].region == i_m;
    }

private:
    Graph *graph_m;
    int i_m;
};

class vertex_predicate_region_bl {
public:
    vertex_predicate_region_bl() : graph_m(0), blacklist_m() {}

    vertex_predicate_region_bl(Graph &graph, int i, std::set<boost::graph_traits<Graph>::vertex_descriptor> &blacklist)
            :
            graph_m(&graph), i_m(i), blacklist_m(&blacklist) {}

    bool operator()(const vertex_id_t &vertex_id) const {
        return (*graph_m)[vertex_id].region == i_m &&
               (*graph_m)[vertex_id].is_concave &&
               (*blacklist_m).find(vertex_id) == (*blacklist_m).end();
    }

private:
    Graph *graph_m;
    int i_m;
    std::set<boost::graph_traits<Graph>::vertex_descriptor> *blacklist_m;
};

class edge_predicate_region_bl {
public:
    edge_predicate_region_bl() : graph_m(0), blacklist_m() {}

    edge_predicate_region_bl(Graph &graph, int i, std::set<boost::graph_traits<Graph>::edge_descriptor> &blacklist) :
            graph_m(&graph), i_m(i), blacklist_m(&blacklist) {}

    bool operator()(const edge_id_t &edge_id) const {
        return (*graph_m)[edge_id].is_concave &&
               (*graph_m)[source(edge_id, (*graph_m))].region == i_m &&
               (*graph_m)[target(edge_id, (*graph_m))].region == i_m &&
               (*blacklist_m).find(edge_id) == (*blacklist_m).end();
    }

private:
    Graph *graph_m;
    int i_m;
    std::set<boost::graph_traits<Graph>::edge_descriptor> *blacklist_m;
};

class edge_predicate_region {
public:
    edge_predicate_region() : graph_m(0) {}

    edge_predicate_region(Graph &graph, int i) : graph_m(&graph), i_m(i) {}

    bool operator()(const edge_id_t &edge_id) const {
        return (*graph_m)[edge_id].region == i_m;
    }

private:
    Graph *graph_m;
    int i_m;
};

bool
connIsConvex(const Eigen::Vector3f &source_centroid, const Eigen::Vector3f &target_centroid,
             const Eigen::Vector3f &source_normal, const Eigen::Vector3f &target_normal, float &normal_angle);

void meshToGraph(pcl::PointCloud<pcl::PointNormal>::Ptr cloud, pcl::PolygonMeshPtr meshData, Graph &g,
                 std::vector<boost::graph_traits<Graph>::vertex_descriptor> &vertex_list,
                 std::vector<boost::graph_traits<Graph>::edge_descriptor> &edge_list);

void labelEdgesByVertexRegions(Graph &g);

void getConcaveEdgesFromGraph(Graph &g, std::vector<boost::graph_traits<Graph>::vertex_descriptor> &vertex_list,
                              std::set<boost::graph_traits<Graph>::edge_descriptor> &concave_edges, bool thinning);

void setConcaveEdges(Graph &g);

void mergeSegments(Graph &g, double min_part,
                   std::vector<boost::graph_traits<Graph>::vertex_descriptor> &vertex_list);

Eigen::VectorXi
planeLineIntersectionBatch(Eigen::Vector3f plane_normal, Eigen::Vector3f plane_point, Eigen::MatrixXf &pb1,
                           Eigen::MatrixXf &pb2,
                           Eigen::MatrixXf &w, Eigen::VectorXf &D, Eigen::VectorXf &N);

void getSegmentEdgesAndVertices(Graph &g, int curr_cluster,
                                std::vector<boost::graph_traits<Graph>::edge_descriptor> &cluster_edge_list,
                                std::vector<boost::graph_traits<Graph>::vertex_descriptor> &cluster_vertex_list,
                                std::vector<Graph::vertex_descriptor> &cluster_concave_vertices,
                                std::vector<boost::graph_traits<Graph>::edge_descriptor> &cluster_concave_edges,
                                Eigen::VectorXi &globalEdgeIndices);

void
MCPCSegment(Graph &g, int ransac_iterations, std::vector<boost::graph_traits<Graph>::vertex_descriptor> &vertex_list,
            std::vector<boost::graph_traits<Graph>::edge_descriptor> &edge_list,
            std::set<boost::graph_traits<Graph>::edge_descriptor> &concave_edges, double min_cut_score);

void mesh2EigenAndCloud(pcl::PolygonMeshPtr &mesh, Eigen::MatrixXd &V, Eigen::MatrixXd &F,
                        pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);

void mesh2EigenAndCloud(pcl::PolygonMeshPtr &mesh, Eigen::MatrixXf &V, Eigen::MatrixXf &F,
                        pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);

void smoothOutliers(Eigen::MatrixXf &PV1, int k);

void normalizeCurvature(Eigen::MatrixXf &PV1);

Eigen::MatrixXf normalizeCurvatureAndCopy(Eigen::MatrixXf &PV1);

void computeVertexNormals(pcl::PolygonMeshPtr &meshData, pcl::PointCloud<pcl::PointNormal>::Ptr &pc, bool inversion);

const char *helpmessage_cpc_mesh_adjacency = "\n\
-- MCPC - Mesh-CPC Segmentation -- :\n\
\n\
Syntax: %s input.obj  [Options] \n\
\n\
Output:\n\
  -o <outname> \n\
          Write segmented Mesh to disk (Type XYZL). If this option is specified without giving a name, the <outputname> defaults to <inputfilename>_out.pcd.\n\
          The content of the file can be changed with the -add and -bin flags\n\
  -novis  Disable visualization\n\
Output options:\n\
  -add    Instead of XYZL, append a label field to the input point cloud which holds the segmentation results (<input_cloud_type>+L)\n\
          If a label field already exists in the input point cloud it will be overwritten by the segmentation\n\
  -bin    Save a binary pcd-file instead of an ascii file \n\
  -so     Additionally write the colored supervoxel image to <outfilename>_svcloud.pcd\n\
  -white  Use white background instead of black \n\
  \n\
Segmentation Parameters: \n\
  -ct <concavity tolerance angle> - Angle threshold in degrees for concave edges to be treated as convex (default 10) \n\
  -st <smoothness threshold> - Invalidate steps. Value from the interval [0,1], where 0 is the strictest and 1 equals 'no smoothness check' (default 0.1)\n\
  -ec - Use extended (less local) convexity check\n\
  -sc - Use sanity criterion to invalidate singular connected patches\n\
  -smooth <mininmal segment size>  - Merge small segments which have fewer points than minimal segment size (default 0)\n\
  \n\
CPCSegmentation Parameters: \n\
  -cut <max_cuts>,<cutting_min_segments>,<min_cut_score> - Plane cutting parameters for splitting of segments\n\
       <max_cuts> - Perform cuts up to this recursion level. Cuts are performed in each segment separately (default 25)\n\
       <cutting_min_segments> - Minumum number of supervoxels in the segment to perform cutting (default 400).\n\
       <min_cut_score> - Minumum score a proposed cut needs to have for being cut (default 0.16)\n\
  -clocal - Use locally constrained cuts (recommended flag)\n\
  -cdir - Use directed weigths (recommended flag) \n\
  -cclean - Use clean cuts. \n\
            Flag set: Only split edges with supervoxels on opposite sites of the cutting-plane \n\
            Flag not set: Split all edges whose centroid is within the seed resolution distance to the cutting-plane\n\
  -citer <num_interations> - Sets the maximum number of iterations for the RANSAC algorithm (default 10000) \n\
  \n";

#endif //FUNCTIONAL_OBJECT_UNDERSTANDING_MESH_UTILS_HPP
