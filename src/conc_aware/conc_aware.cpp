#include <iostream>
#include <random>
#include <memory>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <igl/read_triangle_mesh.h>
#include <igl/edges.h>
#include <igl/gaussian_curvature.h>
#include <igl/grad.h>
#include <igl/decimate.h>
#include <igl/unique.h>
#include <igl/colormap.h>
#include <igl/writeOFF.h>
#include <igl/is_vertex_manifold.h>

#include <igl/opengl/glfw/background_window.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>

#include "custom_isoline.h"
#include "basic_mesh_functions.h"
#include "get_separate_lines.h"
#include "split_mesh.h"
#include "create_laplacian.h"

enum vis_mode{ vis_mesh_color = 0, vis_field, vis_field_gradient, vis_concavity};
enum isoline_vis_mode{ vis_none, vis_local_gs, vis_svs, vis_score, vis_id, vis_gs};

namespace po = boost::program_options;

static void ShowHelpMarker(const char* desc)
{
  ImGui::TextDisabled("(?)");
  if (ImGui::IsItemHovered())
  {
    ImGui::BeginTooltip();
    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
    ImGui::TextUnformatted(desc);
    ImGui::PopTextWrapPos();
    ImGui::EndTooltip();
  }
}

class CustomMenu : public igl::opengl::glfw::imgui::ImGuiMenu
{

public:
    std::vector<std::string> visualization_choices = { "Mesh Color", "Field", "Field Gradient", "Concavity" };
    int visualization_choice_current = vis_mesh_color;
  
    std::vector<std::string> field_choices;
    int field_choice;

    std::vector<std::string> label_choices;
    int label_choice;

    float beta, epsilon, zeta;

    bool redraw;
    bool run_segmentation = false;
    bool segmentation_finished = false;

  //bool draw_iso;

  virtual void draw_custom_window() override
  {
    // Define next window position + size
    ImGui::SetNextWindowPos(ImVec2(180.f * menu_scaling(), 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(300, 200), ImGuiCond_FirstUseEver);
    ImGui::Begin("parameters", nullptr, ImGuiWindowFlags_NoSavedSettings);

    // Numerical Input
    ImGui::InputFloat("Beta", &beta, 0.0f, 1.0f);
    ImGui::SameLine(); ShowHelpMarker("Beta is a weighting coefficient for concave connections used for creating the "
                                        "Graph Laplacian weight Matrix. The smaller Beta, the less will concave regions "
                                        "allow propagation. Thus, a higher Beta value will lead to more uniformly "
                                        "sampled isolines, whereas a low Beta value will result in isolines clustered "
                                        "at concave regions.\n");

    ImGui::InputFloat("Epsilon", &epsilon, 0.0f, 1.0f);
    ImGui::SameLine(); ShowHelpMarker("Small constant to avoid zero division.\n");

    ImGui::InputFloat("Zeta", &zeta, 0.0f, 1.0f);
    ImGui::SameLine(); ShowHelpMarker("Threshold value to decide whether connection of two vertices is convex or "
                                        "concave, based on the normal difference.\n");

    if(segmentation_finished){
        if (ImGui::Combo("Field", &field_choice, field_choices)) redraw = true;
        if (ImGui::Combo("Visualization", &visualization_choice_current, visualization_choices)) redraw = true;
    }

    //if (ImGui::Combo("Label", &label_choice, label_choices)) redraw = true;
    //ImGui::Checkbox("Draw isolines", &draw_iso);

    if (ImGui::Button("Segment")){
      std::cout << "Segmentation Requested by GUI\n";
      run_segmentation = true;
      segmentation_finished = false;
      redraw = true;
    }

    ImGui::End();
  }

};

int main (int argc, char ** argv){

  int index = 0;
  std::string input_file = "";
  Eigen::MatrixXd V;    // Vertices
  Eigen::MatrixXi F;    // Faces
  Eigen::MatrixXi E;    // Edges
  Eigen::MatrixXd N;    // Normals
  std::vector<std::vector<int>> VV;  // list of lists containing at index i the adjacent vertices of vertex set V
  Eigen::MatrixXd dblA; // Area
  Eigen::MatrixXi IF;   // Incident Faces of each e_i in E
  Eigen::MatrixXi OV;   // Opposite Vertices
  Eigen::MatrixXd FN;   // Face Normals
  Eigen::MatrixXd DA;   // Dihedral Angle
  Eigen::MatrixXd D;    // Distance
  Eigen::MatrixXd HL;   // Harmonic Laplacian
  Eigen::MatrixXd L;    // Laplacian
  Eigen::MatrixXd G;    // Gaussian Curvature
  Eigen::MatrixXd P1, P2; // Isolines
  std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> isoline_lines;
  std::vector<Eigen::MatrixXd> isoline_lines_id;
  std::vector<Eigen::MatrixXd> isoline_lines_color;
  Eigen::MatrixXd isoline_colors; // Isoline Colors
  Eigen::MatrixXd g_hat;
  std::vector<int> applied_isolines;
  std::vector<std::vector<double>> isoline_neighbor_length;
  std::vector<std::vector<Eigen::MatrixXd>> isoline_vertices;
  std::vector<std::vector<int>> isoline_face_ids;
  std::vector<double> isoline_length;
  std::vector<double> isoline_svs;
  std::vector<double> isoline_local_gs;
  std::vector<double> isoline_gs;
  std::vector<double> isoline_score;
  std::vector<int> isoline_field_id;
  std::vector<int> candidate_isoline_id;
  std::vector<std::vector<double>> candidate_neighbor_length;
  std::vector<std::vector<Eigen::MatrixXd>> candidate_vertices;
  std::vector<std::vector<int>> candidate_face_ids;
  std::vector<double> candidate_length;
  std::vector<double> candidate_svs;
  std::vector<double> candidate_gs;
  std::vector<double> candidate_score;
  std::vector<int> candidate_field_id;
  std::vector<int> extreme_points;
  int num_fields;
  std::vector<std::set<int>> isoline_face_set;
  std::vector<std::vector<int>> field_to_isoline_ids;
  Eigen::MatrixXi vertex_is_concave;
  bool redraw = false;
  bool draw_isoline = false;
  vis_mode visualization_mode = vis_mesh_color;
  isoline_vis_mode isoline_mode = vis_none;
  std::vector<Eigen::MatrixXd> gradient_magnitude;
  std::vector<Eigen::MatrixXd> fields;
  Eigen::MatrixXd C;
  std::vector<int> extreme_p1;
  std::vector<int> extreme_p2;
  bool indicate_candidates = false;
  std::vector<std::vector<int>> edge_indices;
  bool show_candidates = true;
  bool show_local_gs_score = true;
  double beta, eps, zeta;

  Eigen::MatrixXd vertex_labels;
  Eigen::MatrixXd mesh_label_colored;

  boost::filesystem::path full_path(boost::filesystem::current_path());
  
  std::cout << "Current path is : " << full_path << std::endl;
  full_path /= "../data/models/spot_triangulated_quarter_fair.obj";
  std::cout << "Default data file is : " <<  full_path<< std::endl;

  /// Parse Command Line Arguments
  po::options_description desc("MainOptions");
  desc.add_options()
      ("help,h", "Print help messages")
      ("nogui", "Suppress GUI")
      ("input_file,f",
          //
          po::value<std::string>()->default_value(full_path.string()),
     "Input Mesh File, can be of formats OFF, PLY, STL or OBJ. Needs to be manifold")
    ("beta,b",
     po::value<double>()->default_value(0.75),
     "Concave weight factor")
    ("eps,e",
     po::value<double>()->default_value(1e-6),
     "Small constant to prevent zero division")
    ("zeta,s",
     po::value<double>()->default_value(0.001),
     "Concavity tolerance")
    ("output_file,o",
        po::value<std::string>()->default_value("output.obj"),
    "Output file path");

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
  }
  catch (po::error &e) { // Invalid options
    std::cerr << "Error: " << e.what() << std::endl << std::endl;
    std::cout << "usage : " << std::endl << desc << std::endl;
    return 0;
  }
  try{
      po::notify(vm);
  }catch(std::exception& e)
  {
      std::cerr << "Error: " << e.what() << std::endl;
      return false;
  }
  
  if (vm.count("help")) // print usage
  {
    std::cout << "usage : " << std::endl << desc << std::endl;
    return 0;
  }

  input_file = vm["input_file"].as<std::string>();
  std::string output_file = vm["output_file"].as<std::string>();
  beta = vm["beta"].as<double>();
  eps = vm["eps"].as<double>();
  zeta = vm["zeta"].as<double>();
  

  if (boost::filesystem::is_regular_file(input_file)) {
      igl::read_triangle_mesh(input_file, V, F);   // Read mesh 
  }else{
      std::cerr << "File does not exist!" << std::endl;
      return -1;
  }

  Eigen::MatrixXi B;
  igl::is_vertex_manifold(F, B);
  if (B.minCoeff() == 0) std::cerr << ">> The loaded mesh is not manifold.\n";
  
  auto f = [&](){
    int lap_weighting = 0;
    auto start = std::chrono::steady_clock::now();

    /// Compute Laplacian and Features
    compute_all_features(V, F, E, N, VV, IF, OV, FN, DA, D, G, dblA);

      auto end = std::chrono::steady_clock::now();
      std::cout << "COMPUTED FEATURES. Elapsed time in seconds : "
                << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                << " msec\n";

    std::cout << ">> Computing Laplacian... e=" << eps << std::endl;
    compute_laplacian(V, F, E, G, N, L, vertex_is_concave, beta, eps, zeta);
    //compute_laplacian_harmonic(V, F, E, L, N, beta, zeta);
      end = std::chrono::steady_clock::now();
      std::cout << "COMPUTED LAPLACIAN. Elapsed time in seconds : "
                << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                << " msec\n";
    
      std::vector<T> L_triplets;
      L_triplets.reserve(E.rows() * 2 + V.rows());
      for (int i = 0; i < L.rows(); i++) {
          for (int j = 0; j < L.cols(); j++) {
              if (FP_ZERO != fpclassify(L(i, j))) { // To check L(i, j) != 0.0
                  L_triplets.push_back(T(i, j, L(i, j)));
              }
          }
      }

    std::cout << ">> Finding extreme Points...\n";
    extreme_points = get_extreme_points(F, V, L_triplets, index, E);

      end = std::chrono::steady_clock::now();
      std::cout << extreme_points.size() << " EXTREME POINTS."  << " Elapsed time in seconds : "
                << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                << " msec\n";


    ///Compute Segmentation Fields and choose Isolines as candidates
    std::cout << ">> compute_segmentation_field" << std::endl;
    num_fields = 0;
    int num_isos = 0;

    // for every pair of exteme points (u, v) \in U, compute a corresponding segmetnation field Phi_uv
    for (int point1 : extreme_points) {
      for (int point2 : extreme_points) {
        if (point1 < point2) {

          std::vector<std::vector<double>> isoline_neighbor_length_tmp;
          std::vector<std::vector<Eigen::MatrixXd>> isoline_vertices_tmp;
          std::vector<std::vector<int>> isoline_face_ids_tmp;
          std::vector<double> isoline_length_tmp;
          std::vector<double> isoline_svs_tmp;
          std::vector<double> isoline_gs_tmp;
          std::vector<int> isoline_field_id_tmp;
          bool ret = compute_segmentation_field(V, F, FN, E, L, point1, point2, num_fields, gradient_magnitude,
                                     isoline_neighbor_length_tmp,
                                     isoline_vertices_tmp, isoline_face_ids_tmp,
                                     isoline_length_tmp, isoline_gs_tmp, isoline_field_id_tmp, L_triplets,
                                     fields);

          extreme_p1.push_back(point1);
          extreme_p2.push_back(point2);
          num_fields++;
          
          field_to_isoline_ids.push_back(std::vector<int>());
          for (int i = 0; i < isoline_vertices_tmp.size(); i++){
            isoline_neighbor_length.push_back(isoline_neighbor_length_tmp[i]);
            isoline_vertices.push_back(isoline_vertices_tmp[i]);
            isoline_face_ids.push_back(isoline_face_ids_tmp[i]);
            isoline_length.push_back(isoline_length_tmp[i]);
            isoline_gs.push_back(isoline_gs_tmp[i]);
            isoline_field_id.push_back(isoline_field_id_tmp[i]);
            field_to_isoline_ids.back().push_back(num_isos);
            num_isos++;
          }

          if (ret) { //[Unsolved] why some of the poisson equation solving is failed?
              std::cout << "Something is wrong on compute_segmentation_field" << std::endl;
              continue;
          }
        }
      }
    }
      end = std::chrono::steady_clock::now();
      std::cout << "FIELDS COMPUTED. Elapsed time in seconds : "
                << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                << " sec\n";


    /// Visualize Extreme Points
    redraw = true;

    std::vector<std::set<int>> isoline_ids_sharing_this_face_id(F.rows(), std::set<int>());
      for (int i = 0; i < isoline_vertices.size(); i++){
          //std::set<int> face_set(isoline_face_ids[i].begin(), isoline_face_ids[i].end());
          //isoline_face_set.push_back(face_set);
          for (auto facetile: isoline_face_ids[i]) {
              isoline_ids_sharing_this_face_id[facetile].insert(i);
          }
      }

    ///Compute face sets for caching
    for (int i = 0; i < isoline_vertices.size(); i++){
      std::set<int> face_set(isoline_face_ids[i].begin(), isoline_face_ids[i].end());
      isoline_face_set.push_back(face_set);
    }

    num_isos = 0;
    /// Filter out bad isolines
    for (int i = 0; i < isoline_vertices.size(); i++){

      bool local_max = false;
      /// Check boundary conditions:
      double neigh_left_1_gs = 0;
      double neigh_left_2_gs = 0;
      double neigh_right_1_gs = 0;
      double neigh_right_2_gs = 0;
      bool neigh_left_1_same_id = false;
      bool neigh_left_2_same_id = false;
      bool neigh_right_1_same_id = false;
      bool neigh_right_2_same_id = false;
      if (i > 0){
        neigh_left_1_same_id = isoline_field_id[i] == isoline_field_id[i - 1];
        if (neigh_left_1_same_id)
          neigh_left_1_gs = isoline_gs[i - 1];
      }

      if (i > 1){
        neigh_left_2_same_id = isoline_field_id[i] == isoline_field_id[i - 2];
        if (neigh_left_2_same_id)
          neigh_left_2_gs = isoline_gs[i - 2];
      }

      if (i < isoline_vertices.size() - 1){
        neigh_right_1_same_id = isoline_field_id[i] == isoline_field_id[i + 1];
        if (neigh_right_1_same_id)
          neigh_right_1_gs = isoline_gs[i + 1];
      }

      if (i < isoline_vertices.size() - 2){
        neigh_right_2_same_id = isoline_field_id[i] == isoline_field_id[i + 2];
        if (neigh_right_2_same_id)
          neigh_right_2_gs = isoline_gs[i + 2];
      }

      if (neigh_left_2_gs <= isoline_gs[i] && neigh_left_1_gs <= isoline_gs[i] &&
          neigh_right_1_gs <= isoline_gs[i] && neigh_right_2_gs <= isoline_gs[i]){
        local_max = true;
      }

      /// Remove long isolines that share faces with short isolines
      bool is_longer_and_has_common_face = false;
      /*for (int j = 0; j < isoline_vertices.size(); j++){
        if (i != j){
          std::set<int> common;
          auto face_set1 = isoline_face_set[i];
          auto face_set2 = isoline_face_set[j];

          std::set_intersection (face_set1.begin(), face_set1.end(), face_set2.begin(), face_set2.end(),
                                 std::inserter(common,common.begin()));
          if (common.size() > 0){
            /// Check if length is similar to other isoline
            double l1 = isoline_length[i];
            double l2 = isoline_length[j];
            if (l1 > l2 * 1.5)
              is_longer_and_has_common_face = true;
          }
        }
      }*/

        ////// NEW ROUTINE
        for (auto face : isoline_face_set[i]){
            for (auto line: isoline_ids_sharing_this_face_id[face]){
                if (isoline_length[i] > isoline_length[line] * 1.5)
                    is_longer_and_has_common_face = true;
            }
        }

      /// Final Assignment
      if (!is_longer_and_has_common_face && local_max){
        candidate_vertices.push_back(isoline_vertices[i]);
        candidate_face_ids.push_back(isoline_face_ids[i]);
        candidate_gs.push_back(isoline_gs[i]);
        candidate_length.push_back(isoline_length[i]);
        candidate_neighbor_length.push_back(isoline_neighbor_length[i]);
        candidate_isoline_id.push_back(num_isos);
      }
      num_isos++;
    }

    isoline_local_gs = isoline_gs;

      end = std::chrono::steady_clock::now();
      std::cout << "ISOLINES COMPUTED. Elapsed time in seconds : "
                << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                << " msec\n";

    /// Compute the Shape Variance Score
    //compute_candidate_svs_new(candidate_length, candidate_svs, candidate_face_ids, F, V);
    //compute_candidate_svs_by_sdf(candidate_length, candidate_vertices, candidate_svs, candidate_face_ids, F, V);

    /// Compute the Gradient Magnitude Score
    compute_candidate_gs(gradient_magnitude, candidate_vertices, candidate_face_ids, candidate_length, candidate_gs, g_hat);

    /// Calculate Final Score
    candidate_score.resize(candidate_vertices.size());
    for (int i = 0; i < candidate_score.size(); i++){
        candidate_score[i] = candidate_gs[i];// *candidate_svs[i];
    }

    create_edges_from_isolines(isoline_vertices, isoline_field_id, isoline_lines, num_fields);

    /// Create Edges Indices from Isolines
    int num_edge = 0;
    int offset = 0;
    edge_indices.resize(isoline_vertices.size());
    for (int i = 0; i < isoline_vertices.size(); i++) {
        offset += num_edge;
        num_edge = isoline_vertices[i].size();
        for (int j = offset; j < offset + num_edge; j++)
            edge_indices[i].push_back(j);

    }
    return; /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// Find Valid cuts
    valid_cuts(V, N, F, dblA, IF, E, candidate_face_ids, candidate_score, applied_isolines);

    /// Create Edges from Isolines
    num_edge = 0;
    for (int i = 0; i < applied_isolines.size(); i++){
      num_edge += candidate_vertices[applied_isolines[i]].size();
    }
    P1.resize(num_edge,3);
    P2.resize(num_edge,3);
    isoline_colors = Eigen::MatrixXd::Ones(num_edge, 3);
    int ind = 0;
    for (int i = 0; i < applied_isolines.size(); i++){
      Eigen::MatrixXd last_vertex;
      for (int j = 0; j < candidate_vertices[applied_isolines[i]].size(); j++){
        /// If first element: create edge to very last vertex
        if (j == 0)
          last_vertex = candidate_vertices[applied_isolines[i]].back();
        auto p1 = candidate_vertices[applied_isolines[i]][j];
        auto p2 = last_vertex;
        last_vertex = candidate_vertices[applied_isolines[i]][j];
        P1.row(ind) = p1;
        P2.row(ind) = p2;
        ind++;
      }
    }


    std::vector<std::vector<int>> segmentation_lines;
    for (int i = 0; i < applied_isolines.size(); i++){
      segmentation_lines.push_back(candidate_face_ids[applied_isolines[i]]);
    }

    color_mesh_by_isolines(E, F, segmentation_lines, vertex_labels);

  };

    // Initialize the igl viewer
    igl::opengl::glfw::Viewer viewer;
    if (vm.count("nogui")) {
        GLFWwindow *window;
        igl::opengl::glfw::background_window(window);
        glfwSetWindowSize(window, 200, 200);
        viewer.window = window;
    }

    viewer.core().is_animating = true;

    // Attach a custom menu
    CustomMenu menu;
    viewer.plugins.push_back(&menu);


    menu.label_choices = {"None", "Local Gradient Score", "SVS", "Gradient Score", "Score", "ID"};
    menu.label_choice = 0;

    menu.zeta = zeta;
    menu.beta = beta;
    menu.epsilon = eps;

    menu.run_segmentation = false;

    //menu.draw_iso = draw_isoline;
    menu.redraw = true;

    viewer.data().set_mesh(V, F);

    /// Pre Draw Callback
    viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer &)
    {
      if(menu.redraw) {
        if (menu.run_segmentation){
          std::cout << "Running Segmentation...\n";
          menu.run_segmentation = false;
          beta = menu.beta;
          eps = menu.epsilon;
          zeta = menu.zeta;

          //draw_isoline = menu.draw_iso;
          f();

          std::vector<std::string> items;
          for (int i = 0; i < num_fields; i++){
            items.push_back({ "Field " + std::to_string(i + 1)});
          }
          items.push_back("Normalized");
          gradient_magnitude.push_back(g_hat);
          menu.field_choices = items;

          menu.segmentation_finished = true;

            if (vm.count("nogui")) {
                viewer.launch_shut();
            }

        }

        if (menu.segmentation_finished){
          viewer.data().clear();
          viewer.data().set_mesh(V, F);
          switch (menu.visualization_choice_current) {
            case vis_mesh_color: ///Standard Mode
              if (vertex_labels.rows() == V.rows()) viewer.data().set_colors(mesh_label_colored);
              break;

            case vis_field: //Field Mode
                igl::colormap(igl::ColorMapType::COLOR_MAP_TYPE_JET, fields[menu.field_choice], true, C);
                viewer.data().set_colors(C);
              break;

            case vis_field_gradient: //Field Gradient Mode
                igl::colormap(igl::ColorMapType::COLOR_MAP_TYPE_PARULA,
                              gradient_magnitude[menu.field_choice],
                              true, C);
                viewer.data().set_colors(C);
              break;

            case vis_concavity: ///Concavity Mode
              igl::colormap(igl::ColorMapType::COLOR_MAP_TYPE_PARULA, vertex_is_concave, true, C);
              viewer.data().set_colors(C);
              break;
          }
          redraw = false;

          ///Draw extreme Points
          if (menu.field_choice < menu.field_choices.size() - 1 && menu.field_choices.size() > 0) {
            viewer.data().add_points(V.row(extreme_p1[menu.field_choice]), Eigen::RowVector3d(1.0, 0.0, 0.0));
            viewer.data().add_points(V.row(extreme_p2[menu.field_choice]), Eigen::RowVector3d(0.0, 0.0, 1.0));
          } else {
            for (int i = 0; i < extreme_points.size(); i++) {
              viewer.data().add_points(V.row(extreme_points[i]), Eigen::RowVector3d(1.0, 1.0, 1.0));
            }
          }

          if (draw_isoline) {
            /// Final Cuts, when Normalized Field is selected
              std::cout << "draw  isoline enabled : " << isoline_lines.size() <<std::endl;
            if (isoline_lines.size() == menu.field_choice) {
              viewer.data().add_edges(P1, P2, isoline_colors);
             std::cout << "draw  Final Cuts" << std::endl;
              if (menu.label_choice == 5) {
                for (int i = 0; i < applied_isolines.size(); i++) {
                  viewer.data().add_label(candidate_vertices[applied_isolines[i]][0].row(0), std::to_string(i));
                }
              } else if (menu.label_choice == 4) {
                for (int i = 0; i < applied_isolines.size(); i++) {
                  viewer.data().add_label(candidate_vertices[applied_isolines[i]][0].row(0),
                                          std::to_string(candidate_score[applied_isolines[i]]));
                }
              } else if (menu.label_choice == 2) {
                for (int i = 0; i < applied_isolines.size(); i++) {
                  viewer.data().add_label(candidate_vertices[applied_isolines[i]][0].row(0),
                                          std::to_string(candidate_svs[applied_isolines[i]]));
                }
              } else if (menu.label_choice == 1) {
                for (int i = 0; i < applied_isolines.size(); i++) {
                  viewer.data().add_label(candidate_vertices[applied_isolines[i]][0].row(0),
                                          std::to_string(
                                            isoline_local_gs[candidate_isoline_id[applied_isolines[i]]]));
                }
              } else if (menu.label_choice == 3) {
                for (int i = 0; i < applied_isolines.size(); i++) {
                  viewer.data().add_label(candidate_vertices[applied_isolines[i]][0].row(0),
                                          std::to_string(candidate_gs[applied_isolines[i]]));
                }
              }
            }

            /// All Isolines, when some Segmentation Field is selected
            if (isoline_lines.size() > menu.field_choice) {
                std::cout << "draw  All Isolines" << std::endl;
              for (int i = 0; i < field_to_isoline_ids[menu.field_choice].size(); i++) {
                int iso_id = field_to_isoline_ids[menu.field_choice][i];
                /// Get Candidate ID if isoline is indeed a candidate
                auto it = std::find(candidate_isoline_id.begin(), candidate_isoline_id.end(), iso_id);
                int index = -1;
                if (it == candidate_isoline_id.end()) {
                  /// Isoline is not a candidate
                } else {
                  index = std::distance(candidate_isoline_id.begin(), it);
                }

                /// Add each isoline individually
                Eigen::MatrixXd tmp_p1 = Eigen::MatrixXd::Zero(edge_indices[iso_id].size(), 3);
                Eigen::MatrixXd tmp_p2 = Eigen::MatrixXd::Zero(edge_indices[iso_id].size(), 3);
                Eigen::MatrixXd last_vertex;
                for (int j = 0; j < edge_indices[iso_id].size(); j++) {
                  if (j == 0)
                    last_vertex = isoline_vertices[iso_id].back();
                  auto p1 = isoline_vertices[iso_id][j];
                  auto p2 = last_vertex;
                  last_vertex = isoline_vertices[iso_id][j];
                  tmp_p1.row(j) = p1;
                  tmp_p2.row(j) = p2;
                }
                Eigen::MatrixXd tmp_c = Eigen::MatrixXd::Ones(edge_indices[iso_id].size(), 3);
                if (std::find(candidate_isoline_id.begin(), candidate_isoline_id.end(), iso_id) !=
                    candidate_isoline_id.end() && show_candidates) {
                  for (int j = 0; j < tmp_c.rows(); j++) {
                    tmp_c.row(j) << 1.0, 0.0, 1.0;
                  }
                }
                //viewer.data().add_edges(tmp_p1, tmp_p2, tmp_c);

                /// Add Labels
                if (menu.label_choice == 1) {
                  viewer.data().add_label(isoline_vertices[iso_id][0].row(0),
                                          std::to_string(isoline_local_gs[iso_id]));
                } else if (menu.label_choice == 5) {
                  if (index != -1)
                    viewer.data().add_label(isoline_vertices[iso_id][0].row(0), std::to_string(index));
                } else if (menu.label_choice == 2) {
                  if (index != -1)
                    viewer.data().add_label(isoline_vertices[iso_id][0].row(0), std::to_string(candidate_svs[index]));
                } else if (menu.label_choice == 4) {
                  if (index != -1)
                    viewer.data().add_label(isoline_vertices[iso_id][0].row(0),
                                            std::to_string(candidate_score[index]));
                } else if (menu.label_choice == 3) {
                  if (index != -1)
                    viewer.data().add_label(isoline_vertices[iso_id][0].row(0), std::to_string(candidate_gs[index]));
                }
              }
            }
          }
        }

      }
      return false;
    };

    /// start viewer
    if (vm.count("nogui")) menu.run_segmentation = true;
    
    viewer.launch(true, false, "Mesh Segmentation with Concavity-Aware Fields", 1024,768);

  return 0;
}