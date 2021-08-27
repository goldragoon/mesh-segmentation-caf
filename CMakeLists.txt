cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(mesh-segmentation-caf)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Resolving libraries
list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    SET_PROPERTY(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    SET_PROPERTY(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

option(LIBIGL_WITH_CGAL              "Use CGAL"           ON)
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use IMGUI"          ON)
option(LIBIGL_WITH_PNG               "Use PNG"            ON)
option(LIBIGL_WITH_VIEWER            "Use OpenGL viewer"  ON)
option(LIBIGL_WITH_IMGUI             "Use imgui menu"     ON)
option(LIBIGL_WITH_ANTTWEAKBAR       "Use AntTweakBar"    OFF)
option(LIBIGL_WITH_COMISO            "Use CoMiso"         OFF)
option(LIBIGL_WITH_CORK              "Use Cork"           OFF)
option(LIBIGL_WITH_EMBREE            "Use Embree"         OFF)
option(LIBIGL_WITH_LIM               "Use LIM"            OFF)
option(LIBIGL_WITH_MATLAB            "Use Matlab"         OFF)
option(LIBIGL_WITH_MOSEK             "Use MOSEK"          OFF)
option(LIBIGL_WITH_PYTHON            "Use Python"         OFF)
option(LIBIGL_WITH_TETGEN            "Use Tetgen"         OFF)
option(LIBIGL_WITH_TRIANGLE          "Use Triangle"       OFF)
option(LIBIGL_WITH_XML               "Use XML"            OFF)

find_package(Boost REQUIRED COMPONENTS program_options filesystem)
message(STATUS "BOOST Libraries : " ${BOOST_LIBRARIES})

find_package(Eigen3 REQUIRED)
find_package(suitesparse CONFIG REQUIRED)
find_package(CGAL REQUIRED COMPONENTS Core)
set(CGAL_DO_NOT_WARN_ABOUT_CMAKE_BUILD_TYPE TRUE)

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)
include(libigl)

# Copy data into build directory
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data DESTINATION ${CMAKE_BINARY_DIR})

include( ${CGAL_USE_FILE} )
set(CGAL_DONT_OVERRIDE_CMAKE_FLAGS TRUE CACHE BOOL "Force CGAL to maintain CMAKE flags")
include_directories(
        ${INCLUDE_DIRECTORIES}
	    ${SUITESPARSE_INCLUDE_DIRS}       #${SPQR_INCLUDES}
        ${EIGEN3_INCLUDE_DIR}
        ${LIBIGL_INCLUDE_DIR}
        ${CGAL_INCLUDE_DIR}
        ${OPENGL_INCLUDE_DIRS}
        ${GLEW_INCLUDE_DIRS}
        ${GLFW3_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include)
        
#add_library(merge STATIC src/include/merge.cpp)
#add_library(sdf STATIC src/include/sdf.cpp)
#add_library(mesh_utils STATIC src/include/mesh_utils.cpp)
#target_link_libraries(merge PRIVATE igl::core sdf)
#target_link_libraries(sdf PRIVATE igl::core)
#target_link_libraries(mesh_utils PRIVATE ${PCL_LIBRARIES} igl::core)

add_library(custom_isoline STATIC src/include/custom_isoline.cpp)
target_link_libraries(custom_isoline PRIVATE igl::core)

add_library(split_mesh STATIC src/include/split_mesh.cpp)
target_link_libraries(split_mesh PRIVATE igl::core ${EIGEN3_LIBRARY_DIR})

add_library(basic_mesh_functions STATIC src/include/basic_mesh_functions.cpp)
add_library(get_separate_lines STATIC src/include/get_separate_lines.cpp)

add_library(create_laplacian STATIC src/include/create_laplacian.cpp)
target_include_directories(create_laplacian PUBLIC ${EIGEN3_LIBRARY_DIR})
target_link_libraries(create_laplacian PUBLIC igl::core basic_mesh_functions get_separate_lines)

target_include_directories(basic_mesh_functions  PUBLIC  ${SUITESPARSE_INCLUDE_DIRS}/suitesparse ${EIGEN3_LIBRARY_DIR})
target_link_libraries(basic_mesh_functions PUBLIC 
        ${SUITESPARSE_LIBRARIES}
        igl::core 
        custom_isoline)

add_executable(main src/conc_aware/conc_aware.cpp) # main executable
target_link_libraries(main PUBLIC
        Boost::program_options
        Boost::filesystem
        igl::core
        igl::opengl
        igl::opengl_glfw
        igl::opengl_glfw_imgui
        create_laplacian
        basic_mesh_functions
        get_separate_lines
        split_mesh)