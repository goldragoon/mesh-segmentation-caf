// This file is part of libigl, a simple c++ geometry processing library.
//
// Copyright (C) 2017 Oded Stein <oded.stein@columbia.edu>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.


#ifndef ISOLINES_H
#define ISOLINES_H
//#include "igl/include/igl/igl_inline.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>


//namespace igl
//{
// Constructs isolines for a function z given on a mesh (V,F)
//
//
// Inputs:
//   V  #V by dim list of mesh vertex positions
//   F  #F by 3 list of mesh faces (must be triangles)
//   z  #V by 1 list of function values evaluated at vertices
//   n  the number of desired isolines
// Outputs:
//   isoV  #isoV by dim list of isoline vertex positions
//   isoE  #isoE by 2 list of isoline edge positions
//
//

void isolines(
        const Eigen::MatrixXd &V,
        const Eigen::MatrixXi &F,
        const Eigen::MatrixXd &z,
        const int n,
        Eigen::MatrixXd &isoV,
        Eigen::MatrixXd &isoE,
        std::vector<int> &outFace,
        std::vector<int> &isoI);
//}

//#ifndef IGL_STATIC_LIBRARY
//#  include "custom_isoline.cpp"
//#endif

#endif
