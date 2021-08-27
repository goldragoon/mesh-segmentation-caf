// This file is part of libigl, a simple c++ geometry processing library.
//
// Copyright (C) 2017 Oded Stein <oded.stein@columbia.edu>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.


#include "custom_isoline.h"

#include <vector>
#include <array>
#include <iostream>
#include <fstream>

#include <igl/remove_duplicate_vertices.h>


void isolines(
        const Eigen::MatrixXd &V,
        const Eigen::MatrixXi &F,
        const Eigen::MatrixXd &z,
        const int n,
        Eigen::MatrixXd &isoV,
        Eigen::MatrixXd &isoE,
        std::vector<int> &isoF,
        std::vector<int> &isoI) {
    //Constants
    const int dim = V.cols();
    assert(dim == 2 || dim == 3);
    const int nVerts = V.rows();
    assert(z.rows() == nVerts &&
           "There must be as many function entries as vertices");
    const int nFaces = F.rows();
    const int np1 = n + 1;
    const double min = z.minCoeff(), max = z.maxCoeff();

    //Following http://www.alecjacobson.com/weblog/?p=2529
    typedef typename Eigen::MatrixXd::Scalar Scalar;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vec;
    Vec iso(np1);
    for (int i = 0; i < np1; ++i)
        iso(i) = Scalar(i) / Scalar(n) * (max - min) + min;

    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    std::array<Matrix, 3> t{{Matrix(nFaces, np1),
                                    Matrix(nFaces, np1), Matrix(nFaces, np1)}};
    for (int i = 0; i < nFaces; ++i) {
        for (int k = 0; k < 3; ++k) {
            const Scalar z1 = z(F(i, k)), z2 = z(F(i, (k + 1) % 3));
            for (int j = 0; j < np1; ++j) {
                t[k](i, j) = (iso(j) - z1) / (z2 - z1);
                if (t[k](i, j) < 0 || t[k](i, j) > 1)
                    t[k](i, j) = std::numeric_limits<Scalar>::quiet_NaN();
            }
        }
    }

    //std::vector<int> outFace;
    isoF.clear();
    isoI.clear();

    std::array<std::vector<int>, 3> Fij, Iij;
    for (int i = 0; i < nFaces; ++i) {
        for (int j = 0; j < np1; ++j) {
            for (int k = 0; k < 3; ++k) {
                const int kp1 = (k + 1) % 3, kp2 = (k + 2) % 3;
                if (std::isfinite(t[kp1](i, j)) && std::isfinite(t[kp2](i, j))) {
                    Fij[k].push_back(i);
                    Iij[k].push_back(j);
                    isoF.push_back(i);
                    isoI.push_back(j);
                }
            }
        }
    }

    const int K = Fij[0].size() + Fij[1].size() + Fij[2].size();
    isoV.resize(2 * K, dim);
    int b = 0;
    for (int k = 0; k < 3; ++k) {
        const int kp1 = (k + 1) % 3, kp2 = (k + 2) % 3;
        for (int i = 0; i < Fij[k].size(); ++i) {
            isoV.row(b + i) = (1. - t[kp1](Fij[k][i], Iij[k][i])) *
                              V.row(F(Fij[k][i], kp1)) +
                              t[kp1](Fij[k][i], Iij[k][i]) * V.row(F(Fij[k][i], kp2));
            isoV.row(K + b + i) = (1. - t[kp2](Fij[k][i], Iij[k][i])) *
                                  V.row(F(Fij[k][i], kp2)) +
                                  t[kp2](Fij[k][i], Iij[k][i]) * V.row(F(Fij[k][i], k));
        }
        b += Fij[k].size();
    }

    isoE.resize(K, 2);
    for (int i = 0; i < K; ++i)
        isoE.row(i) << i, K + i;

    int index = 0;
    for (int k = 0; k < 3; ++k) {
        for (int i = 0; i < Fij[k].size(); ++i) {
            isoF[index] = Fij[k][i];
            isoI[index] = Iij[k][i];
            index++;
        }
    }

    //Remove double entries
    typedef typename Eigen::MatrixXd::Scalar LScalar;
    typedef typename Eigen::MatrixXd::Scalar LInt;
    typedef Eigen::Matrix<LInt, Eigen::Dynamic, 1> LIVec;
    typedef Eigen::Matrix<LScalar, Eigen::Dynamic, Eigen::Dynamic> LMat;
    typedef Eigen::Matrix<LInt, Eigen::Dynamic, Eigen::Dynamic> LIMat;
    LIVec dummy1, dummy2;

    //igl::remove_duplicate_vertices(LMat(isoV), LIMat(isoE),
    //                               1.0e-7, isoV, dummy1, dummy2, isoE);

    //std::ofstream myfile;
    //myfile.open ("isoV.txt");
    //myfile << isoV;
    //myfile.close();

    igl::remove_duplicate_vertices(LMat(isoV), LIMat(isoE),
                                   1.0e-7, isoV, dummy1, dummy2, isoE);

}



//#ifdef IGL_STATIC_LIBRARY
// Explicit template instantiation
//template void igl::isolines<Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, int const, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > &, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > &);
//#endif

